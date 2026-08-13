"""Integración del grafo de conocimiento en la recuperación (§8.5).

§8.5 describe cuatro pasos para combinar el grafo con los resultados vectoriales,
y este módulo implementa los tres primeros; el cuarto -la fusión- lo aplica
`retrieval.search`:

  1. Se identifican las entidades mencionadas en la consulta mediante el mismo
     componente NER utilizado durante la construcción del grafo.
  2. Se consulta el grafo para recuperar los fragmentos vinculados a dichas
     entidades y a las entidades directamente relacionadas con ellas (vecinos de
     primer orden).
  3. Los fragmentos recuperados se incorporan al conjunto de candidatos y se les
     asigna una puntuación basada en el número de relaciones relevantes
     encontradas.

────────────────────────────────────────────────────────────────────────────────
DE DONDE SALEN LAS ENTIDADES DE LA CONSULTA

El reconocimiento se ejecuta **una sola vez, fuera de la consulta**, y su
resultado se entrega junto al grafo en `consultas_entidades.json`. Es la misma
decisión que con el índice y con el propio grafo: los tres se construyen aguas
arriba y se entregan construidos. Ejecutar el modelo de NER en tiempo de consulta
obligaria a descargar sus pesos en el entorno de evaluacion, que es coste y
riesgo de instalación sin ninguna contrapartida, dado que las 50 consultas son
fijas y el resultado del reconocimiento es determinista.

Si una consulta no figura en ese archivo -por ejemplo, si se ejecuta el generador
sobre otro conjunto-, se recurre a un enlazado por coincidencia literal contra
los nombres canónicos y las variantes de los nodos del grafo. Es determinista, no
emplea ningún modelo y garantiza que el módulo nunca falle por falta de cache.

────────────────────────────────────────────────────────────────────────────────
POR QUE LA EVIDENCIA SE PONDERA POR LO INFORMATIVA QUE ES LA ENTIDAD

Medido sobre las 50 consultas y los 3.375 nodos del grafo: la coincidencia media
es de 1,2 entidades por consulta, y las que más se repiten son nodos de grado muy
alto. `inteligencia artificial` aparece en diez consultas del fenómeno 1 y tiene
99 aristas; recuperar sus vecinos aporta cientos de fragmentos «sobre IA» a una
consulta sobre IA, es decir, ningún poder discriminante. En el extremo opuesto,
consultas del fenómeno 3 enlazan entidades como `Chocó` o `Norte de Santander`,
con menos de diez aristas, que sí señalan un conjunto pequeño y pertinente.

Por eso el peso de una entidad es inversamente proporcional al logaritmo de su
grado. Es la misma intuición que la frecuencia inversa de documento: una entidad
que se relaciona con todo no informa de nada. Sin esta ponderación, los nodos
concentradores dominarían la señal del grafo en la mayoría de las consultas.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from graph.canonical import canonizar, cargar_alias, clave_canonica

# Cuánto pesa un fragmento alcanzado a través de un vecino de primer orden
# frente a uno que menciona directamente una entidad de la consulta. §8.5 pide
# incluir ambos; el vecino es evidencia indirecta y se pondera como tal.
PESO_VECINO = 0.4

# Cuántos fragmentos como máximo aporta el grafo al conjunto de candidatos. El
# tope existe porque una consulta que enlace un nodo concentrador podría arrastrar
# miles: el grafo debe complementar la búsqueda densa, no sustituirla.
MAX_CANDIDATOS_GRAFO = 60

# Evidencia acumulada a partir de la cual un fragmento se considera plenamente
# respaldado por el grafo. Corresponde, en la práctica, a dos entidades
# específicas -de grado bajo- que coinciden en el mismo fragmento, o a una
# entidad específica que además aporta la relación que la conecta con otra.
SATURACION = 1.0


def _localizar_grafo() -> Path:
    """Localiza la carpeta del grafo ascendiendo desde este archivo."""
    aqui = Path(__file__).resolve()
    for carpeta in aqui.parents:
        for candidata in (carpeta / "base_vectorial", carpeta / "entrega" / "base_vectorial"):
            if (candidata / "grafo" / "grafo.graphml").is_file():
                return candidata / "grafo"
    return aqui.parents[2] / "base_vectorial" / "grafo"


CARPETA_GRAFO = _localizar_grafo()
GRAFO = CARPETA_GRAFO / "grafo.graphml"
CHUNK_INDEX = CARPETA_GRAFO / "chunk_index.json"
ENTIDADES_CONSULTA = CARPETA_GRAFO / "consultas_entidades.json"

_NS = "{http://graphml.graphdrawing.org/xmlns}"


class Grafo:
    """El grafo de conocimiento, cargado para consultarlo en recuperación.

    Se lee el XML directamente y no a través de una biblioteca de grafos: solo
    hacen falta las listas de adyacencia y los `chunk_ids`, y evitarlo ahorra una
    dependencia en el entorno de evaluación.
    """

    def __init__(self, ruta: Path = GRAFO, chunk_index: Path = CHUNK_INDEX) -> None:
        import xml.etree.ElementTree as ET

        raiz = ET.parse(ruta).getroot()
        claves = {k.get("id"): k.get("attr.name") for k in raiz.findall(f"{_NS}key")}
        grafo = raiz.find(f"{_NS}graph")

        self.nombre: Dict[str, str] = {}
        self.chunks_de_nodo: Dict[str, List[str]] = {}
        self.grado: Dict[str, int] = {}
        # Clave canónica -> identificador de nodo. Incluye las variantes
        # superficiales que el nodo registró durante la construcción, de modo que
        # el enlazado por coincidencia literal encuentre también «NATO» y no solo
        # «OTAN».
        self.por_clave: Dict[str, List[str]] = {}

        for nodo in grafo.findall(f"{_NS}node"):
            nid = nodo.get("id")
            datos = {claves.get(d.get("key")): (d.text or "")
                     for d in nodo.findall(f"{_NS}data")}
            self.nombre[nid] = datos.get("nombre", nid)
            self.chunks_de_nodo[nid] = datos.get("chunk_ids", "").split()
            self.grado[nid] = 0
            for variante in {datos.get("nombre", ""), *datos.get("variantes", "").split("|")}:
                clave = clave_canonica(variante)
                if clave:
                    self.por_clave.setdefault(clave, []).append(nid)

        self.vecinos: Dict[str, List[str]] = {}
        # Fragmentos que respaldan cada relación, por par de nodos.
        self.chunks_de_arista: Dict[Tuple[str, str], List[str]] = {}
        for arista in grafo.findall(f"{_NS}edge"):
            o, d = arista.get("source"), arista.get("target")
            datos = {claves.get(x.get("key")): (x.text or "")
                     for x in arista.findall(f"{_NS}data")}
            self.vecinos.setdefault(o, []).append(d)
            self.vecinos.setdefault(d, []).append(o)
            self.grado[o] = self.grado.get(o, 0) + 1
            self.grado[d] = self.grado.get(d, 0) + 1
            chunks = datos.get("chunk_ids", "").split() or \
                [datos.get("chunk_id", "")]
            self.chunks_de_arista.setdefault((o, d), []).extend(c for c in chunks if c)

        self.faiss_id: Dict[str, int] = json.loads(
            chunk_index.read_text(encoding="utf-8")) if chunk_index.is_file() else {}

    def __len__(self) -> int:
        return len(self.nombre)

    def peso_entidad(self, nid: str) -> float:
        """Cuánto informa una entidad, inversamente a su grado.

        Un nodo de grado 1 pesa 1,0; uno de grado 99 pesa alrededor de 0,15. La
        forma logarítmica evita que un solo nodo concentrador anule a los demás
        sin llegar a excluirlo.
        """
        return 1.0 / math.log2(2 + self.grado.get(nid, 0))

    def enlazar(self, entidades: Sequence[Tuple[str, str]],
                alias: Optional[Dict[str, str]] = None) -> List[str]:
        """De entidades `(nombre, tipo)` a identificadores de nodo del grafo.

        Se canoniza con el mismo módulo que se empleó al construir el grafo, de
        modo que las variantes translingües resuelven al mismo nodo.
        """
        if alias is None:
            alias = cargar_alias()
        encontrados: List[str] = []
        for nombre, tipo in entidades:
            canonico, tipo_canonico, clave = canonizar(nombre, tipo, alias)
            directo = f"{tipo_canonico}:{clave}"
            if directo in self.nombre:
                encontrados.append(directo)
                continue
            # El tipo que asigne el reconocedor a la consulta puede no coincidir
            # con el que se le asignó en el corpus; si la clave existe con otro
            # tipo, se acepta igualmente.
            for nid in self.por_clave.get(clave, []):
                encontrados.append(nid)
        # Se eliminan repetidos conservando el orden.
        vistos, salida = set(), []
        for nid in encontrados:
            if nid not in vistos:
                vistos.add(nid)
                salida.append(nid)
        return salida

    def evidencia(self, semillas: Sequence[str],
                  maximo: int = MAX_CANDIDATOS_GRAFO) -> Dict[str, float]:
        """Fragmentos vinculados a las entidades y a sus vecinos, con su puntuación.

        Implementa los pasos 2 y 3 de §8.5. La puntuación de un fragmento es la
        suma de los pesos de las entidades que lo respaldan, de modo que un
        fragmento que aparece por varias entidades de la consulta a la vez puntúa
        más que uno que aparece por una sola. Eso es lo que §8.5 llama «el número
        de relaciones relevantes encontradas», ponderado por lo informativa que
        es cada relación.

        Devuelve `{chunk_id: puntuacion}` acotado al intervalo [0, 1].

        ⚠️ La puntuación **no se normaliza dividiendo por el máximo de la
        consulta**, y la diferencia es la que hace útil el módulo. Con esa
        normalización, el mejor fragmento de cada consulta recibiría siempre
        evidencia 1,0, incluso cuando la única entidad enlazada es un nodo
        concentrador que no discrimina nada: medido, veinte de las cincuenta
        consultas enlazan `inteligencia artificial`, de grado 99, y saturarían.
        Acotar el valor absoluto conserva la escala: un fragmento respaldado solo
        por esa entidad se queda en 0,15, mientras que uno respaldado por dos
        entidades específicas -del orden de cinco aristas cada una- llega a 1,0.
        """
        acumulado: Dict[str, float] = {}

        def sumar(chunks: Sequence[str], peso: float) -> None:
            for cid in chunks:
                if cid in self.faiss_id:
                    acumulado[cid] = acumulado.get(cid, 0.0) + peso

        for nid in semillas:
            peso = self.peso_entidad(nid)
            # Menciones directas de la entidad.
            sumar(self.chunks_de_nodo.get(nid, []), peso)
            # Vecinos de primer orden y las relaciones que los unen.
            for vecino in set(self.vecinos.get(nid, [])):
                par = (nid, vecino) if (nid, vecino) in self.chunks_de_arista \
                    else (vecino, nid)
                sumar(self.chunks_de_arista.get(par, []), peso)
                sumar(self.chunks_de_nodo.get(vecino, []),
                      peso * PESO_VECINO * self.peso_entidad(vecino))

        if not acumulado:
            return {}

        mejores = sorted(acumulado.items(), key=lambda kv: (-kv[1], kv[0]))[:maximo]
        return {cid: min(1.0, v / SATURACION) for cid, v in mejores}


def cargar_entidades_consulta(ruta: Path = ENTIDADES_CONSULTA) -> Dict[str, list]:
    """Lee las entidades reconocidas en cada consulta, si el archivo existe.

    Formato: `{"q001": [["Estados Unidos", "country"], ...], ...}`.
    """
    if not ruta.is_file():
        return {}
    return json.loads(ruta.read_text(encoding="utf-8"))


def entidades_por_coincidencia(consulta: str, grafo: Grafo) -> List[Tuple[str, str]]:
    """Enlazado de respaldo, sin modelo: nombres del grafo presentes en la consulta.

    Recorre los nombres canónicos del grafo y conserva los que aparecen como
    secuencia completa de palabras en la consulta. Es determinista y no emplea
    ningún modelo, de modo que no incurre en §8.3. Solo se usa cuando la consulta
    no figura en el archivo de entidades precalculadas.
    """
    texto = " " + " ".join(clave_canonica(consulta).split()) + " "
    salida: List[Tuple[str, str]] = []
    for clave, nodos in grafo.por_clave.items():
        if len(clave) >= 4 and f" {clave} " in texto:
            for nid in nodos:
                tipo = nid.split(":", 1)[0]
                salida.append((grafo.nombre.get(nid, clave), tipo))
    return salida
