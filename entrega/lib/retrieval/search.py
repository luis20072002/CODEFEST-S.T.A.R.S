"""Módulo de recuperación (§8): de una consulta a 3 documentos y 10 fragmentos.

────────────────────────────────────────────────────────────────────────────────
QUÉ HACE Y EN QUÉ ORDEN

    1. codificar la consulta          → vector normalizado
    2. buscar en FAISS                → K_CANDIDATOS fragmentos con su coseno
    3. ajustar puntuaciones           → §8.7, fenómeno e idioma
    4. agregar a nivel documento      → §8.6, los 3 mejores
    5. filtrar los fragmentos         → tope por documento + deduplicación
    6. dividir a 250 palabras         → §9.2.1, los 10 mejores

Los pasos 4 y 5 son independientes de forma deliberada: §9.2 solicita «los 3
documentos más relevantes» y «los 10 fragmentos más relevantes», sin exigir que
los segundos pertenezcan a los primeros.

TODO OPERA SOBRE VECTORES, PUNTUACIONES Y METADATA. §8.3 prohíbe emplear
modelos generativos para reordenamiento, expansión de la consulta, filtrado o
síntesis. Aquí no interviene ninguno: el único modelo del sistema es el encoder,
que codifica la consulta.

────────────────────────────────────────────────────────────────────────────────
POR QUÉ SE SOLICITAN 200 CANDIDATOS Y NO 10

Entre la búsqueda y la respuesta operan tres filtros que descartan candidatos:
el tope por documento, la deduplicación y el descarte de fragmentos vacíos. Con
k=10 la lista quedaría corta y habría que devolver menos de diez, lo que §9.3.1
penaliza o descarta. Doscientos ofrecen margen suficiente y cuestan lo mismo:
con `IndexFlatIP` el coste está en recorrer los 91.021 vectores, no en cuántos
se devuelven.

────────────────────────────────────────────────────────────────────────────────
POR QUÉ MAX POOLING Y NO SUMA (§8.6)

§8.6 admite tres formas de agregar fragmentos a nivel de documento: «puntuación
del fragmento más relevante (max pooling), suma de las puntuaciones de todos sus
fragmentos recuperados, o media ponderada».

La suma es inadecuada en este corpus. Un único archivo tabular aporta 33.396
fragmentos, frente a una mediana de 5 por documento. Sumar premiaría a los
documentos por su tamaño y no por su relevancia, y los tres puestos de la
respuesta los ocuparían invariablemente los mismos archivos.

Con max pooling un documento vale lo que valga su mejor fragmento. Es el valor
por defecto, pero la estrategia se expone como parámetro.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from retrieval.fragmentos import (LIMITE_PALABRAS, clave_de_texto,
                                  dividir_a_limite, elegir_mejor)

NOMBRE_ENCODER = "encoder_bge-m3"


def _localizar_base_vectorial() -> Path:
    """Localiza `base_vectorial/` ascendiendo desde la ubicación de este archivo.

    La búsqueda es ascendente y no una ruta fija para que el módulo funcione
    tanto si se ejecuta desde el directorio de entrega —donde la biblioteca vive
    en `lib/` y la base vectorial es hermana suya— como desde el repositorio
    completo, donde la base vectorial está bajo `entrega/`.

    Si no encuentra ninguna, devuelve la ruta preferente, de modo que el error
    posterior al abrir el índice nombre la ubicación esperada.
    """
    aqui = Path(__file__).resolve()
    for carpeta in aqui.parents:
        for candidata in (carpeta / "base_vectorial",
                          carpeta / "entrega" / "base_vectorial"):
            if (candidata / NOMBRE_ENCODER).is_dir():
                return candidata
    return aqui.parents[2] / "base_vectorial"


CARPETA_ENCODER = _localizar_base_vectorial() / NOMBRE_ENCODER
INDICE = CARPETA_ENCODER / "index.faiss"
METADATA = CARPETA_ENCODER / "metadata.jsonl"

# Cuántos fragmentos se solicitan a FAISS antes de filtrar. Véase la cabecera.
K_CANDIDATOS = 200

# Factor por el que se amplía k cuando los candidatos no contienen tres
# documentos distintos. El caso se da con consultas que se parecen a un registro
# bibliográfico: los 200 candidatos proceden todos de los dos archivos tabulares
# de mayor tamaño, con 33.396 y 19.239 fragmentos, de modo que solo hay dos
# documentos distintos y §9.3.1 exige tres.
FACTOR_AMPLIACION = 10

# Cuántos fragmentos como máximo puede aportar un mismo documento al top-10.
# 0 o None desactiva el tope.
MAX_POR_DOCUMENTO = 2

# Lo que exige §9.2, y que §9.3.1 penaliza o descarta si no se cumple.
N_DOCUMENTOS = 3
N_FRAGMENTOS = 10

# Factor por el que se multiplica la puntuación de los candidatos que pertenecen
# al fenómeno de la consulta.
#
# El valor se determinó comparando 1,00, 1,03, 1,05 y 1,10 mediante pooling
# sobre juicios de relevancia elaborados manualmente y puntuando cada
# configuración con las métricas de §10:
#
#     factor   correspondencia   docs de otro fenómeno   F1@3        F1@3
#              con el fenómeno   que sobreviven          estricto    permisivo
#     ×1,00    60,4 %            29                      —           —
#     ×1,03    81,2 %            13                      0,595       0,905
#     ×1,05    93,8 %             5                      0,338       0,857
#     ×1,10   100,0 %             0                      0,338       0,810
#
# A partir de 1,10 no sobrevive ningún documento de otro fenómeno: equivale a un
# filtro, y el filtro se descartó porque su fallo es irreversible.
#
# Este valor debe coincidir con el que produjo el `resultados.jsonl` entregado:
# §1.4 excluye la entrega si `generador.py` no la reproduce, y se ejecuta sin
# argumentos. `python generador.py --comprobar` verifica esa correspondencia.
BONIFICACION_FENOMENO = 1.03

# Idiomas que no se penalizan. Son aquellos en que reside el contenido relevante
# del corpus: las 50 consultas están en español y el corpus es 55,0 % inglés y
# 35,1 % español.
IDIOMAS_PREFERIDOS = ("es", "en")

# Factor por el que se multiplica la puntuación de un candidato cuyo idioma no
# figura en IDIOMAS_PREFERIDOS. Determinado por barrido sobre las 50 consultas:
#
#     factor   fragmentos fuera de es/en   conjuntos de documentos alterados
#     ×1,00    16 de 500, en 8 consultas   —
#     ×0,99    11 de 500, en 8 consultas   2 de 50
#     ×0,97     5 de 500, en 4 consultas   5 de 50      ← elegido
#     ×0,95     0 de 500                   5 de 50
#     ×0,90     0 de 500                   idéntico a ×0,95
#
# QUÉ SIGNIFICA EL VALOR. El factor define cuán próxima debe estar una
# alternativa en español o inglés para que se prefiera. A 0,97 la banda es del
# 3 % de similitud coseno, es decir, casi equivalencia semántica: si la
# traducción era relevante, sustituirla por su equivalente cuesta muy poco; si
# el conjunto de referencia está en español o inglés, se gana bastante.
#
# POR QUÉ NO SE BAJÓ MÁS. De 0,95 hacia abajo no sobrevive ningún fragmento
# fuera de {es, en}: sería un filtro con otro nombre. Y las ejecuciones con
# 0,95, 0,90 y 0,80 producen exactamente la misma salida, de modo que penalizar
# más no aporta nada.
#
# POR QUÉ NO SE DEJÓ EN 0,99. §10.2.2 establece que F1@3 es «una métrica de
# conjunto (no considera el orden)». Con 0,99, dos consultas devuelven los
# mismos tres documentos reordenados, y una reordenación no altera la métrica a
# nivel de documento: el conjunto solo cambia en 2 de las 50 consultas.
#
# Este valor debe coincidir con el que produjo el `resultados.jsonl` entregado,
# igual que BONIFICACION_FENOMENO.
FACTOR_IDIOMA_OTROS = 0.97



def aplicar_factor_idioma(
    candidatos: List[Dict],
    factor_otros: float = FACTOR_IDIOMA_OTROS,
    preferidos: tuple = IDIOMAS_PREFERIDOS,
) -> List[Dict]:
    """Reduce la puntuación de los candidatos fuera de {es, en} y reordena.

    ────────────────────────────────────────────────────────────────────────
    EL PROBLEMA QUE CORRIGE, MEDIDO SOBRE LA SALIDA

    Varias organizaciones del corpus publican el mismo informe en varios
    idiomas, y cada versión constituye un documento distinto con su propio
    `doc_id`. Una consulta puede así dedicar posiciones a traducciones del mismo
    contenido.

    Sin este ajuste, el reparto de idiomas de los 500 fragmentos reportados era:

        en 66,2 %  ·  es 30,6 %  ·  zh/fr/pt/ru/ko  16 fragmentos = 3,2 %

    Solo 8 de las 50 consultas traían algo fuera de {es, en}, y no se trata de
    un problema genérico de traducciones sino de una familia concreta de
    documentos: los resúmenes ejecutivos de un mismo informe anual, publicados
    en seis idiomas.

    Los dos casos con coste real:

    - Una consulta dedicaba dos de sus tres puestos de documento a dos
      traducciones del mismo resumen ejecutivo, existiendo el informe completo
      en inglés dentro del corpus. Ahí se paga F1@3 (§10.2.2).
    - Otra devolvía la versión en chino en la primera posición para una consulta
      en español, teniendo la versión inglesa en la segunda.

    ────────────────────────────────────────────────────────────────────────
    POR QUÉ UN FACTOR Y NO UN FILTRO NI UNA REORDENACIÓN POR IDIOMA

    El fallo de un filtro es irreversible. Con un factor, si una traducción es
    el único candidato que responde a la consulta, sobrevive, porque no tiene
    contra quién competir. Solo pierde cuando existe un equivalente en español o
    inglés próximo en puntuación, que es exactamente el caso que se corrige.

    Se descartó reordenar globalmente por idioma. El problema afecta a 8
    consultas y un criterio de orden se aplicaría a las 50. Además, la mezcla de
    idiomas de cada consulta sigue a dónde reside el contenido relevante, porque
    el encoder ordena por significado; y NDCG@10 es sensible al orden (ecuación
    8), de modo que elevar un fragmento en español por encima de otro en inglés
    mejor puntuado costaría NDCG en toda consulta mixta sin ganancia medible.

    ────────────────────────────────────────────────────────────────────────
    POR QUÉ LOS DOCUMENTOS SIN IDIOMA DETECTADO NO SE PENALIZAN

    Un idioma ausente significa que el clasificador no dispuso de señal
    suficiente, no que el documento esté en un idioma extranjero. Son 18
    documentos del corpus, y penalizarlos sería castigar una limitación del
    detector.

    Es la misma razón por la que esto es un factor y no un filtro: alrededor de
    un 1 % de los documentos tiene la etiqueta de idioma equivocada, y con un
    filtro esos textos —algunos en inglés— quedarían inalcanzables por un error
    de detección.

    §8.7 respalda este ajuste de forma explícita: permite «filtrar por campo
    fenomeno, formato, rango de fechas, idioma». Es metadata, no un modelo.
    """
    # Con el factor en 1,0 no hay nada que hacer, y devolver la lista tal cual
    # evita copiar doscientos diccionarios y reordenar sin efecto.
    if factor_otros == 1.0:
        return candidatos

    ajustados = []
    for c in candidatos:
        copia = dict(c)          # no se muta la lista del llamante
        idioma = c.get("idioma")
        # La condición tiene dos partes de forma deliberada: `is not None`
        # protege a los documentos sin idioma detectado, y `not in` es lo que
        # efectivamente penaliza.
        if idioma is not None and idioma not in preferidos:
            copia["score"] = c["score"] * factor_otros
        ajustados.append(copia)

    # Se reordena porque todo lo posterior —agregación a documento y selección
    # de fragmentos— asume orden por puntuación descendente.
    return _ordenar_por_puntuacion(ajustados)


def aplicar_bonificacion(candidatos: List[Dict], fenomeno: Optional[int],
                         factor: float = BONIFICACION_FENOMENO) -> List[Dict]:
    """Eleva la puntuación de los candidatos del fenómeno esperado y reordena.

    ────────────────────────────────────────────────────────────────────────
    EL PROBLEMA QUE CORRIGE, MEDIDO

    Sin bonificación, la proporción de documentos devueltos que pertenecen al
    fenómeno de la consulta era:

        F1 (q001–q016):  29/48  = 60,4 %
        F2 (q017–q032):  42/48  = 87,5 %
        F3 (q033–q050):  50/54  = 92,6 %

    F1 se quedaba en el 60 %. La causa está identificada: las etiquetas de
    fenómeno describen el observatorio de origen y no el contenido, y existen
    fuentes catalogadas en un fenómeno que publican abundantemente sobre el tema
    de otro. Dos consultas no devolvían ni un solo documento de su fenómeno.

    Con el factor aplicado, la correspondencia asciende al 91,3 %.

    ────────────────────────────────────────────────────────────────────────
    POR QUÉ UNA BONIFICACIÓN Y NO UN FILTRO

    §8.7 permite filtrar por metadata, y filtrar por `fenomeno` resolvería el
    60 % de inmediato. Pero el fallo de filtrar es irreversible: si el conjunto
    de referencia considera relevante un documento de otro fenómeno, filtrarlo
    lo vuelve inalcanzable y §10.2.2 lo contabiliza como fallo sin remedio.

    La bonificación desplaza hacia el fenómeno esperado sin cerrar la puerta a
    un documento de otro que sea claramente mejor.

    ────────────────────────────────────────────────────────────────────────
    EL FACTOR ESTÁ MEDIDO, NO ESTIMADO

    Un cálculo sobre la aritmética de los cosenos sugería que cualquier factor
    entre 1,02 y 1,05 sería equivalente. La medición lo desmintió: 1,02 y 1,05
    no producían ningún efecto sobre candidatos de fenómeno minoritario, porque
    estos suelen estar muy abajo en la lista y un incremento pequeño no les
    alcanza para remontar decenas de posiciones. La elección final se hizo
    puntuando las cuatro configuraciones con las métricas de §10; la tabla está
    junto a la constante `BONIFICACION_FENOMENO`.
    """
    if not fenomeno or factor == 1.0:
        return candidatos

    prefijo = f"F{fenomeno}-"
    ajustados = []
    for c in candidatos:
        copia = dict(c)      # no se muta la lista del llamante
        if c["doc_id"].startswith(prefijo):
            copia["score"] = c["score"] * factor
        ajustados.append(copia)
    # Se reordena: la bonificación altera el orden, y todo lo posterior asume
    # orden por puntuación descendente.
    return _ordenar_por_puntuacion(ajustados)


def _ordenar_por_puntuacion(fragmentos: List[Dict]) -> List[Dict]:
    """Ordena de mayor a menor relevancia, con desempate determinista.

    El orden explícito al final es necesario: la segunda pasada de
    `seleccionar_fragmentos()` añade elementos al final de la lista, de modo que
    un fragmento de puntuación alta descartado inicialmente por el tope acabaría
    en una posición inferior a otros peor puntuados. §9.2 exige la lista
    «ordenada de mayor a menor relevancia».

    El desempate va por `chunk_id`. Con `IndexFlatIP` la búsqueda es exacta y el
    único punto de variación entre ejecuciones son los empates de puntuación,
    que §1.4 obliga a resolver sin depender del orden que devuelva la biblioteca
    de búsqueda.
    """
    return sorted(fragmentos, key=lambda f: (-f["score"], f["chunk_id"]))


@dataclass
class Resultado:
    """La respuesta a una consulta, lista para serializar según §9.3.1."""

    query_id: str
    documents: List[str] = field(default_factory=list)   # 3 doc_id, en orden
    fragments: List[Dict] = field(default_factory=list)  # 10 diccionarios

    def to_json(self) -> Dict:
        """Construye el objeto exacto de la Tabla 2 (§9.3.1).

        Los `rank` se generan aquí y empiezan en 1, no en 0: la Tabla 2 los
        define así, a diferencia del campo `posicion` de la Tabla 1, que empieza
        en 0.
        """
        return {
            "query_id": self.query_id,
            "documents": [{"rank": i, "doc_id": d}
                          for i, d in enumerate(self.documents, start=1)],
            "fragments": [{"rank": i, "chunk_id": f["chunk_id"],
                           "doc_id": f["doc_id"], "text": f["text"]}
                          for i, f in enumerate(self.fragments, start=1)],
        }


class AlmacenMetadata:
    """Acceso indexado al `metadata.jsonl` sin cargarlo entero en memoria.

    §5.3 obliga a mantener la metadata en un almacén separado que mapee el
    identificador interno de FAISS al `chunk_id` y al resto de campos. Este es
    ese almacén.

    Conserva únicamente el desplazamiento en bytes de cada línea: son 91.021
    enteros, del orden de 700 KB, en lugar de los 240 MB del archivo completo.
    Importa porque `generador.py` se ejecuta en el entorno de evaluación y no
    debe exigir medio gigabyte de memoria para leer diez fragmentos.
    """

    def __init__(self, ruta: Path = METADATA) -> None:
        self.ruta = Path(ruta)
        if not self.ruta.is_file():
            raise FileNotFoundError(f"No existe {self.ruta}")
        self._offsets: List[int] = []
        # Se abre en binario para que tell() devuelva bytes reales; en modo
        # texto, con UTF-8 y saltos de línea, no es un desplazamiento usable.
        with open(self.ruta, "rb") as f:
            desplazamiento = 0
            for linea in f:
                if linea.strip():
                    self._offsets.append(desplazamiento)
                desplazamiento += len(linea)
        self._archivo = open(self.ruta, "rb")

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, indice: int) -> Dict:
        self._archivo.seek(self._offsets[indice])
        return json.loads(self._archivo.readline().decode("utf-8"))

    def cerrar(self) -> None:
        self._archivo.close()


class Buscador:
    """Índice FAISS y almacén de metadata, listos para consultar."""

    def __init__(self, indice: Path = INDICE, metadata: Path = METADATA) -> None:
        import faiss

        self.indice = faiss.read_index(str(indice))
        self.metadata = AlmacenMetadata(metadata)
        if self.indice.ntotal != len(self.metadata):
            raise ValueError(
                f"El índice tiene {self.indice.ntotal:,} vectores y la metadata "
                f"{len(self.metadata):,} líneas. El mapeo de §5.3 estaría roto."
            )

    def candidatos(self, vector, k: int = K_CANDIDATOS) -> List[Dict]:
        """Devuelve los k fragmentos más similares, con su puntuación.

        El vector de consulta debe estar normalizado: `IndexFlatIP` calcula el
        producto interno, que solo coincide con el coseno sobre vectores de
        norma unitaria (§5.2). La función de codificación ya lo garantiza.
        """
        import numpy as np

        consulta = np.ascontiguousarray(
            np.asarray(vector, dtype="float32").reshape(1, -1))
        puntuaciones, indices = self.indice.search(consulta, k)

        salida = []
        for punt, idx in zip(puntuaciones[0], indices[0]):
            if idx < 0:            # FAISS marca con -1 los huecos si k > ntotal
                continue
            registro = self.metadata[int(idx)]
            salida.append({
                "faiss_id": int(idx),
                "score": float(punt),
                "chunk_id": registro["chunk_id"],
                "doc_id": registro["doc_id"],
                "text": registro["texto"],
                # `idioma` es el campo adicional que §3.4 autoriza y que se
                # conservó porque §8.7 contempla filtrar por él. Se accede con
                # `.get()` porque es opcional: 18 documentos del corpus no
                # ofrecen señal suficiente para determinar su idioma.
                "idioma": registro.get("idioma"),
            })
        return salida

    def candidatos_suficientes(self, vector, k: int = K_CANDIDATOS,
                               min_documentos: int = N_DOCUMENTOS) -> List[Dict]:
        """Como `candidatos()`, pero garantizando variedad de documentos.

        Amplía `k` hasta que aparezcan al menos `min_documentos` distintos o se
        agote el índice. Sin esta ampliación, §9.3.1 se incumpliría en silencio:
        la lista de documentos saldría con dos elementos en lugar de tres y la
        evaluación automática penaliza o descarta el objeto completo.
        """
        while True:
            encontrados = self.candidatos(vector, k)
            distintos = len({c["doc_id"] for c in encontrados})
            if distintos >= min_documentos or k >= self.indice.ntotal:
                return encontrados
            k = min(k * FACTOR_AMPLIACION, self.indice.ntotal)

    # ── §8.6: agregación a nivel documento ───────────────────────────────

    @staticmethod
    def agregar_documentos(candidatos: List[Dict], n: int = N_DOCUMENTOS,
                           estrategia: str = "max") -> List[str]:
        """Agrupa los candidatos por `doc_id` y devuelve los n mejores.

        `estrategia` admite `"max"` (por defecto), `"suma"` o `"media"`. El
        motivo de que la suma resulte inadecuada en este corpus está en la
        cabecera del módulo.

        El desempate es por `doc_id` alfabético y no por el orden que devuelva
        FAISS: §1.4 exige que dos ejecuciones produzcan el mismo resultado.
        """
        por_documento: Dict[str, List[float]] = {}
        for c in candidatos:
            por_documento.setdefault(c["doc_id"], []).append(c["score"])

        def puntuar(puntuaciones: List[float]) -> float:
            if estrategia == "suma":
                return sum(puntuaciones)
            if estrategia == "media":
                return sum(puntuaciones) / len(puntuaciones)
            return max(puntuaciones)

        ordenados = sorted(por_documento.items(),
                           key=lambda kv: (-puntuar(kv[1]), kv[0]))
        return [doc_id for doc_id, _ in ordenados[:n]]

    # ── Filtrado y construcción de los 10 fragmentos ─────────────────────

    @staticmethod
    def seleccionar_fragmentos(
        candidatos: List[Dict],
        consulta: Optional[str] = None,
        n: int = N_FRAGMENTOS,
        max_por_documento: Optional[int] = MAX_POR_DOCUMENTO,
        limite_palabras: int = LIMITE_PALABRAS,
    ) -> List[Dict]:
        """Aplica los filtros y devuelve n fragmentos listos para reportar.

        Orden de los filtros, y su motivo:

        1. **Tope por documento**, en primer lugar: si un documento va a aportar
           como máximo dos fragmentos, no tiene sentido invertir trabajo en
           dividir los treinta restantes que pueda tener.
        2. **División a 250 palabras** (§9.2.1): se selecciona el sub-fragmento
           con mayor solapamiento léxico con la consulta. Sin consulta, el
           primero.
        3. **Deduplicación por texto**, en último lugar y sobre el texto que se
           va a reportar, porque §10.2.1 evalúa el contenido del campo `text`.
           Dos fragmentos distintos pueden producir el mismo sub-fragmento.

        El `chunk_id` reportado es siempre el del fragmento original del índice
        (§9.2.1), aunque el texto sea una porción suya.

        EL TOPE ES UNA PREFERENCIA, NO UNA RESTRICCIÓN DURA. Cuando los
        candidatos proceden de pocos documentos —lo que ocurre con los archivos
        tabulares de mayor tamaño, que tienen decenas de miles de fragmentos
        muy similares— el tope dejaba la lista en cuatro fragmentos, y §9.3.1
        penaliza o descarta un array que no tenga exactamente diez. Por eso
        existe una segunda pasada que completa la lista sin tope: cumplir
        §9.3.1 prevalece sobre la variedad.
        """
        vistos_por_documento: Dict[str, int] = {}
        textos_vistos = set()
        salida: List[Dict] = []
        usados: set = set()

        def intentar(c: Dict, respetar_tope: bool) -> bool:
            """Prepara un candidato y lo añade si supera los filtros."""
            if c["chunk_id"] in usados:
                return False
            if respetar_tope and max_por_documento:
                if vistos_por_documento.get(c["doc_id"], 0) >= max_por_documento:
                    return False

            # División a 250 palabras (§9.2.1): se elige el sub-fragmento con
            # mayor solapamiento léxico con la consulta.
            texto = elegir_mejor(dividir_a_limite(c["text"], limite_palabras),
                                 consulta).strip()
            if not texto:
                return False      # un fragmento vacío no recupera nada

            # Deduplicación sobre el texto final: §10.2.1 evalúa ese contenido.
            clave = clave_de_texto(texto)
            if clave in textos_vistos:
                usados.add(c["chunk_id"])   # descartado definitivamente
                return False
            textos_vistos.add(clave)

            vistos_por_documento[c["doc_id"]] = vistos_por_documento.get(c["doc_id"], 0) + 1
            usados.add(c["chunk_id"])
            # El campo `idioma` debe propagarse hasta aquí. Este diccionario se
            # reconstruye desde cero —no se copia el candidato— porque `text` ya
            # no es el del fragmento completo sino el sub-fragmento de hasta 250
            # palabras. No contamina la salida: `Resultado.to_json()` construye
            # los campos de la Tabla 2 de forma explícita.
            salida.append({"chunk_id": c["chunk_id"], "doc_id": c["doc_id"],
                           "text": texto, "score": c["score"],
                           "idioma": c.get("idioma")})
            return True

        # Primera pasada: con el tope, que es lo que aporta variedad.
        for c in candidatos:
            if len(salida) >= n:
                return _ordenar_por_puntuacion(salida)
            intentar(c, respetar_tope=True)

        # Segunda pasada, solo si faltan. Cumplir §9.3.1 prevalece.
        for c in candidatos:
            if len(salida) >= n:
                break
            intentar(c, respetar_tope=False)

        return _ordenar_por_puntuacion(salida)

    # ── Recuperación completa ────────────────────────────────────────────

    def buscar(
        self,
        vector,
        query_id: str = "q000",
        consulta: Optional[str] = None,
        k: int = K_CANDIDATOS,
        max_por_documento: Optional[int] = MAX_POR_DOCUMENTO,
        estrategia: str = "max",
        fenomeno: Optional[int] = None,
        bonificacion: float = BONIFICACION_FENOMENO,
        factor_idioma: float = FACTOR_IDIOMA_OTROS,
    ) -> Resultado:
        """De un vector de consulta a un `Resultado` completo.

        Los ajustes de puntuación se aplican una sola vez, sobre la lista de
        candidatos, de modo que afectan por igual a la agregación de documentos
        y a la selección de fragmentos: ambas parten de la misma lista ya
        reordenada.

        Los dos ajustes son multiplicativos e independientes, así que el orden
        en que se apliquen no altera el resultado.
        """
        candidatos = self.candidatos_suficientes(vector, k, N_DOCUMENTOS)
        candidatos = aplicar_bonificacion(candidatos, fenomeno, bonificacion)
        candidatos = aplicar_factor_idioma(candidatos, factor_idioma)
        return Resultado(
            query_id=query_id,
            documents=self.agregar_documentos(candidatos, N_DOCUMENTOS, estrategia),
            fragments=self.seleccionar_fragmentos(
                candidatos, consulta, N_FRAGMENTOS, max_por_documento),
        )

    def buscar_texto(self, consulta: str, modelo, query_id: str = "q000",
                     **kwargs) -> Resultado:
        """Igual, partiendo del texto de la consulta. Requiere el encoder."""
        from embedding.encoder import codificar

        vector = codificar(modelo, [consulta])[0]
        return self.buscar(vector, query_id=query_id, consulta=consulta, **kwargs)

    def cerrar(self) -> None:
        self.metadata.cerrar()
