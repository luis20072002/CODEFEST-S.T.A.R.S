"""Recuperación (§8): de una consulta a 3 documentos y 10 fragmentos.

    py -m retrieval.search              # autoprueba con consultas simuladas
    py -m retrieval.search --texto "basura espacial en órbita baja"

────────────────────────────────────────────────────────────────────────────────
QUÉ HACE Y EN QUÉ ORDEN

    1. codificar la consulta          → vector normalizado
    2. buscar en FAISS                → K_CANDIDATOS fragmentos con su coseno
    3. agregar a nivel documento      → §8.6, los 3 mejores
    4. filtrar los fragmentos         → tope por documento + deduplicación
    5. partir a 250 palabras          → §9.2.1, los 10 mejores

Los pasos 3 y 4 son independientes a propósito: §9.2 pide «los 3 documentos más
relevantes» y «los 10 fragmentos más relevantes», sin exigir que los segundos
pertenezcan a los primeros.

⚠️ **Todo opera sobre vectores, puntuaciones y metadata.** §8.3 prohíbe usar
modelos generativos para reranking, expansión de consulta, filtrado o síntesis.
Aquí no hay ninguno: el único modelo es el encoder, que codifica la consulta.

────────────────────────────────────────────────────────────────────────────────
POR QUÉ SE PIDEN 200 CANDIDATOS Y NO 10

Entre la búsqueda y la respuesta hay tres filtros que **quitan** candidatos: el
tope por documento, la deduplicación y el descarte de fragmentos vacíos. Con
`k=10` la lista se quedaría corta y habría que devolver menos de diez, que
§9.3.1 penaliza o descarta. 200 da margen de sobra y cuesta lo mismo: con
`IndexFlatIP` el coste está en recorrer los 91.021 vectores, no en cuántos se
devuelven.

────────────────────────────────────────────────────────────────────────────────
EL TOPE POR DOCUMENTO: QUÉ MIDE Y POR QUÉ EXISTE

Medido sobre las 50 consultas reales (`ESTADO.md` §17): un solo documento tiende
a acaparar el top-10. El caso peor fue la consulta 7, donde **7 de 10 ranks**
salieron de `F3-SIPRI-100`, la traducción coreana de un informe de SIPRI cuya
versión inglesa el jurado probablemente marcó como la relevante. También la
consulta 5, con 4 de sus 5 primeros en `F1-ILIA-005`.

`MAX_POR_DOCUMENTO` corta eso de raíz. Con 2, la consulta 7 habría gastado dos
ranks en coreano en vez de siete, y los otros cinco habrían ido a documentos
distintos. Es configurable para poder medir con y sin él.

────────────────────────────────────────────────────────────────────────────────
POR QUÉ MAX POOLING Y NO SUMA (§8.6)

§8.6 admite tres formas de agregar fragmentos a documento: «puntuación del
fragmento más relevante (max pooling), suma de las puntuaciones de todos sus
fragmentos recuperados, o media ponderada».

**La suma es peligrosa en ESTE corpus.** `F1-AIINDEX-056` tiene **33.396
fragmentos** —los volcados bibliográficos de PubMed— frente a una mediana de 5
por documento (`ESTADO.md` §13). Sumar premiaría a los documentos por ser
grandes, no por ser relevantes, y los tres puestos de `documents` se los
llevarían siempre los mismos tres CSV.

Max pooling no tiene ese sesgo: un documento vale lo que valga su mejor
fragmento. Es el valor por defecto, pero la estrategia es un parámetro para
poder compararlas.
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from retrieval.fragmentos import (LIMITE_PALABRAS, clave_de_texto,
                                  dividir_a_limite, elegir_mejor)

RAIZ = Path(__file__).resolve().parents[2]
CARPETA_ENCODER = RAIZ / "entrega" / "base_vectorial" / "encoder_bge-m3"
INDICE = CARPETA_ENCODER / "index.faiss"
METADATA = CARPETA_ENCODER / "metadata.jsonl"

# Cuántos fragmentos se piden a FAISS antes de filtrar. Ver la cabecera.
K_CANDIDATOS = 200

# Factor por el que se amplía k cuando los candidatos no dan para 3 documentos
# distintos. Lo destapó la autoprueba: una consulta que se parece a un registro
# bibliográfico trae 200 candidatos que salen TODOS de los dos CSV gigantes del
# AI Index (`F1-AIINDEX-056` y `-063`, con 33.396 y 19.239 fragmentos), así que
# solo había 2 documentos distintos y §9.3.1 exige 3.
# Ampliar es barato: con `IndexFlatIP` el coste está en recorrer los 91.021
# vectores, que se hace igual, no en cuántos se devuelven.
FACTOR_AMPLIACION = 10

# Cuántos fragmentos como máximo puede aportar un mismo documento al top-10.
# 0 o None desactiva el tope. Ver la cabecera para la medición que lo motiva.
MAX_POR_DOCUMENTO = 2

# Lo que exige §9.2, y §9.3.1 penaliza o descarta si no cuadra.
N_DOCUMENTOS = 3
N_FRAGMENTOS = 10

# Factor por el que se multiplica la puntuación de los candidatos que
# pertenecen al fenómeno de la consulta.
#
# **1.03, fijado el 2026-08-07 tras compararlo con 1.00, 1.05 y 1.10.** No es un
# valor elegido a ojo: sale de puntuar las cuatro configuraciones con las
# métricas de §10 contra juicios de relevancia etiquetados a mano
# (`evaluation/juicios_muestra.json`). Detalle en `ESTADO.md` §21.
#
# ⚠️ **Este valor tiene que coincidir con el que produjo el `resultados.jsonl`
# entregado.** §1.4 excluye la entrega si `generador.py` no la reproduce, y el
# jurado lo ejecuta sin pasar `--bonificacion`.
BONIFICACION_FENOMENO = 1.03

# Idiomas que NO se penalizan. Son los dos en los que vive el contenido
# relevante del corpus: las 50 consultas están todas en español (`ESTADO.md`
# §5) y el corpus es 55,0% inglés / 35,1% español (§11). Juntos cubren el
# 96,8% de los fragmentos que la salida real devuelve (§17).
IDIOMAS_PREFERIDOS = ("es", "en")

# Factor por el que se multiplica la puntuación de un candidato cuyo `idioma`
# no está en IDIOMAS_PREFERIDOS.
#
# 🔴 **1.0 = DESACTIVADO, y así se queda hasta que esté medido.** Es la
# disciplina que el proyecto ya aprendió con la bonificación por fenómeno
# (`ESTADO.md` §21): los factores 1,02 y 1,05 que parecían razonables sobre la
# aritmética de los cosenos **no hacían nada**, y 1,10 era un filtro
# disfrazado. Aquí el número se elige con `tools/barrer_factor_idioma.py`, no a
# ojo.
#
# ⚠️ **Y hay una razón de §1.4 para no tocarlo todavía**: el `resultados.jsonl`
# entregado se generó SIN factor de idioma. Si este valor deja de ser 1.0, el
# jurado —que corre `generador.py` sin banderas— produciría un archivo distinto
# al entregado y la entrega **se excluye**. Cambiar este número obliga a
# regenerar el entregable y a re-validarlo con `tools.verificar_resultados`.
FACTOR_IDIOMA_OTROS = 1.0


def aplicar_factor_idioma(
    candidatos: List[Dict],
    factor_otros: float = FACTOR_IDIOMA_OTROS,
    preferidos: tuple = IDIOMAS_PREFERIDOS,
) -> List[Dict]:
    """Baja la puntuación de los candidatos que no están en es/en y reordena.

    ────────────────────────────────────────────────────────────────────────
    EL PROBLEMA QUE CORRIGE, MEDIDO SOBRE LA SALIDA REAL

    SWF y SIPRI publican **el mismo informe en varios idiomas**, y en el corpus
    son documentos distintos con `doc_id` distintos. Una consulta puede gastar
    ranks en versiones traducidas del mismo contenido.

    Medido el 2026-08-08 resolviendo los 500 `chunk_id` de la salida contra el
    campo `idioma` del `metadata.jsonl` (`ESTADO.md` §17):

        en 66,2%  ·  es 30,6%  ·  zh/fr/pt/ru/ko  16 fragmentos = 3,2%

    Solo **8 de las 50 consultas** traen algo fuera de {es, en}, y no es un
    problema genérico de traducciones sino **una familia de documentos**: los
    resúmenes ejecutivos del *Global Counterspace Capabilities* de SWF,
    publicados en seis idiomas (`-por`, `-fre`, `-spa`, `-chinese`, …).

    Los dos casos que de verdad cuestan:

    - **`q024` duele a nivel DOCUMENTO, que es donde se paga F1@3 (§10.2.2).**
      Sus tres puestos de `documents` son el execsum español del 2025, el
      portugués del 2026 y un CSIS. El informe completo en inglés existe en el
      corpus (`F2-SWF-124`) pero no entra.
    - **`q022` devuelve `SWF_2025-executive-summary-chinese.pdf` en el rank 1**
      para una consulta en español, teniendo la versión inglesa en el rank 2.

    ────────────────────────────────────────────────────────────────────────
    POR QUÉ UN FACTOR Y NO UN FILTRO NI UN REORDENAMIENTO POR IDIOMA

    Es el mismo razonamiento que `aplicar_bonificacion()`: **el fallo de un
    filtro es irreversible.** Con un factor, si una traducción es el único
    candidato que responde a la consulta, **sobrevive** — no tiene contra quién
    competir. Solo pierde cuando hay un equivalente en es/en cerca en
    puntuación, que es exactamente el caso que se quiere corregir.

    🔴 **Se DESCARTÓ reordenar globalmente es → en → otro.** El problema está
    en 8 consultas y un criterio de orden por idioma se aplica a las 50. Choca
    además con lo que §17 validó con datos: la mezcla de idiomas de cada
    consulta **sigue a dónde vive el contenido relevante**, porque BGE-M3
    ordena por significado. §10.2.1 juzga por el campo `text` y no premia el
    español, y **NDCG@10 es sensible al orden** (fórmula 8): subir un fragmento
    en español por encima de uno en inglés mejor puntuado cuesta NDCG en toda
    consulta mixta a cambio de nada medible.

    ────────────────────────────────────────────────────────────────────────
    POR QUÉ `idioma = None` NO SE PENALIZA

    ⚠️ `None` significa «el detector no tuvo señal suficiente», **no «idioma
    extranjero»**. Son 18 documentos (§11), y penalizarlos sería castigar un
    fallo de `py3langid`, no una traducción.

    Es la misma razón por la que esto es un factor suave y no un filtro:
    `ESTADO.md` §11 mide **~19 documentos (1%) con la etiqueta de idioma mal**
    —los cuatro en inglés detectados como `la` y el `af` de las letras
    duplicadas—. Con un filtro, esos textos en inglés quedarían inalcanzables
    por un error del detector. Comprobado el 2026-08-08: ninguno de los cinco
    aparece en la salida actual, así que hoy no cuesta nada — pero el diseño no
    debe depender de esa suerte.

    §8.7 respalda esto explícitamente: permite «filtrar por campo `fenomeno`,
    `formato`, rango de fechas, **idioma**». Es metadata, no un modelo, así que
    no roza §8.3.
    """
    # Atajo: con el factor en 1.0 esto no hace nada, y devolver la lista tal
    # cual evita copiar 200 diccionarios y reordenar para nada.
    if factor_otros == 1.0:
        return candidatos

    ajustados = []
    for c in candidatos:
        copia = dict(c)          # no se muta la lista del llamante
        idioma = c.get("idioma")
        # La condición tiene dos partes a propósito: `is not None` protege a
        # los documentos sin idioma detectado (ver arriba) y `not in` es lo que
        # de verdad penaliza.
        if idioma is not None and idioma not in preferidos:
            copia["score"] = c["score"] * factor_otros
        ajustados.append(copia)

    # Reordenar, porque todo lo de aguas abajo —agregación a documento y
    # selección de fragmentos— asume orden por puntuación descendente.
    return _ordenar_por_puntuacion(ajustados)


def aplicar_bonificacion(candidatos: List[Dict], fenomeno: Optional[int],
                         factor: float = BONIFICACION_FENOMENO) -> List[Dict]:
    """Sube la puntuación de los candidatos del fenómeno esperado y reordena.

    ────────────────────────────────────────────────────────────────────────
    EL PROBLEMA QUE INTENTA CORREGIR, MEDIDO

    Sobre el `resultados.jsonl` real (2026-08-07), la proporción de documentos
    devueltos que pertenecen al fenómeno de la consulta:

        F1 (q001–q016):  29/48  = 60,4%
        F2 (q017–q032):  42/48  = 87,5%
        F3 (q033–q050):  50/54  = 92,6%

    **F1 se queda en el 60%.** La causa está identificada: SIPRI está
    catalogado como F3 pero publica mucho sobre IA militar, que es el tema de
    F1. Las etiquetas de fenómeno describen **el observatorio de origen, no el
    contenido**. Dos consultas (`q001` y `q003`) no devuelven ni un documento
    de F1.

    ────────────────────────────────────────────────────────────────────────
    POR QUÉ UNA BONIFICACIÓN Y NO UN FILTRO

    §8.7 permite filtrar por metadata, y filtrar por `fenomeno` arreglaría el
    60% de golpe. Pero **el fallo de filtrar es irreversible**: si el *ground
    truth* marca como relevante un documento de otro fenómeno, filtrarlo lo
    hace inalcanzable y §10.2.2 lo cuenta como fallo sin remedio.

    La bonificación es la opción intermedia: empuja hacia el fenómeno esperado
    **sin cerrar la puerta** a un documento de otro que sea claramente mejor.

    ⚠️ **No sabemos cuál es mejor y no se puede medir**: el *ground truth* no
    es público, así que esto es una elección de riesgo, no una optimización.
    Por eso el valor por defecto es `1.0` — desactivada— y hay que pedirla
    explícitamente.

    ────────────────────────────────────────────────────────────────────────
    CÓMO ELEGIR EL FACTOR — MEDIDO, NO ESTIMADO

    ⚠️ La primera versión de este docstring proponía 1,02–1,05 razonando sobre
    la aritmética de los cosenos (se apiñan en torno a 0,60, así que ×1,05 son
    +0,030). **La medición lo desmintió.** Probado sobre el índice real, con
    consultas simuladas cuyos candidatos mezclan fenómenos y bonificando el
    minoritario:

        ×1,02   sin ningún efecto
        ×1,05   sin ningún efecto
        ×1,10   promueve 1 documento y 1 fragmento

    El motivo es que un candidato del fenómeno minoritario suele estar muy
    abajo en la lista, y +0,03 no le alcanza para saltar decenas de puestos.
    Esos casos son adversos a propósito: en una consulta real de F1, los
    documentos de F1 estarán cerca en puntuación y un factor pequeño sí podría
    moverlos.

    **Por eso no hay un valor recomendado a ciegas.** Hay que correr las 50
    consultas con varios factores y mirar cómo se mueve la correspondencia con
    el fenómeno (60,4% en F1 sin bonificación). Eso sí se puede medir; lo que
    no se puede medir es si mejora la recuperación de verdad.
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
    # Reordenar: la bonificación cambia el orden, y todo lo de aguas abajo
    # (agregación a documento y selección de fragmentos) asume orden por
    # puntuación descendente.
    return _ordenar_por_puntuacion(ajustados)


def _ordenar_por_puntuacion(fragmentos: List[Dict]) -> List[Dict]:
    """Ordena de mayor a menor relevancia, con desempate determinista.

    ⚠️ **Hace falta un orden explícito al final, y no es evidente.** La segunda
    pasada de `seleccionar_fragmentos()` añade al final de la lista, así que un
    fragmento con puntuación alta que el tope había descartado acababa en el
    rank 9 por detrás de otros de 0,77. §9.2 pide la lista «ordenada de mayor a
    menor relevancia», y lo destapó la autoprueba.

    El desempate va por `chunk_id`, como fija `ESTADO.md` §8: con `IndexFlatIP`
    la búsqueda es exacta y el único punto de variación entre corridas son los
    empates de puntuación, así que §1.4 exige resolverlos sin depender del
    orden que devuelva FAISS.
    """
    return sorted(fragmentos, key=lambda f: (-f["score"], f["chunk_id"]))


@dataclass
class Resultado:
    """La respuesta a una consulta, lista para serializar según §9.3.1."""

    query_id: str
    documents: List[str] = field(default_factory=list)   # 3 doc_id, en orden
    fragments: List[Dict] = field(default_factory=list)  # 10 dicts

    def to_json(self) -> Dict:
        """Construye el objeto exacto de la Tabla 2 (§9.3.1).

        Los `rank` se generan aquí y empiezan en 1, no en 0: la Tabla 2 los
        define así, al contrario que el campo `posicion` de la Tabla 1, que
        empieza en 0. Es una de las confusiones más fáciles de cometer.
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
    """Acceso por índice al `metadata.jsonl` sin cargarlo entero en memoria.

    §5.3 obliga a mantener la metadata en un almacén separado que mapee el
    identificador interno de FAISS al `chunk_id` y al resto de campos. Este es
    ese almacén.

    **Guarda solo el desplazamiento en bytes de cada línea**, no las líneas.
    Son 91.021 enteros (~700 KB) en lugar de 229 MB de texto y objetos de
    Python. Importa porque `generador.py` lo ejecuta el jurado (§1.4) y no
    conviene exigirle medio giga de RAM para leer diez fragmentos.
    """

    def __init__(self, ruta: Path = METADATA) -> None:
        self.ruta = Path(ruta)
        if not self.ruta.is_file():
            raise FileNotFoundError(f"No existe {self.ruta}")
        self._offsets: List[int] = []
        # Se abre en binario para que tell() dé bytes reales; en modo texto,
        # con UTF-8 y saltos de línea, tell() no es un desplazamiento usable.
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
    """Índice FAISS + almacén de metadata, listos para consultar."""

    def __init__(self, indice: Path = INDICE, metadata: Path = METADATA) -> None:
        import faiss

        self.indice = faiss.read_index(str(indice))
        self.metadata = AlmacenMetadata(metadata)
        if self.indice.ntotal != len(self.metadata):
            raise ValueError(
                f"El índice tiene {self.indice.ntotal:,} vectores y la metadata "
                f"{len(self.metadata):,} líneas. El mapeo de §5.3 estaría roto."
            )

    # ── Búsqueda ─────────────────────────────────────────────────────────

    def candidatos(self, vector, k: int = K_CANDIDATOS) -> List[Dict]:
        """Devuelve los k fragmentos más similares, con su puntuación.

        El vector de consulta debe estar **normalizado**: `IndexFlatIP` calcula
        producto interno, que solo es el coseno sobre vectores de norma 1
        (§5.2). `embedding.encoder.codificar()` ya lo hace.
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
                # `idioma` es el campo extra que §3.4 autoriza y que se
                # conservó justamente porque §8.7 contempla filtrar por él
                # (`ESTADO.md` §14). Lo consume `aplicar_factor_idioma()`.
                # `.get()` y no `[...]`: es opcional y vale `None` en los 18
                # documentos sin señal de idioma suficiente (§11).
                "idioma": registro.get("idioma"),
            })
        return salida

    def candidatos_suficientes(self, vector, k: int = K_CANDIDATOS,
                               min_documentos: int = N_DOCUMENTOS) -> List[Dict]:
        """Como `candidatos()`, pero garantizando variedad de documentos.

        Amplía `k` hasta que aparezcan al menos `min_documentos` distintos o se
        agote el índice. Sin esto, §9.3.1 se incumple en silencio: la lista de
        `documents` saldría con dos elementos en vez de tres y la evaluación
        automática penaliza o descarta el objeto entero.
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

        `estrategia`: `"max"` (por defecto), `"suma"` o `"media"`. El porqué de
        que la suma sea peligrosa aquí está en la cabecera del módulo.

        El desempate es por `doc_id` alfabético, no por el orden que devuelva
        FAISS: §1.4 exige que dos corridas den lo mismo, y `ESTADO.md` §8 fija
        que los empates se resuelvan de forma determinista.
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
        """Aplica los tres filtros y devuelve n fragmentos listos para reportar.

        Orden de los filtros, y por qué ese y no otro:

        1. **Tope por documento** — antes que nada, porque si un documento va a
           aportar como mucho 2 fragmentos, no tiene sentido gastar trabajo en
           partir los otros 30 suyos.
        2. **Partido a 250 palabras** (§9.2.1) — se elige el sub-fragmento con
           más solapamiento léxico con la consulta. Sin consulta, el primero.
        3. **Deduplicación por texto** — la última, **sobre el texto que se va
           a reportar**, porque §10.2.1 evalúa el contenido del campo `text`.
           Dos chunks distintos pueden producir el mismo sub-fragmento.

        El `chunk_id` que se reporta es siempre el del **fragmento original del
        índice** (§9.2.1), aunque el texto sea un trozo suyo.

        ⚠️ **El tope es una preferencia, no una restricción dura**, y esto lo
        destapó la autoprueba. Cuando los candidatos vienen de pocos documentos
        —pasa con los CSV gigantes del AI Index, que tienen decenas de miles de
        fragmentos casi idénticos— el tope dejaba la lista en 4 fragmentos, y
        §9.3.1 penaliza o descarta un array que no tenga exactamente 10.
        Por eso hay una **segunda pasada** que rellena sin tope si hace falta:
        más vale un top-10 poco variado que un objeto descartado.
        """
        vistos_por_documento: Dict[str, int] = {}
        textos_vistos = set()
        salida: List[Dict] = []
        usados: set = set()

        def intentar(c: Dict, respetar_tope: bool) -> bool:
            """Prepara un candidato y lo añade si pasa los filtros."""
            if c["chunk_id"] in usados:
                return False
            if respetar_tope and max_por_documento:
                if vistos_por_documento.get(c["doc_id"], 0) >= max_por_documento:
                    return False

            # Partido a 250 palabras (§9.2.1): se elige el sub-fragmento con
            # más solapamiento léxico con la consulta; sin consulta, el primero.
            texto = elegir_mejor(dividir_a_limite(c["text"], limite_palabras),
                                 consulta).strip()
            if not texto:
                return False      # un fragmento vacío no recupera nada

            # Deduplicación sobre el texto FINAL: §10.2.1 evalúa ese contenido.
            clave = clave_de_texto(texto)
            if clave in textos_vistos:
                usados.add(c["chunk_id"])   # descartado para siempre
                return False
            textos_vistos.add(clave)

            vistos_por_documento[c["doc_id"]] = vistos_por_documento.get(c["doc_id"], 0) + 1
            usados.add(c["chunk_id"])
            salida.append({"chunk_id": c["chunk_id"], "doc_id": c["doc_id"],
                           "text": texto, "score": c["score"]})
            return True

        # Primera pasada: con el tope, que es lo que da variedad.
        for c in candidatos:
            if len(salida) >= n:
                return _ordenar_por_puntuacion(salida)
            intentar(c, respetar_tope=True)

        # Segunda pasada: solo si faltan. Cumplir §9.3.1 manda sobre la variedad.
        for c in candidatos:
            if len(salida) >= n:
                break
            intentar(c, respetar_tope=False)

        return _ordenar_por_puntuacion(salida)

    # ── Todo junto ───────────────────────────────────────────────────────

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

        `fenomeno` + `bonificacion` activan el empujón hacia el fenómeno
        esperado. Se aplica **una sola vez, sobre la lista de candidatos**, así
        que afecta por igual a la agregación de documentos y a la selección de
        fragmentos: las dos parten de la misma lista ya reordenada.

        `factor_idioma` hace lo propio con los idiomas fuera de {es, en}
        (§8.7). Los dos ajustes son **multiplicativos e independientes**, así
        que el orden en que se aplican no cambia el resultado: un candidato de
        otro fenómeno y en coreano acaba con `score × bonificacion ×
        factor_idioma` de todas formas. Se reordena dos veces y gana el último
        orden, que es el correcto.
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


if __name__ == "__main__":
    # Autoprueba. Sin el modelo (4,35 GB) no se puede codificar una consulta de
    # texto, así que se usan **vectores del propio índice como consulta
    # simulada**: buscar el vector del fragmento i debe devolver i el primero.
    # Eso ejercita toda la cañería —FAISS, metadata, agregación, filtros,
    # partido a 250 palabras— sin descargar nada.
    import time

    import numpy as np

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    texto_consulta = None
    if "--texto" in sys.argv:
        texto_consulta = sys.argv[sys.argv.index("--texto") + 1]

    if not INDICE.is_file():
        print(f"No existe {INDICE}. Corre antes `py -m indexing.faiss_index`.")
        raise SystemExit(1)

    print("cargando índice y metadata…")
    inicio = time.perf_counter()
    buscador = Buscador()
    print(f"listo en {time.perf_counter() - inicio:,.1f} s   "
          f"({buscador.indice.ntotal:,} vectores)")

    linea = "─" * 78

    if texto_consulta:
        from embedding.encoder import cargar_modelo

        print("\ncargando el modelo…")
        modelo = cargar_modelo()
        resultados = [buscador.buscar_texto(texto_consulta, modelo, "q001")]
        titulos = [texto_consulta]
    else:
        # Tres posiciones repartidas por el índice, como consultas simuladas.
        matriz = np.load(
            Path(__file__).resolve().parents[1] / "data" / "embeddings.npy",
            mmap_mode="r")
        posiciones = [0, buscador.indice.ntotal // 2, buscador.indice.ntotal - 1]
        resultados, titulos = [], []
        for p in posiciones:
            r = buscador.buscar(np.asarray(matriz[p]), query_id=f"q{p:03d}")
            resultados.append(r)
            titulos.append(f"vector de la posición {p:,} usado como consulta")

    for resultado, titulo in zip(resultados, titulos):
        print(f"\n{linea}\n{titulo}")
        print(f"\ndocumentos (§8.6, max pooling): {resultado.documents}")
        print(f"fragmentos: {len(resultado.fragments)}")
        for i, f in enumerate(resultado.fragments, start=1):
            palabras = len(f["text"].split())
            print(f"  {i:>2}. [{palabras:>3} pal] {f['score']:.4f}  "
                  f"{f['chunk_id']:<22} {f['text'][:70]!r}")

        # Comprobaciones de §9.3.1 sobre el resultado construido.
        docs_ok = len(resultado.documents) == N_DOCUMENTOS
        frags_ok = len(resultado.fragments) == N_FRAGMENTOS
        largos = [len(f["text"].split()) for f in resultado.fragments]
        limite_ok = all(p <= LIMITE_PALABRAS for p in largos)
        por_doc = {}
        for f in resultado.fragments:
            por_doc[f["doc_id"]] = por_doc.get(f["doc_id"], 0) + 1
        tope_ok = all(v <= MAX_POR_DOCUMENTO for v in por_doc.values())
        unicos_ok = len({clave_de_texto(f["text"]) for f in resultado.fragments}) == len(largos)
        puntuaciones = [f["score"] for f in resultado.fragments]
        orden_ok = puntuaciones == sorted(puntuaciones, reverse=True)

        print(f"\n  §9.3.1  3 documentos: {'✔' if docs_ok else '✖'}   "
              f"10 fragmentos: {'✔' if frags_ok else '✖'}   "
              f"≤250 palabras: {'✔' if limite_ok else '✖'}")
        print(f"  §9.2 orden decreciente: {'✔' if orden_ok else '✖'}   "
              f"sin duplicados: {'✔' if unicos_ok else '✖'}")
        # El tope es una preferencia, no un requisito del PDF: si hubo que
        # relajarlo para llegar a 10, se informa, no se marca como fallo.
        print(f"  tope de {MAX_POR_DOCUMENTO} por documento: "
              f"{'respetado' if tope_ok else 'RELAJADO para completar los 10'}   "
              f"reparto: {por_doc}")

    buscador.cerrar()
    print(f"\n{linea}")
