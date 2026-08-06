"""Detección y marcado del idioma predominante del documento (§2.2).

Es el cuarto punto de la lista de §2.2 («Detección y marcado del idioma
predominante del documento»), y el único de los cuatro que `cleaner.py` no
cubre. Llena el campo `language` de `Document`, que hasta ahora siempre venía
en `None`.

PARA QUÉ SIRVE, más allá de cumplir §2.2:

  - §8.7 («Post-filtros basados en vectores o metadata») permite filtrar los
    resultados por metadata, y §10.1 dice que las 50 consultas se reparten
    entre los tres idiomas del corpus. Un campo `language` fiable es lo que
    haría posible esa clase de filtro más adelante.
  - §3.4 permite campos extra en la metadata del fragmento y menciona el
    idioma por su nombre como ejemplo válido.
  - Internamente sirve para elegir el separador de oraciones y para auditar el
    corpus por idioma sin volver a leerlo entero.

POR QUÉ py3langid Y NO OTRA COSA. Es un fork mantenido de `langid.py`: un
clasificador **estadístico** (naive Bayes sobre n-gramas de bytes, con el
modelo preentrenado empotrado en el propio paquete). No es un modelo
generativo ni un decoder, así que no roza la prohibición de §8.3. Licencia BSD,
que entra en el espíritu de licencia libre de §4.3. Y es **determinista**: la
misma entrada da siempre la misma salida, requisito de la reproducibilidad de
§1.4. Otras opciones (`langdetect`, `fasttext`) fallan justo ahí:
`langdetect` usa un muestreo aleatorio internamente y da resultados distintos
entre corridas si no se le fija la semilla.

────────────────────────────────────────────────────────────────────────────
DOS DECISIONES QUE PARECEN MEJORAS Y SON ERRORES

**1. NO se restringe la lista de idiomas a es/en/pt.** La tentación es obvia:
`set_languages(["es", "en", "pt"])` sube la precisión en el 99% del corpus.
Pero `cleaner.py` ya documenta que hay **22 documentos legítimamente** en
árabe, ruso, chino, coreano y japonés (las versiones oficiales de la ONU, los
resúmenes de SWF, el AI Index en chino). Restringiendo, esos 22 se etiquetarían
a la fuerza como es/en/pt y el campo `language` diría una mentira **justo en
los documentos donde más se notaría** si alguien filtra por él.

**2. NO se adivina cuando no hay señal.** Un documento de cuatro caracteres
(«Mapa») o una fila de códigos («ELN AGC 004-22 018-20») no tiene información
de idioma, pero el clasificador devuelve un idioma igual, con la confianza por
los suelos. Medido: texto real da confianza **1.000**, y esos casos sin señal
dan **0.169–0.399**. Por eso hay dos guardas (`MIN_LETTERS` y
`MIN_CONFIDENCE`) y por debajo de ellas se devuelve `None`. Un `None` honesto
es mucho mejor que un `"en"` inventado: `None` se puede detectar y tratar
aparte, una etiqueta falsa se propaga en silencio hasta el índice.
────────────────────────────────────────────────────────────────────────────
"""

import re
from typing import Optional, Tuple

from py3langid.langid import MODEL_FILE, LanguageIdentifier

from core.document import Document

# Mínimo de LETRAS (no de caracteres) para intentar la detección. Se cuentan
# letras y no caracteres a propósito: los formatos tabulares producen textos
# larguísimos hechos de códigos y cifras («b_ADM2_PCODE: BR2109551»), que tienen
# miles de caracteres y casi ninguna palabra. Contar caracteres los daría por
# detectables; contar letras los descarta, que es lo correcto.
MIN_LETTERS = 40

# Confianza mínima para aceptar la etiqueta. El corte en 0.50 sale de la
# medición descrita arriba: hay un hueco enorme entre el ruido (≤0.40) y el
# texto real (1.000), así que cualquier valor dentro de ese hueco sirve y el
# resultado no es sensible al valor exacto elegido.
MIN_CONFIDENCE = 0.50

# `[^\W\d_]` es el idiom de Python para "letra Unicode": `\w` ya excluye
# espacios y puntuación, y de lo que queda se descartan dígitos y guion bajo.
# Hace falta esta forma porque el `re` de la stdlib no soporta `\p{L}`.
LETTER = re.compile(r"[^\W\d_]")

# El identificador se construye UNA sola vez al importar el módulo y se
# reutiliza en cada llamada. Cargar el modelo cuesta ~0.08 s: es poco, pero
# hacerlo dentro de la función lo pagaría 1826 veces (unos 2,5 minutos tirados).
#
# `norm_probs=True` es lo que hace utilizable la confianza. Sin ese argumento,
# `classify()` devuelve la LOG-probabilidad sin normalizar —valores como
# -107681.69, que dependen del largo del texto y no se pueden comparar entre
# documentos ni umbralizar. Con él, la salida es un número entre 0 y 1.
_identifier = LanguageIdentifier.from_pickled_model(MODEL_FILE, norm_probs=True)


def detect_language(text: str) -> Tuple[Optional[str], float]:
    """Devuelve `(codigo_iso_639_1, confianza)` para el texto dado.

    El código es de dos letras (`"es"`, `"en"`, `"pt"`, `"ar"`, `"zh"`…) o
    `None` si el texto no tiene señal suficiente para decidir; en ese caso la
    confianza que acompaña sigue siendo la que reportó el clasificador, para
    poder auditar después qué tan cerca del umbral se quedó.

    Se le pasa el texto **completo**, no un recorte de los primeros N
    caracteres. Se comprobó sobre una muestra de 37 documentos que recortar a
    5.000 caracteres da el mismo resultado en 37/37 y es 12× más rápido, pero
    la diferencia es de segundos sobre el corpus entero y el texto completo es
    más robusto en el caso que sí existe aquí: un informe con dos páginas de
    portada y créditos en inglés que luego continúa en español.
    """
    # `sum(1 for _ in finditer(...))` cuenta sin construir la lista de matches.
    # Sobre un XLSX de seis millones de palabras, materializar esa lista sería
    # el paso más caro de todo el módulo.
    if sum(1 for _ in LETTER.finditer(text)) < MIN_LETTERS:
        return None, 0.0

    # `datatype="uint32"` NO es opcional en este corpus, aunque la librería
    # traiga "uint16" por defecto. py3langid cuenta cuántas veces aparece cada
    # rasgo (n-grama de bytes) en un vector de enteros de ese tipo, y uint16
    # solo llega a 65.535. Los archivos tabulares desbordan ese techo sin
    # esfuerzo —el XLSX del AI Index tiene seis millones de palabras y un rasgo
    # que aparece 111.826 veces— y la librería revienta con
    # `OverflowError: Python integer 111826 out of bounds for uint16`.
    # Comprobado que no cambia el resultado en los textos normales: solo amplía
    # el techo del contador.
    language, confidence = _identifier.classify(text, datatype="uint32")
    confidence = float(confidence)
    if confidence < MIN_CONFIDENCE:
        return None, confidence
    return language, confidence


def tag_document(document: Document) -> Document:
    """Rellena `language` in-place y anota la confianza en `metadata`.

    A diferencia de `cleaner.clean_document()`, que devuelve una copia, aquí sí
    se modifica el Document recibido: el cleaner necesita conservar el original
    para poder comparar antes y después, pero etiquetar el idioma no destruye
    nada, así que copiar solo gastaría memoria.

    La confianza se guarda en `metadata` y no en un campo propio de `Document`
    porque no es información del documento sino de **nuestra medición** sobre
    él. Sirve para auditar: permite listar los documentos que se quedaron justo
    por encima del umbral sin volver a clasificarlos.
    """
    language, confidence = detect_language(document.text)
    document.language = language
    document.metadata["language_confidence"] = round(confidence, 4)
    return document


if __name__ == "__main__":
    # Diagnóstico: `py -m preprocess.language [ruta.jsonl]` desde src/.
    # Detecta el idioma del corpus YA LIMPIO y resume el reparto, sin escribir
    # nada. Si no se le pasa ruta usa el .jsonl limpio; si aún no existe,
    # avisa en vez de reventar con un traceback.
    import sys
    from collections import Counter
    from pathlib import Path

    from core.store import StoreError, read_documents

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    datos = Path(__file__).resolve().parents[1] / "data"
    ruta = Path(sys.argv[1]) if len(sys.argv) > 1 else datos / "documentos_limpios.jsonl"

    if not ruta.is_file():
        print(f"No existe {ruta}.")
        print("Corre antes:  py -m preprocess.runner")
        raise SystemExit(1)

    idiomas: Counter = Counter()
    por_fenomeno: dict = {1: Counter(), 2: Counter(), 3: Counter(), None: Counter()}
    sin_idioma: list = []
    dudosos: list = []

    try:
        for documento in read_documents(ruta):
            language, confidence = detect_language(documento.text)
            etiqueta = language or "SIN_IDIOMA"
            idiomas[etiqueta] += 1
            por_fenomeno[documento.phenomenon][etiqueta] += 1
            if language is None:
                sin_idioma.append((documento.doc_id, documento.format,
                                   len(documento.text.split()), confidence))
            elif confidence < 0.90:
                dudosos.append((confidence, documento.doc_id, language,
                                documento.text[:70].replace("\n", " ")))
    except StoreError as error:
        raise SystemExit(f"Error leyendo {ruta}: {error}")

    total = sum(idiomas.values())
    print(f"{ruta}\n")
    print(f"documentos: {total}\n")

    print("reparto de idiomas:")
    for codigo, n in idiomas.most_common():
        print(f"  {codigo:12} {n:5}  ({100 * n / total:5.1f}%)")

    print("\npor fenómeno:")
    for fenomeno in (1, 2, 3, None):
        cuenta = por_fenomeno[fenomeno]
        if cuenta:
            print(f"  F{fenomeno}: {dict(cuenta.most_common())}")

    print(f"\nsin idioma asignado: {len(sin_idioma)}")
    for doc_id, formato, palabras, confianza in sin_idioma[:25]:
        print(f"  {doc_id:20} {formato:5} {palabras:8,} pal  confianza={confianza:.3f}")
    if len(sin_idioma) > 25:
        print(f"  … y {len(sin_idioma) - 25} más")

    print(f"\netiquetados con confianza < 0.90 (revisar a mano): {len(dudosos)}")
    for confianza, doc_id, language, muestra in sorted(dudosos)[:20]:
        print(f"  {confianza:.3f}  {doc_id:20} {language:4}  {muestra!r}")
