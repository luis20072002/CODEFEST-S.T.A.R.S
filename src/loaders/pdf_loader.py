"""Loader de documentos PDF del corpus.

Sigue el mismo principio de §2.1 que el JSONLoader: leer el archivo y
estructurar explícitamente texto vs. metadata, sin recorrer el objeto de
forma genérica y sin mezclar campos descriptivos (autor, fecha, número de
páginas) dentro del cuerpo del documento.

El loader SOLO lee y estructura. No limpia ni normaliza (eso es §2.2, y va en
el Preprocessor), no detecta idioma, no fragmenta.

CONTRATO: `load()` devuelve `list[Document]`, no un único `Document`. La
mayoría de los PDF producen una lista de un elemento (el texto del propio
PDF), pero cuando el PDF trae figuras o tablas insertadas como imagen (no
como tabla nativa del PDF), cada una se procesa vía OCR (`OCRLoader`) y se
agrega como un `Document` adicional e independiente en la misma lista — no
se mezcla con el texto del cuerpo, porque son unidades de contenido
distintas (una figura/tabla no es un párrafo del artículo).

Cómo se decide qué imagen vale la pena mandar a OCR: NO por tamaño ni por
heurísticas sobre el contenido de los píxeles. Se busca una leyenda tipo
"Figura 3", "Tabla 2", "Gráfico 1" (o su equivalente en inglés/portugués)
cerca de la imagen en el propio PDF — esa es la señal que el documento ya
trae para decir "esto es una figura/tabla", y es mucho más confiable que
adivinar. Una imagen sin leyenda cercana (logos, adornos, marcas de agua) se
descarta sin pasar por OCR.
"""

import re
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF: extracción de imágenes embebidas + sus coordenadas
import pdfplumber

from core.catalog import CatalogEntry
from core.document import Document
from loaders.base_loader import BaseLoader
from loaders.ocr_loader import OCRLoadError, OCRLoader

# Patrón de leyenda de figura/tabla en español, inglés y portugués (el corpus
# mezcla los tres — ver comentario de BODY_FIELDS en json_loader.py sobre
# fuentes en portugués como INPE). Solo mira el INICIO de la línea: una
# leyenda real empieza así, no basta con que la palabra aparezca en medio de
# una oración ("como muestra la figura 2, ...", eso NO es la leyenda misma).
CAPTION_PATTERN = re.compile(
    r"^\s*(figura|fig\.?|tabla|cuadro|gr[aá]fico|grafica|diagrama|"
    r"figure|table|chart|quadro)\s*\.?\s*\d*",
    re.IGNORECASE,
)

# Distancia vertical máxima (en puntos PDF, ~1/72") entre una imagen y el
# bloque de texto candidato a ser su leyenda. Las leyendas van pegadas a la
# imagen (arriba o abajo); un texto a 40pt es "la línea de al lado", más
# lejos que eso ya es párrafo de cuerpo normal que no es leyenda de nada.
CAPTION_MAX_DISTANCE = 40

# Campos de metadata del PDF que sí aportan significado y se conservan.
# El resto de lo que trae el diccionario `pdf.metadata` (claves propietarias
# de cada editor: Producer, CreationDate en formato PDF crudo, etc.) se
# normaliza a nombres en minúscula y SÍ se conserva, salvo lo que está en
# METADATA_DISCARD: son campos técnicos de la herramienta que generó el PDF,
# no describen el contenido (paralelo a RECORD_SKIP_KEYS del JSONLoader).
METADATA_DISCARD = frozenset({
    "trapped", "gts_pdfxversion", "gts_pdfxconformance",
})

# Umbral (en palabras) por debajo del cual el documento se considera un
# posible "stub": o el PDF está escaneado (sin capa de texto) y pdfplumber no
# extrajo nada, o solo trae una portada. Mismo rol que SHORT_BODY_WORDS en el
# JSONLoader: no descarta el documento, solo lo señala para revisión manual.
SHORT_BODY_WORDS = 40

# Regla para separar párrafos dentro del texto que devuelve una página.
# pdfplumber no reconstruye párrafos: entrega líneas ya envueltas al ancho de
# la página. Cuando SÍ trae una línea en blanco real entre bloques (común en
# reportes con espaciado editorial) se respeta como frontera de párrafo;
# cuando no la trae, la página completa se trata como un único bloque de
# texto, porque partir por saltos de línea simples cortaría oraciones a la
# mitad (el salto de línea ahí es solo ajuste de ancho, no de contenido).
BLANK_LINE = re.compile(r"\n\s*\n")


class PDFLoadError(Exception):
    pass


class PDFLoader(BaseLoader):
    """Convierte un archivo .pdf del corpus en una lista de Document:
    el texto del cuerpo (siempre) + una figura/tabla por cada imagen con
    leyenda que se haya podido reconocer vía OCR (si las hay)."""

    def __init__(self) -> None:
        self._ocr_loader = OCRLoader()

    # `entry` trae doc_id / source / phenomenon / format ya resueltos por el
    # catálogo, igual que en JSONLoader: el loader no vuelve a tocar el Excel.
    def load(self, path: str | Path, entry: CatalogEntry) -> list[Document]:
        paginas, pdf_metadata = self._read(path)

        bloques = self._paragraphs_por_pagina(paginas)
        text = "\n\n".join(bloques)

        figuras = self._extraer_figuras_con_leyenda(path)

        metadata = self._build_metadata(pdf_metadata, paginas, figuras)
        # `titulo` es un campo propio de Document, no debe quedar duplicado
        # dentro de metadata_adicional. Se saca de ahí si vino en los
        # metadatos del PDF (puede no venir: muchos PDFs no traen Title en
        # sus propiedades de documento).
        titulo = metadata.pop("title", None)

        documento_principal = Document(
            doc_id=entry.doc_id,
            fuente=entry.source,
            formato=entry.format,
            fenomeno=entry.phenomenon,
            idioma=None,           # lo llena el Preprocessor (§2.2), no el loader
            titulo=titulo,
            texto=text,
            metadata_adicional=metadata,
        )

        documentos = [documento_principal]
        documentos.extend(self._figuras_a_documentos(figuras, entry))
        return documentos

    # ---------- lectura de texto ----------

    @staticmethod
    def _read(path: str | Path) -> tuple[list[str], dict[str, Any]]:
        # pdfplumber sobre pypdf puro: conserva mejor el orden espacial del
        # texto en documentos a varias columnas, que es lo habitual en los
        # informes de CSIS/ESA/SIPRI del corpus (ver pdf-reading skill).
        try:
            with pdfplumber.open(path) as pdf:
                paginas = [pagina.extract_text() or "" for pagina in pdf.pages]
                pdf_metadata = dict(pdf.metadata or {})
        except Exception as error:
            # pdfplumber lanza distintas excepciones según la causa (PDF
            # cifrado, corrupto, con estructura rota); se homogeneizan todas
            # bajo PDFLoadError, igual que JSONLoadError homogeneiza
            # json.JSONDecodeError.
            raise PDFLoadError(f"No se pudo leer el PDF en {path}: {error}") from error

        return paginas, pdf_metadata

    # ---------- extracción de figuras/tablas embebidas como imagen ----------

    @staticmethod
    def _extraer_figuras_con_leyenda(path: str | Path) -> list[dict[str, Any]]:
        # PyMuPDF (fitz) en vez de pdfplumber para esta parte: da acceso
        # directo a los bytes ya decodificados de cada imagen (`extract_image`)
        # y a las coordenadas de texto por bloque, sin tener que lidiar a mano
        # con el filtro de compresión del stream (DCTDecode, FlateDecode...).
        try:
            documento = fitz.open(path)
        except Exception as error:
            raise PDFLoadError(f"No se pudo abrir el PDF en {path} para extraer figuras: {error}") from error

        figuras = []
        try:
            for num_pagina, pagina in enumerate(documento, start=1):
                bloques_texto = pagina.get_text("blocks")
                for img_info in pagina.get_images(full=True):
                    xref = img_info[0]
                    rects = pagina.get_image_rects(xref)
                    if not rects:
                        continue  # imagen referenciada pero no dibujada en la página (recurso reusado)

                    leyenda = _buscar_leyenda(bloques_texto, rects[0])
                    if leyenda is None:
                        continue  # sin leyenda cercana: no es candidata a figura/tabla, se descarta

                    base_image = documento.extract_image(xref)
                    figuras.append({
                        "bytes": base_image["image"],
                        "extension": base_image["ext"],
                        "pagina": num_pagina,
                        "leyenda": leyenda,
                    })
        finally:
            documento.close()

        return figuras

    def _figuras_a_documentos(
        self, figuras: list[dict[str, Any]], entry: CatalogEntry
    ) -> list[Document]:
        documentos = []
        for indice, figura in enumerate(figuras, start=1):
            doc_id_figura = f"{entry.doc_id}_fig{indice:02d}"
            try:
                doc_figura = self._ocr_loader.load_from_bytes(
                    figura["bytes"],
                    doc_id=doc_id_figura,
                    fuente=entry.source,
                    formato="pdf_figura",   # distingue de "pdf" en formato aguas abajo
                    fenomeno=entry.phenomenon,
                    # La leyenda ("Figura 1: ...") es, semánticamente, el
                    # título de este mini-documento — va al campo `titulo`
                    # dedicado en vez de vivir solo dentro de metadata.
                    titulo=figura["leyenda"],
                    extra_metadata={
                        "documento_origen": entry.doc_id,
                        "pagina": figura["pagina"],
                    },
                )
            except OCRLoadError:
                # Una figura que falla en OCR no debería tumbar el PDF
                # completo: se omite esa figura y se sigue con el resto.
                continue
            if doc_figura.texto:
                documentos.append(doc_figura)
        return documentos

    # ---------- estructuración del cuerpo ----------

    def _paragraphs_por_pagina(self, paginas: list[str]) -> list[str]:
        bloques: list[str] = []
        for texto_pagina in paginas:
            bloques.extend(_paragraphs_from_page(texto_pagina))
        return bloques

    # ---------- metadata ----------

    def _build_metadata(
        self, pdf_metadata: dict[str, Any], paginas: list[str], figuras: list[dict[str, Any]]
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {}

        # Los metadatos de PDF vienen con claves en CamelCase heredadas del
        # estándar (Title, Author, CreationDate...). Se normalizan a
        # snake_case en minúscula para que queden consistentes con el resto
        # del pipeline (los JSON ya traen sus claves en minúscula).
        for clave, valor in pdf_metadata.items():
            clave_norm = _snake_case(clave)
            if clave_norm in METADATA_DISCARD:
                continue
            texto = _as_text(valor)
            if texto:
                metadata[clave_norm] = texto

        # Diagnóstico estructural: no es un campo del PDF, se calcula. Sirve
        # para que el mismo tipo de alarma que usa el __main__ del JSONLoader
        # ("menos de 20 palabras") pueda distinguir aquí entre "portada corta"
        # y "PDF escaneado sin capa de texto".
        metadata["n_paginas"] = len(paginas)
        metadata["paginas_sin_texto"] = sum(1 for p in paginas if not p.strip())
        # No es "n_figuras": este es el conteo de figuras/tablas CANDIDATAS
        # (con leyenda detectada), antes de saber si el OCR sobre ellas dio
        # texto útil. El conteo real de Document generados se ve contando los
        # elementos de la lista que devuelve load().
        metadata["n_figuras_candidatas"] = len(figuras)

        return metadata


# ---------- utilidades ----------


def _buscar_leyenda(bloques_texto: list[tuple], rect) -> str | None:
    """Busca, entre los bloques de texto de la página, el más cercano a
    `rect` (el rectángulo de la imagen) cuyo inicio matchee un patrón de
    leyenda de figura/tabla. Devuelve la primera línea de ese bloque, o None
    si ningún bloque cercano matchea."""
    candidatos = []
    for x0, y0, x1, y1, texto, *_resto in bloques_texto:
        primera_linea = texto.strip().splitlines()[0] if texto.strip() else ""
        if not primera_linea or not CAPTION_PATTERN.match(primera_linea):
            continue
        # Distancia vertical entre el bloque de texto y la imagen: si el
        # bloque está arriba de la imagen, mide desde su borde inferior (y1)
        # hasta el borde superior de la imagen (rect.y0); si está debajo, al
        # revés. Se toma la menor de las dos por si acaso.
        distancia = min(abs(y0 - rect.y1), abs(rect.y0 - y1))
        if distancia <= CAPTION_MAX_DISTANCE:
            candidatos.append((distancia, primera_linea))

    if not candidatos:
        return None
    candidatos.sort(key=lambda c: c[0])
    return candidatos[0][1]


def _paragraphs_from_page(texto_pagina: str) -> list[str]:
    """Convierte el texto crudo de una página en una lista de párrafos."""
    if not texto_pagina or not texto_pagina.strip():
        return []

    crudo = texto_pagina.replace("\r\n", "\n").replace("\r", "\n").strip()

    if BLANK_LINE.search(crudo):
        # Hay líneas en blanco reales: se respetan como frontera de párrafo,
        # igual que "\n\n" separa bloques en el JSONLoader.
        partes = BLANK_LINE.split(crudo)
    else:
        # Sin líneas en blanco: la página es un único bloque de prosa. Unir
        # por saltos de línea simples (que aquí son solo ajuste de ancho de
        # columna, no fin de párrafo) evitaría cortar oraciones a la mitad.
        partes = [crudo]

    parrafos = []
    for parte in partes:
        # Dentro de cada bloque, los saltos de línea simples SÍ son ajuste de
        # ancho: se colapsan a espacio para reconstruir la oración continua.
        texto = " ".join(linea.strip() for linea in parte.split("\n") if linea.strip())
        if texto:
            parrafos.append(texto)
    return parrafos


def _snake_case(clave: str) -> str:
    # "CreationDate" -> "creation_date". Las claves de pdfplumber.metadata son
    # pocas y conocidas, así que una regex simple basta (no hace falta un
    # normalizador completo de camelCase con acrónimos).
    return re.sub(r"(?<!^)(?=[A-Z])", "_", clave).lower()


def _as_text(value: Any) -> str:
    """Aplana un valor de metadata de PDF a texto plano, sin inventar formato."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


if __name__ == "__main__":
    import sys
    from collections import Counter

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from core.catalog import Catalog

    root = Path(__file__).resolve().parents[1] / "data" / "data_raw"
    catalog = Catalog.from_excel(root / "Indice_Datos_Codefest.xlsx")
    loader = PDFLoader()

    documentos, fallos, vacios, escaneados = [], [], [], []
    figuras_ocr = []
    for entrada in catalog.entries(format="pdf"):
        try:
            resultado = loader.load(root / entrada.source, entrada)
        except PDFLoadError as error:
            fallos.append((entrada.source, str(error)))
            continue
        principal, *figuras = resultado
        documentos.append(principal)
        figuras_ocr.extend(figuras)

    # Un documento con muy poco texto casi siempre significa que el PDF está
    # escaneado (sin capa de texto). Ya no significa "necesitaría OCR y no lo
    # tiene": las figuras/tablas SÍ pasan por OCR ahora; esto sigue señalando
    # el caso de un PDF completo escaneado como imagen (sin capa de texto en
    # el cuerpo), que es un caso distinto y este loader no cubre.
    for doc in documentos:
        if len(doc.texto.split()) < 20:
            vacios.append(doc)
        if doc.metadata_adicional.get("paginas_sin_texto", 0) == doc.metadata_adicional.get("n_paginas", -1):
            escaneados.append(doc)

    print(f"documentos PDF (cuerpo) cargados : {len(documentos)}")
    print(f"figuras/tablas vía OCR generadas : {len(figuras_ocr)}")
    print(f"fallos de lectura                : {len(fallos)}")
    print(f"con menos de 20 palabras          : {len(vacios)}")
    print(f"posiblemente escaneados           : {len(escaneados)}")

    palabras = sorted(len(d.texto.split()) for d in documentos)
    if palabras:
        print(f"palabras  min={palabras[0]}  mediana={palabras[len(palabras) // 2]}  "
              f"max={palabras[-1]}  total={sum(palabras):,}")

    print("por fenómeno :", dict(sorted(Counter(d.fenomeno for d in documentos).items())))

    for source, error in fallos[:10]:
        print(f"  FALLO: {source} -> {error}")
    for doc in vacios[:10]:
        print(f"  CORTO ({len(doc.texto.split())} pal.): {doc.fuente}")
    for doc in escaneados[:10]:
        print(f"  ESCANEADO: {doc.fuente}")
    for doc in figuras_ocr[:10]:
        print(f"  FIGURA: {doc.doc_id} ({doc.titulo!r}, "
              f"{len(doc.texto.split())} pal. OCR)")