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

# ---------------------------------------------------------------------------
# Política de OCR (decidida el 2026-08-04 con medición, ver ESTADO.md §9)
# ---------------------------------------------------------------------------

# OCR de figuras/tablas embebidas: DESACTIVADO.
#
# Se midió sobre 178 figuras de 60 PDFs repartidos por todo el corpus
# (`py -m tools.inspeccionar_ocr 60`): el 62% daba OCR vacío, el 12% ruido, el
# 7% menos de cinco palabras, y solo el 19% algo aprovechable — degradado
# además de forma sistemática: la sigla «AI» salía como «Al» en el 100% de los
# casos de la muestra, justo el término central de F1.
#
# Contra la Tabla 1 («texto: Texto original del fragmento, sin modificaciones»)
# eso es meter al índice texto que el documento no dice. Y al desactivarlo
# desaparece de paso el problema de ESTADO.md §7: los doc_id inventados
# `<doc_id>_figNN` y el `formato = "pdf_figura"`, que no es ninguno de los
# valores que la Tabla 1 enumera.
#
# La causa de fondo no es Tesseract: es que a una gráfica de barras no hay
# texto que sacarle, y las imágenes embebidas venían a su resolución nativa,
# casi siempre baja. Se deja como interruptor y no borrado, porque
# `_extract_figures_with_caption` sigue siendo útil para el diagnóstico
# (`tools/inspeccionar_ocr.py`) y porque la decisión es reversible si se
# encuentra una forma de filtrar por calidad.
OCR_FIGURAS = False

# OCR de página completa para PDF escaneados: ACTIVADO.
#
# Caso distinto y medido aparte: cuando la página ENTERA es la imagen del
# documento (texto corrido, tipografía de imprenta), el OCR funciona bien. En
# `ALERTAS_informes001.pdf` —0 caracteres de capa de texto— rasterizar y pasar
# Tesseract dio 605 palabras de prosa correcta, con acentos y todo.
# Son los 51 documentos que hoy entran al índice como vectores vacíos.
# §2.1 recomienda OCR explícitamente para este caso.
OCR_PAGINA_COMPLETA = True

# Se rasteriza a 150 DPI y no a 300: medido sobre una página, a 300 DPI se
# PERDÍAN acentos («Defensoria», «dinamicas») que a 150 sí se leían bien, y
# además cuesta 4× más. Provisional: está medido sobre una sola página.
DPI_OCR_PAGINA = 150

# Por debajo de estos caracteres en TODO el documento se considera escaneado y
# se dispara el OCR de página completa. Se decide a nivel de documento y no de
# página para no rasterizar páginas en blanco sueltas de PDFs que sí tienen
# texto: eso multiplicaría el coste sobre los 757 PDF para no ganar nada.
UMBRAL_DOC_ESCANEADO = 200

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
    """Convierte un archivo .pdf del corpus en una lista con UN Document: el
    texto del cuerpo.

    Si el PDF está escaneado (sin capa de texto útil), ese cuerpo se
    reconstruye por OCR de página completa. El OCR de figuras embebidas está
    desactivado — ver el bloque «Política de OCR» arriba, con la medición que
    sustenta las dos decisiones."""

    def __init__(self) -> None:
        self._ocr_loader = OCRLoader()

    # `entry` trae doc_id / source / phenomenon / format ya resueltos por el
    # catálogo, igual que en JSONLoader: el loader no vuelve a tocar el Excel.
    def load(self, path: str | Path, entry: CatalogEntry) -> list[Document]:
        pages, pdf_metadata = self._read(path)

        blocks = self._paragraphs_per_page(pages)
        text = "\n\n".join(blocks)

        # Rescate de escaneados: si el PDF apenas trae capa de texto, el
        # cuerpo se reconstruye rasterizando cada página y pasándola por OCR.
        ocr_de_pagina = False
        if OCR_PAGINA_COMPLETA and len(text.strip()) < UMBRAL_DOC_ESCANEADO:
            paginas_ocr = self._ocr_paginas_completas(path)
            texto_ocr = "\n\n".join(p for p in paginas_ocr if p.strip())
            # Solo se reemplaza si el OCR aportó MÁS que lo que ya había: un
            # PDF legítimamente corto (una portada, un oficio de media página)
            # no debe empeorar por pasar por aquí.
            if len(texto_ocr.strip()) > len(text.strip()):
                text = texto_ocr
                ocr_de_pagina = True

        figures = self._extract_figures_with_caption(path) if OCR_FIGURAS else []

        metadata = self._build_metadata(pdf_metadata, pages, figures)
        # Trazabilidad: deja constancia de que este texto NO se leyó de la capa
        # de texto del PDF sino que lo reconoció un OCR. Importa para auditar
        # el índice y para el informe técnico.
        if ocr_de_pagina:
            metadata["ocr_pagina_completa"] = True
            metadata["ocr_dpi"] = DPI_OCR_PAGINA
        # `title` es un campo propio de Document, no debe quedar duplicado
        # dentro de metadata. Se saca de ahí si vino en los metadatos del PDF
        # (puede no venir: muchos PDFs no traen Title en sus propiedades de
        # documento).
        title = metadata.pop("title", None)

        main_document = Document(
            doc_id=entry.doc_id,
            source=entry.source,
            format=entry.format,
            phenomenon=entry.phenomenon,
            language=None,         # lo llena el Preprocessor (§2.2), no el loader
            title=title,
            text=text,
            metadata=metadata,
        )

        documents = [main_document]
        documents.extend(self._figures_to_documents(figures, entry))
        return documents

    # ---------- lectura de texto ----------

    @staticmethod
    def _read(path: str | Path) -> tuple[list[str], dict[str, Any]]:
        # pdfplumber sobre pypdf puro: conserva mejor el orden espacial del
        # texto en documentos a varias columnas, que es lo habitual en los
        # informes de CSIS/ESA/SIPRI del corpus (ver pdf-reading skill).
        try:
            with pdfplumber.open(path) as pdf:
                pages = [page.extract_text() or "" for page in pdf.pages]
                pdf_metadata = dict(pdf.metadata or {})
        except Exception as error:
            # pdfplumber lanza distintas excepciones según la causa (PDF
            # cifrado, corrupto, con estructura rota); se homogeneizan todas
            # bajo PDFLoadError, igual que JSONLoadError homogeneiza
            # json.JSONDecodeError.
            raise PDFLoadError(f"No se pudo leer el PDF en {path}: {error}") from error

        return pages, pdf_metadata

    # ---------- extracción de figuras/tablas embebidas como imagen ----------

    @staticmethod
    def _extract_figures_with_caption(path: str | Path) -> list[dict[str, Any]]:
        # PyMuPDF (fitz) en vez de pdfplumber para esta parte: da acceso
        # directo a los bytes ya decodificados de cada imagen (`extract_image`)
        # y a las coordenadas de texto por bloque, sin tener que lidiar a mano
        # con el filtro de compresión del stream (DCTDecode, FlateDecode...).
        try:
            document = fitz.open(path)
        except Exception as error:
            raise PDFLoadError(f"No se pudo abrir el PDF en {path} para extraer figuras: {error}") from error

        figures = []
        try:
            for page_number, page in enumerate(document, start=1):
                text_blocks = page.get_text("blocks")
                for img_info in page.get_images(full=True):
                    xref = img_info[0]
                    rects = page.get_image_rects(xref)
                    if not rects:
                        continue  # imagen referenciada pero no dibujada en la página (recurso reusado)

                    caption = _find_caption(text_blocks, rects[0])
                    if caption is None:
                        continue  # sin leyenda cercana: no es candidata a figura/tabla, se descarta

                    base_image = document.extract_image(xref)
                    figures.append({
                        "bytes": base_image["image"],
                        "extension": base_image["ext"],
                        "pagina": page_number,
                        "leyenda": caption,
                    })
        finally:
            document.close()

        return figures

    # ---------- OCR de página completa (PDF escaneados) ----------

    def _ocr_paginas_completas(self, path: str | Path) -> list[str]:
        """Rasteriza cada página y le pasa OCR. Devuelve el texto por página.

        Se usa fitz y no pdfplumber porque `get_pixmap(dpi=...)` renderiza la
        página entera —incluido lo que sea imagen— a un mapa de bits, que es
        justo lo que necesita Tesseract. pdfplumber solo sabe leer la capa de
        texto, que en estos archivos está vacía.
        """
        try:
            documento = fitz.open(path)
        except Exception as error:
            raise PDFLoadError(
                f"No se pudo abrir el PDF en {path} para el OCR de página: {error}"
            ) from error

        textos: list[str] = []
        try:
            for numero, pagina in enumerate(documento, start=1):
                pixmap = pagina.get_pixmap(dpi=DPI_OCR_PAGINA)
                try:
                    textos.append(self._ocr_loader.text_from_bytes(
                        pixmap.tobytes("png"),
                        contexto=f"{Path(path).name} p.{numero}",
                    ))
                except OCRLoadError:
                    # Una página ilegible no debe tumbar el documento entero:
                    # se pierde esa página y se sigue con las demás.
                    textos.append("")
        finally:
            documento.close()

        return textos

    def _figures_to_documents(
        self, figures: list[dict[str, Any]], entry: CatalogEntry
    ) -> list[Document]:
        documents = []
        for index, figure in enumerate(figures, start=1):
            figure_doc_id = f"{entry.doc_id}_fig{index:02d}"
            try:
                figure_doc = self._ocr_loader.load_from_bytes(
                    figure["bytes"],
                    doc_id=figure_doc_id,
                    source=entry.source,
                    format="pdf_figura",   # distingue de "pdf" en formato aguas abajo
                    phenomenon=entry.phenomenon,
                    # La leyenda ("Figura 1: ...") es, semánticamente, el
                    # título de este mini-documento — va al campo `title`
                    # dedicado en vez de vivir solo dentro de metadata.
                    title=figure["leyenda"],
                    extra_metadata={
                        "documento_origen": entry.doc_id,
                        "pagina": figure["pagina"],
                    },
                )
            except OCRLoadError:
                # Una figura que falla en OCR no debería tumbar el PDF
                # completo: se omite esa figura y se sigue con el resto.
                continue
            if figure_doc.text:
                documents.append(figure_doc)
        return documents

    # ---------- estructuración del cuerpo ----------

    def _paragraphs_per_page(self, pages: list[str]) -> list[str]:
        blocks: list[str] = []
        for page_text in pages:
            blocks.extend(_paragraphs_from_page(page_text))
        return blocks

    # ---------- metadata ----------

    def _build_metadata(
        self, pdf_metadata: dict[str, Any], pages: list[str], figures: list[dict[str, Any]]
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {}

        # Los metadatos de PDF vienen con claves en CamelCase heredadas del
        # estándar (Title, Author, CreationDate...). Se normalizan a
        # snake_case en minúscula para que queden consistentes con el resto
        # del pipeline (los JSON ya traen sus claves en minúscula).
        for key, value in pdf_metadata.items():
            normalized_key = _snake_case(key)
            if normalized_key in METADATA_DISCARD:
                continue
            text = _as_text(value)
            if text:
                metadata[normalized_key] = text

        # Diagnóstico estructural: no es un campo del PDF, se calcula. Sirve
        # para que el mismo tipo de alarma que usa el __main__ del JSONLoader
        # ("menos de 20 palabras") pueda distinguir aquí entre "portada corta"
        # y "PDF escaneado sin capa de texto".
        metadata["n_paginas"] = len(pages)
        metadata["paginas_sin_texto"] = sum(1 for p in pages if not p.strip())
        # No es "n_figuras": este es el conteo de figuras/tablas CANDIDATAS
        # (con leyenda detectada), antes de saber si el OCR sobre ellas dio
        # texto útil. El conteo real de Document generados se ve contando los
        # elementos de la lista que devuelve load().
        metadata["n_figuras_candidatas"] = len(figures)

        return metadata


# ---------- utilidades ----------


def _find_caption(text_blocks: list[tuple], rect) -> str | None:
    """Busca, entre los bloques de texto de la página, el más cercano a
    `rect` (el rectángulo de la imagen) cuyo inicio matchee un patrón de
    leyenda de figura/tabla. Devuelve la primera línea de ese bloque, o None
    si ningún bloque cercano matchea."""
    candidates = []
    for x0, y0, x1, y1, text, *_rest in text_blocks:
        first_line = text.strip().splitlines()[0] if text.strip() else ""
        if not first_line or not CAPTION_PATTERN.match(first_line):
            continue
        # Distancia vertical entre el bloque de texto y la imagen: si el
        # bloque está arriba de la imagen, mide desde su borde inferior (y1)
        # hasta el borde superior de la imagen (rect.y0); si está debajo, al
        # revés. Se toma la menor de las dos por si acaso.
        distance = min(abs(y0 - rect.y1), abs(rect.y0 - y1))
        if distance <= CAPTION_MAX_DISTANCE:
            candidates.append((distance, first_line))

    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0])
    return candidates[0][1]


def _paragraphs_from_page(page_text: str) -> list[str]:
    """Convierte el texto crudo de una página en una lista de párrafos."""
    if not page_text or not page_text.strip():
        return []

    raw = page_text.replace("\r\n", "\n").replace("\r", "\n").strip()

    if BLANK_LINE.search(raw):
        # Hay líneas en blanco reales: se respetan como frontera de párrafo,
        # igual que "\n\n" separa bloques en el JSONLoader.
        parts = BLANK_LINE.split(raw)
    else:
        # Sin líneas en blanco: la página es un único bloque de prosa. Unir
        # por saltos de línea simples (que aquí son solo ajuste de ancho de
        # columna, no fin de párrafo) evitaría cortar oraciones a la mitad.
        parts = [raw]

    paragraphs = []
    for part in parts:
        # Dentro de cada bloque, los saltos de línea simples SÍ son ajuste de
        # ancho: se colapsan a espacio para reconstruir la oración continua.
        text = " ".join(line.strip() for line in part.split("\n") if line.strip())
        if text:
            paragraphs.append(text)
    return paragraphs


def _snake_case(key: str) -> str:
    # "CreationDate" -> "creation_date". Las claves de pdfplumber.metadata son
    # pocas y conocidas, así que una regex simple basta (no hace falta un
    # normalizador completo de camelCase con acrónimos).
    return re.sub(r"(?<!^)(?=[A-Z])", "_", key).lower()


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

    documents, failures, short, scanned = [], [], [], []
    ocr_figures = []
    for entry in catalog.entries(format="pdf"):
        try:
            result = loader.load(root / entry.source, entry)
        except PDFLoadError as error:
            failures.append((entry.source, str(error)))
            continue
        main_doc, *figures = result
        documents.append(main_doc)
        ocr_figures.extend(figures)

    # Un documento con muy poco texto casi siempre significa que el PDF está
    # escaneado (sin capa de texto). Ya no significa "necesitaría OCR y no lo
    # tiene": las figuras/tablas SÍ pasan por OCR ahora; esto sigue señalando
    # el caso de un PDF completo escaneado como imagen (sin capa de texto en
    # el cuerpo), que es un caso distinto y este loader no cubre.
    for doc in documents:
        if len(doc.text.split()) < 20:
            short.append(doc)
        if doc.metadata.get("paginas_sin_texto", 0) == doc.metadata.get("n_paginas", -1):
            scanned.append(doc)

    print(f"documentos PDF (cuerpo) cargados : {len(documents)}")
    print(f"figuras/tablas vía OCR generadas : {len(ocr_figures)}")
    print(f"fallos de lectura                : {len(failures)}")
    print(f"con menos de 20 palabras          : {len(short)}")
    print(f"posiblemente escaneados           : {len(scanned)}")

    word_counts = sorted(len(d.text.split()) for d in documents)
    if word_counts:
        print(f"palabras  min={word_counts[0]}  mediana={word_counts[len(word_counts) // 2]}  "
              f"max={word_counts[-1]}  total={sum(word_counts):,}")

    print("por fenómeno :", dict(sorted(Counter(d.phenomenon for d in documents).items())))

    for source, error in failures[:10]:
        print(f"  FALLO: {source} -> {error}")
    for doc in short[:10]:
        print(f"  CORTO ({len(doc.text.split())} pal.): {doc.source}")
    for doc in scanned[:10]:
        print(f"  ESCANEADO: {doc.source}")
    for doc in ocr_figures[:10]:
        print(f"  FIGURA: {doc.doc_id} ({doc.title!r}, "
              f"{len(doc.text.split())} pal. OCR)")
