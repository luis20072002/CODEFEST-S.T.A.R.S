"""Loader de documentos PDF del corpus.

Sigue el mismo principio de §2.1 que el JSONLoader: leer el archivo y
estructurar explícitamente texto vs. metadata, sin recorrer el objeto de
forma genérica y sin mezclar campos descriptivos (autor, fecha, número de
páginas) dentro del cuerpo del documento.

El loader SOLO lee y estructura. No limpia ni normaliza (eso es §2.2, y va en
el Preprocessor), no detecta idioma, no fragmenta y no hace OCR. Si un PDF
está escaneado (sin capa de texto), el loader lo deja constar en metadata en
vez de inventar contenido.
"""

import re
from pathlib import Path
from typing import Any

import pdfplumber

from core.catalog import CatalogEntry
from core.document import Document
from loaders.base_loader import BaseLoader

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
    """Convierte un archivo .pdf del corpus en un Document."""

    # `entry` trae doc_id / source / phenomenon / format ya resueltos por el
    # catálogo, igual que en JSONLoader: el loader no vuelve a tocar el Excel.
    def load(self, path: str | Path, entry: CatalogEntry) -> Document:
        paginas, pdf_metadata = self._read(path)

        bloques = self._paragraphs_por_pagina(paginas)
        text = "\n\n".join(bloques)

        metadata = self._build_metadata(pdf_metadata, paginas)

        return Document(
            doc_id=entry.doc_id,
            source=entry.source,
            format=entry.format,
            phenomenon=entry.phenomenon,
            language=None,        # lo llena el Preprocessor (§2.2), no el loader
            text=text,
            metadata=metadata,
        )

    # ---------- lectura ----------

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

    # ---------- estructuración del cuerpo ----------

    def _paragraphs_por_pagina(self, paginas: list[str]) -> list[str]:
        bloques: list[str] = []
        for texto_pagina in paginas:
            bloques.extend(_paragraphs_from_page(texto_pagina))
        return bloques

    # ---------- metadata ----------

    def _build_metadata(
        self, pdf_metadata: dict[str, Any], paginas: list[str]
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

        return metadata


# ---------- utilidades ----------


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
    for entrada in catalog.entries(format="pdf"):
        try:
            documentos.append(loader.load(root / entrada.source, entrada))
        except PDFLoadError as error:
            fallos.append((entrada.source, str(error)))

    # Un documento con muy poco texto casi siempre significa que el PDF está
    # escaneado (sin capa de texto) y necesitaría OCR, que este loader
    # deliberadamente no hace. Es la señal de alarma equivalente a la del
    # JSONLoader, pero aquí se distingue explícitamente ese caso.
    for doc in documentos:
        if len(doc.text.split()) < 20:
            vacios.append(doc)
        if doc.metadata.get("paginas_sin_texto", 0) == doc.metadata.get("n_paginas", -1):
            escaneados.append(doc)

    print(f"documentos PDF cargados  : {len(documentos)}")
    print(f"fallos de lectura        : {len(fallos)}")
    print(f"con menos de 20 palabras : {len(vacios)}")
    print(f"posiblemente escaneados  : {len(escaneados)}")

    palabras = sorted(len(d.text.split()) for d in documentos)
    if palabras:
        print(f"palabras  min={palabras[0]}  mediana={palabras[len(palabras) // 2]}  "
              f"max={palabras[-1]}  total={sum(palabras):,}")

    print("por fenómeno :", dict(sorted(Counter(d.phenomenon for d in documentos).items())))

    for source, error in fallos[:10]:
        print(f"  FALLO: {source} -> {error}")
    for doc in vacios[:10]:
        print(f"  CORTO ({len(doc.text.split())} pal.): {doc.source}")
    for doc in escaneados[:10]:
        print(f"  ESCANEADO: {doc.source}")