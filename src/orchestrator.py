"""Orquestador del pipeline de carga.

Es el único módulo que sabe qué loader corresponde a cada `format` del
catálogo. Todo lo que viene después (Preprocessor, Chunker, Encoder) recibe
la lista de `Document` ya estandarizados y no necesita saber si un documento
vino de un JSON, un PDF, un CSV/XLSX o un PBF.

No decide NADA sobre el contenido de los documentos (eso es responsabilidad
de cada loader): solo enruta por `format`, agrega los resultados y lleva la
cuenta de qué falló y por qué.
"""

import time
from collections import Counter
from pathlib import Path

from core.catalog import Catalog, CatalogEntry
from core.document import Document
from loaders.json_loader import JSONLoader, JSONLoadError
from loaders.ocr_loader import OCRLoader, OCRLoadError, ensure_tesseract
from loaders.pbf_loader import PBFLoader, PBFLoadError
from loaders.pdf_loader import PDFLoader, PDFLoadError
from loaders.tabular_loader import TabularLoader, TabularLoadError
from loaders.base_loader import BaseLoader

# Un loader por formato. Si el día de mañana el corpus trae un formato nuevo
# (HTML, DOCX...), lo único que hay que tocar es este diccionario y escribir
# el loader correspondiente — nada más del pipeline necesita cambiar.
#
# "image" está listo para cuando el catálogo traiga archivos de imagen
# sueltos (hoy no existen: OCRLoader solo se usa internamente desde
# PDFLoader para las figuras/tablas embebidas).
LOADERS: dict[str, BaseLoader] = {
    "json": JSONLoader(),
    "pdf": PDFLoader(),
    "csv": TabularLoader(),
    "xlsx": TabularLoader(),
    "pbf": PBFLoader(),
    "image": OCRLoader(),
}

# Excepciones de carga de cada loader, homogeneizadas aquí para que el
# orquestador solo necesite un único except. Una excepción NO listada aquí
# (un bug real, no un archivo corrupto) se deja propagar: silenciarla
# escondería un error de programación bajo un simple "fallo de lectura".
LOAD_ERRORS = (JSONLoadError, PDFLoadError, TabularLoadError, PBFLoadError, OCRLoadError)


class OrchestrationResult:
    """Resultado de correr load_all: documentos cargados + registro de fallos."""

    def __init__(self) -> None:
        self.documents: list[Document] = []
        self.failures: list[tuple[str, str, str]] = []  # (source, format, error)

    def add_documents(self, documents: list[Document]) -> None:
        self.documents.extend(documents)

    def add_failure(self, entry: CatalogEntry, error: str) -> None:
        self.failures.append((entry.source, entry.format, error))

    # ---------- resumen ----------

    def print_summary(self, elapsed_seconds: float | None = None) -> None:
        print(f"documentos cargados : {len(self.documents)}")
        print(f"fallos               : {len(self.failures)}")
        if elapsed_seconds is not None:
            print(f"tiempo total         : {elapsed_seconds:.1f}s")

        by_format = Counter(d.format for d in self.documents)
        print("por formato :", dict(sorted(by_format.items())))

        by_phenomenon = Counter(d.phenomenon for d in self.documents)
        print("por fenómeno :", dict(sorted(by_phenomenon.items(), key=lambda kv: (kv[0] is None, kv[0]))))

        empty = [d for d in self.documents if not d.text.strip()]
        print(f"documentos sin texto : {len(empty)}")

        if self.failures:
            print("\nfallos por formato :", dict(sorted(Counter(f for _, f, _ in self.failures).items())))
            print("primeros fallos:")
            for source, fmt, error in self.failures[:15]:
                print(f"  [{fmt}] {source} -> {error}")

        if empty:
            print("\nprimeros documentos sin texto:")
            for doc in empty[:15]:
                print(f"  [{doc.format}] {doc.doc_id} ({doc.source})")


def load_all(root: Path, catalog: Catalog, verbose: bool = True) -> OrchestrationResult:
    """Recorre TODAS las entradas del catálogo (sin filtrar por formato) y
    devuelve todos los Document que se lograron cargar, junto con el
    registro de lo que falló.

    `verbose=True` imprime progreso en vivo. Es necesario porque el PDF con
    OCR puede tardar varios segundos por archivo (cada figura/tabla candidata
    pasa por Tesseract) — sin esto, una corrida de varios minutos es
    indistinguible de un cuelgue."""
    result = OrchestrationResult()
    entries = list(catalog.entries())
    total = len(entries)

    # Chequeo previo del entorno de OCR. Se hace ANTES del bucle, y solo si de
    # verdad va a hacer falta, por dos razones:
    #   1. Fallar en el segundo 0 en vez del minuto 20. El primer formato que
    #      usa OCR aparece a mitad del recorrido; sin esto, una máquina mal
    #      configurada se descubre cuando ya se procesaron cientos de archivos.
    #   2. Dejar constancia de la versión exacta del motor con la que se
    #      construyó el índice, que es lo que hay que citar en el informe
    #      técnico (§1.4): pip freeze no captura dependencias de sistema.
    # Si Tesseract no está o le faltan idiomas, esto lanza y detiene la corrida:
    # es preferible a construir un índice al que le falta todo el OCR.
    if any(entry.format in ("pdf", "image") for entry in entries):
        entorno = ensure_tesseract()
        if verbose:
            print(f"OCR: Tesseract {entorno['version']} · lang={entorno['lang']}", flush=True)

    for index, entry in enumerate(entries, start=1):
        if verbose:
            # flush=True: en algunos entornos (PowerShell redirigiendo a
            # archivo, IDEs) stdout se buffers y no se ve nada hasta el
            # final aunque el proceso sí esté avanzando.
            print(f"[{index}/{total}] {entry.format:6s} {entry.source}", flush=True)

        loader = LOADERS.get(entry.format)
        if loader is None:
            result.add_failure(entry, f"sin loader registrado para el formato {entry.format!r}")
            continue

        try:
            documents = loader.load(root / entry.source, entry)
        except LOAD_ERRORS as error:
            result.add_failure(entry, str(error))
            if verbose:
                print(f"    FALLO: {error}", flush=True)
            continue

        result.add_documents(documents)

    return result


if __name__ == "__main__":
    root = Path(__file__).resolve().parent / "data" / "data_raw"
    catalog = Catalog.from_excel(root / "Indice_Datos_Codefest.xlsx")

    start = time.time()
    result = load_all(root, catalog)
    elapsed = time.time() - start

    result.print_summary(elapsed_seconds=elapsed)
