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
from loaders.ocr_loader import OCRLoader, OCRLoadError
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
        self.documentos: list[Document] = []
        self.fallos: list[tuple[str, str, str]] = []  # (source, format, error)

    def agregar_documentos(self, documentos: list[Document]) -> None:
        self.documentos.extend(documentos)

    def agregar_fallo(self, entry: CatalogEntry, error: str) -> None:
        self.fallos.append((entry.source, entry.format, error))

    # ---------- resumen ----------

    def imprimir_resumen(self, tiempo_segundos: float | None = None) -> None:
        print(f"documentos cargados : {len(self.documentos)}")
        print(f"fallos               : {len(self.fallos)}")
        if tiempo_segundos is not None:
            print(f"tiempo total         : {tiempo_segundos:.1f}s")

        por_formato = Counter(d.formato for d in self.documentos)
        print("por formato :", dict(sorted(por_formato.items())))

        por_fenomeno = Counter(d.fenomeno for d in self.documentos)
        print("por fenómeno :", dict(sorted(por_fenomeno.items(), key=lambda kv: (kv[0] is None, kv[0]))))

        vacios = [d for d in self.documentos if not d.texto.strip()]
        print(f"documentos sin texto : {len(vacios)}")

        if self.fallos:
            print("\nfallos por formato :", dict(sorted(Counter(f for _, f, _ in self.fallos).items())))
            print("primeros fallos:")
            for source, formato, error in self.fallos[:15]:
                print(f"  [{formato}] {source} -> {error}")

        if vacios:
            print("\nprimeros documentos sin texto:")
            for doc in vacios[:15]:
                print(f"  [{doc.formato}] {doc.doc_id} ({doc.fuente})")


def load_all(root: Path, catalog: Catalog, verbose: bool = True) -> OrchestrationResult:
    """Recorre TODAS las entradas del catálogo (sin filtrar por formato) y
    devuelve todos los Document que se lograron cargar, junto con el
    registro de lo que falló.

    `verbose=True` imprime progreso en vivo. Es necesario porque el PDF con
    OCR puede tardar varios segundos por archivo (cada figura/tabla candidata
    pasa por Tesseract) — sin esto, una corrida de varios minutos es
    indistinguible de un cuelgue."""
    resultado = OrchestrationResult()
    entradas = list(catalog.entries())
    total = len(entradas)

    for indice, entry in enumerate(entradas, start=1):
        if verbose:
            # flush=True: en algunos entornos (PowerShell redirigiendo a
            # archivo, IDEs) stdout se buffers y no se ve nada hasta el
            # final aunque el proceso sí esté avanzando.
            print(f"[{indice}/{total}] {entry.format:6s} {entry.source}", flush=True)

        loader = LOADERS.get(entry.format)
        if loader is None:
            resultado.agregar_fallo(entry, f"sin loader registrado para el formato {entry.format!r}")
            continue

        try:
            documentos = loader.load(root / entry.source, entry)
        except LOAD_ERRORS as error:
            resultado.agregar_fallo(entry, str(error))
            if verbose:
                print(f"    FALLO: {error}", flush=True)
            continue

        resultado.agregar_documentos(documentos)

    return resultado


if __name__ == "__main__":
    root = Path(__file__).resolve().parent / "data" / "data_raw"
    catalog = Catalog.from_excel(root / "Indice_Datos_Codefest.xlsx")

    inicio = time.time()
    resultado = load_all(root, catalog)
    duracion = time.time() - inicio

    resultado.imprimir_resumen(tiempo_segundos=duracion)