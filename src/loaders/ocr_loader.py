"""Loader OCR de imágenes del corpus.

Dos formas de uso:

1. Como loader normal del catálogo (`load`), para el día en que el corpus
   traiga archivos de imagen sueltos (format="image") — hoy no existen, pero
   la interfaz queda lista.
2. Como utilidad interna (`load_from_bytes`) para convertir a Document una
   imagen que NO tiene su propia entrada en el catálogo porque vive embebida
   dentro de otro archivo (p. ej. una figura o tabla dentro de un PDF). Esta
   es la vía que usa `PDFLoader`.

El loader SOLO ejecuta OCR y estructura el resultado. No decide qué imagen
vale la pena procesar (esa decisión —¿es una figura/tabla real o un logo?—
la toma quien llama, con el contexto que tiene: PDFLoader la toma mirando si
hay una leyenda "Figura X"/"Tabla X" cerca de la imagen en el PDF).
"""

import io
import os
import shutil
from pathlib import Path
from typing import Any

import pytesseract
from PIL import Image

from core.catalog import CatalogEntry
from core.document import Document
from loaders.base_loader import BaseLoader

# spa+eng cubre la mayoría del corpus (español + fuentes en inglés como
# CSIS/ESA/SIPRI). El idioma `por` YA está instalado en la máquina de trabajo,
# pero no se activa todavía: añadirlo cambia el texto que sale del OCR y por
# tanto el contenido del índice, así que es una decisión que se toma midiendo
# sobre una muestra (ver Fase 1 de TAREAS.md), no por si acaso.
OCR_LANG = "spa+eng"

# Rutas donde el instalador de Windows deja el ejecutable. Solo se consultan si
# `tesseract` no aparece en el PATH — el caso típico es una terminal que se
# abrió antes de instalarlo.
_RUTAS_HABITUALES = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"/usr/bin/tesseract",
    r"/usr/local/bin/tesseract",
)

# Resultado de la comprobación del entorno, cacheado. Sin cache, cada figura de
# cada PDF volvería a lanzar el ejecutable dos veces (versión + idiomas): con
# 759 PDFs eso son miles de procesos de más.
_entorno_ocr: dict[str, Any] | None = None


class OCRLoadError(Exception):
    pass


class TesseractNoDisponibleError(Exception):
    """El motor de OCR no está instalado, no se encuentra, o le faltan idiomas.

    ⚠️ NO hereda de OCRLoadError, y es a propósito. Es un problema del ENTORNO,
    no del documento que se estaba procesando. `PDFLoader` captura OCRLoadError
    por figura y sigue adelante (una figura ilegible no debe tumbar un PDF de
    200 páginas); si este error entrara en esa jerarquía, una máquina sin
    Tesseract se tragaría el fallo en las 759 iteraciones y produciría un índice
    SIN nada de OCR, sin un solo mensaje de error. Al quedar fuera, propaga
    hasta arriba y detiene la corrida con un mensaje que dice qué instalar.
    """


def ensure_tesseract() -> dict[str, Any]:
    """Localiza el binario de Tesseract y verifica que tenga los idiomas de OCR_LANG.

    Devuelve un dict con la versión y los idiomas encontrados, que sirve para
    dejar constancia en el informe técnico de con qué se construyó el índice.

    Por qué esto existe, y por qué falla en vez de continuar:

    - `pytesseract` es solo un envoltorio; el motor es un ejecutable aparte que
      no instala pip. Si no está, el error nativo es un `FileNotFoundError`
      críptico que no dice qué hacer.
    - Si el idioma pedido NO está instalado, Tesseract **no siempre falla**:
      puede caer a otro idioma y devolver texto igualmente. Eso produciría un
      índice distinto sin un solo error en pantalla, que es el peor escenario
      posible para reproducir la base vectorial. Mejor romper aquí y ruidoso.
    """
    global _entorno_ocr
    if _entorno_ocr is not None:
        return _entorno_ocr

    # Orden de búsqueda: la variable de entorno gana (permite fijar una versión
    # concreta sin tocar el PATH del sistema), luego el PATH, luego las rutas
    # habituales del instalador.
    binario = os.environ.get("TESSERACT_CMD")
    if binario and not Path(binario).is_file():
        # Se avisa en vez de seguir buscando: si alguien se tomó el trabajo de
        # fijar la variable, un typo en la ruta es un error suyo que quiere ver,
        # no algo que debamos resolver a sus espaldas con otro binario.
        raise TesseractNoDisponibleError(
            f"La variable TESSERACT_CMD apunta a {binario!r}, que no existe."
        )
    if not binario:
        binario = shutil.which("tesseract")
    if not binario:
        binario = next((ruta for ruta in _RUTAS_HABITUALES if Path(ruta).is_file()), None)

    if not binario:
        raise TesseractNoDisponibleError(
            "No se encontró el ejecutable de Tesseract (el motor de OCR).\n"
            "  pytesseract es solo un envoltorio: NO trae el motor.\n"
            "  Windows : winget install --id UB-Mannheim.TesseractOCR -e\n"
            "  Debian  : sudo apt install tesseract-ocr tesseract-ocr-spa tesseract-ocr-eng\n"
            "  Si ya está instalado, la terminal probablemente se abrió ANTES de\n"
            "  instalarlo y no ve el PATH nuevo: abre otra, o define TESSERACT_CMD\n"
            "  con la ruta completa a tesseract.exe."
        )

    pytesseract.pytesseract.tesseract_cmd = binario

    try:
        version = str(pytesseract.get_tesseract_version())
        disponibles = set(pytesseract.get_languages(config=""))
    except Exception as error:
        raise TesseractNoDisponibleError(
            f"Se encontró {binario} pero no responde: {error}"
        ) from error

    # OCR_LANG es una cadena tipo "spa+eng": cada componente es un idioma que
    # tiene que existir como .traineddata en la carpeta tessdata.
    pedidos = set(OCR_LANG.split("+"))
    faltantes = sorted(pedidos - disponibles)
    if faltantes:
        raise TesseractNoDisponibleError(
            f"Tesseract {version} está instalado pero le faltan idiomas: {', '.join(faltantes)}.\n"
            f"  OCR_LANG pide: {OCR_LANG}\n"
            f"  Disponibles  : {', '.join(sorted(disponibles))}\n"
            "  El instalador de Windows solo trae 'eng' y 'osd'. Descarga los\n"
            "  .traineddata que falten de github.com/tesseract-ocr/tessdata_fast\n"
            "  y cópialos a la carpeta tessdata\\ (requiere permisos de admin).\n"
            "  Ojo: sin esto el OCR NO se detiene, usa otro idioma y ensucia el\n"
            "  índice en silencio."
        )

    _entorno_ocr = {"binario": binario, "version": version, "lang": OCR_LANG}
    return _entorno_ocr


class OCRLoader(BaseLoader):
    """Convierte una imagen (de archivo o de bytes en memoria) en un Document vía OCR."""

    # ---------- caso 1: imagen como entrada propia del catálogo ----------

    def load(self, path: str | Path, entry: CatalogEntry) -> list[Document]:
        try:
            image = Image.open(path)
            image.load()   # fuerza la lectura ahora; si el archivo está
                           # corrupto, falla aquí y no más adelante en el OCR
        except Exception as error:
            raise OCRLoadError(f"No se pudo abrir la imagen en {path}: {error}") from error

        text = self._ocr(image)
        metadata = {"ancho_px": image.width, "alto_px": image.height}

        return [Document(
            doc_id=entry.doc_id,
            source=entry.source,
            format=entry.format,
            phenomenon=entry.phenomenon,
            language=None,         # lo llena el Preprocessor (§2.2), no el loader
            text=text,
            metadata=metadata,
        )]

    # ---------- caso 2: imagen embebida en otro documento (p. ej. un PDF) ----------

    def load_from_bytes(
        self,
        image_bytes: bytes,
        *,
        doc_id: str,
        source: str,
        format: str,
        phenomenon: int | None,
        title: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> Document:
        """Construye un Document a partir de los bytes de una imagen que no
        tiene entrada propia en el catálogo. Quien llama es responsable de
        armar un doc_id que no choque con el del documento contenedor (p. ej.
        `f"{entry.doc_id}_fig01"`)."""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.load()
        except Exception as error:
            raise OCRLoadError(f"No se pudo decodificar la imagen embebida ({doc_id}): {error}") from error

        text = self._ocr(image)
        metadata = {"ancho_px": image.width, "alto_px": image.height}
        if extra_metadata:
            metadata.update(extra_metadata)

        return Document(
            doc_id=doc_id,
            source=source,
            format=format,
            phenomenon=phenomenon,
            language=None,
            title=title,
            text=text,
            metadata=metadata,
        )

    # ---------- OCR ----------

    @staticmethod
    def _ocr(image: Image.Image) -> str:
        # Comprobación perezosa: la primera imagen paga el chequeo del entorno,
        # las demás lo leen de la cache. Importar el módulo no lanza nada, así
        # que el orquestador puede importarse en una máquina sin Tesseract.
        ensure_tesseract()
        try:
            text = pytesseract.image_to_string(image, lang=OCR_LANG)
        except Exception as error:
            raise OCRLoadError(f"Tesseract falló al procesar la imagen: {error}") from error
        # Tesseract deja líneas en blanco de sobra entre bloques reconocidos;
        # se recorta al final, no se colapsa el interior (eso ya es limpieza
        # de §2.2, no estructuración).
        return text.strip()


if __name__ == "__main__":
    # Diagnóstico del entorno de OCR: `py -m loaders.ocr_loader` desde src/.
    # Sirve para comprobar la máquina ANTES de lanzar una corrida larga, y para
    # copiar la versión exacta al informe técnico.
    from PIL import ImageDraw

    try:
        entorno = ensure_tesseract()
    except TesseractNoDisponibleError as error:
        print("✗ El entorno de OCR NO está listo:\n")
        print(error)
        raise SystemExit(1)

    print(f"✓ Tesseract {entorno['version']}")
    print(f"  binario  : {entorno['binario']}")
    print(f"  OCR_LANG : {entorno['lang']}")
    print(f"  idiomas  : {', '.join(sorted(pytesseract.get_languages(config='')))}")

    # Prueba de extremo a extremo sobre una imagen generada al vuelo: comprueba
    # que el motor de verdad reconoce texto, no solo que el ejecutable existe.
    prueba = Image.new("RGB", (700, 100), "white")
    ImageDraw.Draw(prueba).text((10, 40), "Figura 3: seguridad orbital LEO", fill="black")
    reconocido = OCRLoader._ocr(prueba)
    print(f"\n  prueba de OCR -> {reconocido!r}")
    print("\n✓ Listo para correr el orquestador." if reconocido else
          "\n✗ El OCR no devolvió texto; revisa la instalación.")
