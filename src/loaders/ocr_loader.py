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
from pathlib import Path
from typing import Any

import pytesseract 
from PIL import Image

from core.catalog import CatalogEntry
from core.document import Document
from loaders.base_loader import BaseLoader

# spa+eng cubre la mayoría del corpus (español + fuentes en inglés como
# CSIS/ESA/SIPRI). Si el corpus trae PDFs en portugués con figuras/tablas que
# necesiten OCR (p. ej. INPE), hay que instalar el paquete de idioma
# `tesseract-ocr-por` y añadir "por" aquí; no se agrega por defecto para no
# asumir un paquete que puede no estar instalado en la máquina de destino.
OCR_LANG = "spa+eng"


class OCRLoadError(Exception):
    pass


class OCRLoader(BaseLoader):
    """Convierte una imagen (de archivo o de bytes en memoria) en un Document vía OCR."""

    # ---------- caso 1: imagen como entrada propia del catálogo ----------

    def load(self, path: str | Path, entry: CatalogEntry) -> list[Document]:
        try:
            imagen = Image.open(path)
            imagen.load()  # fuerza la lectura ahora; si el archivo está
                            # corrupto, falla aquí y no más adelante en el OCR
        except Exception as error:
            raise OCRLoadError(f"No se pudo abrir la imagen en {path}: {error}") from error

        texto = self._ocr(imagen)
        metadata = {"ancho_px": imagen.width, "alto_px": imagen.height}

        return [Document(
            doc_id=entry.doc_id,
            fuente=entry.source,
            formato=entry.format,
            fenomeno=entry.phenomenon,
            idioma=None,           # lo llena el Preprocessor (§2.2), no el loader
            texto=texto,
            metadata_adicional=metadata,
        )]

    # ---------- caso 2: imagen embebida en otro documento (p. ej. un PDF) ----------

    def load_from_bytes(
        self,
        image_bytes: bytes,
        *,
        doc_id: str,
        fuente: str,
        formato: str,
        fenomeno: int | None,
        titulo: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> Document:
        """Construye un Document a partir de los bytes de una imagen que no
        tiene entrada propia en el catálogo. Quien llama es responsable de
        armar un doc_id que no choque con el del documento contenedor (p. ej.
        `f"{entry.doc_id}_fig01"`)."""
        try:
            imagen = Image.open(io.BytesIO(image_bytes))
            imagen.load()
        except Exception as error:
            raise OCRLoadError(f"No se pudo decodificar la imagen embebida ({doc_id}): {error}") from error

        texto = self._ocr(imagen)
        metadata = {"ancho_px": imagen.width, "alto_px": imagen.height}
        if extra_metadata:
            metadata.update(extra_metadata)

        return Document(
            doc_id=doc_id,
            fuente=fuente,
            formato=formato,
            fenomeno=fenomeno,
            idioma=None,
            titulo=titulo,
            texto=texto,
            metadata_adicional=metadata,
        )

    # ---------- OCR ----------

    @staticmethod
    def _ocr(imagen: Image.Image) -> str:
        try:
            texto = pytesseract.image_to_string(imagen, lang=OCR_LANG)
        except Exception as error:
            raise OCRLoadError(f"Tesseract falló al procesar la imagen: {error}") from error
        # Tesseract deja líneas en blanco de sobra entre bloques reconocidos;
        # se recorta al final, no se colapsa el interior (eso ya es limpieza
        # de §2.2, no estructuración).
        return texto.strip()