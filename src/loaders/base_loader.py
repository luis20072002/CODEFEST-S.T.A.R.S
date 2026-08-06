from abc import ABC, abstractmethod
from pathlib import Path

from core.catalog import CatalogEntry
from core.document import Document


class BaseLoader(ABC):
    """Interfaz común de todos los loaders del pipeline.

    `entry` es obligatorio: todo loader necesita el CatalogEntry para
    construir el Document (doc_id, source, format, phenomenon) — sin él no
    hay forma de poblar esos campos. La firma anterior (`load(self, path)`)
    no coincidía con cómo lo implementaba ya JSONLoader ni con el resto de
    loaders del pipeline; se corrige aquí para que quede documentado tal
    como se usa en la práctica.

    El retorno es `list[Document]`, no un único `Document`: la mayoría de
    los loaders devuelven una lista de un elemento, pero PDFLoader puede
    devolver varios (el cuerpo del PDF + una figura/tabla por cada imagen
    procesada vía OCR).
    """

    @abstractmethod
    def load(self, path: str | Path, entry: CatalogEntry) -> list[Document]:
        pass