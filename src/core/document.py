from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Document:
    """
    Representación estandarizada de un documento, independiente de su
    formato de origen (PDF, JSON, HTML, CSV, ...). Todos los loaders del
    pipeline devuelven una instancia de esta clase, y todo lo que viene
    después (Preprocessor, Chunker, Encoder) trabaja únicamente sobre
    Document, sin conocer el formato original.

    Los nombres de los campos coinciden deliberadamente con la metadata
    obligatoria a nivel de documento definida en la especificación
    (doc_id, fuente, formato, fenomeno), para que el Chunker pueda
    propagarlos directamente a cada Chunk sin necesidad de traducirlos.
    """

    doc_id: str
    fuente: str                        # nombre o URL del archivo original
    formato: str                       # "pdf", "html", "json", "csv", "xlsx", ...
    texto: str = ""                    # texto limpio, listo para el Chunker
    fenomeno: Optional[int] = None     # 1, 2 o 3 (puede asignarse tras cargar)
    titulo: Optional[str] = None
    idioma: Optional[str] = None

    # Cualquier información adicional que no forme parte de los campos
    # obligatorios (autores, fecha, tags, url, etc.) vive aquí y no se
    # mezcla nunca con `texto`.
    metadata_adicional: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.doc_id:
            raise ValueError("doc_id no puede estar vacío")
        if not self.fuente:
            raise ValueError("fuente no puede estar vacía")
        if self.fenomeno is not None and self.fenomeno not in (1, 2, 3):
            raise ValueError(f"fenomeno debe ser 1, 2 o 3 (recibido: {self.fenomeno})")

    def __repr__(self) -> str:
        preview = (self.texto[:60] + "…") if len(self.texto) > 60 else self.texto
        return (
            f"Document(doc_id={self.doc_id!r}, formato={self.formato!r}, "
            f"fenomeno={self.fenomeno!r}, texto={preview!r})"
        )