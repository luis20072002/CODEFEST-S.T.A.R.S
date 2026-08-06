from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Document:
    """
    Representación estandarizada de un documento, independiente de su
    formato de origen (PDF, JSON, CSV, PBF, imagen...). Todos los loaders del
    pipeline devuelven instancias de esta clase, y todo lo que viene después
    (Preprocessor, Chunker, Encoder) trabaja únicamente sobre Document, sin
    conocer el formato original.

    Los nombres de los campos están en INGLÉS, igual que en CatalogEntry
    (core/catalog.py), para que todo el proyecto use una sola convención y no
    haya que traducir mentalmente al pasar de un módulo a otro.

    La metadata obligatoria de la Tabla 1 va en español, así que esa
    traducción se hace UNA sola vez, al serializar el metadata.jsonl de la
    entrega, y no en cada clase del pipeline. La correspondencia es directa:

        source → fuente          format     → formato
        text   → texto           phenomenon → fenomeno
    """

    # Campos obligatorios: sin ellos el documento no es identificable ni
    # trazable hasta el archivo original.
    doc_id: str
    source: str                        # nombre o URL del archivo original
    format: str                        # "pdf", "json", "csv", "xlsx", "pbf", ...

    # Campos con valor por defecto. En una dataclass TIENEN que ir después de
    # los obligatorios: Python arma __init__ en este mismo orden, y un
    # parámetro sin default no puede seguir a uno que sí lo tiene.
    text: str = ""                     # texto limpio, listo para el Chunker
    phenomenon: Optional[int] = None   # 1, 2 o 3 (puede asignarse tras cargar)
    title: Optional[str] = None
    language: Optional[str] = None

    # Cualquier información adicional que no forme parte de los campos
    # obligatorios (autores, fecha, tags, url, etc.) vive aquí y no se
    # mezcla nunca con `text`.
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.doc_id:
            raise ValueError("doc_id no puede estar vacío")
        if not self.source:
            raise ValueError("source no puede estar vacío")
        if self.phenomenon is not None and self.phenomenon not in (1, 2, 3):
            raise ValueError(f"phenomenon debe ser 1, 2 o 3 (recibido: {self.phenomenon})")

    def __repr__(self) -> str:
        preview = (self.text[:60] + "…") if len(self.text) > 60 else self.text
        return (
            f"Document(doc_id={self.doc_id!r}, format={self.format!r}, "
            f"phenomenon={self.phenomenon!r}, text={preview!r})"
        )
