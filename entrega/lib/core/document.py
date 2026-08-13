"""Representación estandarizada de un documento del corpus.

Un `Document` corresponde a un archivo individual provisto por ADL, que es la
definición de documento que fija §2.3. Todos los extractores del pipeline
devuelven instancias de esta clase, y las etapas posteriores —normalización,
fragmentación y codificación— trabajan únicamente sobre ella, sin conocer el
formato de origen.

Los identificadores del código están en inglés y los nombres de campo de la
Tabla 1 en español. La traducción se realiza en un único punto del sistema,
`Chunk.to_metadata_record()`, al serializar el `metadata.jsonl`:

    source → fuente          format     → formato
    text   → texto           phenomenon → fenomeno

Concentrar la traducción en un solo lugar evita que dos módulos mantengan
listas de campos distintas y que el almacén de metadata acabe con claves que no
son las que exige la Tabla 1.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Document:
    """Un archivo del corpus, ya extraído a texto plano."""

    # Campos obligatorios: sin ellos el documento no es trazable hasta el
    # archivo original que entregó ADL.
    doc_id: str                        # DOC_ID oficial del índice de datos
    source: str                        # ruta relativa del archivo original
    format: str                        # extensión real: "pdf", "json", "csv", …

    # Campos con valor por defecto. En una dataclass deben declararse después
    # de los obligatorios, porque el constructor se genera en este mismo orden.
    text: str = ""                     # texto normalizado, listo para fragmentar
    phenomenon: Optional[int] = None   # 1, 2 o 3 (Tabla 1)
    title: Optional[str] = None
    language: Optional[str] = None

    # Información descriptiva que no forma parte del contenido del documento
    # (autores, fecha, etiquetas, URL). §2.1 recomienda conservarla como
    # metadata en lugar de mezclarla con el cuerpo del texto, y por eso vive
    # aquí y nunca se concatena a `text`.
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Valida lo que, de ser incorrecto, no produciría ningún error visible.

        Un `phenomenon` fuera de {1, 2, 3} o un `source` vacío no interrumpen el
        pipeline por su cuenta: producen un `metadata.jsonl` que incumple la
        Tabla 1 varias etapas más adelante.
        """
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
