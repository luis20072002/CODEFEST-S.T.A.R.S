from dataclasses import dataclass,field
from typing import Any

@dataclass
class Document:
    doc_id: str
    source: str
    format: str

    text: str
 
    metadata:dict[str,Any] = field(default_factory=dict)