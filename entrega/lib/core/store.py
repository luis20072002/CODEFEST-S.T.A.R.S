"""Persistencia en JSON Lines de los objetos intermedios del pipeline.

Este módulo escribe y relee los archivos de trabajo que enlazan las etapas de
construcción del índice: los documentos extraídos, los documentos normalizados y
los fragmentos previos a la codificación.

No debe confundirse con el `metadata.jsonl` entregado. Aquel contiene fragmentos
con los nombres de campo en español que exige la Tabla 1 y lo produce la etapa
de serialización llamando a `Chunk.to_metadata_record()`. Los archivos de este
módulo contienen objetos del pipeline con los nombres en inglés y no forman
parte de la entrega.

POR QUÉ JSON LINES. Es el mismo formato que §1.4 exige para el almacén de
metadata, de modo que todo el pipeline trabaja con el formato final desde el
principio; y es el único que permite ir escribiendo a medida que se procesa, sin
mantener el corpus completo en memoria.

DOS DETALLES DE CODIFICACIÓN QUE NO SON OPCIONALES

- `encoding="utf-8"` explícito. En Windows la codificación preferida del sistema
  es cp1252: sin este argumento los acentos se corrompen y un carácter fuera de
  ese juego —una comilla tipográfica, un guion largo— interrumpe la escritura a
  mitad de proceso.
- `ensure_ascii=False`. Con el valor por defecto, `json` escapa todo lo no ASCII
  y «órbita» se almacenaría como «\\u00f3rbita». Sigue siendo JSON válido, pero
  §10.2.1 evalúa los fragmentos por el contenido de su campo de texto y conviene
  que los archivos sean legibles y auditables tal cual.
"""

# El comité fijó Python >= 3.9.5 como entorno de evaluación, y este módulo anota
# `str | Path` (PEP 604), que no existe hasta 3.10. Las anotaciones se evalúan al
# definir la función, así que sin esta línea el import falla con TypeError. Con
# ella quedan como cadenas y no se evalúan nunca. No cambia el comportamiento en
# 3.10+, y aquí no hay dataclasses ni introspección de anotaciones que lo note.
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Iterator

from core.chunk import Chunk
from core.document import Document


class StoreError(Exception):
    pass


def write_documents(path: str | Path, documents: Iterable[Document]) -> int:
    """Escribe los `Document` en un `.jsonl` y devuelve cuántos escribió.

    Acepta cualquier iterable, no solo una lista, de modo que la extracción
    puede entregar los documentos según los produce sin acumularlos en memoria.

    Escribe primero en un archivo temporal y renombra al terminar. Si el proceso
    se interrumpe, el archivo válido de la ejecución anterior permanece intacto
    en lugar de quedar parcialmente sobrescrito.
    """
    destino = Path(path)
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporal = destino.with_suffix(destino.suffix + ".parcial")

    n = 0
    try:
        with open(temporal, "w", encoding="utf-8", newline="\n") as f:
            for documento in documents:
                f.write(json.dumps(asdict(documento), ensure_ascii=False))
                f.write("\n")
                n += 1
    except OSError as error:
        raise StoreError(f"No se pudo escribir {destino}: {error}") from error

    # replace() sustituye el destino de forma atómica si ya existía.
    temporal.replace(destino)
    return n


def read_documents(path: str | Path) -> Iterator[Document]:
    """Lee un `.jsonl` y devuelve los `Document` uno a uno.

    Es un generador: no carga el archivo completo en memoria.

    Reconstruye objetos `Document` y no diccionarios, de modo que se revalida
    `__post_init__`: una línea con un `phenomenon` inválido o un `doc_id` vacío
    falla aquí y no tres etapas más adelante.
    """
    origen = Path(path)
    if not origen.is_file():
        raise StoreError(f"No existe el archivo de documentos: {origen}")

    with open(origen, encoding="utf-8") as f:
        for numero, linea in enumerate(f, start=1):
            linea = linea.strip()
            if not linea:
                continue      # tolera una línea en blanco final
            try:
                yield Document(**json.loads(linea))
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                # El número de línea es lo único que permite localizar el
                # problema en un archivo de decenas de miles de líneas.
                raise StoreError(f"{origen}: línea {numero} inválida: {error}") from error


def write_chunks(path: str | Path, chunks: Iterable[Chunk]) -> int:
    """Escribe los `Chunk` en un `.jsonl` y devuelve cuántos escribió.

    Misma mecánica que `write_documents`: archivo temporal y renombrado, UTF-8
    explícito y `ensure_ascii=False`.
    """
    destino = Path(path)
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporal = destino.with_suffix(destino.suffix + ".parcial")

    n = 0
    try:
        with open(temporal, "w", encoding="utf-8", newline="\n") as f:
            for fragmento in chunks:
                f.write(json.dumps(asdict(fragmento), ensure_ascii=False))
                f.write("\n")
                n += 1
    except OSError as error:
        raise StoreError(f"No se pudo escribir {destino}: {error}") from error

    temporal.replace(destino)
    return n


def read_chunks(path: str | Path) -> Iterator[Chunk]:
    """Lee un `.jsonl` de fragmentos y devuelve los `Chunk` uno a uno.

    Igual que `read_documents`, reconstruye objetos y revalida el contrato: una
    `position` negativa o un `num_tokens` en cero fallan aquí y no al construir
    el índice.
    """
    origen = Path(path)
    if not origen.is_file():
        raise StoreError(f"No existe el archivo de fragmentos: {origen}")

    with open(origen, encoding="utf-8") as f:
        for numero, linea in enumerate(f, start=1):
            linea = linea.strip()
            if not linea:
                continue
            try:
                yield Chunk(**json.loads(linea))
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                raise StoreError(f"{origen}: línea {numero} inválida: {error}") from error
