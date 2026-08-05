"""Persistencia intermedia de los Document, entre extracción y chunking.

⚠️ NO CONFUNDIR con el `metadata.jsonl` de la entrega (§1.4). Aquel lleva
**fragmentos** con los nombres de campo en español de la Tabla 1. Este lleva
**documentos** con los nombres en inglés del pipeline, es un archivo de trabajo
interno y no se entrega.

POR QUÉ EXISTE. La primera corrida completa del orquestador (2026-08-04) no
guardó nada: los 3569 Document murieron con el proceso, y volver a mirarlos
costaba re-extraer los 1826 archivos otra vez. Con esto, la extracción se paga
UNA vez y todo lo que viene después (cleaner, detección de idioma, chunking,
encoder) parte de un archivo que se lee en segundos.

FORMATO: JSON Lines, un Document por línea, UTF-8. Es el mismo formato que
pide §1.4 para la entrega, así que el equipo trabaja con el formato final desde
el principio, y además es el único que permite ir escribiendo a medida que se
extrae, sin tener que construir la lista entera en memoria.

DOS DETALLES DE CODIFICACIÓN QUE NO SON OPCIONALES EN WINDOWS:

- `encoding="utf-8"` explícito. En esta máquina `locale.getpreferredencoding()`
  es **cp1252**: sin este argumento, `open(..., "w")` escribe en cp1252, los
  acentos salen mal y un carácter fuera de ese juego (una comilla tipográfica,
  un guion largo) **revienta la escritura a media corrida**.
- `ensure_ascii=False`. Con el valor por defecto (True) `json` escapa todo lo
  no-ASCII y "órbita" acabaría como "\\u00f3rbita". Es JSON válido, pero
  §10.2.1 evalúa los fragmentos por el CONTENIDO del campo de texto: conviene
  que el archivo se pueda leer y auditar tal cual.
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Iterator

from core.document import Document


class StoreError(Exception):
    pass


def write_documents(path: str | Path, documents: Iterable[Document]) -> int:
    """Escribe los Document en un .jsonl y devuelve cuántos escribió.

    Acepta cualquier iterable, no solo una lista: así el día que el orquestador
    entregue los documentos según los va produciendo, esto no necesita cambiar
    ni mantener nada en memoria.

    Escribe primero en un archivo temporal y renombra al final. Si la corrida
    se corta a la mitad (o se cae la máquina), el .jsonl bueno de la corrida
    anterior sigue intacto en vez de quedar medio pisado — que es justo el
    fallo que este módulo existe para evitar.
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

    # replace() sobreescribe el destino de forma atómica si ya existía.
    temporal.replace(destino)
    return n


def read_documents(path: str | Path) -> Iterator[Document]:
    """Lee un .jsonl y va devolviendo Document, uno a uno.

    Es un generador: no carga el archivo entero en memoria. Para tenerlos todos
    en una lista basta con `list(read_documents(ruta))`.

    Reconstruye Document de verdad (no diccionarios), así que se revalida el
    `__post_init__`: si una línea trae un `phenomenon` inválido o un `doc_id`
    vacío, salta aquí y no tres etapas más adelante.
    """
    origen = Path(path)
    if not origen.is_file():
        raise StoreError(f"No existe el archivo de documentos: {origen}")

    with open(origen, encoding="utf-8") as f:
        for numero, linea in enumerate(f, start=1):
            linea = linea.strip()
            if not linea:
                continue      # tolera una línea en blanco al final del archivo
            try:
                yield Document(**json.loads(linea))
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                # El número de línea es lo único que permite encontrar el
                # problema en un archivo de miles de líneas.
                raise StoreError(f"{origen}: línea {numero} inválida: {error}") from error


if __name__ == "__main__":
    # Diagnóstico: `py -m core.store [ruta.jsonl]` desde src/.
    # Verifica que el archivo se puede releer entero y resume qué hay dentro.
    import sys
    from collections import Counter

    ruta = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).resolve().parents[1] / "data" / "documentos.jsonl"
    )

    documentos = list(read_documents(ruta))
    print(f"{ruta}\n")
    print(f"documentos            : {len(documentos)}")
    print(f"palabras totales      : {sum(len(d.text.split()) for d in documentos):,}")
    print(f"documentos sin texto  : {sum(1 for d in documentos if not d.text.strip())}")
    print("por formato  :", dict(sorted(Counter(d.format for d in documentos).items())))
    print("por fenómeno :", dict(sorted(Counter(d.phenomenon for d in documentos).items(),
                                        key=lambda kv: (kv[0] is None, kv[0]))))
    rescatados = [d for d in documentos if d.metadata.get("ocr_pagina_completa")]
    print(f"rescatados por OCR de página completa : {len(rescatados)}")
