"""Almacén de metadata (§5.3): produce el `metadata.jsonl` de la entrega.

    py -m indexing.metadata              # chunks.jsonl → entrega/…/metadata.jsonl

────────────────────────────────────────────────────────────────────────────────
QUÉ ES ESTE ARCHIVO Y POR QUÉ EXISTE

§5.3 explica la separación: «FAISS almacena únicamente los vectores numéricos y
sus identificadores enteros internos. La metadata asociada a cada fragmento
(Tabla 1) debe mantenerse en un almacén separado que mapee el identificador
interno de FAISS al chunk_id y al resto de los campos».

Es decir, el índice no sabe qué texto tiene cada vector. Este archivo es la otra
mitad, y **es un entregable de §1.4**: va en
`entrega/base_vectorial/encoder_<nombre>/metadata.jsonl`.

⚠️ **NO confundir con `chunks.jsonl`.** Aquel es un archivo de trabajo interno,
con los nombres de campo en inglés del pipeline y los diagnósticos del chunker.
Este lleva **los nombres en español de la Tabla 1** y solo lo que se entrega.

────────────────────────────────────────────────────────────────────────────────
EL ORDEN ES EL CONTRATO, Y ES LO ÚNICO QUE NO SE PUEDE EQUIVOCAR

    línea i de metadata.jsonl  ==  fila i de embeddings.npy  ==  id interno i de FAISS

Las tres cosas se derivan del mismo sitio —el orden de `chunks.jsonl`— y por eso
coinciden. Si alguna se desalineara, cada consulta devolvería el texto de otro
fragmento **sin lanzar ninguna excepción**: los resultados saldrían plausibles y
mal, y las métricas de §10 no lo distinguirían de un mal encoder.

Por eso este módulo hace dos cosas contra ese riesgo:

1. Escribe un campo **`faiss_id`** explícito con el número de línea. §3.4
   autoriza campos adicionales, y este convierte el invariante en algo
   **comprobable** en vez de confiado: `tools/verificar_indice.py` recupera el
   vector de una fila y verifica que FAISS devuelve ese mismo `faiss_id`.
2. **No ordena, no filtra y no salta nada.** Recorre `chunks.jsonl` en su orden
   y escribe una línea por fragmento, siempre.

────────────────────────────────────────────────────────────────────────────────
LA TRADUCCIÓN AL ESPAÑOL NO SE HACE AQUÍ

La hace `Chunk.to_metadata_record()`, que es el único punto del proyecto donde
aparecen las claves de la Tabla 1. Este módulo la llama; no rehace el
diccionario. Si lo rehiciera habría dos listas de campos y un día dejarían de
coincidir — y el modo de fallo sería un `metadata.jsonl` con las claves
equivocadas, que incumple la Tabla 1 entera.
"""

import json
import sys
from pathlib import Path

from core.store import read_chunks

RAIZ = Path(__file__).resolve().parents[2]
DATOS = Path(__file__).resolve().parents[1] / "data"
CHUNKS = DATOS / "chunks.jsonl"

# §1.4 fija el árbol de la entrega: `base_vectorial/encoder_<nombre>/`. El
# nombre sale del modelo elegido (`BAAI/bge-m3`), quedándose con la parte del
# modelo y no la del autor, que es lo que lo identifica.
CARPETA_ENCODER = RAIZ / "entrega" / "base_vectorial" / "encoder_bge-m3"
METADATA = CARPETA_ENCODER / "metadata.jsonl"


def write_metadata(
    ruta_chunks: Path = CHUNKS,
    salida: Path = METADATA,
    *,
    con_faiss_id: bool = True,
    contar_tokens=None,
) -> int:
    """Escribe el `metadata.jsonl` de la entrega y devuelve cuántas líneas puso.

    `con_faiss_id=False` deja únicamente los ocho campos de la Tabla 1 más
    `idioma`. Se puede usar si algún día se prefiere un archivo mínimo, pero
    entonces se pierde la comprobación de alineación de
    `tools/verificar_indice.py`, que es barata y protege el fallo más caro del
    proyecto.

    `contar_tokens` **recalcula `num_tokens` con el tokenizador del encoder**.
    Existe por un caso real (2026-08-06): la codificación se lanzó sobre un
    `chunks.jsonl` construido con el contador de PALABRAS, así que su campo
    `num_tokens` contenía palabras. La Tabla 1 define ese campo como «Número de
    tokens del fragmento», de modo que entregarlo así sería declarar una cosa
    por otra.

    Se corrige **aquí y no reescribiendo `chunks.jsonl`** por dos razones:

    - El texto de los fragmentos **no cambia**, así que los vectores ya
      calculados siguen siendo válidos: no hay que re-codificar nada.
    - `chunks.jsonl` es un archivo interno y su `num_tokens` refleja
      honestamente el presupuesto con el que se construyó. Reescribirlo
      borraría ese rastro.
    """
    salida.parent.mkdir(parents=True, exist_ok=True)
    temporal = salida.with_suffix(salida.suffix + ".parcial")

    n = 0
    # `newline="\n"` y UTF-8 explícitos: esta máquina usa cp1252 por defecto y
    # §9.3/§10.2.1 evalúan el CONTENIDO del texto. Un acento mal escrito aquí
    # es un fragmento que deja de emparejar.
    with open(temporal, "w", encoding="utf-8", newline="\n") as f:
        for fragmento in read_chunks(ruta_chunks):
            registro = fragmento.to_metadata_record()
            if contar_tokens is not None:
                registro["num_tokens"] = contar_tokens(fragmento.text)
            if con_faiss_id:
                registro["faiss_id"] = n
            f.write(json.dumps(registro, ensure_ascii=False))
            f.write("\n")
            n += 1

    temporal.replace(salida)
    return n


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not CHUNKS.is_file():
        print(f"No existe {CHUNKS}. Corre antes `py -m chunking.chunker --tokens`.")
        raise SystemExit(1)

    # `--tokens` recalcula num_tokens con el tokenizador del encoder. Hace falta
    # cuando el chunks.jsonl se generó con el contador de palabras (sin
    # `--tokens` en el chunker): la Tabla 1 pide tokens, no palabras.
    contar = None
    if "--tokens" in sys.argv[1:]:
        # Solo el tokenizador (~17 MB), no los pesos del modelo (4,35 GB):
        # contar tokens no necesita la red neuronal.
        from embedding.encoder import cargar_tokenizador, contador_de_tokens

        print("cargando el tokenizador del encoder (~17 MB)…")
        contar = contador_de_tokens(cargar_tokenizador())

    print(f"origen : {CHUNKS}")
    print(f"salida : {METADATA}")
    print(f"num_tokens: {'recalculado con el tokenizador' if contar else 'tal como viene de chunks.jsonl'}")

    n = write_metadata(contar_tokens=contar)
    tam = METADATA.stat().st_size / 1024 / 1024

    print(f"\nlíneas escritas : {n:,}")
    print(f"tamaño          : {tam:,.1f} MB")

    # Se relee la primera línea y se comprueban los 8 obligatorios de la Tabla 1
    # sobre el archivo YA escrito, no sobre el objeto en memoria: es la única
    # forma de detectar un problema de serialización.
    from core.chunk import TABLA1_FIELDS

    with open(METADATA, encoding="utf-8") as f:
        primera = json.loads(f.readline())

    obligatorios = set(TABLA1_FIELDS.values())
    faltan = obligatorios - primera.keys()
    print(f"\ncampos de la Tabla 1 en la primera línea: "
          f"{len(obligatorios) - len(faltan)}/8" + (f"  ✖ FALTAN {faltan}" if faltan else "  ✔"))
    print("claves:", list(primera))
    print("\nprimera línea (texto recortado):")
    muestra = dict(primera)
    muestra["texto"] = muestra["texto"][:80] + "…"
    print(json.dumps(muestra, ensure_ascii=False, indent=2))
