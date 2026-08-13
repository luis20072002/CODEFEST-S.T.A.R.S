"""Prueba cross-lingual: ¿una consulta en español recupera documentos en inglés?

    py -m tools.prueba_crosslingual              # con las consultas de ejemplo
    py -m tools.prueba_crosslingual 5            # top-5 en vez de top-10
    py -m tools.prueba_crosslingual --pdf        # con las 50 consultas reales

────────────────────────────────────────────────────────────────────────────────
POR QUÉ ESTA PRUEBA ES LA MÁS IMPORTANTE QUE SE PUEDE HACER HOY

El *ground truth* de §10 no es público, así que **NDCG@10 y F1@3 no se pueden
medir**. Lo que sí se puede comprobar es la hipótesis sobre la que descansa toda
la Fase 5, y que está documentada en `ESTADO.md` §5:

    las 50 consultas están TODAS en español
    y el corpus es 55% inglés, 35% español, 6% portugués

Si el encoder alinea bien es→en, una consulta en español debe recuperar
fragmentos en inglés cuando el contenido relevante esté en inglés. Si todo lo
que devuelve está en español, el encoder está agrupando por **idioma** en vez de
por **significado**, y entonces el 55% del corpus es prácticamente
inalcanzable — sin que ninguna prueba técnica lo delate.

────────────────────────────────────────────────────────────────────────────────
CÓMO SE LEE EL RESULTADO

Se compara el reparto de idiomas de lo recuperado contra el del corpus:

    corpus:  en 55%  ·  es 35%  ·  pt 6%

  - Reparto parecido al del corpus → el encoder ignora el idioma y ordena por
    significado. Es lo que queremos.
  - Casi todo en español → sesgo monolingüe. Habría que cambiar de encoder, y
    cuanto antes se sepa mejor.
  - Casi todo en inglés → no es necesariamente malo (el corpus tiene más
    inglés), pero conviene mirar si los textos son realmente pertinentes.

⚠️ **Esto es un diagnóstico, no una métrica.** Un reparto sano no garantiza un
buen NDCG; un reparto enfermo sí garantiza uno malo.

────────────────────────────────────────────────────────────────────────────────
NO USA FAISS A PROPÓSITO

Con 91.021 vectores, la similitud coseno es un producto matricial de `numpy`
sobre `embeddings.npy`. Así la prueba se puede correr en cualquier máquina que
tenga el modelo, sin depender de que el índice esté construido — que es justo la
situación en la que suele hacer falta.
"""

import json
import sys
from collections import Counter
from pathlib import Path

DATOS = Path(__file__).resolve().parents[1] / "data"
CHUNKS = DATOS / "chunks.jsonl"
EMBEDDINGS = DATOS / "embeddings.npy"
PREGUNTAS = DATOS / "Extracto_Preguntas_50_v2.pdf"

# Reparto de idiomas del corpus, medido en `ESTADO.md` §11. Es la referencia
# contra la que se juzga lo recuperado.
CORPUS = {"en": 55.0, "es": 35.1, "pt": 5.9}

# Consultas de ejemplo, en español, una por fenómeno. NO son las de ADL: son
# nuestras, escritas para que el diagnóstico se pueda correr sin el PDF de
# preguntas, que no está en el repositorio (`.gitignore` excluye *.pdf).
CONSULTAS_EJEMPLO = [
    "¿Cómo se usa la inteligencia artificial en sistemas de armas autónomos?",
    "riesgo de colisión por basura espacial en órbita baja terrestre",
    "impacto de la minería ilegal en territorios indígenas de la Amazonía",
]


def cargar_consultas_del_pdf(ruta: Path) -> list:
    """Extrae las 50 consultas reales del PDF de ADL, si está disponible."""
    import re

    from pypdf import PdfReader

    texto = "\n".join((p.extract_text() or "") for p in PdfReader(ruta).pages)
    # Las consultas vienen como `q001 ... ¿...?`; se parte por el identificador.
    partes = re.split(r"\bq0*(\d{1,3})\b", texto)
    consultas = []
    for i in range(1, len(partes) - 1, 2):
        cuerpo = " ".join(partes[i + 1].split())
        if len(cuerpo) > 15:
            consultas.append(cuerpo[:300])
    return consultas


def main() -> int:
    import numpy as np

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    argumentos = sys.argv[1:]
    usar_pdf = "--pdf" in argumentos
    if usar_pdf:
        argumentos.remove("--pdf")
    top_k = int(argumentos[0]) if argumentos else 10

    for ruta in (CHUNKS, EMBEDDINGS):
        if not ruta.is_file():
            print(f"No existe {ruta}.")
            return 1

    if usar_pdf:
        if not PREGUNTAS.is_file():
            print(f"No existe {PREGUNTAS}. Súbelo o corre sin --pdf.")
            return 1
        consultas = cargar_consultas_del_pdf(PREGUNTAS)
        print(f"consultas: {len(consultas)} leídas de {PREGUNTAS.name}")
    else:
        consultas = CONSULTAS_EJEMPLO
        print(f"consultas: {len(consultas)} de ejemplo (nuestras, no las de ADL)")

    # Metadata de los fragmentos. Solo idioma, doc_id y un recorte del texto:
    # cargar los 236 MB completos no hace falta para un diagnóstico.
    print("leyendo chunks.jsonl…")
    idiomas, doc_ids, chunk_ids, textos = [], [], [], []
    with open(CHUNKS, encoding="utf-8") as f:
        for linea in f:
            if not linea.strip():
                continue
            d = json.loads(linea)
            idiomas.append(d.get("language") or "?")
            doc_ids.append(d["doc_id"])
            chunk_ids.append(d["chunk_id"])
            textos.append(d["text"][:160].replace("\n", " "))

    matriz = np.load(EMBEDDINGS, mmap_mode="r")
    if matriz.shape[0] != len(idiomas):
        print(f"✖ {matriz.shape[0]:,} vectores y {len(idiomas):,} chunks: no cuadran.")
        return 1
    print(f"índice: {matriz.shape[0]:,} vectores de dimensión {matriz.shape[1]}")

    from embedding.encoder import cargar_modelo, codificar

    print("\ncargando el modelo… (la primera vez descarga 4,35 GB)")
    modelo = cargar_modelo()

    print("codificando las consultas…")
    vectores = codificar(modelo, consultas)

    # Producto interno sobre vectores normalizados = coseno (§5.2). Se hace por
    # bloques para no materializar 91.021 × n_consultas de golpe.
    print("buscando…\n")
    matriz = np.asarray(matriz, dtype=np.float32)
    similitudes = matriz @ vectores.T          # (n_chunks, n_consultas)

    global_reparto: Counter = Counter()
    linea = "─" * 78

    for i, consulta in enumerate(consultas):
        col = similitudes[:, i]
        # argpartition es O(n) y basta: solo hace falta el top-k, no el orden
        # completo de los 91.021.
        mejores = np.argpartition(-col, top_k)[:top_k]
        mejores = mejores[np.argsort(-col[mejores])]

        reparto = Counter(idiomas[j] for j in mejores)
        global_reparto.update(reparto)

        print(f"{linea}\nCONSULTA {i + 1}: {consulta[:110]}")
        print(f"idiomas del top-{top_k}: {dict(reparto)}")
        for rango, j in enumerate(mejores[:5], start=1):
            print(f"  {rango}. [{idiomas[j]}] {col[j]:.4f}  {chunk_ids[j]}")
            print(f"     {textos[j][:120]}")
        print()

    total = sum(global_reparto.values())
    print(f"{linea}\nREPARTO GLOBAL DE LO RECUPERADO ({total} fragmentos)")
    print(f"{'idioma':<8} {'recuperado':>12} {'en el corpus':>14}")
    for idioma, cuantos in global_reparto.most_common():
        pct = 100 * cuantos / total
        ref = CORPUS.get(idioma)
        print(f"{idioma:<8} {pct:>11.1f}% {(f'{ref:.1f}%' if ref else '—'):>14}")

    pct_es = 100 * global_reparto.get("es", 0) / total
    print(f"\n{linea}")
    if pct_es > 85:
        print("⚠️  SESGO MONOLINGÜE: casi todo lo recuperado está en español, y el")
        print("    corpus es 55% inglés. El encoder estaría agrupando por idioma y no")
        print("    por significado. Revisar la elección del encoder (§4.3).")
    else:
        print("✔  El encoder recupera en varios idiomas: no hay sesgo monolingüe")
        print("    evidente. Recuerda que esto es un diagnóstico, no una métrica —")
        print("    hay que MIRAR si los textos de arriba son realmente pertinentes.")
    print(linea)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
