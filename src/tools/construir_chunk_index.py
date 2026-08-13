"""Construye `chunk_index.json`, el mapa `chunk_id → faiss_id` que pide §8.5.

    py -m tools.construir_chunk_index [--salida <ruta>]

────────────────────────────────────────────────────────────────────────────────
PARA QUÉ SIRVE Y POR QUÉ HACE FALTA

`retrieval.grafo.Grafo.evidencia()` solo acumula los fragmentos que estén en el
mapa `faiss_id`: sin este archivo devuelve `{}` **siempre y en silencio**, así
que la integración de §8.5 parece funcionar y no hace nada. Es exactamente el
tipo de fallo mudo que este proyecto verifica.

El mapa no se calcula en tiempo de consulta —recorrer las 91.021 líneas del
`metadata.jsonl` cuesta más que la búsqueda entera— y tampoco requiere modelo ni
GPU: sale de leer el propio `metadata.jsonl` entregado, cuyo campo `faiss_id` es
el identificador interno del índice (§5.3).

────────────────────────────────────────────────────────────────────────────────
POR QUÉ SOLO LOS FRAGMENTOS QUE EL GRAFO NOMBRA

El mapa completo tendría 91.021 entradas y ~2,5 MB. Se restringe a los
`chunk_id` que el grafo referencia —en sus nodos y en sus aristas— porque
`evidencia()` no puede proponer ningún otro: el resultado es **idéntico** y el
archivo, una fracción del tamaño. Cada byte de más en `entrega/` es superficie
que el jurado tiene que descargar para nada.

La corrida informa además de cuántos `chunk_id` del grafo **no** aparecen en el
`metadata.jsonl`. Debe ser 0: es la trazabilidad de §7.2, la misma que comprueba
el bloque B de `tools.verificar_grafo`. Si sale distinto de 0, el grafo y el
índice no son de la misma pasada y la integración estaría mintiendo.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Set

from retrieval.grafo import CARPETA_GRAFO, GRAFO, Grafo
from retrieval.search import METADATA

SALIDA = CARPETA_GRAFO / "chunk_index.json"


def chunks_referenciados(grafo: Grafo) -> Set[str]:
    """Todos los `chunk_id` que el grafo nombra, en nodos y en aristas."""
    referidos: Set[str] = set()
    for chunks in grafo.chunks_de_nodo.values():
        referidos.update(c for c in chunks if c)
    for chunks in grafo.chunks_de_arista.values():
        referidos.update(c for c in chunks if c)
    return referidos


def construir(referidos: Set[str], metadata: Path = METADATA) -> Dict[str, int]:
    """Una sola pasada sobre el `metadata.jsonl` para resolver los que interesan."""
    mapa: Dict[str, int] = {}
    with open(metadata, encoding="utf-8") as f:
        for numero, linea in enumerate(f):
            if not linea.strip():
                continue
            registro = json.loads(linea)
            cid = registro["chunk_id"]
            if cid in referidos:
                # §5.3: `faiss_id` es el identificador interno del índice. Se
                # prefiere el campo explícito y se cae al número de línea solo
                # si no estuviera, que es la relación que §5.3 garantiza.
                mapa[cid] = int(registro.get("faiss_id", numero))
    return mapa


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    salida = SALIDA
    if "--salida" in sys.argv:
        salida = Path(sys.argv[sys.argv.index("--salida") + 1])

    if not GRAFO.is_file():
        print(f"No existe {GRAFO}.")
        return 1
    if not METADATA.is_file():
        print(f"No existe {METADATA}. Descarga la base vectorial (README §3).")
        return 1

    linea = "─" * 78
    print(linea)
    print(f"grafo    : {GRAFO}")
    print(f"metadata : {METADATA}")

    grafo = Grafo(GRAFO, chunk_index=Path("no-existe"))
    referidos = chunks_referenciados(grafo)
    print(f"nodos {len(grafo):,} · fragmentos referenciados por el grafo: {len(referidos):,}")

    mapa = construir(referidos)
    faltan = referidos - set(mapa)

    print(f"resueltos contra el metadata : {len(mapa):,}")
    if faltan:
        print(f"🔴 {len(faltan):,} `chunk_id` del grafo NO existen en el metadata.")
        print("   El grafo y el índice no son de la misma pasada (§7.2).")
        for cid in sorted(faltan)[:5]:
            print(f"     {cid}")
        return 1
    print("✔ los 0 faltantes que exige §7.2: el grafo y el índice se corresponden")

    # Claves ordenadas para que dos corridas den el mismo archivo byte a byte,
    # que es lo que exige la reproducibilidad de §1.4.
    salida.parent.mkdir(parents=True, exist_ok=True)
    texto = json.dumps(dict(sorted(mapa.items())), ensure_ascii=False, indent=0)
    salida.write_text(texto, encoding="utf-8")

    print(f"escrito  : {salida}  ({salida.stat().st_size:,} B)")
    print(linea)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
