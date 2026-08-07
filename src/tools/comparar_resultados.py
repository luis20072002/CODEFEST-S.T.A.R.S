"""Compara dos `resultados.jsonl`: cuánto cambian y hacia dónde.

    py -m tools.comparar_resultados a.jsonl b.jsonl
    py -m tools.comparar_resultados a.jsonl          # solo describe uno

Sirve para elegir el factor de `--bonificacion` con datos en vez de a ojo.

⚠️ **LO QUE ESTO NO MIDE.** El *ground truth* no es público, así que **no puede
decir cuál de las dos salidas es mejor**. Mide dos cosas distintas:

  - **Correspondencia con el fenómeno**: qué porcentaje de los documentos
    devueltos pertenece al fenómeno de la consulta, según el reparto verificado
    `q001`–`q016`→F1, `q017`–`q032`→F2, `q033`–`q050`→F3 (`ESTADO.md` §5).
    Sin bonificación la referencia medida es **F1 60,4% · F2 87,5% · F3 92,6%**.
  - **Cuánto se mueve la salida**: cuántos puestos cambian entre una y otra.

Que la correspondencia suba **no significa que la recuperación mejore**: solo
significa que la bonificación está haciendo lo que dice hacer. Si el *ground
truth* está etiquetado por tema y no por fuente, subirla podría empeorar el
resultado. Es una decisión de riesgo, y este módulo solo da los números para
tomarla con los ojos abiertos.
"""

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

from retrieval.consultas import fenomeno_de_consulta


def cargar(ruta: Path) -> Dict[str, dict]:
    salida = {}
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            if linea.strip():
                objeto = json.loads(linea)
                salida[objeto["query_id"]] = objeto
    return salida


def correspondencia(resultados: Dict[str, dict]) -> Dict[int, tuple]:
    """Documentos del fenómeno esperado, por fenómeno de la consulta."""
    aciertos: Counter = Counter()
    totales: Counter = Counter()
    for query_id, objeto in resultados.items():
        fenomeno = fenomeno_de_consulta(query_id)
        if not fenomeno:
            continue
        for d in objeto["documents"]:
            totales[fenomeno] += 1
            if d["doc_id"].startswith(f"F{fenomeno}-"):
                aciertos[fenomeno] += 1
    return {f: (aciertos[f], totales[f]) for f in sorted(totales)}


def describir(nombre: str, resultados: Dict[str, dict]) -> None:
    print(f"\n{nombre}")
    print(f"  consultas : {len(resultados)}")
    docs = [d["doc_id"] for o in resultados.values() for d in o["documents"]]
    frags = [f["chunk_id"] for o in resultados.values() for f in o["fragments"]]
    print(f"  documentos distintos : {len(set(docs))} de {len(docs)} puestos")
    print(f"  fragmentos distintos : {len(set(frags))} de {len(frags)} puestos")
    print("  correspondencia con el fenómeno de la consulta:")
    total_a = total_t = 0
    for fenomeno, (a, t) in correspondencia(resultados).items():
        total_a, total_t = total_a + a, total_t + t
        print(f"    F{fenomeno}: {a:>3}/{t:<3} = {100*a/t:5.1f}%")
    if total_t:
        print(f"    global: {total_a}/{total_t} = {100*total_a/total_t:.1f}%")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    a = cargar(Path(sys.argv[1]))
    describir(sys.argv[1], a)

    if len(sys.argv) < 3:
        return 0

    b = cargar(Path(sys.argv[2]))
    describir(sys.argv[2], b)

    linea = "─" * 74
    print(f"\n{linea}\nCUÁNTO CAMBIA de la primera a la segunda")

    docs_cambiados = frags_cambiados = 0
    consultas_iguales = 0
    movidas: List[tuple] = []
    for query_id in sorted(a):
        if query_id not in b:
            continue
        da = [d["doc_id"] for d in a[query_id]["documents"]]
        db = [d["doc_id"] for d in b[query_id]["documents"]]
        fa = [f["chunk_id"] for f in a[query_id]["fragments"]]
        fb = [f["chunk_id"] for f in b[query_id]["fragments"]]
        # Se compara como CONJUNTO en documentos porque F1@3 no considera el
        # orden (§10.2.2), y como lista en fragmentos porque NDCG@10 sí.
        distintos_doc = len(set(da) - set(db))
        distintos_frag = sum(1 for x, y in zip(fa, fb) if x != y)
        docs_cambiados += distintos_doc
        frags_cambiados += distintos_frag
        if not distintos_doc and not distintos_frag:
            consultas_iguales += 1
        elif distintos_doc:
            movidas.append((query_id, da, db))

    print(f"  consultas idénticas      : {consultas_iguales} de {len(a)}")
    print(f"  puestos de documento que cambian : {docs_cambiados} de {3*len(a)}")
    print(f"  puestos de fragmento que cambian : {frags_cambiados} de {10*len(a)}")

    if movidas:
        print("\n  consultas donde cambian los documentos (hasta 10):")
        for query_id, da, db in movidas[:10]:
            print(f"    {query_id}  {da}")
            print(f"    {'':<5}→ {db}")
    print(linea)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
