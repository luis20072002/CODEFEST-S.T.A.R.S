"""Puntúa dos `resultados.jsonl` con las métricas de §10, sobre juicios propios.

    py -m tools.comparar_configuraciones a.jsonl b.jsonl [c.jsonl …]

────────────────────────────────────────────────────────────────────────────────
QUÉ ES ESTO Y QUÉ NO ES

§10.1 dice que el *ground truth* de ADL «no es pública durante el reto», así que
**estos números no son la nota**. Lo que sí permiten es lo que en recuperación
de información se llama **pooling** (la metodología de TREC): se juntan los
documentos que devuelven los sistemas a comparar, se etiqueta ese conjunto a
mano, y se puntúa a los dos sobre el mismo material. La comparación **relativa**
es válida; los valores absolutos no son comparables con los del jurado.

Limitaciones que hay que decir en voz alta:

  - Los juicios los hizo el equipo, no ADL (`evaluation/juicios_muestra.json`).
  - Solo cubren las **7 consultas** en que las configuraciones difieren. En las
    otras 43 devuelven lo mismo y no discriminan, así que meterlas solo diluiría
    la diferencia hacia cero.
  - Un documento fuera del pool cuenta como irrelevante. Es la suposición
    estándar del pooling y sesga **a favor** de los sistemas comparados, por
    igual a todos.

**Se compara por `fuente`, no por `doc_id`**, como exige §10.2.2. La traducción
la hace el `metadata.jsonl`, que es justo su función según §5.3.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

from evaluation.metrics import f1_at_3, promedio
from indexing.metadata import METADATA

JUICIOS = Path(__file__).resolve().parents[1] / "evaluation" / "juicios_muestra.json"


def cargar_resultados(ruta: Path) -> Dict[str, dict]:
    return {json.loads(l)["query_id"]: json.loads(l)
            for l in open(ruta, encoding="utf-8") if l.strip()}


def mapa_fuentes(doc_ids: set) -> Dict[str, str]:
    """`doc_id` → `fuente`, leyendo el metadata.jsonl (§5.3)."""
    salida: Dict[str, str] = {}
    with open(METADATA, encoding="utf-8") as f:
        for linea in f:
            if not linea.strip():
                continue
            registro = json.loads(linea)
            if registro["doc_id"] in doc_ids and registro["doc_id"] not in salida:
                salida[registro["doc_id"]] = registro["fuente"]
            if len(salida) == len(doc_ids):
                break
    return salida


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    rutas = [Path(a) for a in sys.argv[1:]]
    if len(rutas) < 2:
        print(__doc__)
        return 1

    juicios = json.loads(JUICIOS.read_text(encoding="utf-8"))
    consultas = [q for q in juicios if not q.startswith("_")]

    resultados = {r.stem: cargar_resultados(r) for r in rutas}

    # Todo doc_id que aparezca, para traducirlo a fuente una sola vez.
    necesarios = {d["doc_id"] for r in resultados.values() for q in consultas
                  for d in r[q]["documents"]}
    necesarios |= {d for q in consultas for d in juicios[q] if not d.startswith("_")}
    fuente_de = mapa_fuentes(necesarios)

    linea = "─" * 78
    print(f"{linea}")
    print(f"juicios  : {JUICIOS.name}  ({len(consultas)} consultas etiquetadas a mano)")
    print(f"sistemas : {', '.join(resultados)}")
    print(f"⚠️  Comparación RELATIVA sobre juicios propios. NO es la nota de §10.")

    for umbral, etiqueta in ((2, "estricto: solo relevancia 2"),
                             (1, "permisivo: relevancia ≥ 1")):
        print(f"\n{linea}\nF1@3 — {etiqueta}")
        print(f"{'consulta':<10}" + "".join(f"{n:>22}" for n in resultados))

        por_sistema: Dict[str, List[float]] = {n: [] for n in resultados}
        for q in consultas:
            relevantes = {fuente_de[d] for d, g in juicios[q].items()
                          if not d.startswith("_") and g >= umbral and d in fuente_de}
            fila = f"{q:<10}"
            for nombre, r in resultados.items():
                devueltos = {fuente_de[d["doc_id"]] for d in r[q]["documents"]
                             if d["doc_id"] in fuente_de}
                m = f1_at_3(devueltos, relevantes)
                por_sistema[nombre].append(m["F1@3"])
                aciertos = len(devueltos & relevantes)
                fila += f"{m['F1@3']:>14.3f} ({aciertos}/{min(len(relevantes), 3)})"
            print(fila)

        print(f"{'MEDIA':<10}" + "".join(f"{promedio(v):>22.3f}"
                                         for v in por_sistema.values()))

    print(f"\n{linea}")
    print("Recordatorio: los juicios son del equipo, no de ADL, y cubren solo las")
    print("consultas donde las configuraciones difieren. Sirven para elegir entre")
    print("ellas, no para estimar la puntuación real.")
    print(linea)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
