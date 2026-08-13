"""Validación de `resultados.jsonl` contra §9.3.1 y §10.2.

    py -m tools.verificar_resultados
    py -m tools.verificar_resultados ../entrega/resultados_simulado.jsonl

Mismo patrón que los otros verificadores: veredicto y código de salida 1.

POR QUÉ NO SE REVISA A OJO

§9.3.1 es un «Requisito obligatorio» y no admite matices:

    «Objetos con campos faltantes, arrays con un número diferente de elementos
    (distinto de 3 para documentos y de 10 para fragmentos), o fragmentos que
    superen las 250 palabras serán penalizados o descartados durante la
    evaluación automática.»

Un archivo de 50 líneas con diez fragmentos cada una son 500 textos. Contar
palabras a ojo en 500 textos no es viable, y un solo fallo puede costar la
consulta entera.

LAS SEIS PRUEBAS:

  A. ESTRUCTURA — 50 líneas, JSON válido, `query_id` presente y en orden.
  B. TAMAÑOS — exactamente 3 documentos y 10 fragmentos por consulta.
  C. CAMPOS — los de la Tabla 2, sin faltar ninguno ni venir vacíos.
  D. RANKS — 1..3 y 1..10, consecutivos y sin repetir.
  E. 250 PALABRAS — ningún fragmento se pasa.
  F. TRAZABILIDAD (§10.2.2) — **todo `doc_id` y todo `chunk_id` de la salida
     existe en el `metadata.jsonl`**. Es la prueba que más fácil se olvida y la
     que más caro cuesta: §10.2.2 empareja los documentos por `fuente`, y la
     única vía de `doc_id` a `fuente` es el `metadata.jsonl`. Un `doc_id` que no
     esté ahí **no puntúa aunque sea el documento correcto**.

No escribe nada.
"""

import json
import sys
from pathlib import Path

from indexing.metadata import METADATA
from retrieval.fragmentos import LIMITE_PALABRAS
from retrieval.search import N_DOCUMENTOS, N_FRAGMENTOS

RAIZ = Path(__file__).resolve().parents[2]
RESULTADOS = RAIZ / "entrega" / "resultados.jsonl"

LINEAS_ESPERADAS = 50
CAMPOS_DOCUMENTO = ("rank", "doc_id")
CAMPOS_FRAGMENTO = ("rank", "chunk_id", "doc_id", "text")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ruta = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTADOS
    if not ruta.is_file():
        print(f"No existe {ruta}. Corre antes `python entrega/generador.py`.")
        return 1

    # Los identificadores válidos salen del metadata.jsonl, que es el único
    # sitio que mapea doc_id → fuente (§10.2.2).
    print("leyendo metadata.jsonl…")
    doc_ids, chunk_ids = set(), set()
    with open(METADATA, encoding="utf-8") as f:
        for linea in f:
            if not linea.strip():
                continue
            registro = json.loads(linea)
            doc_ids.add(registro["doc_id"])
            chunk_ids.add(registro["chunk_id"])
    print(f"  {len(doc_ids):,} doc_id y {len(chunk_ids):,} chunk_id conocidos\n")

    # Acumuladores por prueba
    malformadas, sin_query_id = [], []          # A
    tamanos_malos = []                          # B
    campos_faltantes = []                       # C
    ranks_malos = []                            # D
    pasados = []                                # E
    doc_desconocido, chunk_desconocido = [], [] # F
    query_ids = []
    total = 0

    with open(ruta, encoding="utf-8") as f:
        for numero, linea in enumerate(f, start=1):
            if not linea.strip():
                continue
            total += 1
            try:
                objeto = json.loads(linea)
            except json.JSONDecodeError as error:
                malformadas.append((numero, str(error)[:70]))
                continue

            query_id = objeto.get("query_id")
            if not query_id:
                sin_query_id.append(numero)
            else:
                query_ids.append(query_id)

            documentos = objeto.get("documents") or []
            fragmentos = objeto.get("fragments") or []

            # ── B. tamaños ──────────────────────────────────────────────
            if len(documentos) != N_DOCUMENTOS or len(fragmentos) != N_FRAGMENTOS:
                tamanos_malos.append((query_id, len(documentos), len(fragmentos)))

            # ── C, D, F sobre documentos ────────────────────────────────
            for d in documentos:
                faltan = [c for c in CAMPOS_DOCUMENTO if not d.get(c)]
                if faltan:
                    campos_faltantes.append((query_id, "documents", faltan))
                if d.get("doc_id") and d["doc_id"] not in doc_ids:
                    doc_desconocido.append((query_id, d["doc_id"]))
            esperados = list(range(1, len(documentos) + 1))
            if [d.get("rank") for d in documentos] != esperados:
                ranks_malos.append((query_id, "documents",
                                    [d.get("rank") for d in documentos]))

            # ── C, D, E, F sobre fragmentos ─────────────────────────────
            for fr in fragmentos:
                faltan = [c for c in CAMPOS_FRAGMENTO if not fr.get(c)]
                if faltan:
                    campos_faltantes.append((query_id, "fragments", faltan))
                texto = fr.get("text") or ""
                palabras = len(texto.split())
                if palabras > LIMITE_PALABRAS:
                    pasados.append((query_id, fr.get("rank"), palabras))
                if fr.get("doc_id") and fr["doc_id"] not in doc_ids:
                    doc_desconocido.append((query_id, fr["doc_id"]))
                if fr.get("chunk_id") and fr["chunk_id"] not in chunk_ids:
                    chunk_desconocido.append((query_id, fr["chunk_id"]))
            esperados = list(range(1, len(fragmentos) + 1))
            if [fr.get("rank") for fr in fragmentos] != esperados:
                ranks_malos.append((query_id, "fragments",
                                    [fr.get("rank") for fr in fragmentos]))

    linea_sep = "─" * 74
    print(f"{linea_sep}\narchivo: {ruta}")
    print(f"líneas : {total}")

    print(f"\n{linea_sep}\nA. ESTRUCTURA — {LINEAS_ESPERADAS} líneas y JSON válido")
    lineas_ok = total == LINEAS_ESPERADAS
    print(f"   {'✔ PASA' if lineas_ok else f'✖ FALLA: hay {total}'}: número de líneas")
    print("   ✔ PASA: todas las líneas son JSON válido" if not malformadas
          else f"   ✖ FALLA: {len(malformadas)} malformadas: {malformadas[:3]}")
    print("   ✔ PASA: todas tienen query_id" if not sin_query_id
          else f"   ✖ FALLA: sin query_id en las líneas {sin_query_id[:10]}")
    repetidos = len(query_ids) != len(set(query_ids))
    print("   ✔ PASA: query_id sin repetir" if not repetidos
          else "   ✖ FALLA: hay query_id repetidos")

    print(f"\n{linea_sep}\nB. TAMAÑOS — {N_DOCUMENTOS} documentos y {N_FRAGMENTOS} fragmentos")
    if tamanos_malos:
        print(f"   ✖ FALLA: {len(tamanos_malos)} consultas con tamaños distintos")
        for qid, nd, nf in tamanos_malos[:10]:
            print(f"     {qid}: {nd} documentos, {nf} fragmentos")
    else:
        print("   ✔ PASA")

    print(f"\n{linea_sep}\nC. CAMPOS de la Tabla 2")
    print("   ✔ PASA: ningún campo faltante ni vacío" if not campos_faltantes
          else f"   ✖ FALLA: {len(campos_faltantes)} casos: {campos_faltantes[:5]}")

    print(f"\n{linea_sep}\nD. RANKS consecutivos desde 1")
    print("   ✔ PASA" if not ranks_malos
          else f"   ✖ FALLA: {len(ranks_malos)} casos: {ranks_malos[:5]}")

    print(f"\n{linea_sep}\nE. LÍMITE de {LIMITE_PALABRAS} palabras por fragmento")
    if pasados:
        print(f"   ✖ FALLA: {len(pasados)} fragmentos se pasan")
        for qid, rank, palabras in sorted(pasados, key=lambda x: -x[2])[:10]:
            print(f"     {qid} rank {rank}: {palabras} palabras")
    else:
        print("   ✔ PASA: ninguno supera el límite")

    print(f"\n{linea_sep}\nF. TRAZABILIDAD (§10.2.2) — todo id existe en metadata.jsonl")
    print("   ✔ PASA: todos los doc_id son conocidos" if not doc_desconocido
          else f"   ✖ FALLA: {len(doc_desconocido)} desconocidos: {doc_desconocido[:5]}")
    print("   ✔ PASA: todos los chunk_id son conocidos" if not chunk_desconocido
          else f"   ✖ FALLA: {len(chunk_desconocido)} desconocidos: {chunk_desconocido[:5]}")

    fallo = bool(not lineas_ok or malformadas or sin_query_id or repetidos
                 or tamanos_malos or campos_faltantes or ranks_malos or pasados
                 or doc_desconocido or chunk_desconocido)
    print(f"\n{linea_sep}")
    print("VEREDICTO: ✖ HAY FALLOS — §9.3.1 penaliza o descarta esto" if fallo
          else "VEREDICTO: ✔ TODAS LAS PRUEBAS PASAN")
    print(linea_sep)
    return 1 if fallo else 0


if __name__ == "__main__":
    raise SystemExit(main())
