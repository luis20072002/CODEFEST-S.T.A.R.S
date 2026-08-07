"""Verificación del índice y su metadata: que el mapeo de §5.3 sea real.

    py -m tools.verificar_indice

Mismo patrón que `verificar_cleaner` y `verificar_chunker`: veredicto y código
de salida 1 si algo falla.

POR QUÉ ESTA HERRAMIENTA ES LA MÁS IMPORTANTE DE LAS TRES

Un índice mal alineado con su metadata **no da ningún error**. Devuelve
resultados perfectamente plausibles que corresponden a otro fragmento, y las
métricas de §10 no lo distinguen de un encoder malo: verías un NDCG bajo y
buscarías el problema en el modelo, no en el mapeo.

LAS CINCO PRUEBAS:

  A. CUENTAS — que las tres numeraciones tengan el mismo tamaño:
     índice FAISS, `metadata.jsonl` y `chunks.jsonl`.

  B. IDA Y VUELTA — la prueba que de verdad demuestra el mapeo. Se toma el
     vector de la posición i **del propio índice**, se busca, y el resultado
     mejor tiene que ser i con puntuación ~1,0. Si el orden de inserción se
     hubiera desordenado, esto lo caza.

  C. TABLA 1 — que cada línea del `metadata.jsonl` traiga los ocho campos
     obligatorios **con los nombres en español**, no vacíos, con `posicion`
     entero desde 0 y `num_tokens` positivo.

  D. ALINEACIÓN CON LOS CHUNKS — que la línea i del `metadata.jsonl` sea el
     fragmento i de `chunks.jsonl`: mismo `chunk_id` y mismo texto. Es lo que
     garantiza que el `texto` que se reporte en `resultados.jsonl` es el del
     vector que se recuperó (§10.2.1 evalúa por el contenido del campo).

  E. `faiss_id` — que el campo coincida con el número de línea.

No escribe nada.
"""

import json
import sys
from pathlib import Path

from core.chunk import TABLA1_FIELDS
from core.store import read_chunks
from embedding.encoder import EMBEDDINGS
from indexing.faiss_index import INDICE
from indexing.metadata import CHUNKS, METADATA

# Cuántas posiciones se prueban en la ida y vuelta. No hacen falta las 143.962:
# un desorden de inserción afectaría a todas, así que una muestra sistemática
# —determinista, no aleatoria— lo detecta igual y tarda segundos.
MUESTRAS_IDA_VUELTA = 200

# Cuánto puede alejarse de 1,0 el coseno para seguir considerándose un empate
# entre vectores indistinguibles. Las diferencias medidas por el relleno de los
# lotes son del orden de 1e-6 (`ESTADO.md` §14); 1e-4 deja margen de sobra y
# sigue muy lejos de cualquier desalineamiento real.
TOLERANCIA_EMPATE = 1e-4


def main() -> int:
    import faiss
    import numpy as np

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    for ruta in (INDICE, METADATA, CHUNKS):
        if not ruta.is_file():
            print(f"No existe {ruta}.")
            return 1

    linea = "─" * 74
    indice = faiss.read_index(str(INDICE))

    # ── A. cuentas ──────────────────────────────────────────────────────────
    with open(METADATA, encoding="utf-8") as f:
        n_metadata = sum(1 for l in f if l.strip())
    with open(CHUNKS, encoding="utf-8") as f:
        n_chunks = sum(1 for l in f if l.strip())

    print(f"{linea}\nA. CUENTAS")
    print(f"   índice FAISS   : {indice.ntotal:,}")
    print(f"   metadata.jsonl : {n_metadata:,}")
    print(f"   chunks.jsonl   : {n_chunks:,}")
    cuentas_ok = indice.ntotal == n_metadata == n_chunks
    print("   ✔ PASA: las tres coinciden" if cuentas_ok
          else "   ✖ FALLA: no coinciden — el mapeo de §5.3 no puede ser correcto")

    # ── B. ida y vuelta ─────────────────────────────────────────────────────
    print(f"\n{linea}\nB. IDA Y VUELTA — buscar la fila i de embeddings.npy debe devolver i")
    paso = max(1, indice.ntotal // MUESTRAS_IDA_VUELTA)
    posiciones = list(range(0, indice.ntotal, paso))[:MUESTRAS_IDA_VUELTA]

    # ⚠️ Se consulta con los vectores de `embeddings.npy`, NO con
    # `indice.reconstruct(i)`. La primera versión de esta prueba usaba
    # `reconstruct()` y era mucho más débil de lo que parecía: devuelve lo que
    # el índice guarda en la posición i, así que buscar eso encuentra i
    # **aunque el índice estuviera construido a partir de otra cosa**. Solo
    # comprobaba la coherencia interna de FAISS.
    # Partiendo del `.npy` sí se comprueba lo que importa: que la fila i del
    # archivo que produjo el encoder acabó en la posición i del índice.
    fuente = np.load(EMBEDDINGS, mmap_mode="r")
    if fuente.shape[0] != indice.ntotal:
        print(f"   ✖ FALLA: embeddings.npy tiene {fuente.shape[0]:,} filas y el índice "
              f"{indice.ntotal:,}")
        return 1
    consultas = np.ascontiguousarray(fuente[posiciones], dtype=np.float32)
    puntuaciones, encontrados = indice.search(consultas, 1)

    fallos_b = []
    empates = 0
    for p, e, s in zip(posiciones, encontrados, puntuaciones):
        obtenido = int(e[0])
        if obtenido == p:
            continue
        # Un empate entre vectores indistinguibles no es un desalineamiento:
        # FAISS devuelve uno de los dos y es lo correcto. El corpus tiene 3.597
        # chunks con texto repetido (`ESTADO.md` §15), así que esto pasa de
        # verdad — 7 de 200 en la primera corrida.
        #
        # ⚠️ El criterio es el COSENO, no la igualdad byte a byte. Antes se
        # comparaba con `array_equal` y fallaba en Colab: dos chunks con el
        # mismo texto codificados en lotes distintos difieren ~4e-06 por el
        # relleno (`ESTADO.md` §14), así que su coseno es 1,000000 pero sus
        # bytes no coinciden. Un desalineamiento de verdad daría un coseno
        # bajo, no 1,0, así que este criterio sigue detectándolo.
        if float(s[0]) >= 1.0 - TOLERANCIA_EMPATE:
            empates += 1
            continue
        fallos_b.append((int(p), obtenido, float(s[0])))

    if fallos_b:
        print(f"   ✖ FALLA: {len(fallos_b)} de {len(posiciones)} apuntan a otro vector")
        for esperado, obtenido, punt in fallos_b[:10]:
            print(f"     posición {esperado} → devolvió {obtenido} (coseno {punt:.6f})")
    else:
        print(f"   ✔ PASA: las {len(posiciones)} posiciones probadas apuntan a su vector")
    if empates:
        print(f"   (informativo: {empates} resolvieron a un duplicado exacto — "
              f"vector idéntico, no es desalineamiento)")

    # ── C, D, E: recorrido conjunto de metadata y chunks ────────────────────
    obligatorios = set(TABLA1_FIELDS.values())
    faltantes: list = []       # C
    invalidos: list = []       # C
    desalineados: list = []    # D
    faiss_id_malo: list = []   # E

    with open(METADATA, encoding="utf-8") as f:
        for numero, (cruda, fragmento) in enumerate(zip(f, read_chunks(CHUNKS))):
            registro = json.loads(cruda)

            faltan = obligatorios - {k for k, v in registro.items() if v not in (None, "")}
            if faltan:
                faltantes.append((numero, sorted(faltan)))
            if (not isinstance(registro.get("posicion"), int)
                    or registro["posicion"] < 0
                    or not isinstance(registro.get("num_tokens"), int)
                    or registro["num_tokens"] <= 0):
                invalidos.append((numero, registro.get("posicion"), registro.get("num_tokens")))

            if (registro.get("chunk_id") != fragmento.chunk_id
                    or registro.get("texto") != fragmento.text):
                desalineados.append((numero, registro.get("chunk_id"), fragmento.chunk_id))

            if "faiss_id" in registro and registro["faiss_id"] != numero:
                faiss_id_malo.append((numero, registro["faiss_id"]))

    print(f"\n{linea}\nC. TABLA 1 — los 8 campos en español")
    print("   ✔ PASA: todas las líneas completas" if not faltantes
          else f"   ✖ FALLA: {len(faltantes)} líneas con campos vacíos: {faltantes[:5]}")
    print("   ✔ PASA: posicion y num_tokens válidos" if not invalidos
          else f"   ✖ FALLA: {len(invalidos)} líneas con valores inválidos: {invalidos[:5]}")

    print(f"\n{linea}\nD. ALINEACIÓN metadata ↔ chunks (chunk_id y texto)")
    print("   ✔ PASA: cada línea corresponde a su fragmento" if not desalineados
          else f"   ✖ FALLA: {len(desalineados)} líneas desalineadas: {desalineados[:5]}")

    print(f"\n{linea}\nE. faiss_id == número de línea")
    print("   ✔ PASA" if not faiss_id_malo
          else f"   ✖ FALLA: {len(faiss_id_malo)} discrepancias: {faiss_id_malo[:5]}")

    fallo = bool(not cuentas_ok or fallos_b or faltantes or invalidos
                 or desalineados or faiss_id_malo)
    print(f"\n{linea}")
    print("VEREDICTO: ✖ HAY FALLOS — no entregar así" if fallo
          else "VEREDICTO: ✔ TODAS LAS PRUEBAS PASAN")
    print(linea)
    return 1 if fallo else 0


if __name__ == "__main__":
    raise SystemExit(main())
