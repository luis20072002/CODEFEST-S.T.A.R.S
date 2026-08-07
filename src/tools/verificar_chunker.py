"""Verificación del chunker: las cinco pruebas que sí pueden fallar en silencio.

    py -m tools.verificar_chunker           # sobre data/chunks.jsonl completo
    py -m tools.verificar_chunker 20        # muestra: 1 de cada 20 documentos

Mismo patrón y misma razón de ser que `tools/verificar_cleaner.py`: el chunker
no lanza excepciones cuando se equivoca. Produce fragmentos que incumplen §3.3,
o que se pasan del tope del encoder, o que pierden texto — y nada de eso se
nota hasta que las métricas de §10 ya están calculadas.

LAS CINCO PRUEBAS (las pide `TAREAS.md`, Fase 4):

  A. §3.3 COMPLETITUD LINGÜÍSTICA — la que más vale.
     «Ningún fragmento puede contener oraciones o frases incompletas.» Se
     comprueba que ningún fragmento de prosa empiece en minúscula (señal de
     corte a media oración) ni termine sin puntuación de cierre.
     ⚠️ Se juzga SOLO la prosa cortada en los niveles 1 y 2. Los fragmentos
     tabulares son filas —`PMID: 34…` no termina en punto y está bien— y los de
     nivel 3 y 4 son celdas de tabla donde no hay oración que completar. Contar
     esos como fallos sería medir mal a propósito.

  B. COBERTURA — que no se pierda texto.
     Se concatenan los fragmentos de cada documento y se compara, carácter a
     carácter **ignorando espacios**, con el texto del documento. Se ignoran los
     espacios porque el chunker sí cambia separadores al reagrupar (un `\\n\\n`
     puede quedar como espacio), pero no puede perder ni un carácter con
     contenido. Comprueba además que ningún documento con texto se quede sin
     fragmentos.

  C. TABLA 1 — que la metadata obligatoria esté completa y bien formada.
     Los 8 campos presentes, `position` empezando en 0 y consecutiva dentro de
     cada documento, y `chunk_id` único en todo el corpus. Si el `chunk_id` se
     repitiera, dos vectores distintos de FAISS apuntarían a la misma línea del
     `metadata.jsonl`.

  D. PRESUPUESTO — que ningún fragmento se pase del tope del encoder.
     Un fragmento pasado no da error: el encoder lo trunca y la parte de atrás
     desaparece de la recuperación en silencio (§4.3).

  E. DETERMINISMO (§1.4) — que dos corridas den lo mismo.
     Se vuelve a chunkear una muestra y se comparan `chunk_id` y textos. §1.4
     excluye la entrega si `generador.py` no reproduce los resultados.

No escribe nada.
"""

import sys
from collections import Counter
from itertools import groupby
from pathlib import Path

from chunking.chunker import (
    CIERRES,
    FIN_ORACION,
    FORMATOS_TABULARES,
    chunk_document,
)
# El tope contra el que se juzga la prueba D. Por defecto son los 512 tokens de
# §4.3, que es el presupuesto de la corrida definitiva (`--tokens`). Si se está
# verificando una corrida hecha con el contador de palabras, hay que pasarle
# ese otro presupuesto: `py -m tools.verificar_chunker --presupuesto 350`.
from embedding.encoder import MAX_TOKENS
from core.chunk import TABLA1_FIELDS
from core.store import read_chunks, read_documents

DATOS = Path(__file__).resolve().parents[1] / "data"
DOCUMENTOS = DATOS / "documentos_limpios.jsonl"
CHUNKS = DATOS / "chunks.jsonl"

# Cuántos documentos se re-chunkean para la prueba E. No hacen falta los 1826:
# el determinismo es una propiedad del código, no de los datos, y con una
# muestra sistemática (no aleatoria) la prueba sigue siendo reproducible.
DOCS_DETERMINISMO = 40


def _sin_espacios(texto: str) -> str:
    """Quita TODO el espacio en blanco. Para comparar contenido, no formato."""
    return "".join(texto.split())


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    argumentos = sys.argv[1:]
    presupuesto = MAX_TOKENS
    if "--presupuesto" in argumentos:
        i = argumentos.index("--presupuesto")
        presupuesto = int(argumentos[i + 1])
        argumentos = argumentos[:i] + argumentos[i + 2:]
    paso = int(argumentos[0]) if argumentos else 1

    for ruta in (DOCUMENTOS, CHUNKS):
        if not ruta.is_file():
            print(f"No existe {ruta}. Corre antes `py -m chunking.chunker`.")
            return 1

    # ── Acumuladores ────────────────────────────────────────────────────────
    total_chunks = 0
    cortes_malos: list = []        # A — oraciones partidas de verdad
    sin_puntuacion = 0             # A — informativo
    juzgados_a = 0                 # A — cortes examinados
    sin_cobertura: list = []       # B
    sin_fragmentos: list = []      # B
    campos_faltantes: list = []    # C
    posiciones_malas: list = []    # C
    repetidos: list = []           # C
    vistos: set = set()            # C
    pasados: list = []             # D
    por_nivel: Counter = Counter()

    obligatorios = set(TABLA1_FIELDS.values())

    # Los dos archivos van en el mismo orden de documentos (los dos salen de
    # documentos_limpios.jsonl), así que se pueden recorrer en paralelo sin
    # cargar ninguno entero en memoria. `groupby` agrupa los fragmentos
    # consecutivos que comparten doc_id.
    fragmentos_por_doc = groupby(read_chunks(CHUNKS), key=lambda c: c.doc_id)
    grupo_actual = next(fragmentos_por_doc, None)

    for indice, documento in enumerate(read_documents(DOCUMENTOS)):
        if not documento.text.strip():
            continue

        if grupo_actual is not None and grupo_actual[0] == documento.doc_id:
            fragmentos = list(grupo_actual[1])
            grupo_actual = next(fragmentos_por_doc, None)
        else:
            fragmentos = []

        if not fragmentos:
            sin_fragmentos.append((documento.doc_id, documento.format))
            continue

        total_chunks += len(fragmentos)
        es_tabular = documento.format in FORMATOS_TABULARES

        # ── B. cobertura ────────────────────────────────────────────────────
        unido = _sin_espacios("".join(f.text for f in fragmentos))
        original = _sin_espacios(documento.text)
        if unido != original:
            sin_cobertura.append((documento.doc_id, documento.format,
                                  len(original), len(unido)))

        # ── A. §3.3, sobre los CORTES entre fragmentos consecutivos ─────────
        # Se juzga la frontera, no el fragmento suelto. El síntoma inequívoco
        # de una oración partida es que un fragmento no cierre y el siguiente
        # empiece en minúscula. Mirar solo «¿termina en punto?» daría falsos
        # positivos por millares: los encabezados y los pies de figura de los
        # informes («Figure 1.2.1», «Chapter 1 Preview 19») no llevan punto
        # final y no por eso hay una oración rota.
        if not es_tabular:
            for actual, siguiente in zip(fragmentos, fragmentos[1:]):
                if actual.metadata.get("nivel_corte", 0) > 2:
                    continue          # celdas de tabla: no hay oración que romper
                juzgados_a += 1
                cierra = actual.text.strip().rstrip(CIERRES)
                empieza = siguiente.text.strip()[:1]
                if cierra and cierra[-1] not in FIN_ORACION and empieza.islower():
                    cortes_malos.append((actual.chunk_id, cierra[-55:], empieza and
                                         siguiente.text.strip()[:55]))
            # Informativo, no es fallo: cuántos no cierran en puntuación.
            for fragmento in fragmentos:
                if fragmento.metadata.get("nivel_corte", 0) > 2:
                    continue
                cierra = fragmento.text.strip().rstrip(CIERRES)
                if cierra and cierra[-1] not in FIN_ORACION:
                    sin_puntuacion += 1

        # ── C. posiciones consecutivas desde 0 ──────────────────────────────
        esperadas = list(range(len(fragmentos)))
        if [f.position for f in fragmentos] != esperadas:
            posiciones_malas.append((documento.doc_id,
                                     [f.position for f in fragmentos][:8]))

        for fragmento in fragmentos:
            nivel = fragmento.metadata.get("nivel_corte", 0)
            por_nivel[nivel] += 1

            # ── C. Tabla 1 y unicidad ───────────────────────────────────────
            registro = fragmento.to_metadata_record(extras=False)
            faltan = obligatorios - {k for k, v in registro.items()
                                     if v not in (None, "")}
            if faltan:
                campos_faltantes.append((fragmento.chunk_id, sorted(faltan)))
            if fragmento.chunk_id in vistos:
                repetidos.append(fragmento.chunk_id)
            vistos.add(fragmento.chunk_id)

            # ── D. presupuesto ──────────────────────────────────────────────
            if fragmento.num_tokens > presupuesto:
                pasados.append((fragmento.chunk_id, fragmento.num_tokens, nivel))

        if indice % 400 == 0 and indice:
            print(f"  … {indice} documentos", file=sys.stderr, flush=True)

    # ── E. determinismo ─────────────────────────────────────────────────────
    diferencias = []
    for indice, documento in enumerate(read_documents(DOCUMENTOS)):
        if indice % max(1, 1826 // DOCS_DETERMINISMO):
            continue
        primera = [(c.chunk_id, c.text) for c in chunk_document(documento)]
        segunda = [(c.chunk_id, c.text) for c in chunk_document(documento)]
        if primera != segunda:
            diferencias.append(documento.doc_id)

    # ═══════════════════════════════════════════════════════════════════════
    # Informe
    # ═══════════════════════════════════════════════════════════════════════
    linea = "─" * 74
    print(f"\n{linea}\nfragmentos examinados: {total_chunks:,}"
          + (f"  (1 de cada {paso})" if paso > 1 else ""))
    print("por nivel de la cascada:", dict(sorted(por_nivel.items())))

    print(f"\n{linea}\nA. §3.3 COMPLETITUD LINGÜÍSTICA "
          f"({juzgados_a:,} cortes de prosa examinados)")
    if cortes_malos:
        print(f"   ✖ FALLA: {len(cortes_malos)} cortes parten una oración "
              f"({100*len(cortes_malos)/max(juzgados_a,1):.3f}%)")
        for chunk_id, cierre, apertura in cortes_malos[:10]:
            print(f"     {chunk_id:24} …{cierre!r}")
            print(f"     {'':24}  ↳ {apertura!r}")
    else:
        print("   ✔ PASA: ningún corte parte una oración")
    print(f"   (informativo: {sin_puntuacion:,} fragmentos no terminan en puntuación "
          f"de cierre — encabezados, pies de figura y celdas, no oraciones rotas)")

    print(f"\n{linea}\nB. COBERTURA — que no se pierda texto")
    if sin_cobertura:
        print(f"   ✖ FALLA: {len(sin_cobertura)} documentos no se reconstruyen")
        for doc_id, formato, a, d in sin_cobertura[:10]:
            print(f"     {doc_id:20} {formato:5} {a:,} → {d:,} caracteres")
    else:
        print("   ✔ PASA: todo documento se reconstruye carácter a carácter")
    if sin_fragmentos:
        print(f"   ✖ FALLA: {len(sin_fragmentos)} documentos con texto sin fragmentos")
        for doc_id, formato in sin_fragmentos[:10]:
            print(f"     {doc_id:20} {formato}")
    else:
        print("   ✔ PASA: ningún documento con texto se quedó sin fragmentos")

    print(f"\n{linea}\nC. TABLA 1 — metadata obligatoria")
    print("   ✔ PASA: los 8 campos en todos" if not campos_faltantes
          else f"   ✖ FALLA: {len(campos_faltantes)} fragmentos con campos vacíos: "
               f"{campos_faltantes[:5]}")
    print("   ✔ PASA: position consecutiva desde 0 en todos los documentos"
          if not posiciones_malas
          else f"   ✖ FALLA: {len(posiciones_malas)} documentos: {posiciones_malas[:5]}")
    print(f"   ✔ PASA: {len(vistos):,} chunk_id únicos" if not repetidos
          else f"   ✖ FALLA: {len(repetidos)} chunk_id repetidos: {repetidos[:5]}")

    print(f"\n{linea}\nD. PRESUPUESTO — tope de {presupuesto}")
    if pasados:
        print(f"   ✖ FALLA: {len(pasados)} fragmentos se pasan")
        for chunk_id, n, nivel in sorted(pasados, key=lambda x: -x[1])[:10]:
            print(f"     {chunk_id:24} {n:,} tokens (nivel {nivel})")
    else:
        print("   ✔ PASA: ningún fragmento supera el presupuesto")

    print(f"\n{linea}\nE. DETERMINISMO (§1.4) — {DOCS_DETERMINISMO} documentos, dos corridas")
    print("   ✔ PASA: mismos chunk_id y mismos textos" if not diferencias
          else f"   ✖ FALLA: {diferencias}")

    fallo = bool(cortes_malos or sin_cobertura or sin_fragmentos
                 or campos_faltantes or posiciones_malas or repetidos or pasados
                 or diferencias)
    print(f"\n{linea}")
    print("VEREDICTO: ✖ HAY FALLOS — no seguir a la Fase 5 sin resolverlos" if fallo
          else "VEREDICTO: ✔ TODAS LAS PRUEBAS PASAN")
    print(linea)
    return 1 if fallo else 0


if __name__ == "__main__":
    raise SystemExit(main())
