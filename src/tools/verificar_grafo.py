"""Verificador del grafo de conocimiento (§7). Seis pruebas, código de salida.

    py -m tools.verificar_grafo
    py -m tools.verificar_grafo ../entrega/base_vectorial/grafo/grafo_demo.graphml

Mismo patrón que los otros cinco verificadores del proyecto: veredicto explícito y
**código de salida 1 si algo falla**, para poder encadenarlo.

────────────────────────────────────────────────────────────────────────────────
QUÉ COMPRUEBA, Y POR QUÉ CADA COSA

A. **Estructura (§7.1).** El grafo tiene que ser dirigido y sus aristas tipadas.
   `T ⊆ E × R × E` no admite una arista sin relación.
B. **Trazabilidad (§7.2).** Todo `chunk_id` de una arista existe en el
   `metadata.jsonl` entregado. Es la prueba F de `verificar_resultados` aplicada
   al grafo: §7.2 exige que cada tripleta permita «rastrear la evidencia textual»,
   y un `chunk_id` que no esté en el índice no rastrea nada.
C. **Canonicalización.** Ningún par de nodos del mismo tipo comparte clave
   canónica. Si dos la comparten, la canonicalización se aplicó a medias y el
   grafo tiene el mismo hecho repartido entre dos nodos.
D. **Ida y vuelta.** Escribir y releer devuelve el mismo grafo con los atributos
   intactos. Es lo que garantiza que el entregable no pierda nada al abrirlo
   quien lo evalúe.
E. **Cobertura.** Cuántos documentos del corpus aportan al menos una tripleta, y
   su reparto por fenómeno. Un grafo que solo cubre un fenómeno no representa el
   corpus que §1.3 define.
F. **Sin modelos generativos (§4.2, §8.3).** Se revisan los imports de todo el
   paquete `graph/` buscando bibliotecas de decoders. No es una declaración de
   buena fe: es una comprobación.
G. **Interoperabilidad.** Que el archivo abra en un visor: `id` de arista únicos
   (GraphML los exige globalmente únicos) y `label` en los nodos. **Se lee el
   XML crudo, no el grafo de networkx**, y esa es su razón de ser — ver el
   comentario del bloque. Existe porque el entregable pasó las otras seis y no
   abría en ningún visor.

────────────────────────────────────────────────────────────────────────────────
LO QUE ESTE VERIFICADOR **NO** PUEDE COMPROBAR

Que las tripletas sean **verdaderas**. Que `(Beijing, regula, China)` sea correcto
no lo dice ninguna prueba automática — solo leerlo. Igual que
`verificar_resultados` comprueba la forma y no el fondo (`ESTADO.md` §20), esto
comprueba que el grafo está bien construido, no que diga cosas ciertas. La
precisión medida a ojo está en `ESTADO.md` §18.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from graph.canonical import clave_canonica, sugerir_alias  # noqa: E402

GRAFO = RAIZ / "entrega" / "base_vectorial" / "grafo" / "grafo.graphml"
METADATA = (RAIZ / "entrega" / "base_vectorial" / "encoder_bge-m3"
            / "metadata.jsonl")

# Bibliotecas y llamadas de arquitectura decoder. §4.2 prohíbe los decoders «en
# las etapas de construcción del índice», y el grafo es construcción.
#
# ⚠️ `transformers` NO está en la lista: es la biblioteca de GLiNER, que es un
# encoder. Lo que se busca son las clases y bibliotecas **generativas**, no
# cualquier uso de transformers — prohibir la biblioteca entera confundiría la
# arquitectura con el paquete que la sirve, que es el error que §4.2 invita a
# cometer.
GENERATIVOS = re.compile(
    r"\b(openai|anthropic|cohere|google\.generativeai|litellm|ollama|llama_cpp|"
    r"vllm|AutoModelForCausalLM|AutoModelForSeq2SeqLM|"
    r"VisionEncoderDecoder|GPT2|LlamaFor|MistralFor|T5For)\b")

LINEA = "─" * 74


def _leer(ruta: Path):
    """Lee el GraphML como MultiDiGraph, con las dos precauciones que hacen falta.

    `force_multigraph=True` porque `read_graphml` devuelve un `DiGraph` cuando no
    hay aristas paralelas, y entonces una comparación de tipos falla sin que nada
    esté mal. `edge_key_type=str` porque las claves son nombres de relación y por
    omisión se convertirían a entero. Las dos están documentadas en
    `graph/build.py`.
    """
    import networkx as nx
    return nx.read_graphml(ruta, force_multigraph=True, edge_key_type=str)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ruta = Path(sys.argv[1]) if len(sys.argv) > 1 else GRAFO
    if not ruta.is_file():
        print(f"✘ no existe {ruta}")
        print("  Constrúyelo con `py -m graph.build` (o `--demo` para probar).")
        return 1

    es_demo = "demo" in ruta.name
    fallos: List[str] = []
    avisos: List[str] = []

    G = _leer(ruta)
    print(f"archivo : {ruta.name}   ({ruta.stat().st_size / 1e6:,.2f} MB)")
    print(f"grafo   : {G.number_of_nodes():,} nodos · {G.number_of_edges():,} aristas")
    if es_demo:
        print("\n⚠️  Es un grafo de DEMO: sus identificadores son sintéticos, así que")
        print("    la prueba B (trazabilidad) no se le puede aplicar.")

    # ── A. Estructura (§7.1) ─────────────────────────────────────────────────
    print(f"\n{LINEA}\nA. ESTRUCTURA (§7.1) — dirigido y con relaciones tipadas\n{LINEA}")
    if G.is_directed():
        print("   ✔ PASA: el grafo es dirigido")
    else:
        fallos.append("§7.1 define un grafo DIRIGIDO y este no lo es")

    sin_relacion = sin_doc = sin_chunk = 0
    for _, _, d in G.edges(data=True):
        if not str(d.get("relacion", "")).strip():
            sin_relacion += 1
        if not str(d.get("doc_id", "")).strip():
            sin_doc += 1
        if not str(d.get("chunk_id", "")).strip():
            sin_chunk += 1
    for nombre, n, exigido in (("relacion", sin_relacion, "§7.1 (T ⊆ E × R × E)"),
                               ("doc_id", sin_doc, "§7.2"),
                               ("chunk_id", sin_chunk, "§7.2")):
        if n:
            fallos.append(f"{n:,} aristas sin `{nombre}`, que {exigido} exige")
        else:
            print(f"   ✔ PASA: todas las aristas tienen `{nombre}`")

    sin_tipo = sum(1 for _, d in G.nodes(data=True)
                   if not str(d.get("tipo", "")).strip())
    if sin_tipo:
        fallos.append(f"{sin_tipo:,} nodos sin `tipo`")
    else:
        print("   ✔ PASA: todos los nodos tienen `tipo`")

    # ── B. Trazabilidad (§7.2) ───────────────────────────────────────────────
    print(f"\n{LINEA}\nB. TRAZABILIDAD (§7.2) — todo chunk_id existe en metadata\n{LINEA}")
    ids_grafo: Set[str] = set()
    docs_grafo: Set[str] = set()
    for _, _, d in G.edges(data=True):
        if d.get("chunk_id"):
            ids_grafo.add(d["chunk_id"])
        # `chunk_ids` es la cadena serializada con las evidencias adicionales.
        ids_grafo.update(str(d.get("chunk_ids", "")).split())
        if d.get("doc_id"):
            docs_grafo.add(d["doc_id"])

    docs_corpus: Set[str] = set()
    fenomenos_corpus: Counter = Counter()
    if es_demo:
        print("   ·  omitida: grafo de demo")
    elif not METADATA.is_file():
        avisos.append(f"no está {METADATA.name}; no se pudo comprobar la "
                      f"trazabilidad ni la cobertura")
        print(f"   ⚠️ no está {METADATA.name}")
    else:
        print(f"   leyendo {METADATA.name}…")
        conocidos: Set[str] = set()
        with open(METADATA, encoding="utf-8") as f:
            for linea in f:
                r = json.loads(linea)
                docs_corpus.add(r["doc_id"])
                fenomenos_corpus[str(r["fenomeno"])] += 1
                if r["chunk_id"] in ids_grafo:
                    conocidos.add(r["chunk_id"])
        huerfanos = ids_grafo - conocidos
        if huerfanos:
            fallos.append(
                f"{len(huerfanos):,} de {len(ids_grafo):,} chunk_id del grafo NO "
                f"están en metadata.jsonl: su evidencia no se puede rastrear "
                f"(§7.2). Ejemplos: {sorted(huerfanos)[:3]}")
        else:
            print(f"   ✔ PASA: los {len(ids_grafo):,} chunk_id del grafo existen "
                  f"en metadata.jsonl")

    # ── C. Canonicalización ──────────────────────────────────────────────────
    print(f"\n{LINEA}\nC. CANONICALIZACIÓN — un nodo por entidad\n{LINEA}")
    por_clave: Dict[Tuple[str, str], List[str]] = {}
    for n, d in G.nodes(data=True):
        nombre = d.get("nombre", n)
        tipo = d.get("tipo", "?")
        por_clave.setdefault((tipo, clave_canonica(nombre)), []).append(nombre)
    colisiones = {k: v for k, v in por_clave.items() if len(v) > 1}
    if colisiones:
        fallos.append(f"{len(colisiones)} grupos de nodos comparten clave "
                      f"canónica: la canonicalización se aplicó a medias y el "
                      f"mismo hecho está repartido entre varios nodos")
        for k, v in list(colisiones.items())[:5]:
            fallos.append(f"      {k}: {v}")
    else:
        print(f"   ✔ PASA: ningún par de nodos del mismo tipo comparte clave")

    # Las sugerencias no son un fallo: son trabajo de curación para la tabla de
    # alias, y `sugerir_alias()` propone sin aplicar a propósito (`ESTADO.md` §18).
    nodos_para_sugerir = {
        (d.get("tipo", "?"), clave_canonica(d.get("nombre", n))):
            {"nombre": d.get("nombre", n), "n": G.degree(n), "variantes": {}}
        for n, d in G.nodes(data=True)}
    sugerencias = sugerir_alias(nodos_para_sugerir, minimo=1)
    if sugerencias:
        print(f"   ℹ️  {len(sugerencias)} fusiones CANDIDATAS para revisar a mano "
              f"(no son un fallo):")
        for corto, largo, motivo in sugerencias[:10]:
            print(f"        {corto!r} ←→ {largo!r}   {motivo}")
        avisos.append(f"{len(sugerencias)} candidatas a alias sin revisar; "
                      f"añadir las buenas a data/alias_entidades.json")

    # ── D. Ida y vuelta ──────────────────────────────────────────────────────
    print(f"\n{LINEA}\nD. IDA Y VUELTA — el archivo no pierde nada al releerse\n{LINEA}")
    import tempfile

    import networkx as nx

    with tempfile.TemporaryDirectory() as tmp:
        copia = Path(tmp) / "ida_y_vuelta.graphml"
        nx.write_graphml(G, copia, encoding="utf-8")
        H = _leer(copia)
    mismos_nodos = set(G.nodes) == set(H.nodes)
    mismas_aristas = (sorted((u, v, k) for u, v, k in G.edges(keys=True))
                      == sorted((u, v, k) for u, v, k in H.edges(keys=True)))
    # Se comparan también los atributos, que es donde vive la evidencia de §7.2.
    atributos_ok = all(
        G.edges[u, v, k] == H.edges[u, v, k] for u, v, k in G.edges(keys=True))
    for nombre, ok in (("nodos", mismos_nodos), ("aristas", mismas_aristas),
                       ("atributos de arista", atributos_ok)):
        if ok:
            print(f"   ✔ PASA: {nombre} idénticos")
        else:
            fallos.append(f"la ida y vuelta cambia los {nombre}")

    # ── E. Cobertura ─────────────────────────────────────────────────────────
    print(f"\n{LINEA}\nE. COBERTURA — qué parte del corpus aporta al grafo\n{LINEA}")
    fen_grafo = Counter(d[1] for d in docs_grafo if d[:1] == "F")
    print(f"   documentos con tripletas : {len(docs_grafo):,}")
    if docs_corpus:
        pct = 100 * len(docs_grafo) / len(docs_corpus)
        print(f"   documentos del corpus    : {len(docs_corpus):,}  "
              f"→ cobertura {pct:.1f}%")
        faltan = {d[1] for d in docs_corpus if d[:1] == "F"} - set(fen_grafo)
        if faltan:
            fallos.append(f"el grafo no cubre los fenómenos {sorted(faltan)}; "
                          f"§1.3 define el corpus como los TRES")
        else:
            print(f"   ✔ PASA: los tres fenómenos están representados")
    print(f"   reparto por fenómeno     : "
          f"{ {f'F{k}': v for k, v in sorted(fen_grafo.items())} }")

    aisladas = sum(1 for _, g in G.degree if g == 0)
    if aisladas:
        avisos.append(f"{aisladas:,} nodos sin ninguna arista: no aportan nada "
                      f"al grafo y se pueden podar")

    # ── F. Sin modelos generativos (§4.2, §8.3) ──────────────────────────────
    print(f"\n{LINEA}\nF. SIN DECODERS (§4.2, §8.3) — imports del paquete graph/\n{LINEA}")
    sospechosos: List[str] = []
    for py in sorted((RAIZ / "src" / "graph").glob("*.py")):
        for i, linea_py in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
            # Solo las líneas de import; una mención en un comentario o docstring
            # que explique la prohibición no es una violación de ella.
            if not re.match(r"\s*(import|from)\s", linea_py):
                continue
            m = GENERATIVOS.search(linea_py)
            if m:
                sospechosos.append(f"{py.name}:{i}  {linea_py.strip()}")
    if sospechosos:
        fallos.append("§4.2 prohíbe los decoders en la construcción del índice, "
                      "y hay imports generativos:")
        fallos.extend(f"      {s}" for s in sospechosos)
    else:
        print("   ✔ PASA: ningún import de biblioteca o clase generativa")

    try:
        from graph.ner import MODELO, REVISION
        print(f"   modelo NER : {MODELO}")
        print(f"   arquitectura: encoder bidireccional (mDeBERTa-v3) · "
              f"licencia Apache 2.0")
        if REVISION:
            print(f"   ✔ PASA: revisión fijada ({REVISION[:16]}…), §1.4")
        else:
            fallos.append("la REVISIÓN del modelo NER no está fijada: §1.4 exige "
                          "el commit y no solo el nombre")
    except ImportError as e:
        avisos.append(f"no se pudo leer la declaración del modelo NER: {e}")

    # ── G. Interoperabilidad ─────────────────────────────────────────────────
    print(f"\n{LINEA}\nG. INTEROPERABILIDAD — que el archivo abra en un visor\n{LINEA}")
    # 🔴 Esta prueba lee el **XML crudo**, y esa es su razón de ser. networkx
    # interpreta el `id` de arista como la clave del multigrafo, que solo es
    # única *por par de nodos*, así que al leer el archivo la repetición
    # desaparece y `G` se ve perfectamente sano: **toda comprobación que parta de
    # `G` —incluida la D, la de ida y vuelta— es ciega a esto por construcción**.
    #
    # Lo destapó un visor, no una prueba. El grafo entregado tenía las 3.460
    # aristas repartiéndose **13 valores de `id`** (el nombre de la relación,
    # porque `build.py` usaba `key=relacion`), y **graphology** —el motor de
    # Gephi Lite y de casi todos los visores web— aborta la carga al ver el
    # segundo repetido:
    #
    #     Graph.mergeDirectedEdgeWithKey: inconsistency detected when attempting
    #     to merge the "coopera_con" edge with … vs. (…)
    #
    # §7 es bonus, o sea que existe para que alguien lo mire: un entregable que
    # no abre en un visor vale lo mismo que no entregarlo.
    ids_arista = re.findall(r'<edge [^>]*\bid="([^"]*)"',
                            ruta.read_text(encoding="utf-8"))
    repetidos = [i for i, n in Counter(ids_arista).items() if n > 1]
    if repetidos:
        fallos.append(
            f"las {len(ids_arista):,} aristas comparten solo "
            f"{len(set(ids_arista)):,} valores de `id`, y GraphML los exige "
            f"únicos: los visores basados en graphology no cargarán el archivo. "
            f"Repetidos: {sorted(repetidos)[:5]}")
        fallos.append("      se arregla sin reconstruir: "
                      "`py -m graph.build --reexportar <ruta>`")
    elif ids_arista:
        print(f"   ✔ PASA: los {len(ids_arista):,} `id` de arista son únicos")
    else:
        avisos.append("las aristas del XML no traen `id`; no se pudo comprobar "
                      "que sean únicos")

    sin_label = sum(1 for _, d in G.nodes(data=True)
                    if not str(d.get("label", "")).strip())
    if sin_label:
        avisos.append(f"{sin_label:,} nodos sin `label`: los visores pintarán su "
                      f"identificador (`LOC:estados unidos`) en vez del nombre. "
                      f"Se añade con `py -m graph.build --reexportar <ruta>`")
    else:
        print("   ✔ PASA: todos los nodos traen `label` para el visor")

    # ── Veredicto ────────────────────────────────────────────────────────────
    print(f"\n{LINEA}")
    for a in avisos:
        print(f"⚠️  {a}")
    if fallos:
        for f in fallos:
            print(f"✘ {f}")
        print(f"{LINEA}\nVEREDICTO: ✘ {len(fallos)} comprobación(es) fallan")
        return 1
    print(f"{LINEA}\nVEREDICTO: ✔ TODAS LAS PRUEBAS PASAN")
    print("Esto comprueba que el grafo está bien CONSTRUIDO, no que sus tripletas")
    print("sean ciertas. Eso solo se ve leyéndolas — ver `ESTADO.md` §18.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
