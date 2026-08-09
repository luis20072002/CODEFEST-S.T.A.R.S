"""Construcción y exportación del grafo de conocimiento (§7).

    py -m graph.build --demo        # con entidades de prueba, sin GLiNER
    py -m graph.build               # la pasada completa (necesita GPU) ~1,6 h
    py -m graph.build --limite 500  # solo las primeras 500 unidades, para probar

Salida: `entrega/base_vectorial/grafo/grafo.graphml` — **dentro** de
`base_vectorial/`, como fija el árbol de §1.4.

────────────────────────────────────────────────────────────────────────────────
QUÉ CONSTRUYE

§7.1 pide `G = (E, R, T)` con `T ⊆ E × R × E`: **grafo dirigido con relaciones
tipadas**. Se usa `nx.MultiDiGraph`:

  · **dirigido**, porque `(EE.UU., desarrolla, X)` no es lo mismo que su inverso;
  · **multi**, porque dos entidades pueden estar unidas por varias relaciones
    distintas —un país puede a la vez `desarrolla` y `regula` una tecnología— y
    un grafo simple las machacaría dejando solo la última.

────────────────────────────────────────────────────────────────────────────────
🔴 LA TRAMPA DE GRAPHML: SOLO ADMITE ESCALARES

GraphML no tiene tipo lista. Un atributo solo puede ser cadena, entero, flotante
o booleano, y `nx.write_graphml()` **falla con una excepción** si se le pasa una
lista o un diccionario.

Eso choca de frente con §7.2, que es textual: «Cada tripleta mantiene una
referencia al `doc_id` **y al `chunk_id` de origen**, lo que permite rastrear la
evidencia textual de cada relación». Una misma tripleta aparece en muchos
fragmentos, así que la referencia es naturalmente una lista.

**Se resuelve con las dos vías a la vez, y ninguna sobra:**

1. **Una arista por tripleta distinta**, con `doc_id` y `chunk_id` del **primer**
   fragmento donde se observó — el que sirve de evidencia canónica.
2. **Los demás fragmentos, serializados** en `chunk_ids` como cadena separada por
   espacios, con un tope de `MAX_EVIDENCIAS`. Sin tope, una relación frecuente
   como `(EE.UU., desarrolla, IA)` arrastraría miles de identificadores y el
   archivo se dispararía.

Así cada arista cumple §7.2 con un `chunk_id` escalar **y** conserva la
trazabilidad múltiple de forma legible.

⚠️ **Segunda trampa, al releer: `read_graphml` devuelve un `DiGraph`, no un
`MultiDiGraph`, si el grafo no tiene aristas paralelas.** No es pérdida de datos
—nodos, aristas y atributos vuelven intactos— pero una prueba de ida y vuelta que
compare el tipo falla sin que nada esté mal. Hay que releer con
`read_graphml(ruta, force_multigraph=True)`.

⚠️ **Tercera: la clave de la arista.** Se escribe `key=relacion`, y al releer con
`force_multigraph` la clave vuelve; pero `read_graphml` la convierte a `int` por
omisión (`edge_key_type=int`), así que con nombres de relación hay que pasar
`edge_key_type=str`. Por eso la relación se guarda **además** como atributo
`relacion`: el atributo es la fuente de verdad y la clave es comodidad.

────────────────────────────────────────────────────────────────────────────────
LO QUE NO SE HACE, Y POR QUÉ

**No se vincula cada entidad con todos sus chunks como nodos del grafo.** §7.3
dice que «cada entidad **puede** vincularse con los chunks del índice FAISS que la
mencionan», y hacerlo con nodos convertiría el grafo en bipartito y añadiría
decenas de miles de nodos que no son entidades. La vinculación se conserva como
atributo `chunk_ids` del nodo, con el mismo tope.

**No se integra en la recuperación (§8.5).** Decisión de `ESTADO.md` §18: §8.5
dice «**puede** combinarlo», §7 concede la puntuación por implementar el
componente, e integrarlo obligaría a regenerar `resultados.jsonl` y volver a
probar §1.4 a cambio de una mejora que **no se puede medir** sin *ground truth*.
"""

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graph.canonical import cargar_alias  # noqa: E402
from graph.relations import Triple, extraer_relaciones  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
SALIDA = RAIZ / "entrega" / "base_vectorial" / "grafo" / "grafo.graphml"

# Cuántos `chunk_id` se conservan por arista y por nodo. Con tope, el archivo
# crece con el número de tripletas distintas y no con el de menciones.
MAX_EVIDENCIAS = 10

# Posiciones de fragmento que se saltan. La `#0000` de un PDF es casi siempre
# portada, índice o créditos, y medido en `ESTADO.md` §18 producía tripletas como
# `(CSET, desarrolla, Jason Ly)` a partir de «produced by» y de listas de
# empleados. No es contenido del documento, es su carátula.
POSICIONES_EXCLUIDAS = {0}


# ── Construcción ─────────────────────────────────────────────────────────────

def _clave_arista(t: Triple) -> Tuple[str, str, str, str, str]:
    """Identidad de una tripleta: sujeto, tipo, relación, objeto, tipo."""
    return (t.subject, t.subject_type, t.relation, t.object, t.object_type)


def construir(tripletas: Iterable[Triple],
              max_evidencias: int = MAX_EVIDENCIAS):
    """Agrega tripletas en un `MultiDiGraph` listo para exportar.

    Las tripletas repetidas **no crean aristas nuevas**: incrementan su `peso` y
    añaden su `chunk_id` a la evidencia. El peso es la señal de cuántas veces el
    corpus afirma el mismo hecho, y §8.5 lo contempla explícitamente al hablar de
    puntuar «basada en el número de relaciones relevantes encontradas».
    """
    import networkx as nx

    G = nx.MultiDiGraph()
    aristas: Dict[Tuple, Dict] = {}
    chunks_por_nodo: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    fenomenos_por_nodo: Dict[Tuple[str, str], set] = defaultdict(set)

    for t in tripletas:
        clave = _clave_arista(t)
        arista = aristas.get(clave)
        if arista is None:
            aristas[clave] = arista = {
                "relacion": t.relation,
                "peso": 0,
                "doc_id": t.doc_id,          # primera evidencia: la canónica
                "chunk_id": t.chunk_id,
                "chunk_ids": [],
                "evidencia": t.evidence,
                "pasiva": t.passive,
            }
        arista["peso"] += 1
        if (t.chunk_id and len(arista["chunk_ids"]) < max_evidencias
                and t.chunk_id not in arista["chunk_ids"]):
            arista["chunk_ids"].append(t.chunk_id)

        # El fenómeno se lee del `doc_id` (`F2-ESA-028#0007`), que es lo que hay
        # a mano; sirve para saber en qué fenómenos aparece cada entidad, que es
        # lo que §8.7 contempla filtrar.
        fen = t.doc_id[1] if t.doc_id[:1] == "F" else "?"
        for nombre, tipo in ((t.subject, t.subject_type),
                             (t.object, t.object_type)):
            fenomenos_por_nodo[(nombre, tipo)].add(fen)
            lista = chunks_por_nodo[(nombre, tipo)]
            if t.chunk_id and len(lista) < max_evidencias and t.chunk_id not in lista:
                lista.append(t.chunk_id)

    # Nodos primero, para que lleven sus atributos aunque no tengan aristas.
    for (nombre, tipo), chunks in chunks_por_nodo.items():
        G.add_node(f"{tipo}:{nombre}", nombre=nombre, tipo=tipo,
                   fenomenos="".join(sorted(fenomenos_por_nodo[(nombre, tipo)])),
                   chunk_ids=" ".join(chunks), n_chunks=len(chunks))

    for (suj, t_suj, rel, obj, t_obj), a in aristas.items():
        # ⚠️ `chunk_ids` se serializa a cadena AQUÍ: GraphML no admite listas y
        # `write_graphml` lanzaría una excepción con la lista de Python.
        G.add_edge(f"{t_suj}:{suj}", f"{t_obj}:{obj}", key=rel,
                   relacion=rel, peso=a["peso"],
                   doc_id=a["doc_id"], chunk_id=a["chunk_id"],
                   chunk_ids=" ".join(a["chunk_ids"]),
                   evidencia=a["evidencia"][:200], pasiva=bool(a["pasiva"]))
    return G


def exportar(G, destino: Path = SALIDA) -> Path:
    """Escribe el GraphML, a temporal y renombrando.

    El temporal es la misma precaución que en `core/store.py` y en
    `generador.py`: una corrida interrumpida no debe dejar un `.graphml` a medias,
    que es peor que no dejar ninguno porque **parece** un entregable válido.
    """
    import networkx as nx

    destino.parent.mkdir(parents=True, exist_ok=True)
    temporal = destino.with_suffix(destino.suffix + ".parcial")
    nx.write_graphml(G, temporal, encoding="utf-8", prettyprint=True)
    temporal.replace(destino)
    return destino


def resumen(G) -> str:
    """Las cifras que hay que citar en el informe técnico."""
    import networkx as nx

    tipos = Counter(d.get("tipo", "?") for _, d in G.nodes(data=True))
    rels = Counter(d.get("relacion", "?") for _, _, d in G.edges(data=True))
    grados = sorted(G.degree, key=lambda x: -x[1])[:10]
    lineas = [
        f"nodos            : {G.number_of_nodes():,}",
        f"aristas          : {G.number_of_edges():,}",
        f"por tipo de nodo : {dict(tipos.most_common())}",
        f"por relación     : {dict(rels.most_common())}",
        f"componentes      : {nx.number_weakly_connected_components(G):,}",
        "más conectados   : " + ", ".join(f"{n.split(':',1)[1]} ({g})"
                                          for n, g in grados),
    ]
    return "\n  ".join(lineas)


# ── La pasada sobre el corpus ────────────────────────────────────────────────

def tripletas_del_corpus(limite: Optional[int] = None,
                         lote: int = 8) -> Iterable[Triple]:
    """Recorre el corpus con GLiNER y va emitiendo tripletas. **Necesita GPU.**

    Se hace en flujo y no acumulando: las 127.070 unidades con sus entidades no
    caben cómodamente en memoria, y el grafo se agrega incrementalmente.
    """
    from graph import ner

    alias = cargar_alias()
    modelo = ner.cargar_modelo()
    contar = ner.contador_combinado(modelo)

    pendientes: List[dict] = []
    vistas = 0
    t0 = time.perf_counter()

    def procesar(bloque: List[dict]) -> Iterable[Triple]:
        entidades, _ = ner.extraer_entidades(
            modelo, [u["texto"] for u in bloque], ner.ETIQUETAS_EN,
            umbral=ner.UMBRAL, lote=lote)
        for u, ents in zip(bloque, entidades):
            if len(ents) < 2:
                continue          # una sola entidad no forma tripleta
            yield from extraer_relaciones(
                u["texto"], ents, doc_id=u["doc_id"], chunk_id=u["chunk_id"],
                alias=alias)

    for registro in ner.leer_registros():
        # Se salta la carátula: ver POSICIONES_EXCLUIDAS.
        if int(registro.get("posicion", -1)) in POSICIONES_EXCLUIDAS:
            continue
        for unidad in ner.unidades_de_registro(registro, contar, ner.PRESUPUESTO):
            pendientes.append(unidad)
            vistas += 1
            if len(pendientes) >= lote * 8:
                yield from procesar(pendientes)
                pendientes = []
                if vistas % 4000 < lote * 8:
                    ritmo = vistas / (time.perf_counter() - t0)
                    restantes = (ner.UNIDADES_TOTALES - vistas) / ritmo / 60
                    print(f"  {vistas:,} unidades · {ritmo:.1f} u/s · "
                          f"quedan ~{restantes:.0f} min", flush=True)
            if limite and vistas >= limite:
                break
        if limite and vistas >= limite:
            break
    if pendientes:
        yield from procesar(pendientes)


def tripletas_de_demo() -> List[Triple]:
    """Tripletas de un texto sintético, para probar el constructor sin GLiNER.

    Cubre lo que importa del constructor y no del extractor: que las repetidas
    sumen peso en vez de duplicar aristas, que dos relaciones distintas entre las
    mismas entidades **no** se machaquen, y que la evidencia se serialice.
    """
    from graph.relations import gazetteer, buscar_entidades
    from graph.canonical import VARIANTES_SEMILLA

    alias = cargar_alias()
    patron = gazetteer(VARIANTES_SEMILLA)
    textos = [
        # Coordinación: dos sujetos, dos tripletas desde una oración.
        ("EE.UU. y China desarrollan armas antisatélite avanzadas.", "F1-X-001", "#0001"),
        # Repetición: la misma tripleta que la anterior → debe sumar `peso`, no
        # duplicar la arista.
        ("Estados Unidos desarrolla armas antisatélite desde 2019.", "F1-X-002", "#0002"),
        # 🔴 Arista PARALELA: mismo par de entidades que arriba, relación distinta.
        # Es lo único que ejercita de verdad el `MultiDiGraph`; con un grafo simple
        # esta tripleta machacaría la de `desarrolla` y el corpus perdería un hecho.
        ("Estados Unidos opera armas antisatélite en órbita.", "F1-X-003", "#0006"),
        ("El Tratado del Espacio Exterior regula las armas antisatélite.", "F2-Y-001", "#0003"),
        ("La OTAN coopera con Estados Unidos en defensa espacial.", "F2-Y-002", "#0004"),
        # Sin verbo del inventario («amenazan»): no debe emitir tripleta.
        ("Los desechos orbitales amenazan la órbita baja terrestre.", "F2-Y-003", "#0005"),
    ]
    salida: List[Triple] = []
    for texto, doc, chunk in textos:
        ents = buscar_entidades(texto, patron)
        salida += extraer_relaciones(texto, ents, doc_id=doc,
                                     chunk_id=f"{doc}{chunk}", alias=alias)
    return salida


# ── Entrada ──────────────────────────────────────────────────────────────────

def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Construye grafo.graphml (§7)")
    parser.add_argument("--demo", action="store_true",
                        help="entidades sintéticas, sin GLiNER ni GPU")
    parser.add_argument("--limite", type=int, default=None,
                        help="tope de unidades, para una corrida corta")
    parser.add_argument("--salida", type=Path, default=SALIDA)
    parser.add_argument("--max-evidencias", type=int, default=MAX_EVIDENCIAS)
    args = parser.parse_args()

    if args.demo and args.salida == SALIDA:
        # Que una prueba no pise el entregable, igual que en `generador.py`.
        args.salida = SALIDA.with_name("grafo_demo.graphml")
        print("⚠️  MODO DEMO: entidades sintéticas, el grafo NO es el del corpus.")
        print(f"    Salida: {args.salida.name}\n")

    inicio = time.perf_counter()
    fuente = tripletas_de_demo() if args.demo else tripletas_del_corpus(args.limite)

    print("construyendo…")
    G = construir(fuente, args.max_evidencias)
    ruta = exportar(G, args.salida)

    linea = "─" * 74
    print(f"\n{linea}")
    print(f"  {resumen(G)}")
    print(f"\n  archivo : {ruta}")
    print(f"  tamaño  : {ruta.stat().st_size / 1e6:,.2f} MB")
    print(f"  tiempo  : {time.perf_counter() - inicio:,.1f} s")
    print(f"{linea}")
    print("Siguiente: py -m tools.verificar_grafo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
