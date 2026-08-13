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
from graph.canonical import cargar_alias, clave_canonica  # noqa: E402
from graph.relations import Triple, extraer_relaciones  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
SALIDA = RAIZ / "entrega" / "base_vectorial" / "grafo" / "grafo.graphml"

# El núcleo mirable de `--nucleo`. Vive en `src/data/` y **no** en `entrega/`:
# es una herramienta para verlo y para la figura del informe, no un entregable.
NUCLEO = RAIZ / "src" / "data" / "grafo_nucleo.graphml"

# Cuántos `chunk_id` se conservan por arista y por nodo. Con tope, el archivo
# crece con el número de tripletas distintas y no con el de menciones.
MAX_EVIDENCIAS = 10

# Posiciones de fragmento que se saltan. La `#0000` de un PDF es casi siempre
# portada, índice o créditos, y medido en `ESTADO.md` §18 producía tripletas como
# `(CSET, desarrolla, Jason Ly)` a partir de «produced by» y de listas de
# empleados. No es contenido del documento, es su carátula.
POSICIONES_EXCLUIDAS = {0}


# ── Construcción ─────────────────────────────────────────────────────────────

def id_nodo(nombre: str, tipo: str) -> str:
    """El identificador de un nodo en el grafo: `tipo:clave_canónica`.

    🔴 **Se indexa por la CLAVE, no por el nombre, y esto fue un bug real.** La
    primera versión usaba `f"{tipo}:{nombre}"`, y el resultado lo destapó la
    prueba C del verificador sobre el grafo completo: **93 grupos de nodos
    compartían clave canónica**, entre ellos `['Israel', 'ISRAEL']`,
    `['India', 'INDIA', 'india']` y `['Canada', 'Canadá']`.

    La causa: `clave_canonica()` colapsa correctamente mayúsculas y tildes, pero
    `canonizar()` **conserva el nombre tal como vino** cuando no hay entrada en la
    tabla de alias — porque §7 no pide reescribir el texto. Indexar por el nombre
    volvía a separar todo lo que la clave había unido, y la canonicalización solo
    funcionaba para las entidades que la tabla de alias enumeraba.

    `canonical.agrupar_por_identidad()` ya lo hacía bien; el error fue
    reimplementar aquí el agrupamiento en vez de usar esa identidad.
    """
    return f"{tipo}:{clave_canonica(nombre)}"


def _clave_arista(t: Triple) -> Tuple[str, str, str, str, str]:
    """Identidad de una tripleta: sujeto, tipo, relación, objeto, tipo.

    Los extremos se identifican por su **clave**, para que la arista de
    `('ISRAEL', ataca, X)` y la de `('Israel', ataca, X)` sean la misma.
    """
    return (clave_canonica(t.subject), t.subject_type, t.relation,
            clave_canonica(t.object), t.object_type)


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
    chunks_por_nodo: Dict[str, List[str]] = defaultdict(list)
    fenomenos_por_nodo: Dict[str, set] = defaultdict(set)
    tipo_por_nodo: Dict[str, str] = {}
    # Cuántas veces se ha visto cada forma superficial de un nodo. El nombre que
    # se muestra es el **más frecuente**, con desempate alfabético para que sea
    # determinista (§1.4): «Israel» debe ganar a «ISRAEL» porque aparece más, no
    # porque llegara antes en el recorrido.
    formas_por_nodo: Dict[str, Counter] = defaultdict(Counter)

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
            nodo = id_nodo(nombre, tipo)
            tipo_por_nodo[nodo] = tipo
            formas_por_nodo[nodo][nombre] += 1
            fenomenos_por_nodo[nodo].add(fen)
            lista = chunks_por_nodo[nodo]
            if t.chunk_id and len(lista) < max_evidencias and t.chunk_id not in lista:
                lista.append(t.chunk_id)

    # Nodos primero, para que lleven sus atributos aunque no tengan aristas.
    for nodo, chunks in chunks_por_nodo.items():
        formas = formas_por_nodo[nodo]
        # `-n` primero y luego el nombre: más frecuente gana, y a igualdad el
        # orden alfabético decide. Sin el desempate, dos corridas podrían mostrar
        # nombres distintos para el mismo nodo.
        nombre = min(formas.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        G.add_node(nodo, nombre=nombre, tipo=tipo_por_nodo[nodo],
                   fenomenos="".join(sorted(fenomenos_por_nodo[nodo])),
                   chunk_ids=" ".join(chunks), n_chunks=len(chunks),
                   # Las demás formas se conservan: son la trazabilidad de qué
                   # colapsó en este nodo, y lo que se mira para curar la tabla.
                   variantes=" | ".join(sorted(formas)[:8]),
                   n_variantes=len(formas))

    for (suj, t_suj, rel, obj, t_obj), a in aristas.items():
        # ⚠️ `chunk_ids` se serializa a cadena AQUÍ: GraphML no admite listas y
        # `write_graphml` lanzaría una excepción con la lista de Python.
        G.add_edge(f"{t_suj}:{suj}", f"{t_obj}:{obj}", key=rel,
                   relacion=rel, peso=a["peso"],
                   doc_id=a["doc_id"], chunk_id=a["chunk_id"],
                   chunk_ids=" ".join(a["chunk_ids"]),
                   evidencia=a["evidencia"][:200], pasiva=bool(a["pasiva"]))
    return G


def fusionar_por_clave(G):
    """Repara un grafo ya construido fusionando nodos que comparten clave.

    💡 **Existe para no repetir las dos horas de GPU.** El grafo de la primera
    pasada se construyó con el bug de `id_nodo()` —nodos indexados por nombre— y
    reconstruirlo desde el corpus costaría otra pasada completa. Esta función hace
    la fusión sobre el `.graphml` ya escrito, en segundos y sin modelo.

    Conserva de cada grupo el nombre **más frecuente por grado**, suma los pesos
    de las aristas que colapsan y une sus evidencias respetando el tope.
    """
    import networkx as nx

    grupos: Dict[str, List[str]] = defaultdict(list)
    for n, d in G.nodes(data=True):
        grupos[id_nodo(d.get("nombre", n), d.get("tipo", "?"))].append(n)

    H = nx.MultiDiGraph()
    destino_de: Dict[str, str] = {}
    for nuevo, viejos in grupos.items():
        # Nombre representativo: mayor grado, y a igualdad orden alfabético.
        elegido = min(viejos, key=lambda v: (-G.degree(v),
                                             G.nodes[v].get("nombre", v)))
        d = dict(G.nodes[elegido])
        chunks, fen, formas = [], set(), set()
        for v in viejos:
            destino_de[v] = nuevo
            dv = G.nodes[v]
            formas.add(dv.get("nombre", v))
            fen.update(str(dv.get("fenomenos", "")))
            for c in str(dv.get("chunk_ids", "")).split():
                if c not in chunks and len(chunks) < MAX_EVIDENCIAS:
                    chunks.append(c)
        d.update(chunk_ids=" ".join(chunks), n_chunks=len(chunks),
                 fenomenos="".join(sorted(x for x in fen if x)),
                 variantes=" | ".join(sorted(formas)[:8]),
                 n_variantes=len(formas))
        H.add_node(nuevo, **d)

    for u, v, k, d in G.edges(keys=True, data=True):
        nu, nv = destino_de[u], destino_de[v]
        if nu == nv:
            continue          # el bucle que aparece al fusionar no informa nada
        if H.has_edge(nu, nv, k):
            actual = H.edges[nu, nv, k]
            actual["peso"] = int(actual.get("peso", 1)) + int(d.get("peso", 1))
            ids = actual.get("chunk_ids", "").split()
            for c in str(d.get("chunk_ids", "")).split():
                if c not in ids and len(ids) < MAX_EVIDENCIAS:
                    ids.append(c)
            actual["chunk_ids"] = " ".join(ids)
        else:
            H.add_edge(nu, nv, key=k, **dict(d))
    return H


def nucleo(G, peso_minimo: int = 2):
    """Devuelve el esqueleto **mirable** del grafo. NO es el entregable.

    El grafo completo no se puede leer en un visor, y está medido por qué: 603
    componentes, de los cuales **454 son parejas sueltas** —una única tripleta que
    no conecta con nada— y 92 son tríos. El 35,2 % de los nodos vive en esas islas
    y el 88,3 % tiene grado 1 o 2. Aunque se pinte perfecto, es una maraña
    rodeada de confeti.

    Este filtro deja lo que sí cuenta algo, en dos pasos:

    1. **Aristas con `peso ≥ peso_minimo`**, o sea hechos que el corpus afirma más
       de una vez. Es un filtro con significado, no un recorte estético: §8.5
       habla justamente de puntuar «basada en el número de relaciones relevantes
       encontradas». Una tripleta vista una sola vez es la que más probablemente
       venga de los falsos positivos de `ESTADO.md` §18.
    2. **Solo el componente mayor**, que quita las islas que sobreviven al paso 1.

    Medido sobre el grafo entregado:

    | peso mínimo | tras filtrar | solo el gigante |
    |---|---|---|
    | 2 (por defecto) | 678 nodos · 660 aristas · 131 comp. | **365 nodos · 475 aristas** |
    | 3 | 236 nodos · 207 aristas · 54 comp. | **110 nodos · 132 aristas** |

    Con 2 se explora; con **3 sale la figura del `informe_tecnico.pdf`**, que a
    110 nodos se lee impresa.

    ⚠️ **Se escribe fuera de `entrega/`.** Un `.graphml` de más en la carpeta del
    entregable invita a que el jurado evalúe el recorte en vez del grafo, y §7
    pide el grafo del corpus, no su resumen.
    """
    import networkx as nx

    H = nx.MultiDiGraph()
    H.graph.update(G.graph)
    H.add_nodes_from(G.nodes(data=True))
    for u, v, k, d in G.edges(keys=True, data=True):
        if int(d.get("peso", 1)) >= peso_minimo:
            H.add_edge(u, v, key=k, **d)
    # Los nodos que se quedan sin ninguna arista tras el filtro ya no pintan nada.
    H.remove_nodes_from([n for n, g in H.degree() if g == 0])
    if H.number_of_nodes() == 0:
        return H
    mayor = max(nx.weakly_connected_components(H), key=len)
    return H.subgraph(mayor).copy()


def exportar(G, destino: Path = SALIDA) -> Path:
    """Escribe el GraphML, a temporal y renombrando.

    El temporal es la misma precaución que en `core/store.py` y en
    `generador.py`: una corrida interrumpida no debe dejar un `.graphml` a medias,
    que es peor que no dejar ninguno porque **parece** un entregable válido.

    🔴 **CUARTA TRAMPA: el `id` de una arista es GLOBALMENTE único en GraphML.**

    `write_graphml` escribe la **clave del multigrafo** en ese atributo, y aquí la
    clave es el nombre de la relación (ver `construir`). El archivo salía con
    3.460 aristas repartiéndose **13 valores de `id`**: las 357 `coopera_con`
    llevaban todas `id="coopera_con"`.

    networkx no se queja porque él interpreta ese `id` como clave **por par de
    nodos**, no global — así que ida y vuelta le cuadra y las seis pruebas de
    `tools/verificar_grafo.py` pasaban igual. Pero **graphology**, que es el motor
    de Gephi Lite y de casi todos los visores web, indexa las aristas por `id` y
    aborta la carga al ver el segundo repetido:

        Graph.mergeDirectedEdgeWithKey: inconsistency detected when attempting
        to merge the "coopera_con" edge with "ORG:ai index" source &
        "ORG:mckinsey & company" target vs. ("ORG:ai index", "ORG:epoch ai")

    Un entregable que no abre en un visor vale lo mismo que no entregarlo, y §7
    es bonus precisamente por si alguien lo mira.

    ⚠️ **La relación no se pierde** al dejar de ser el `id`: viaja como atributo
    `relacion`, que este módulo ya declaraba la fuente de verdad y la clave, mera
    comodidad. Lo único que cambia es que al releer con `edge_key_type=str` la
    clave es `e123` en vez de `coopera_con`.

    ⚠️ **Se arregla renumerando la CLAVE, no con `edge_id_from_attribute`.** Esa
    bandera del escritor también da ids únicos, pero **repite el valor como un
    `<data>` más** en las 3.460 aristas: un campo redundante en el entregable y
    ~100 KB de archivo a cambio de nada. Renumerar la clave en una copia deja el
    archivo exactamente igual que antes salvo por el `id`.
    """
    import networkx as nx

    # Copia con la clave de arista sustituida por un identificador propio. Se
    # trabaja sobre una copia para no alterar el grafo que tenga el llamador —
    # `main()` imprime `resumen(G)` después de exportar.
    #
    # El orden de `G.edges()` es el de inserción, así que la numeración es
    # **determinista**: dos exportaciones del mismo grafo dan el mismo archivo,
    # que es lo que exige §1.4.
    H = nx.MultiDiGraph()
    H.graph.update(G.graph)
    for n, datos in G.nodes(data=True):
        # `label` es lo que los visores pintan **por convención**. Sin él muestran
        # el identificador, que va en minúscula, sin tildes y con el tipo delante
        # (`LOC:estados unidos` en vez de `Estados Unidos`), porque se construye
        # desde la clave canónica. No es un dato nuevo: es una copia de `nombre`
        # puesta donde el lector la busca, igual que el `id` de arista de abajo.
        H.add_node(n, **{**datos, "label": datos.get("nombre", n)})
    for i, (u, v, _, datos) in enumerate(G.edges(keys=True, data=True)):
        # Se descarta un `id` que viniera en los datos: lo dejaba una versión
        # anterior de esta función y volvería a escribirse como `<data>`.
        H.add_edge(u, v, key=f"e{i}",
                   **{k: x for k, x in datos.items() if k != "id"})

    destino.parent.mkdir(parents=True, exist_ok=True)
    temporal = destino.with_suffix(destino.suffix + ".parcial")
    nx.write_graphml(H, temporal, encoding="utf-8", prettyprint=True)
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
        # ⚠️ Se lee el atributo `nombre`, NO el identificador: desde que los nodos
        # se indexan por clave canónica, el id va en minúsculas y sin tildes
        # (`LOC:estados unidos`), que no es lo que un humano quiere leer.
        "más conectados   : " + ", ".join(
            f"{G.nodes[n].get('nombre', n)} ({g})" for n, g in grados),
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
    parser.add_argument("--fusionar", type=Path, default=None,
                        help="repara un .graphml existente fusionando los nodos "
                             "que comparten clave canónica. Segundos, sin modelo")
    parser.add_argument("--reexportar", type=Path, default=None,
                        help="reescribe un .graphml existente con el escritor "
                             "actual, sin reconstruirlo. Segundos, sin modelo")
    parser.add_argument("--nucleo", type=Path, default=None,
                        help="extrae el esqueleto mirable de un .graphml "
                             "(aristas corroboradas + componente mayor) a "
                             "src/data/. NO es el entregable")
    parser.add_argument("--peso-minimo", type=int, default=2,
                        help="peso mínimo de arista para --nucleo (2 explora, "
                             "3 da la figura del informe)")
    args = parser.parse_args()

    if args.nucleo is not None:
        import networkx as nx

        # Que un recorte no pise el entregable, igual que hace `--demo`.
        if args.salida == SALIDA:
            args.salida = NUCLEO
        print(f"leyendo {args.nucleo.name}…")
        G = nx.read_graphml(args.nucleo, force_multigraph=True,
                            edge_key_type=str)
        H = nucleo(G, args.peso_minimo)
        ruta = exportar(H, args.salida)
        linea = "─" * 74
        print(f"\n{linea}")
        print(f"⚠️  ESTO NO ES EL ENTREGABLE: es el grafo recortado para poder")
        print(f"    mirarlo. El de §7 sigue siendo {SALIDA.name} entero.\n")
        print(f"  entero  : {G.number_of_nodes():,} nodos · "
              f"{G.number_of_edges():,} aristas")
        print(f"  núcleo  : {H.number_of_nodes():,} nodos · "
              f"{H.number_of_edges():,} aristas  (peso >= {args.peso_minimo}, "
              f"un solo componente)")
        print(f"\n  {resumen(H)}")
        print(f"\n  archivo : {ruta}   ({ruta.stat().st_size / 1e3:,.0f} KB)")
        print(f"{linea}")
        return 0

    # `--reexportar` no toca el grafo: lo lee y lo vuelve a escribir. Existe para
    # aplicar un arreglo del ESCRITOR —como los `id` de arista únicos— sin repetir
    # las ~2 h de GPU de la pasada completa.
    #
    # ⚠️ **Para esto NO sirve `--fusionar`.** Sobre un grafo ya fusionado sus
    # grupos son de un solo nodo, y entonces recalcula `variantes` a partir de ese
    # único miembro: `iHEALTH | iHealth` se quedaría en `iHEALTH` y `n_variantes`
    # volvería a 1. Se perdería la trazabilidad de qué formas colapsó cada nodo.
    if args.reexportar is not None:
        import networkx as nx

        print(f"leyendo {args.reexportar.name}…")
        G = nx.read_graphml(args.reexportar, force_multigraph=True,
                            edge_key_type=str)
        ruta = exportar(G, args.salida)
        linea = "─" * 74
        print(f"\n{linea}")
        print(f"  {G.number_of_nodes():,} nodos · {G.number_of_edges():,} aristas "
              f"(sin cambios: solo se reescribe el archivo)")
        print(f"\n  archivo : {ruta}   ({ruta.stat().st_size / 1e6:,.2f} MB)")
        print(f"{linea}\nSiguiente: py -m tools.verificar_grafo")
        return 0

    # `--fusionar` no construye nada: repara. Se atiende antes de todo lo demás.
    if args.fusionar is not None:
        import networkx as nx

        print(f"leyendo {args.fusionar.name}…")
        G = nx.read_graphml(args.fusionar, force_multigraph=True,
                            edge_key_type=str)
        antes = (G.number_of_nodes(), G.number_of_edges())
        H = fusionar_por_clave(G)
        ruta = exportar(H, args.salida)
        linea = "─" * 74
        print(f"\n{linea}")
        print(f"  antes   : {antes[0]:,} nodos · {antes[1]:,} aristas")
        print(f"  después : {H.number_of_nodes():,} nodos · "
              f"{H.number_of_edges():,} aristas")
        print(f"  fusionados: {antes[0] - H.number_of_nodes():,} nodos")
        print(f"\n  {resumen(H)}")
        print(f"\n  archivo : {ruta}   ({ruta.stat().st_size / 1e6:,.2f} MB)")
        print(f"{linea}\nSiguiente: py -m tools.verificar_grafo")
        return 0

    if args.demo and args.salida == SALIDA:
        # Que una prueba no pise el entregable, igual que en `generador.py`.
        # Y que tampoco se quede DENTRO de `entrega/`: un grafo de demo en la
        # carpeta del entregable invita a que se evalúe lo que no es, el mismo
        # motivo por el que `--nucleo` escribe fuera.
        args.salida = NUCLEO.with_name("grafo_demo.graphml")
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
