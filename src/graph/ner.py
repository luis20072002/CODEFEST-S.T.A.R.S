"""Banco de pruebas de GLiNER para el grafo (§7.2). Pensado para GPU.

    py -m graph.ner              # banco completo (~pocos minutos en GPU)
    py -m graph.ner 400          # con 400 unidades en vez de 200

Mismo papel que `py -m embedding.encoder 32` tuvo para la codificación: **decide
si la pasada completa cabe en una sesión** antes de lanzarla, y responde con
medición las preguntas que quedaron abiertas en `TAREAS.md`, Fase 7.

────────────────────────────────────────────────────────────────────────────────
QUÉ MIDE, Y POR QUÉ CADA COSA

1. **`max_len` real del modelo.** El presupuesto de la unidad se fijó en 320
   tokens suponiendo que la ventana es 384. Si es otra, el presupuesto cambia.
2. **El presupuesto real de texto**, barriendo 384 → 224. Medido el 2026-08-09:
   GLiNER truncó unidades que el tokenizador contaba en 320 y él contó en **401**,
   porque su cuenta incluye el prompt de etiquetas y sus propios marcadores. El
   presupuesto **no se puede deducir de `max_len`**: se elige como el mayor valor
   donde el contador de truncamientos da cero con las nueve etiquetas.
3. **Unidades por segundo**, con el conjunto de etiquetas real y no con dos de
   juguete: el coste crece con el número de etiquetas.
4. **Etiquetas en inglés frente a español.** El corpus es trilingüe y GLiNER es
   zero-shot; cuál rinde mejor no se puede razonar, hay que verlo.
5. **Los 13 documentos bibliográficos excluidos.** `ESTADO.md` §18 los excluye con
   un argumento **estructural** —un registro con `PMID:` es una cita— y eso no es
   una medición de entidades. Aquí se cierra: se pasa GLiNER sobre una muestra y
   se cuenta qué tipos devuelve. Si son nombres de autor, la regla queda medida.
6. **Tripletas de punta a punta**, encadenando con `graph.relations`, para ver
   con los ojos si las entidades específicas de GLiNER mejoran la precisión
   respecto al gazetteer de 45 entidades genéricas.

────────────────────────────────────────────────────────────────────────────────
DE DÓNDE LEE EL TEXTO

De **`metadata.jsonl`**, no de `chunks.jsonl`. Los dos tienen el texto, pero el
primero ya viaja a la máquina de cálculo porque `generador.py` lo necesita, y son
240 MB en vez de 247 MB adicionales. Trae además `formato`, que hace falta para
separar prosa de tabular, y el `chunk_id` que §7.2 exige en cada tripleta.

────────────────────────────────────────────────────────────────────────────────
LA REVISIÓN ESTÁ FIJADA (§1.4)

`REVISION = "443d26d6…"`, resuelta el 2026-08-09. Es el mismo riesgo que con
BGE-M3 (§8): sin fijar el commit, otra descarga puede dar otro modelo y otras
entidades **en silencio**. El banco imprime el sha vigente en HuggingFace y lo
compara con el fijado; si dejan de coincidir, el repo del modelo se movió y hay
que decidir si se actualiza.
"""

import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODELO = "urchade/gliner_multi-v2.1"

# Revisión fijada el 2026-08-09, resuelta con `huggingface_hub.model_info()`.
# Es la misma disciplina que con BGE-M3 (§8): sin fijar el commit, otra descarga
# puede dar otro modelo y otras entidades **sin dar ningún error**.
REVISION: Optional[str] = "443d26d654e0324125a96bebd8e796c14ff2efe6"

# Umbral de confianza. Más bajo da más entidades **y más falsos positivos**, y un
# grafo lleno de entidades espurias no cumple el ejemplo de §7.1. El banco lo
# barre para que se elija con datos.
UMBRAL = 0.5

# Presupuesto de texto de la unidad, en tokens.
#
# 🔴 **288, y NO 320 ni 384.** El razonamiento de que con `max_len = 384` cabían
# 320 de texto más ~64 de prompt **era falso**, y lo desmintió la medición:
#
#   · GLiNER truncó **7 de 300** unidades construidas con presupuesto 320,
#     avisando «Sentence of length 401 has been truncated to 384»;
#   · esas 7 son exactamente las que tienen **más de 310 tokens** de texto, así
#     que el sobrecoste real de GLiNER es **74 tokens**, no 64;
#   · y de esos 74, solo ~32 son las etiquetas contadas a mano. Los ~42 restantes
#     son marcadores internos del modelo, que **no se pueden deducir** — de ahí
#     que el presupuesto haya que medirlo y no calcularlo.
#
# Con 288 el texto más largo que produce la cascada mide 286 tokens: 286 + 74 =
# 360, con 24 de margen para que las etiquetas en español cuesten 4 más que en
# inglés. El coste es pasar de ~1,4 h a ~1,6 h de GPU: **doce minutos a cambio de
# que ninguna unidad se escanee a medias en silencio.**
PRESUPUESTO = 288

# Los tipos que se le piden a GLiNER, anclados en los cinco de §7.1 más los dos
# que los fenómenos del reto exigen. Las dos versiones se comparan en el banco.
ETIQUETAS_EN = ["person", "organization", "country", "location",
                "weapon system", "military technology", "space object",
                "treaty or regulation", "armed group"]

ETIQUETAS_ES = ["persona", "organización", "país", "lugar",
                "sistema de armas", "tecnología militar", "objeto espacial",
                "tratado o norma", "grupo armado"]

# Marcadores inequívocos de volcado bibliográfico. `Authors:` NO está: casa
# igual con una cita de PubMed que con un catálogo legítimo, y excluía por error
# `F1-DAIO-002` (`ESTADO.md` §18).
BIBLIOGRAFICO = re.compile(r"\b(PMID|NCT Number)\s*:", re.I)

FORMATOS_TABULARES = {"csv", "xlsx", "pbf"}

METADATA = (Path(__file__).resolve().parents[2] / "entrega" / "base_vectorial"
            / "encoder_bge-m3" / "metadata.jsonl")


# ── Carga del modelo ─────────────────────────────────────────────────────────

def cargar_modelo(nombre: str = MODELO, revision: Optional[str] = REVISION):
    """Carga GLiNER y lo mueve a GPU si hay.

    Se importa aquí dentro y no arriba para que el resto del módulo —las
    constantes, el troceado— se pueda importar sin pagar la carga de torch.
    """
    import torch
    from gliner import GLiNER

    kwargs = {"revision": revision} if revision else {}
    modelo = GLiNER.from_pretrained(nombre, **kwargs)
    if not revision:
        print("⚠️  REVISION sin fijar: §1.4 exige el commit, no solo el nombre.")
    if torch.cuda.is_available():
        modelo = modelo.to("cuda")
        print(f"dispositivo : cuda ({torch.cuda.get_device_name(0)})")
    else:
        print("dispositivo : cpu  ⚠️ la pasada completa no es viable así")
    return modelo


def sha_del_modelo(nombre: str = MODELO) -> str:
    """El commit actual del repo del modelo en HuggingFace, para fijarlo."""
    try:
        from huggingface_hub import model_info
        return model_info(nombre).sha or "?"
    except Exception as e:                     # sin red, o versión antigua
        return f"no se pudo resolver ({type(e).__name__})"


# ── Troceado en unidades ─────────────────────────────────────────────────────

def contador_de_tokens(modelo=None):
    """Devuelve la función que cuenta tokens con el tokenizador de GLiNER.

    Si se pasa el modelo, usa su propio tokenizador —lo correcto—. Si no, baja el
    de mDeBERTa aparte, que es el backbone: son unos MB frente a los ~1,2 GB del
    modelo, y permite trocear en una máquina sin GPU. Es el mismo truco que
    `embedding.encoder.cargar_tokenizador()`.
    """
    if modelo is not None:
        tk = modelo.data_processor.transformer_tokenizer
    else:
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base")
    return lambda t: len(tk.encode(t))


def leer_registros(ruta: Path = METADATA, excluir_biblio: bool = True
                   ) -> Iterable[dict]:
    """Recorre `metadata.jsonl` decidiendo qué entra al grafo.

    Devuelve los registros que se escanean: prosa completa más los tabulares que
    no son volcados bibliográficos. La clasificación se hace **por documento**, no
    por formato — medido en `ESTADO.md` §18: los tabulares no son una población
    homogénea y excluirlos en bloque tiraría catálogos con entidades reales.
    """
    clase: Dict[str, bool] = {}                # doc_id -> es bibliográfico
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            r = json.loads(linea)
            if r["formato"] in FORMATOS_TABULARES:
                d = r["doc_id"]
                if d not in clase:
                    clase[d] = bool(BIBLIOGRAFICO.search(r["texto"][:1200]))
                if excluir_biblio and clase[d]:
                    continue
            yield r


def unidades_de_registro(registro: dict, contar_tokens,
                         presupuesto: int = PRESUPUESTO) -> List[dict]:
    """Trocea un fragmento del índice en unidades que caben en GLiNER.

    Reutiliza la cascada de `chunking.chunker` —recibe el contador y el
    presupuesto por parámetro justamente para esto— y con ello hereda la garantía
    de §3.3 de no partir oraciones. Aquí eso importa por una razón propia: **una
    ventana que corta una entidad por la mitad la pierde.**

    ⚠️ Cada unidad **hereda el `doc_id` y el `chunk_id` del fragmento padre**, no
    genera identificadores nuevos. §7.2 exige que la tripleta referencie el
    `chunk_id` de origen, y ese id tiene que existir en el `metadata.jsonl`
    entregado o la evidencia no se puede rastrear.
    """
    from chunking.chunker import chunk_document
    from core.document import Document

    doc = Document(doc_id=registro["doc_id"], source=registro.get("fuente", ""),
                   format=registro["formato"], text=registro["texto"],
                   phenomenon=int(registro["fenomeno"]))
    return [{"texto": u.text, "doc_id": registro["doc_id"],
             "chunk_id": registro["chunk_id"], "num_tokens": u.num_tokens}
            for u in chunk_document(doc, contar_tokens=contar_tokens,
                                    presupuesto=presupuesto)]


def _sorteo_estable(doc_id: str, divisor: int) -> bool:
    """¿Este documento entra en la muestra? Determinista entre ejecuciones.

    Se usa un hash **estable** y no `hash()`, que en Python está aleatorizado por
    proceso salvo que se fije `PYTHONHASHSEED`: con `hash()` la muestra cambiaría
    en cada corrida y las cifras del banco no serían comparables entre sí.
    """
    import hashlib
    h = hashlib.md5(doc_id.encode("utf-8")).hexdigest()
    return int(h, 16) % divisor == 0


def muestra_de_unidades(n: int, contar_tokens, solo_biblio: bool = False,
                        por_documento: int = 2, divisor: int = 7,
                        presupuesto: int = PRESUPUESTO) -> List[dict]:
    """Toma `n` unidades repartidas por **documentos**, de forma determinista.

    🔴 **El reparto es por documento y no por registro, y la diferencia importa.**
    La primera versión saltaba un registro de cada 37, pero los documentos de AI
    Index tienen **miles de chunks cada uno**, así que la muestra entera salía de
    tres documentos de AI Index — prosa estadística sobre bibliometría, donde no
    hay ni un sistema de armas ni un tratado. La medición de tipos de entidad que
    produjo era, por tanto, una medición del AI Index y no del corpus.

    `por_documento` limita cuántas unidades aporta cada documento, de modo que
    para llenar la muestra haya que recorrer muchos documentos distintos.

    🔴 **Y con eso no basta: hace falta sortear Y estratificar.**
    `metadata.jsonl` va ordenado por `doc_id`, y eso sesgó la muestra dos veces:

    - limitando solo las unidades por documento, salía de los **99 primeros
      documentos** del archivo — 53 % AI Index, cero F2 y cero F3;
    - añadiendo el sorteo por hash mejoró a 8 observatorios, pero **seguía sin un
      solo documento de F3**, que son 888 de los 1.826 y se llevan 18 de las 50
      consultas. Las 200 unidades se llenaban antes de llegar a `F3-*`.

    Así que la cuota es **por fenómeno**: cada uno aporta como mucho un tercio de
    la muestra, y el recorrido sigue aunque F1 ya esté lleno. §1.3 define el
    corpus como los tres fenómenos; una muestra que solo mide uno no mide el
    corpus.
    """
    unidades: List[dict] = []
    vistos: Dict[str, int] = {}
    cuota = max(1, n // 3)
    por_fenomeno: Dict[int, int] = {1: 0, 2: 0, 3: 0}

    for r in leer_registros(excluir_biblio=not solo_biblio):
        if solo_biblio:
            es_biblio = (r["formato"] in FORMATOS_TABULARES
                         and BIBLIOGRAFICO.search(r["texto"][:1200]))
            if not es_biblio:
                continue
        elif not _sorteo_estable(r["doc_id"], divisor):
            continue

        fen = int(r["fenomeno"])
        if not solo_biblio and por_fenomeno.get(fen, 0) >= cuota:
            continue                    # este fenómeno ya llenó su cuota
        d = r["doc_id"]
        if vistos.get(d, 0) >= por_documento:
            continue
        nuevas = unidades_de_registro(
            r, contar_tokens, presupuesto)[:por_documento]
        if not nuevas:
            continue

        vistos[d] = vistos.get(d, 0) + len(nuevas)
        por_fenomeno[fen] = por_fenomeno.get(fen, 0) + len(nuevas)
        unidades.extend(nuevas)
        # Solo se corta cuando los TRES han llenado, no al alcanzar `n`: cortar
        # antes es lo que dejaba F3 fuera.
        if solo_biblio and len(unidades) >= n:
            break
        if all(c >= cuota for c in por_fenomeno.values()):
            break
    return unidades[:n]


# ── Inferencia ───────────────────────────────────────────────────────────────

def extraer_entidades(modelo, textos: Sequence[str], etiquetas: Sequence[str],
                      umbral: float = UMBRAL, lote: int = 8
                      ) -> Tuple[List[List[dict]], int]:
    """Pasa GLiNER por una lista de textos. Devuelve `(entidades, truncados)`.

    La salida de las entidades ya trae `start`, `end`, `text` y `label`, que es
    exactamente lo que `graph.relations.extraer_relaciones()` espera.

    🔴 **Se cuentan los truncamientos, y eso es la mitad del valor de esta
    función.** GLiNER avisa con un `UserWarning` —«Sentence of length 401 has been
    truncated to 384»— y en un cuaderno ese aviso se pierde entre el ruido. Pero
    un texto truncado significa que **la última parte de la unidad no se escanea y
    sus entidades no existen**, sin ningún error. Contarlos convierte el
    presupuesto en algo que se elige con datos: se baja hasta que el contador da
    cero.

    ⚠️ La cuenta de GLiNER **no es la del tokenizador**: incluye el prompt de
    etiquetas y sus propios marcadores. Una unidad de 320 tokens medidos con
    mDeBERTa puede darle 401 a GLiNER. Por eso el presupuesto no se puede deducir
    de `max_len`, hay que medirlo aquí.
    """
    import warnings

    salida: List[List[dict]] = []
    truncados = 0
    for i in range(0, len(textos), lote):
        trozo = list(textos[i:i + lote])
        with warnings.catch_warnings(record=True) as avisos:
            warnings.simplefilter("always")
            try:
                salida.extend(modelo.batch_predict_entities(
                    trozo, list(etiquetas), threshold=umbral))
            except AttributeError:             # versiones sin batch_
                salida.extend(modelo.predict_entities(t, list(etiquetas),
                                                      threshold=umbral)
                              for t in trozo)
        truncados += sum(1 for a in avisos if "truncated" in str(a.message))
    return salida, truncados


# ── Banco ────────────────────────────────────────────────────────────────────

def _resumen_tipos(entidades: Iterable[List[dict]]) -> Dict[str, int]:
    from collections import Counter
    c: Counter = Counter()
    for lista in entidades:
        for e in lista:
            c[e["label"]] += 1
    return dict(c.most_common())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    n_unidades = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    linea = "─" * 74

    # ── 1. Entorno y procedencia ─────────────────────────────────────────────
    print(f"{linea}\n1. ENTORNO\n{linea}")
    print(f"modelo      : {MODELO}")
    print(f"sha en HF   : {sha_del_modelo()}")
    print(f"REVISION    : {REVISION or '🔴 SIN FIJAR — pegar el sha de arriba'}")
    if not METADATA.is_file():
        print(f"\n✘ no está {METADATA}")
        print("  La base vectorial no viaja en git; hay que copiarla (ESTADO.md §16).")
        raise SystemExit(1)

    modelo = cargar_modelo()
    contar = contador_de_tokens(modelo)

    # ── 2. La ventana real ───────────────────────────────────────────────────
    print(f"\n{linea}\n2. VENTANA DEL MODELO\n{linea}")
    cfg = getattr(modelo, "config", None)
    max_len = getattr(cfg, "max_len", None)
    print(f"config.max_len   : {max_len}")
    print(f"config.max_width : {getattr(cfg, 'max_width', None)}")
    print(f"presupuesto      : {PRESUPUESTO} tokens de TEXTO")
    if isinstance(max_len, int):
        margen = max_len - PRESUPUESTO
        print(f"margen para el prompt de etiquetas: {margen} tokens")
        if margen < 60:
            print("  ⚠️ margen escaso para 9 etiquetas: bajar PRESUPUESTO")

    # ── 3. ¿Cuánto texto cabe de verdad? ─────────────────────────────────────
    print(f"\n{linea}\n3. PRESUPUESTO REAL: BARRIDO SOBRE UNIDADES DEL CORPUS\n{linea}")
    # ⚠️ Dos versiones anteriores de esta prueba NO medían nada, y las dos por el
    # mismo motivo: usaban una sonda sintética.
    #
    #   · la primera construía el relleno repitiendo una frase sin contar, así que
    #     medía 669 tokens y se truncaba en TODOS los casos → 0 entidades siempre;
    #   · la segunda lo construía en tokens, pero el bucle paraba **por debajo**
    #     del presupuesto, así que ninguna sonda cruzaba la frontera y todos los
    #     presupuestos salían «truncado no», incluido 384.
    #
    # La lección es la de siempre en este proyecto: **medir sobre los datos
    # reales.** GLiNER cuenta la longitud a su manera —incluye el prompt de
    # etiquetas y sus marcadores— y esa cuenta no se puede reproducir a mano. Lo
    # único fiable es trocear unidades de verdad con cada presupuesto y contar
    # cuántas avisa GLiNER de haber truncado.
    print("presup.  unidades  truncadas   veredicto")
    for presupuesto in (384, 320, 288, 256, 224):
        muestra = muestra_de_unidades(80, contar, presupuesto=presupuesto)
        _, trunc = extraer_entidades(
            modelo, [u["texto"] for u in muestra], ETIQUETAS_EN)
        pct = 100 * trunc / len(muestra) if muestra else 0
        veredicto = "✔ ninguna" if trunc == 0 else f"🔴 {pct:.1f}% se trunca"
        print(f"  {presupuesto:3d}    {len(muestra):5d}     {trunc:5d}     {veredicto}")
    print("\n  El presupuesto bueno es el MAYOR con 0 truncadas: por encima, el")
    print("  final de esas unidades no se escanea y sus entidades no existen.")

    # ── 4. Troceado y rendimiento ────────────────────────────────────────────
    print(f"\n{linea}\n4. RENDIMIENTO\n{linea}")
    print(f"troceando {n_unidades} unidades de metadata.jsonl…")
    t0 = time.perf_counter()
    unidades = muestra_de_unidades(n_unidades, contar)
    print(f"  {len(unidades)} unidades en {time.perf_counter() - t0:,.1f} s")
    excedidas = sum(1 for u in unidades if u["num_tokens"] > PRESUPUESTO)
    print(f"  por encima del presupuesto: {excedidas}  (debe ser 0)")

    textos = [u["texto"] for u in unidades]
    for nombre, etiquetas in (("EN", ETIQUETAS_EN), ("ES", ETIQUETAS_ES)):
        t0 = time.perf_counter()
        ents, truncados = extraer_entidades(modelo, textos, etiquetas)
        seg = time.perf_counter() - t0
        ritmo = len(textos) / seg if seg else 0
        total = sum(len(e) for e in ents)
        # 118.788 es el total a escanear, medido en `ESTADO.md` §18.
        horas = 118_788 / ritmo / 3600 if ritmo else float("inf")
        print(f"\n  etiquetas {nombre}: {ritmo:5.1f} unidades/s → "
              f"**{horas:4.1f} h** para 118.788")
        print(f"    entidades: {total} ({total/len(textos):.1f} por unidad)")
        print(f"    truncadas: {truncados} de {len(textos)} "
              f"{'🔴 bajar PRESUPUESTO' if truncados else '✔'}")
        print(f"    por tipo : {_resumen_tipos(ents)}")
        if nombre == "EN":
            ents_en = ents

    # ── 5. Umbral ────────────────────────────────────────────────────────────
    print(f"\n{linea}\n5. UMBRAL DE CONFIANZA\n{linea}")
    # ⚠️ El recuento por umbral **no decide nada**: más entidades no es mejor, y un
    # grafo lleno de entidades espurias no cumple el ejemplo de §7.1. Lo que decide
    # es mirar las **marginales** —las que un umbral admite y el siguiente
    # rechaza—, porque son exactamente las que están en discusión. Imprimirlas
    # convierte la elección en una revisión de diez segundos en vez de una
    # corazonada sobre un número.
    por_umbral = {}
    for u in (0.3, 0.5, 0.7):
        e, _ = extraer_entidades(modelo, textos[:60], ETIQUETAS_EN, umbral=u)
        total = sum(len(x) for x in e)
        # Clave por (texto, etiqueta) para poder restar conjuntos entre umbrales.
        por_umbral[u] = {(x["text"], x["label"]) for lista in e for x in lista}
        print(f"  umbral {u}: {total:4d} entidades en 60 unidades "
              f"({total/60:.1f} por unidad · {len(por_umbral[u])} distintas)")

    for bajo, alto in ((0.3, 0.5), (0.5, 0.7)):
        marginales = sorted(por_umbral[bajo] - por_umbral[alto])
        print(f"\n  Las que {bajo} admite y {alto} rechaza ({len(marginales)}), "
              f"muestra de 25:")
        for t, l in marginales[:25]:
            print(f"      [{l:22s}] {t}")
    print(f"\n  ¿Son entidades de verdad y del dominio? Si al bajar el umbral solo")
    print(f"  entra ruido, el umbral alto es el bueno.")

    # ── 6. Los 13 documentos bibliográficos excluidos ────────────────────────
    print(f"\n{linea}\n6. ¿QUÉ RINDEN LOS 13 DOCUMENTOS EXCLUIDOS?\n{linea}")
    print("Cierra con medición la regla que hoy es solo estructural (§18).")
    biblio = muestra_de_unidades(40, contar, solo_biblio=True)
    if biblio:
        e_bib, _ = extraer_entidades(modelo, [u["texto"] for u in biblio],
                                     ETIQUETAS_EN)
        print(f"  {len(biblio)} unidades bibliográficas → "
              f"{sum(len(x) for x in e_bib)} entidades")
        print(f"  por tipo : {_resumen_tipos(e_bib)}")
        muestras = [x["text"] for lista in e_bib for x in lista][:20]
        print(f"  ejemplos : {muestras}")
        print("  ✔ si son nombres de autor y términos médicos, la exclusión")
        print("    queda medida y no solo argumentada.")
    else:
        print("  · no se encontraron unidades bibliográficas")

    # ── 7. Tripletas de punta a punta ────────────────────────────────────────
    print(f"\n{linea}\n7. TRIPLETAS CON ENTIDADES REALES\n{linea}")
    from graph.canonical import cargar_alias
    from graph.relations import extraer_relaciones

    alias = cargar_alias()
    tripletas = []
    for u, ents in zip(unidades, ents_en):
        if len(ents) >= 2:
            tripletas += extraer_relaciones(
                u["texto"], ents, doc_id=u["doc_id"],
                chunk_id=u["chunk_id"], alias=alias)
    print(f"  {len(unidades)} unidades → {len(tripletas)} tripletas")
    for t in tripletas[:20]:
        print(f"    ({t.subject}, {t.relation}, {t.object})"
              f"{'  ⇐ pasiva' if t.passive else ''}")
        print(f"        {t.chunk_id}  «…{t.evidence}…»")
    print("\n  ⚠️ Compárese con el gazetteer de 45 entidades genéricas, donde de")
    print("     12 tripletas unas 7-8 eran defendibles (§18). Si aquí la")
    print("     proporción no mejora, el inventario de verbos es el problema.")

    print(f"\n{linea}")
    print("SIGUIENTE: pegar el sha en REVISION y lanzar la pasada completa.")
    print(linea)
