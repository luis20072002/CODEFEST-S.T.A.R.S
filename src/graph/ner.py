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
2. **Si el prompt de etiquetas ocupa la ventana.** GLiNER concatena los tipos
   antes del texto. Con nueve etiquetas eso puede comerse 40-50 tokens, y
   entonces el presupuesto de TEXTO tiene que bajar. Se comprueba
   empíricamente: se busca una entidad puesta **al final** de un texto largo, con
   pocas etiquetas y con muchas. Si con muchas desaparece, el prompt desplaza.
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
🔴 ANTES DE LA PASADA DEFINITIVA: FIJAR LA REVISIÓN

`REVISION` está en `None`, y así **no cumple §1.4**. Es el mismo riesgo que con
BGE-M3: sin fijar el commit, otra descarga puede dar otro modelo y otras
entidades, en silencio. El banco **imprime el sha resuelto**; hay que pegarlo
aquí abajo antes de la corrida buena.
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

# Presupuesto de la unidad, en tokens. Medido en `ESTADO.md` §18: con 320 salen
# ~105.285 unidades de prosa y **0 excedidas**. Se confirma contra `max_len`.
PRESUPUESTO = 320

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


def muestra_de_unidades(n: int, contar_tokens, solo_biblio: bool = False,
                        paso: int = 37) -> List[dict]:
    """Toma `n` unidades repartidas por el corpus, de forma determinista.

    `paso` salta registros para que la muestra no salga toda del primer
    documento, que en este corpus sería AI Index y no representa nada.
    """
    unidades: List[dict] = []
    for i, r in enumerate(leer_registros(excluir_biblio=not solo_biblio)):
        if solo_biblio:
            es_biblio = (r["formato"] in FORMATOS_TABULARES
                         and BIBLIOGRAFICO.search(r["texto"][:1200]))
            if not es_biblio:
                continue
        elif i % paso:
            continue
        unidades.extend(unidades_de_registro(r, contar_tokens))
        if len(unidades) >= n:
            break
    return unidades[:n]


# ── Inferencia ───────────────────────────────────────────────────────────────

def extraer_entidades(modelo, textos: Sequence[str], etiquetas: Sequence[str],
                      umbral: float = UMBRAL, lote: int = 8) -> List[List[dict]]:
    """Pasa GLiNER por una lista de textos y devuelve entidades por texto.

    La salida de `predict_entities` ya trae `start`, `end`, `text` y `label`, que
    es exactamente lo que `graph.relations.extraer_relaciones()` espera.
    """
    salida: List[List[dict]] = []
    for i in range(0, len(textos), lote):
        trozo = list(textos[i:i + lote])
        try:
            salida.extend(modelo.batch_predict_entities(
                trozo, list(etiquetas), threshold=umbral))
        except AttributeError:                 # versiones sin batch_
            salida.extend(modelo.predict_entities(t, list(etiquetas),
                                                  threshold=umbral)
                          for t in trozo)
    return salida


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

    # ── 3. ¿El prompt de etiquetas desplaza el texto? ─────────────────────────
    print(f"\n{linea}\n3. ¿EL PROMPT DE ETIQUETAS OCUPA LA VENTANA?\n{linea}")
    # Se pone una entidad inequívoca AL FINAL de un texto que llena el
    # presupuesto. Si con 9 etiquetas desaparece y con 2 se encuentra, el prompt
    # está desplazando el final del texto fuera de la ventana.
    relleno = ("El informe analiza la situación con detalle y aporta datos. " * 60)
    sonda = relleno + " El acuerdo fue firmado en Bogotá, Colombia."
    for etiquetas in (["country"], ETIQUETAS_EN):
        ents = extraer_entidades(modelo, [sonda], etiquetas)[0]
        halla_final = any(e["text"].strip().rstrip(".") in ("Colombia", "Bogotá")
                          for e in ents)
        print(f"  {len(etiquetas)} etiqueta(s): {len(ents):3d} entidades · "
              f"encuentra la del final: {'sí' if halla_final else '🔴 NO'}")
    print("  (si con 9 no la encuentra y con 1 sí, hay que bajar PRESUPUESTO)")

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
        ents = extraer_entidades(modelo, textos, etiquetas)
        seg = time.perf_counter() - t0
        ritmo = len(textos) / seg if seg else 0
        total = sum(len(e) for e in ents)
        # 118.788 es el total a escanear, medido en `ESTADO.md` §18.
        horas = 118_788 / ritmo / 3600 if ritmo else float("inf")
        print(f"\n  etiquetas {nombre}: {ritmo:5.1f} unidades/s → "
              f"**{horas:4.1f} h** para 118.788")
        print(f"    entidades: {total} ({total/len(textos):.1f} por unidad)")
        print(f"    por tipo : {_resumen_tipos(ents)}")
        if nombre == "EN":
            ents_en = ents

    # ── 5. Umbral ────────────────────────────────────────────────────────────
    print(f"\n{linea}\n5. UMBRAL DE CONFIANZA\n{linea}")
    for u in (0.3, 0.5, 0.7):
        e = extraer_entidades(modelo, textos[:60], ETIQUETAS_EN, umbral=u)
        total = sum(len(x) for x in e)
        print(f"  umbral {u}: {total:4d} entidades en 60 unidades "
              f"({total/60:.1f} por unidad)")
    print("  ⚠️ más entidades no es mejor: un grafo con entidades espurias no")
    print("     cumple el ejemplo de §7.1. Mirar las de umbral bajo a ojo.")

    # ── 6. Los 13 documentos bibliográficos excluidos ────────────────────────
    print(f"\n{linea}\n6. ¿QUÉ RINDEN LOS 13 DOCUMENTOS EXCLUIDOS?\n{linea}")
    print("Cierra con medición la regla que hoy es solo estructural (§18).")
    biblio = muestra_de_unidades(40, contar, solo_biblio=True)
    if biblio:
        e_bib = extraer_entidades(modelo, [u["texto"] for u in biblio],
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
