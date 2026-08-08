"""Barre varios valores de `--factor-idioma` sobre las 50 consultas reales.

    py -m tools.barrer_factor_idioma                    # usa los vectores en cache
    py -m tools.barrer_factor_idioma 0.99 0.97 0.90     # los factores que se quieran
    py -m tools.barrer_factor_idioma --guardar          # codifica y CACHEA (necesita el modelo)

Existe porque `ESTADO.md` §17 decidió el factor por idioma pero **prohíbe
elegirlo a ojo**, y con razón: en §21 los factores 1,02 y 1,05 de la
bonificación por fenómeno, que parecían razonables sobre la aritmética de los
cosenos, resultaron **no hacer absolutamente nada**. Un factor multiplicativo
sobre puntuaciones que se apiñan en torno a 0,60 mueve mucho menos de lo que la
intuición dice.

────────────────────────────────────────────────────────────────────────────
💡 EL TRUCO QUE HACE ESTO BARATO: 200 KB EN VEZ DE 4,35 GB

Aplicar un factor es aritmética sobre los candidatos que ya devolvió FAISS, así
que **lo único que necesita el modelo es codificar las 50 consultas** — y eso se
hace UNA vez. Los 50 vectores son `50 × 1024 float32 = 200 KB`, así que se
guardan en `data/consultas_vectores.npy` y a partir de ahí este barrido, y
cualquier otro experimento de recuperación, corre en una máquina sin el modelo.

    En Colab, una sola vez:   py -m tools.barrer_factor_idioma --guardar
    Traerse                    src/data/consultas_vectores.npy   (200 KB)
    Aquí, todas las veces:     py -m tools.barrer_factor_idioma 0.99 0.97 0.90

⚠️ El cache guarda **modelo y revisión** junto a los vectores. Si no coinciden
con los de `embedding/encoder.py` se rechaza en vez de mezclar vectores de dos
modelos en el mismo espacio, que no daría ningún error y produciría un ranking
sin sentido. Es la misma precaución que `codificar_corpus()` toma con
`embeddings.progreso.json` (`ESTADO.md` §14).

────────────────────────────────────────────────────────────────────────────
⚠️ LO QUE ESTO NO MIDE

**No dice qué factor recupera mejor.** El *ground truth* no es público (§10.1),
así que no hay NDCG@10 ni F1@3 reales que comparar. Lo que mide es:

  - **cuántos fragmentos fuera de {es, en} sobreviven** — el problema concreto
    que el factor existe para corregir (16 de 500 en la salida sin factor);
  - **cuánto se mueve la salida** — si un factor no mueve nada, es cosmético;
  - **si mata la puerta de escape** — un factor tan bajo que deja 0 fragmentos
    fuera de es/en **es un filtro con otro nombre**, y eso es justo lo que
    §17 descartó. Fue el criterio que hundió el ×1,10 de la bonificación (§21).
  - **el efecto colateral sobre el fenómeno**, porque los dos ajustes se
    multiplican sobre la misma lista de candidatos.

El objetivo no es llevar el 3,2% a cero. Es quitar las traducciones que
compiten con un equivalente en es/en **conservando** las que son el único
candidato que responde.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from retrieval.consultas import CONSULTAS, cargar_consultas, fenomeno_de_consulta
from retrieval.search import (BONIFICACION_FENOMENO, IDIOMAS_PREFERIDOS,
                              MAX_POR_DOCUMENTO, Buscador)

RAIZ = Path(__file__).resolve().parents[1]
CACHE = RAIZ / "data" / "consultas_vectores.npy"
CACHE_META = CACHE.with_suffix(".json")

# Valores por defecto del barrido. Se empieza en 1.0 —sin factor— porque es la
# referencia contra la que se comparan los demás, y se baja de forma no lineal:
# lo que importa es encontrar el umbral donde el factor deja de ser cosmético,
# y no se sabe de antemano si está en 0,99 o en 0,80.
FACTORES = (1.0, 0.99, 0.97, 0.95, 0.90, 0.80)


# ── El cache de los 50 vectores ─────────────────────────────────────────────

def guardar_vectores(consultas: List[tuple]) -> np.ndarray:
    """Codifica las 50 consultas y las cachea. **Requiere el modelo.**"""
    from embedding.encoder import MODELO, REVISION, cargar_modelo, codificar

    print(f"cargando el modelo {MODELO} (revisión {REVISION[:8]}…)")
    modelo = cargar_modelo()
    textos = [texto for _, texto in consultas]
    print(f"codificando {len(textos)} consultas…")
    # `codificar()` ya normaliza (§5.2), que es lo que exige `IndexFlatIP`.
    matriz = np.asarray(codificar(modelo, textos), dtype="float32")

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(CACHE, matriz)
    # La procedencia va aparte, en JSON, para poder leerla sin cargar numpy y
    # para que sea evidente al abrirla con un editor.
    CACHE_META.write_text(json.dumps({
        "modelo": MODELO,
        "revision": REVISION,
        "consultas": [qid for qid, _ in consultas],
        "forma": list(matriz.shape),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✔ {CACHE.name}: {matriz.shape} · {CACHE.stat().st_size / 1024:,.0f} KB")
    return matriz


def cargar_vectores(consultas: List[tuple]) -> Optional[np.ndarray]:
    """Lee el cache, o `None` si no existe o no corresponde a este modelo."""
    if not (CACHE.is_file() and CACHE_META.is_file()):
        return None

    from embedding.encoder import MODELO, REVISION

    meta = json.loads(CACHE_META.read_text(encoding="utf-8"))
    # Tres comprobaciones, y las tres han fallado alguna vez en proyectos así:
    # otro modelo, otra revisión del mismo modelo, u otro conjunto de consultas.
    if meta.get("modelo") != MODELO or meta.get("revision") != REVISION:
        print(f"⚠️  el cache es de {meta.get('modelo')} rev "
              f"{str(meta.get('revision'))[:8]}… y el código pide {MODELO} rev "
              f"{REVISION[:8]}…. Se ignora: mezclar dos espacios vectoriales no "
              f"da ningún error y produce un ranking sin sentido.")
        return None
    if meta.get("consultas") != [qid for qid, _ in consultas]:
        print("⚠️  el cache no corresponde a estas consultas. Se ignora.")
        return None

    return np.load(CACHE)


# ── Una corrida completa con un factor dado ─────────────────────────────────

def corrida(buscador: Buscador, consultas: List[tuple], vectores: np.ndarray,
            factor: float, bonificacion: float) -> Dict[str, dict]:
    """Las 50 consultas con un `factor_idioma`, devuelto como `{qid: resultado}`.

    Se reutiliza el mismo `buscador` en todo el barrido: cargar el índice de
    355 MB una vez y no una por factor.
    """
    salida = {}
    for (query_id, texto), vector in zip(consultas, vectores):
        fenomeno = fenomeno_de_consulta(query_id) if bonificacion != 1.0 else None
        resultado = buscador.buscar(
            vector, query_id=query_id, consulta=texto,
            max_por_documento=MAX_POR_DOCUMENTO, fenomeno=fenomeno,
            bonificacion=bonificacion, factor_idioma=factor)
        salida[query_id] = resultado
    return salida


def idiomas_de(resultado) -> List[Optional[str]]:
    """Los 10 idiomas de los fragmentos, en orden de rank."""
    # `idioma` viaja en el candidato desde `Buscador.candidatos()`, así que aquí
    # no hay que volver a abrir el metadata.jsonl.
    return [f.get("idioma") for f in resultado.fragments]


def resumir(nombre: str, resultados: Dict[str, dict],
            base: Optional[Dict[str, dict]] = None) -> dict:
    """Cuenta lo que interesa de una corrida, comparándola con la base."""
    idiomas = Counter()
    fuera_por_consulta = {}
    for qid, r in resultados.items():
        idis = idiomas_de(r)
        idiomas.update(idis)
        fuera = [i for i in idis if i is not None and i not in IDIOMAS_PREFERIDOS]
        if fuera:
            fuera_por_consulta[qid] = fuera

    total_frag = sum(idiomas.values())
    fuera_total = sum(len(v) for v in fuera_por_consulta.values())

    # Correspondencia con el fenómeno de la consulta: efecto colateral, porque
    # los dos factores se multiplican sobre la misma lista de candidatos.
    ok = tot = 0
    for qid, r in resultados.items():
        esperado = fenomeno_de_consulta(qid)
        for doc_id in r.documents:
            tot += 1
            if doc_id.startswith(f"F{esperado}-"):
                ok += 1

    movidos_doc = movidos_frag = 0
    if base is not None:
        for qid in resultados:
            if resultados[qid].documents != base[qid].documents:
                movidos_doc += 1
            if ([f["chunk_id"] for f in resultados[qid].fragments]
                    != [f["chunk_id"] for f in base[qid].fragments]):
                movidos_frag += 1

    return {
        "nombre": nombre,
        "fuera": fuera_total,
        "pct_fuera": 100 * fuera_total / total_frag if total_frag else 0.0,
        "consultas_afectadas": len(fuera_por_consulta),
        "detalle": fuera_por_consulta,
        "corresp": 100 * ok / tot if tot else 0.0,
        "movidos_doc": movidos_doc,
        "movidos_frag": movidos_frag,
        "idiomas": idiomas,
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Barre valores de --factor-idioma sobre las 50 consultas")
    parser.add_argument("factores", type=float, nargs="*", default=list(FACTORES),
                        help=f"factores a probar (por defecto {FACTORES})")
    parser.add_argument("--guardar", action="store_true",
                        help="codifica las consultas y cachea los vectores "
                             "(requiere el modelo, 4,35 GB)")
    parser.add_argument("--bonificacion", type=float, default=BONIFICACION_FENOMENO,
                        help="factor por fenómeno que se mantiene fijo durante "
                             "el barrido")
    parser.add_argument("--consultas", type=Path, default=CONSULTAS)
    args = parser.parse_args()

    consultas = cargar_consultas(args.consultas)
    print(f"consultas : {len(consultas)} de {args.consultas.name}")

    vectores = None if args.guardar else cargar_vectores(consultas)
    if vectores is None:
        if not args.guardar:
            print(f"\nNo hay vectores en cache ({CACHE.name}).")
            print("Córrelo una vez con --guardar en una máquina con el modelo:")
            print("    py -m tools.barrer_factor_idioma --guardar")
            print("y tráete src/data/consultas_vectores.npy (200 KB). Después "
                  "este barrido corre sin el modelo.")
            return 1
        vectores = guardar_vectores(consultas)
    else:
        print(f"vectores  : {CACHE.name} en cache, {vectores.shape} "
              f"— no hace falta el modelo")

    if len(vectores) != len(consultas):
        print(f"⚠️  {len(vectores)} vectores para {len(consultas)} consultas.")
        return 1

    print("\ncargando índice y metadata…")
    buscador = Buscador()
    print(f"listo ({buscador.indice.ntotal:,} vectores)")
    if args.bonificacion != 1.0:
        print(f"bonificación por fenómeno fija en ×{args.bonificacion} "
              f"durante todo el barrido")

    # La primera corrida del barrido es la referencia. Si el usuario no incluyó
    # 1.0 se añade delante, porque sin base no hay con qué comparar «cuánto se
    # mueve».
    factores = list(args.factores)
    if 1.0 not in factores:
        factores.insert(0, 1.0)

    linea = "─" * 78
    base = None
    filas = []
    for factor in factores:
        print(f"\n{linea}\nfactor_idioma = {factor}")
        resultados = corrida(buscador, consultas, vectores, factor,
                             args.bonificacion)
        if base is None:
            base = resultados
        fila = resumir(f"×{factor}", resultados, base)
        filas.append(fila)
        print(f"  fragmentos fuera de es/en : {fila['fuera']:3d} de 500 "
              f"({fila['pct_fuera']:.1f}%) en {fila['consultas_afectadas']} consultas")
        print(f"  correspondencia fenómeno  : {fila['corresp']:.1f}%")
        print(f"  cambian vs ×1.0           : {fila['movidos_doc']} listas de "
              f"documentos · {fila['movidos_frag']} de fragmentos")
        if fila["detalle"]:
            for qid, fuera in sorted(fila["detalle"].items()):
                print(f"     {qid}: {'/'.join(fuera)}")

    # ── La tabla que se copia al informe / a ESTADO.md ──────────────────────
    print(f"\n{linea}\nRESUMEN\n{linea}")
    print(f"{'factor':>8} {'fuera es/en':>12} {'consultas':>10} "
          f"{'fenómeno':>9} {'docs≠':>6} {'frags≠':>7}")
    for f in filas:
        print(f"{f['nombre']:>8} {f['fuera']:>7d}/500 {f['consultas_afectadas']:>10d} "
              f"{f['corresp']:>8.1f}% {f['movidos_doc']:>6d} {f['movidos_frag']:>7d}")

    print(f"\n{linea}")
    print("CÓMO LEER ESTO (§17):")
    print("  · Un factor que deja 'docs≠' y 'frags≠' en 0 es COSMÉTICO: no hace")
    print("    nada y no vale la pena asumir el riesgo de §1.4 por él.")
    print("  · Un factor que deja 'fuera es/en' en 0 es un FILTRO con otro")
    print("    nombre, y §17 descartó el filtro porque su fallo es irreversible.")
    print("  · El objetivo está en medio: quitar las traducciones que compiten")
    print("    con un equivalente en es/en y CONSERVAR las que son el único")
    print("    candidato. Mirar q018/q022/q024 (los execsum de SWF) a mano.")
    print("  · Esto NO mide calidad de recuperación: el ground truth no es")
    print("    público (§10.1). Es una decisión de riesgo con números, no una")
    print("    optimización.")

    buscador.cerrar()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
