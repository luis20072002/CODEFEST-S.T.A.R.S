"""Mide qué le hace a la recuperación activar el grafo de §8.5.

    py -m tools.medir_grafo [--bonificaciones 0.03,0.05,0.10] [--escribir]

────────────────────────────────────────────────────────────────────────────────
POR QUÉ EXISTE

El comité aclaró por escrito que el bonus del grafo «para que sea válido lo deben
integrar a la recuperación, el solo construirlo no es válido». Eso obliga a
activar §8.5, y activarlo cambia `resultados.jsonl`, que es el archivo cuya
reproducibilidad decide §1.4. Antes de tocar la entrega hay que saber **qué
cambia y cuánto**, que es justo lo que `ESTADO.md` §23 declaraba sin medir.

────────────────────────────────────────────────────────────────────────────────
CÓMO SE MIDE, Y POR QUÉ ASÍ

Se comparan corridas que solo difieren en `bonificacion_grafo`. Todo lo demás
—índice, metadata, vectores de consulta, bonificación por fenómeno, factor de
idioma— es idéntico, así que **toda diferencia observada es del grafo**.

⚠️ Se usan los **vectores de consulta precalculados** (`consultas_vectores.npy`),
no el modelo. `ESTADO.md` §22 midió que esa vía difiere de la del modelo en 4 de
las 50 consultas por casi-empates de float32, así que **esta herramienta no
produce el entregable**. Para un A/B sí es la vía correcta y no un atajo: los dos
lados de la comparación arrastran exactamente el mismo sesgo, de modo que se
cancela en la diferencia. Lo que aquí se mide es el efecto del grafo, no el valor
absoluto de la salida.

A nivel de documento se compara el **conjunto**, no el orden: §10.2.2 dice que
F1@3 «es una métrica de conjunto (no considera el orden)», así que una
reordenación de los mismos tres documentos vale cero y contarla como cambio
exageraría el efecto.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from retrieval.consultas import cargar_consultas, fenomeno_de_consulta
from retrieval.grafo import GRAFO, Grafo
from retrieval.search import Buscador, Resultado

DATOS = Path(__file__).resolve().parents[1] / "data"
VECTORES = DATOS / "consultas_vectores.npy"
ORDEN = DATOS / "consultas_vectores.json"
CHUNK_INDEX = DATOS / "chunk_index.json"

BONIFICACIONES = (0.03, 0.05, 0.10)


def cargar_vectores() -> Tuple[List[str], "object"]:
    """Los 50 vectores de consulta ya codificados, con su orden de identificadores."""
    import numpy as np

    orden = json.loads(ORDEN.read_text(encoding="utf-8"))["consultas"]
    return orden, np.load(VECTORES)


def corrida(buscador: Buscador, orden: List[str], vectores,
            textos: Dict[str, str], bonificacion: float) -> Dict[str, Resultado]:
    """Las 50 consultas con un valor concreto de `bonificacion_grafo`."""
    salida: Dict[str, Resultado] = {}
    for i, query_id in enumerate(orden):
        salida[query_id] = buscador.buscar(
            vectores[i],
            query_id=query_id,
            consulta=textos.get(query_id),
            fenomeno=fenomeno_de_consulta(query_id),
            bonificacion_grafo=bonificacion,
        )
    return salida


def diagnostico_enlazado(buscador: Buscador, orden: List[str],
                         textos: Dict[str, str]) -> List[Tuple[str, int, int]]:
    """Cuántas entidades enlaza cada consulta y cuántos fragmentos aporta el grafo.

    Es la comprobación de que el grafo **dispara**. Sin ella, una bonificación
    que no cambia nada es indistinguible de una integración rota: `evidencia()`
    devuelve `{}` en silencio si falta `chunk_index.json`, que es precisamente el
    fallo mudo que `ESTADO.md` §23 describe.
    """
    filas = []
    for query_id in orden:
        entidades = buscador.entidades_de_consulta(query_id, textos.get(query_id))
        semillas = buscador.grafo.enlazar(entidades)
        evidencia = buscador.grafo.evidencia(semillas)
        filas.append((query_id, len(semillas), len(evidencia)))
    return filas


def _docs(r: Resultado) -> frozenset:
    return frozenset(r.documents)


def _frags(r: Resultado) -> tuple:
    return tuple(f["chunk_id"] for f in r.fragments)


def comparar(base: Dict[str, Resultado],
             otra: Dict[str, Resultado]) -> Tuple[List[str], List[str]]:
    """Consultas cuyo CONJUNTO de documentos cambia, y cuyos fragmentos cambian."""
    docs = [q for q in base if _docs(base[q]) != _docs(otra[q])]
    frags = [q for q in base if _frags(base[q]) != _frags(otra[q])]
    return docs, frags


def escribir(resultados: Dict[str, Resultado], ruta: Path) -> None:
    """Serializa una corrida como `resultados.jsonl` para poder puntuarla."""
    with open(ruta, "w", encoding="utf-8", newline="\n") as f:
        for query_id in sorted(resultados):
            f.write(json.dumps(resultados[query_id].to_json(),
                               ensure_ascii=False) + "\n")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    bonificaciones = list(BONIFICACIONES)
    if "--bonificaciones" in sys.argv:
        crudo = sys.argv[sys.argv.index("--bonificaciones") + 1]
        bonificaciones = [float(x) for x in crudo.split(",")]

    for ruta in (VECTORES, ORDEN, CHUNK_INDEX, GRAFO):
        if not ruta.is_file():
            print(f"Falta {ruta}.")
            if ruta == CHUNK_INDEX:
                print("Constrúyelo con `py -m tools.construir_chunk_index "
                      "--salida data/chunk_index.json`.")
            return 1

    linea = "─" * 78
    print(linea)
    print("cargando grafo, índice y metadata…")
    grafo = Grafo(GRAFO, chunk_index=CHUNK_INDEX)
    buscador = Buscador(grafo=grafo)
    orden, vectores = cargar_vectores()
    textos = dict(cargar_consultas())
    print(f"grafo {len(grafo):,} nodos · {len(grafo.faiss_id):,} fragmentos "
          f"resolubles · {len(orden)} consultas")

    # ── ¿dispara siquiera? ───────────────────────────────────────────────
    print(f"\n{linea}\nA. ENLAZADO — ¿el grafo encuentra algo en cada consulta?")
    filas = diagnostico_enlazado(buscador, orden, textos)
    sin_entidad = [q for q, s, _ in filas if s == 0]
    sin_evidencia = [q for q, _, e in filas if e == 0]
    semillas_tot = sum(s for _, s, _ in filas)
    print(f"   entidades enlazadas   : {semillas_tot} en total, "
          f"media {semillas_tot / len(filas):.2f} por consulta")
    print(f"   consultas sin entidad : {len(sin_entidad)}  {sin_entidad[:8]}")
    print(f"   consultas sin evidencia: {len(sin_evidencia)}")
    if len(sin_evidencia) == len(filas):
        print("   🔴 NINGUNA consulta recibe evidencia: la integración está rota,")
        print("      no es que el grafo no aporte. Revisa `chunk_index.json`.")
        return 1
    con = [(q, s, e) for q, s, e in filas if e]
    print(f"   consultas con evidencia: {len(con)}")
    for q, s, e in con[:10]:
        print(f"      {q}  {s:>2} entidades → {e:>3} fragmentos")

    # ── ¿cambia la salida? ───────────────────────────────────────────────
    print(f"\n{linea}\nB. EFECTO — qué cambia frente a la salida sin grafo")
    base = corrida(buscador, orden, vectores, textos, 0.0)
    corridas = {"sin_grafo": base}

    print(f"   {'bonificación':<14}{'docs cambian':>14}{'frags cambian':>16}   consultas")
    for b in bonificaciones:
        otra = corrida(buscador, orden, vectores, textos, b)
        corridas[f"grafo_{b:g}".replace(".", "")] = otra
        docs, frags = comparar(base, otra)
        print(f"   {b:<14g}{len(docs):>14}{len(frags):>16}   "
              f"{','.join(sorted(set(docs) | set(frags))[:6])}")

    if "--escribir" in sys.argv:
        print(f"\n{linea}\nC. ARCHIVOS para `tools.comparar_configuraciones`")
        for nombre, r in corridas.items():
            ruta = DATOS / f"resultados_{nombre}.jsonl"
            escribir(r, ruta)
            print(f"   {ruta}  ({ruta.stat().st_size:,} B)")

    buscador.cerrar()
    print(f"\n{linea}")
    print("⚠️  Vectores precalculados: sirve para el A/B, NO produce el entregable.")
    print(linea)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
