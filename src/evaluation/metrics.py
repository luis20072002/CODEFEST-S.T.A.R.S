"""Métricas de evaluación de §10: NDCG@10 y F1@3.

    py -m evaluation.metrics            # autoprueba contra valores calculados a mano

────────────────────────────────────────────────────────────────────────────────
LAS DOS MÉTRICAS, TAL COMO LAS DEFINE EL PDF

**NDCG@10 (§10.2.1), a nivel fragmento.** Con el vector de relevancia
`r = (r₁, …, r₁₀)`, donde `rᵢ ≥ 0` es la relevancia del fragmento en la
posición `i` de la lista entregada:

        DCG@k  = Σᵢ  rᵢ / log₂(i + 1)
        NDCG@k = DCG@k / IDCG@k

⚠️ **Ganancia LINEAL, no exponencial.** La fórmula (8) del PDF es `rᵢ` en el
numerador, no `2^rᵢ − 1`. Muchas implementaciones (incluida la de
`sklearn.metrics.ndcg_score` con `gains="exponential"`) usan la exponencial y
dan otro número. Con relevancias binarias coinciden; con relevancia graduada,
no. El PDF dice «en qué grado», así que la graduada es el caso real.

**F1@3 (§10.2.2), a nivel documento.** Métrica de **conjunto**: no considera el
orden.

        P@3  = |D̂ ∩ D*| / 3
        R@3  = |D̂ ∩ D*| / mín(|D*|, 3)
        F1@3 = 2·P@3·R@3 / (P@3 + R@3)

El denominador de R@3 se limita a `mín(|D*|, 3)` «para no penalizar casos en
los que el número de documentos relevantes es inferior a 3».

Las dos se promedian sobre las 50 consultas (fórmulas 10 y 14).

────────────────────────────────────────────────────────────────────────────────
🔴 LA CLAVE DE EMPAREJAMIENTO NO SON NUESTROS IDENTIFICADORES

§10.2.1 lo dice con todas las letras, y es lo más fácil de olvidar al montar
una evaluación propia:

> «La relevancia de cada fragmento se juzga sobre su contenido textual (campo
> `text`). El `chunk_id` **no** es la clave de emparejamiento con el ground
> truth… Análogamente, a nivel de documento el emparejamiento se realiza a
> través del campo **`fuente`** (archivo original provisto por ADL), no del
> `doc_id` arbitrario asignado por el equipo.»

Por eso `f1_at_3()` recibe **fuentes**, no `doc_id`. Convertir de uno a otro es
justo para lo que existe el `metadata.jsonl` (§5.3), y es la razón por la que
todo `doc_id` de la salida tiene que estar ahí.

────────────────────────────────────────────────────────────────────────────────
⚠️ ESTE MÓDULO NO PUEDE DECIRTE CÓMO VAS A PUNTUAR

§10.1: el *ground truth* «no es pública durante el reto». Sin ella estas
funciones no dan la nota real. Sirven para dos cosas legítimas:

  1. **Comparar dos configuraciones nuestras** contra un conjunto de relevancia
     etiquetado a mano, que es lo que hace `tools/comparar_configuraciones.py`.
  2. Tener la implementación lista y **verificada** para el día que haya
     etiquetas, en vez de escribirla con prisas.
"""

import math
import sys
from typing import Dict, Iterable, Sequence, Set


def dcg_at_k(relevancias: Sequence[float], k: int = 10) -> float:
    """DCG con ganancia lineal, fórmula (8) del PDF.

    `log₂(i + 1)` con `i` empezando en 1, así que la primera posición divide
    entre `log₂(2) = 1`: el primer puesto no se descuenta.
    """
    return sum(r / math.log2(i + 1)
               for i, r in enumerate(relevancias[:k], start=1))


def ndcg_at_k(relevancias: Sequence[float], ideales: Sequence[float] | None = None,
              k: int = 10) -> float:
    """NDCG@k, fórmula (9).

    `ideales` son **todas** las relevancias disponibles para esa consulta, de
    donde sale el ranking ideal. Si se omite, se asume que el ideal es reordenar
    lo entregado — que es lo correcto solo cuando la lista entregada contiene
    ya todos los fragmentos relevantes, y en general **no lo es**: si el equipo
    dejó fuera un fragmento muy relevante, el IDCG debe contarlo igual. Por eso
    conviene pasarlo siempre.

    Devuelve 0.0 si el ideal es 0, en vez de dividir entre cero: una consulta
    sin nada relevante no puntúa, no rompe la media.
    """
    referencia = list(ideales) if ideales is not None else list(relevancias)
    idcg = dcg_at_k(sorted(referencia, reverse=True), k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(relevancias, k) / idcg


def f1_at_3(devueltos: Iterable[str], relevantes: Iterable[str],
            n: int = 3) -> Dict[str, float]:
    """P@3, R@3 y F1@3 de §10.2.2. Devuelve las tres.

    ⚠️ `devueltos` y `relevantes` son **fuentes**, no `doc_id` (§10.2.1).

    Se usan conjuntos porque es una métrica de conjunto: si la salida repitiera
    un documento, contaría una sola vez — que es lo que hace el jurado.

    `P@3` divide entre `n` fijo (3) y no entre los devueltos: el PDF escribe
    `|D̂| = 3` porque §9.2 exige exactamente tres. Si por un error la salida
    trajera menos, dividir entre los realmente devueltos maquillaría el fallo.
    """
    devueltos_set: Set[str] = set(devueltos)
    relevantes_set: Set[str] = set(relevantes)
    aciertos = len(devueltos_set & relevantes_set)

    precision = aciertos / n
    denominador_r = min(len(relevantes_set), n)
    recall = aciertos / denominador_r if denominador_r else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"P@3": precision, "R@3": recall, "F1@3": f1}


def promedio(valores: Iterable[float]) -> float:
    """Media sobre las consultas, fórmulas (10) y (14)."""
    valores = list(valores)
    return sum(valores) / len(valores) if valores else 0.0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    linea = "─" * 74
    fallos = 0

    def comprobar(nombre: str, obtenido: float, esperado: float,
                  tolerancia: float = 1e-4) -> None:
        global fallos
        ok = abs(obtenido - esperado) <= tolerancia
        fallos += 0 if ok else 1
        print(f"  {'✔' if ok else '✖'} {nombre:<52} {obtenido:.4f} "
              f"(esperado {esperado:.4f})")

    print(f"{linea}\nNDCG — contra valores calculados a mano")

    # Caso clásico de libro, verificable a mano:
    #   DCG  = 3/1 + 2/1.58496 + 3/2 + 0 + 1/2.58496 + 2/2.80735 = 6.8611
    #   IDCG = 3/1 + 3/1.58496 + 2/2 + 2/2.32193 + 1/2.58496 + 0  = 7.1410
    #   NDCG = 0.9608
    r = [3, 2, 3, 0, 1, 2]
    comprobar("DCG@6 de [3,2,3,0,1,2]", dcg_at_k(r, 6), 6.8611)
    comprobar("NDCG@6 de [3,2,3,0,1,2]", ndcg_at_k(r, r, 6), 0.9608)

    comprobar("ranking perfecto → 1.0", ndcg_at_k([3, 2, 1], [3, 2, 1], 3), 1.0)
    comprobar("todo irrelevante → 0.0", ndcg_at_k([0, 0, 0], [0, 0, 0], 3), 0.0)

    # El ideal se toma de FUERA de lo entregado: hay un fragmento de
    # relevancia 3 que el equipo no devolvió, y el NDCG tiene que bajar.
    comprobar("ideal externo penaliza lo no devuelto",
              ndcg_at_k([1, 1], [3, 1, 1], 2), (1 / 1 + 1 / 1.58496) / (3 / 1 + 1 / 1.58496))

    # Ganancia lineal, no exponencial: con [2,0] la lineal da 1.0 (ya está
    # ordenado); si alguien metiera 2^r-1 el número cambiaría en otros casos.
    comprobar("primera posición sin descuento", dcg_at_k([5], 1), 5.0)

    print(f"\n{linea}\nF1@3 — contra valores calculados a mano")

    # 2 aciertos de 3 devueltos, con 5 relevantes:
    #   P = 2/3, R = 2/mín(5,3) = 2/3, F1 = 2/3
    r1 = f1_at_3(["a", "b", "c"], ["a", "b", "x", "y", "z"])
    comprobar("2 aciertos, 5 relevantes → F1 2/3", r1["F1@3"], 2 / 3)

    # 1 solo documento relevante y lo acertamos:
    #   P = 1/3, R = 1/mín(1,3) = 1, F1 = 2·(1/3)·1 / (1/3+1) = 0.5
    r2 = f1_at_3(["a", "b", "c"], ["a"])
    comprobar("1 relevante y acertado → F1 0.5", r2["F1@3"], 0.5)
    comprobar("  …su R@3 no se penaliza por |D*|<3", r2["R@3"], 1.0)

    r3 = f1_at_3(["a", "b", "c"], ["x", "y", "z"])
    comprobar("ningún acierto → F1 0.0", r3["F1@3"], 0.0)

    r4 = f1_at_3(["a", "b", "c"], ["a", "b", "c"])
    comprobar("los 3 acertados → F1 1.0", r4["F1@3"], 1.0)

    # Duplicados: la métrica es de conjunto, así que repetir no suma.
    r5 = f1_at_3(["a", "a", "b"], ["a", "b", "c"])
    comprobar("documento repetido cuenta una vez", r5["F1@3"], 2 / 3)

    print(f"\n{linea}")
    print("VEREDICTO: ✔ TODAS LAS COMPROBACIONES PASAN" if not fallos
          else f"VEREDICTO: ✖ {fallos} COMPROBACIONES FALLAN")
    print(linea)
    raise SystemExit(1 if fallos else 0)
