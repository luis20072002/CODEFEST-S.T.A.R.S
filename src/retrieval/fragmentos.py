"""División de fragmentos al límite de 250 palabras (§9.2.1).

    py -m retrieval.fragmentos          # mide la división sobre todo chunks.jsonl

────────────────────────────────────────────────────────────────────────────────
POR QUÉ ESTE MÓDULO EXISTE, CON EL NÚMERO QUE LO JUSTIFICA

§9.2 exige que cada uno de los 10 fragmentos reportados tenga «como máximo 250
palabras», y §9.3.1 avisa de que los que se pasen «serán penalizados o
descartados durante la evaluación automática».

Medido sobre los 91.021 chunks del índice:

    mediana        : 324 palabras
    percentil 10   : 256          ← incluso el decil más corto se pasa
    superan 250    : 82.444  (90,6%)

**Nueve de cada diez fragmentos no se pueden reportar tal como están.** No es un
caso excepcional que se resuelva con un `if`: es el caso normal.

⚠️ **El límite NO se acata en el índice, se acata al generar la salida.** Los
chunks grandes están bien donde están —dan vectores mejores, con más contexto— y
solo se parten al construir `resultados.jsonl`. Por eso este módulo vive en
`retrieval/` y no toca ni el chunker ni la base vectorial.

────────────────────────────────────────────────────────────────────────────────
LAS REGLAS DE §9.2.1, TEXTUALES

  - «Si un chunk recuperado tiene más de 250 palabras, debe dividirse en
    sub-fragmentos que respeten el límite. La división debe respetar el
    requisito de completitud lingüística (sin oraciones cortadas).»
  - «Cuando un mismo chunk se divide en varios sub-fragmentos, todos ellos
    comparten el mismo chunk_id; esto es aceptable, ya que la relevancia se
    evalúa sobre el campo text y el chunk_id cumple aquí una función de
    trazabilidad, no de emparejamiento.»
  - «Cada sub-fragmento debe ocupar su propia posición (rank) en la lista de 10
    fragmentos.»

De la última se sigue algo que conviene tener claro: **§9.2.1 no obliga a
reportar todos los sub-fragmentos de un chunk.** Lo que prohíbe es meter dos en
un mismo rank. Eso deja abierta la decisión de `elegir_mejor()`, más abajo.

────────────────────────────────────────────────────────────────────────────────
POR QUÉ SE REUTILIZA EL CORTADOR DE ORACIONES DEL CHUNKER

§9.2.1 remite a la misma completitud lingüística de §3.3, así que la lógica es
la misma que ya está escrita, probada sobre los 1826 documentos y verificada por
`tools/verificar_chunker.py`. Tener una segunda implementación aquí sería
garantizar que un día divergen.

`ESTADO.md` §8 prohibía que `generador.py` importara el chunker; esa regla se
corrigió el 2026-08-06 y ahora prohíbe solo los **loaders y el orquestador**, que
son los que arrastran `pdfplumber` y `pytesseract` a la máquina del jurado.
`chunking/chunker.py` no depende de nada fuera de la librería estándar.
"""

import re
import sys
import unicodedata
from typing import List, Optional, Sequence

from chunking.chunker import SECUNDARIOS, dividir_en_oraciones

# §9.2 y §9.3.1. No es negociable ni aproximable: un fragmento de 251 palabras
# puede costar la entrada entera.
LIMITE_PALABRAS = 250

# Palabras que no discriminan nada al elegir el mejor sub-fragmento. No pretende
# ser exhaustiva: solo evita que «de», «la» o «the» dominen el solapamiento.
VACIAS = frozenset("""
el la los las un una unos unas de del al a en con por para sin sobre entre
que qué cual cuál cuales como cómo cuando cuándo donde dónde quien quién
es son ser sea fue han ha hay su sus lo se no ni o u y e
the a an of in on at to for with by from and or as is are was were be been
this that these those it its
""".split())


def contar_palabras(texto: str) -> int:
    """Palabras según §9.2.1: separadas por espacios en blanco.

    Es la misma cuenta que `Chunk.word_count`. Se repite aquí porque este módulo
    trabaja con cadenas sueltas (los sub-fragmentos), no con objetos `Chunk`.
    """
    return len(texto.split())


def dividir_a_limite(texto: str, limite: int = LIMITE_PALABRAS) -> List[str]:
    """Parte un texto en trozos de `limite` palabras o menos, sin cortar oraciones.

    Devuelve la lista completa de sub-fragmentos, en orden, cubriendo todo el
    texto. Si ya cabe, devuelve `[texto]` sin tocarlo.

    Es la misma cascada del chunker, pero medida en palabras y en dos niveles:

      1. agrupar **oraciones completas** hasta el límite  ← §9.2.1 literal
      2. si una sola oración se pasa, separadores secundarios y, en último
         extremo, corte por longitud

    El nivel 2 existe porque el corpus tiene unidades que **no son oraciones**
    aunque lo parezcan: tablas presupuestales extraídas en línea recta e índices
    con puntos guía. Ahí no hay oración que respetar. Está medido en
    `ESTADO.md` §13: son ~893 casos en 157 documentos, el 3% de la prosa.
    """
    if contar_palabras(texto) <= limite:
        return [texto]

    partes: List[str] = []
    actual: List[str] = []
    cuenta = 0

    def cerrar() -> None:
        nonlocal actual, cuenta
        if actual:
            partes.append(" ".join(actual))
            actual, cuenta = [], 0

    for oracion in dividir_en_oraciones(texto):
        n = contar_palabras(oracion)

        if n > limite:
            # Una «oración» que por sí sola no cabe. Se cierra lo acumulado y se
            # trocea aparte, para no arrastrar el problema al grupo siguiente.
            cerrar()
            partes.extend(_partir_unidad_larga(oracion, limite))
            continue

        if cuenta + n > limite:
            cerrar()
        actual.append(oracion)
        cuenta += n

    cerrar()
    return partes or [texto]


def _partir_unidad_larga(texto: str, limite: int) -> List[str]:
    """Trocea algo que no cabe y que el cortador de oraciones no pudo partir.

    Primero prueba los separadores secundarios del chunker (puntos guía, `;`,
    `:`, salto de línea). Si ninguno reduce el tamaño, corta por longitud entre
    palabras — nunca dentro de una, porque partir una palabra la vuelve
    inencontrable, que es el mismo daño que causó el bug del `\\x07`
    (`ESTADO.md` §11).
    """
    for patron in SECUNDARIOS:
        trozos = [t.strip() for t in patron.split(texto) if t.strip()]
        if len(trozos) > 1:
            salida: List[str] = []
            actual: List[str] = []
            cuenta = 0
            for trozo in trozos:
                n = contar_palabras(trozo)
                if actual and cuenta + n > limite:
                    salida.append(" ".join(actual))
                    actual, cuenta = [], 0
                if n > limite:
                    if actual:
                        salida.append(" ".join(actual))
                        actual, cuenta = [], 0
                    salida.extend(_cortar_por_longitud(trozo, limite))
                    continue
                actual.append(trozo)
                cuenta += n
            if actual:
                salida.append(" ".join(actual))
            return salida
    return _cortar_por_longitud(texto, limite)


def _cortar_por_longitud(texto: str, limite: int) -> List[str]:
    """Último recurso: trocear por número de palabras."""
    palabras = texto.split()
    return [" ".join(palabras[i:i + limite]) for i in range(0, len(palabras), limite)] \
        or [texto]


# ══════════════════════════════════════════════════════════════════════════════
# Elegir qué sub-fragmento se reporta
# ══════════════════════════════════════════════════════════════════════════════

def _normalizar(texto: str) -> str:
    """Minúsculas y sin tildes, para que «órbita» y «orbita» cuenten igual."""
    descompuesto = unicodedata.normalize("NFD", texto.lower())
    return "".join(c for c in descompuesto if unicodedata.category(c) != "Mn")


def _terminos(texto: str) -> set:
    return {p for p in re.findall(r"[^\W\d_]{3,}", _normalizar(texto))
            if p not in VACIAS}


def elegir_mejor(subfragmentos: Sequence[str], consulta: Optional[str] = None) -> str:
    """Devuelve el sub-fragmento más pertinente para la consulta.

    ────────────────────────────────────────────────────────────────────────
    POR QUÉ SE ELIGE UNO Y NO SE REPORTAN TODOS

    §9.2.1 permite las dos cosas: lo único que prohíbe es meter dos
    sub-fragmentos en un mismo rank. Con el 90,6% de los chunks pasados de 250
    palabras, la diferencia es grande:

      - reportando todos    → los 10 ranks salen de ~5 chunks distintos, y la
        mitad de un chunk que no contiene la parte relevante entra igual y
        puntúa 0. NDCG@10 penaliza tener irrelevantes arriba.
      - reportando el mejor → los 10 ranks salen de 10 chunks distintos, cada
        uno con su porción pertinente.

    ────────────────────────────────────────────────────────────────────────
    POR QUÉ SE ELIGE POR SOLAPAMIENTO LÉXICO Y NO POR COSENO

    Lo «natural» sería codificar cada sub-fragmento y quedarse con el de mayor
    similitud. Se descartó por coste: `generador.py` lo ejecuta el jurado
    (§1.4), y con BGE-M3 en CPU cada codificación son ~9 s. Cincuenta consultas
    × diez chunks × dos mitades serían horas en una máquina sin GPU, y una
    entrega que no se puede reproducir queda excluida.

    El solapamiento léxico es determinista, cuesta microsegundos y **no es un
    modelo**, así que no roza la prohibición de §8.3 —que habla de modelos
    generativos, no de contar palabras en común—.

    Sin consulta, devuelve el primero: es el comienzo del chunk y por tanto el
    trozo con más contexto propio.
    """
    if not subfragmentos:
        return ""
    if len(subfragmentos) == 1 or not consulta:
        return subfragmentos[0]

    terminos = _terminos(consulta)
    if not terminos:
        return subfragmentos[0]

    def puntuar(trozo: str) -> tuple:
        comunes = terminos & _terminos(trozo)
        # Desempate por el orden original (índice negativo) para que sea
        # determinista: §1.4 exige que dos corridas den lo mismo.
        return (len(comunes), -subfragmentos.index(trozo))

    return max(subfragmentos, key=puntuar)


# ══════════════════════════════════════════════════════════════════════════════
# Deduplicación
# ══════════════════════════════════════════════════════════════════════════════

def clave_de_texto(texto: str) -> str:
    """Clave para decidir si dos fragmentos dicen lo mismo.

    Normaliza **solo el espacio en blanco**, y nada más. No baja a minúsculas ni
    quita puntuación a propósito: dos textos que difieren en mayúsculas
    difieren de verdad, y agruparlos sería descartar contenido distinto.

    El espacio sí se normaliza porque está **medido**: de los tres pares de
    chunks del corpus que producen exactamente el mismo vector, los tres tienen
    textos idénticos salvo por uno a cinco caracteres de espaciado
    (`F2-CSIS-053#0020` y `#0065`, entre otros). El tokenizador del encoder ya
    los trata como iguales; el deduplicador debe hacer lo mismo.
    """
    return " ".join(texto.split())


def deduplicar_por_texto(items: Sequence, texto_de=lambda x: x,
                         limite: Optional[int] = None) -> List:
    """Quita los items cuyo texto ya apareció antes. Conserva el orden.

    ────────────────────────────────────────────────────────────────────────
    POR QUÉ HACE FALTA, CON LA MEDICIÓN

    El índice tiene **91.021 chunks pero solo 87.427 textos únicos**: 3.594
    (el 3,9%) repiten contenido. No es un fallo del pipeline, es el corpus, y
    son dos patrones distintos:

      - filas tabulares idénticas repartidas entre varios documentos
        (`F3-AMAZONUW-004#0007` aparece en 11 documentos), y
      - secciones repetidas dentro de un mismo informe
        (`F2-CSIS-053#0020` y `#0065` son el mismo párrafo).

    Si dos de esos entran en el top-10, ocupan **dos de los diez ranks con el
    mismo contenido**. §10.2.1 evalúa la relevancia sobre el campo `text`, así
    que el segundo no aporta nada y desplaza a un fragmento que sí podría
    sumar. Deduplicar es puro beneficio.

    ⚠️ **Conviene aplicarlo sobre el texto FINAL que se va a reportar**, ya
    partido a 250 palabras, no sobre el chunk entero: es ese texto el que
    evalúa el jurado. Y como quita candidatos, `search.py` tiene que pedir a
    FAISS **más de 10** para que después de deduplicar sigan quedando 10.

    No usa modelos de ningún tipo: es comparación de cadenas, así que no roza
    §8.3.
    """
    vistos = set()
    salida: List = []
    for item in items:
        clave = clave_de_texto(texto_de(item))
        if clave in vistos:
            continue
        vistos.add(clave)
        salida.append(item)
        if limite is not None and len(salida) >= limite:
            break
    return salida


def preparar_fragmento(texto: str, consulta: Optional[str] = None,
                       limite: int = LIMITE_PALABRAS) -> str:
    """Lo que `generador.py` va a llamar: texto listo para reportar.

    Garantiza que la salida tiene `limite` palabras o menos. El `chunk_id` que
    se reporte junto a esto debe seguir siendo el del chunk **original** del
    índice (§9.2.1).
    """
    return elegir_mejor(dividir_a_limite(texto, limite), consulta)


if __name__ == "__main__":
    # Diagnóstico sobre el índice real: comprueba que la división cumple el
    # límite en los 91.021 y que no pierde texto.
    import json
    import time
    from collections import Counter
    from pathlib import Path

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    CHUNKS = Path(__file__).resolve().parents[1] / "data" / "chunks.jsonl"
    if not CHUNKS.is_file():
        print(f"No existe {CHUNKS}.")
        raise SystemExit(1)

    inicio = time.perf_counter()
    total = 0
    pasados = 0            # sub-fragmentos que incumplen el límite: debe ser 0
    perdidos = 0           # chunks cuya división no reconstruye el texto
    reparto = Counter()    # en cuántos trozos se parte cada chunk
    tamanos: list = []
    claves = Counter()     # para medir el duplicado de contenido

    with open(CHUNKS, encoding="utf-8") as f:
        for linea in f:
            if not linea.strip():
                continue
            texto = json.loads(linea)["text"]
            total += 1
            claves[clave_de_texto(texto)] += 1
            trozos = dividir_a_limite(texto)
            reparto[len(trozos)] += 1
            for trozo in trozos:
                n = contar_palabras(trozo)
                tamanos.append(n)
                if n > LIMITE_PALABRAS:
                    pasados += 1
            # Cobertura: la unión de los trozos, sin espacios, tiene que ser el
            # texto original sin espacios. Mismo criterio que la prueba B del
            # verificador del chunker.
            if "".join("".join(t.split()) for t in trozos) != "".join(texto.split()):
                perdidos += 1

    tamanos.sort()
    segundos = time.perf_counter() - inicio
    linea = "─" * 74

    print(f"{linea}\nchunks procesados : {total:,}   en {segundos:,.1f} s")
    print(f"sub-fragmentos    : {len(tamanos):,}")
    print(f"palabras: mediana {tamanos[len(tamanos)//2]}  min {tamanos[0]}  max {tamanos[-1]}")
    print(f"\n{linea}")
    print(f"§9.2.1 — sub-fragmentos de más de {LIMITE_PALABRAS} palabras: {pasados}"
          + ("  ✔ PASA" if not pasados else "  ✖ FALLA"))
    print(f"cobertura — chunks que pierden texto al dividirse: {perdidos}"
          + ("  ✔ PASA" if not perdidos else "  ✖ FALLA"))
    print(f"\n{linea}\nen cuántos trozos se parte cada chunk:")
    for trozos, cuantos in sorted(reparto.items()):
        print(f"   {trozos:>2} trozo(s) : {cuantos:>7,}  ({100*cuantos/total:5.1f}%)")

    # Duplicación de contenido: cuántos ranks se ahorrarían deduplicando.
    excedente = sum(c - 1 for c in claves.values() if c > 1)
    grupos = sum(1 for c in claves.values() if c > 1)
    print(f"\n{linea}\nDEDUPLICACIÓN (§10.2.1 evalúa por el campo `text`)")
    print(f"   textos únicos        : {len(claves):,} de {total:,}")
    print(f"   chunks que repiten   : {excedente:,}  ({100*excedente/total:.1f}%) "
          f"en {grupos:,} grupos")
    print(f"   el grupo más grande  : {max(claves.values())} copias del mismo texto")
    print(linea)
    raise SystemExit(1 if (pasados or perdidos) else 0)
