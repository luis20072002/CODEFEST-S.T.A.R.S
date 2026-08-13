"""División de los fragmentos al límite de 250 palabras (§9.2.1).

────────────────────────────────────────────────────────────────────────────────
POR QUÉ ESTE MÓDULO EXISTE

§9.2 exige que cada uno de los 10 fragmentos reportados tenga «como máximo 250
palabras», y §9.3.1 advierte de que los que se excedan «serán penalizados o
descartados durante la evaluación automática».

Medido sobre los 91.021 fragmentos del índice entregado:

    mediana        : 324 palabras
    percentil 10   : 256          ← incluso el decil más corto se excede
    superan 250    : 82.444  (90,6 %)

Nueve de cada diez fragmentos no pueden reportarse tal como están almacenados.
No es un caso excepcional resoluble con una condición: es el caso normal.

EL LÍMITE SE APLICA AL CONSTRUIR LA SALIDA, NO AL ÍNDICE. Los fragmentos
extensos producen vectores con más contexto y por tanto mejores; solo se
dividen en el momento de construir `resultados.jsonl`. Por eso este módulo
pertenece a la recuperación y no altera ni la fragmentación ni la base
vectorial.

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

De la última se sigue una consecuencia relevante: §9.2.1 no obliga a reportar
todos los sub-fragmentos de un chunk. Lo que prohíbe es situar dos en una misma
posición del ranking. Esa es la decisión que resuelve `elegir_mejor()`.

────────────────────────────────────────────────────────────────────────────────
POR QUÉ SE REUTILIZA EL DIVISOR DE ORACIONES DE LA FRAGMENTACIÓN

§9.2.1 remite a la misma completitud lingüística de §3.3, de modo que la lógica
requerida es la que ya está implementada y verificada sobre los 1.826
documentos. Una segunda implementación de la misma regla acabaría divergiendo de
la primera.

El módulo de fragmentación no depende de nada fuera de la biblioteca estándar,
por lo que importarlo aquí no añade ninguna dependencia a la reproducción de los
resultados.
"""

import re
import unicodedata
from typing import List, Optional, Sequence

from chunking.chunker import SECUNDARIOS, dividir_en_oraciones

# §9.2 y §9.3.1. Un fragmento de 251 palabras puede ser descartado.
LIMITE_PALABRAS = 250

# Palabras que no discriminan al elegir el mejor sub-fragmento. No pretende ser
# una lista exhaustiva: solo evita que artículos y preposiciones dominen el
# cálculo de solapamiento.
VACIAS = frozenset("""
el la los las un una unos unas de del al a en con por para sin sobre entre
que qué cual cuál cuales como cómo cuando cuándo donde dónde quien quién
es son ser sea fue han ha hay su sus lo se no ni o u y e
the a an of in on at to for with by from and or as is are was were be been
this that these those it its
""".split())


def contar_palabras(texto: str) -> int:
    """Palabras según §9.2.1: unidades separadas por espacios en blanco."""
    return len(texto.split())


def dividir_a_limite(texto: str, limite: int = LIMITE_PALABRAS) -> List[str]:
    """Divide un texto en porciones de `limite` palabras o menos, sin cortar oraciones.

    Devuelve la lista completa de sub-fragmentos, en orden, cubriendo todo el
    texto. Si ya cabe, devuelve el texto sin modificar.

    Es la misma cascada de la fragmentación, medida en palabras y en dos niveles:

      1. agrupar **oraciones completas** hasta el límite  ← §9.2.1 literal
      2. si una sola oración se excede, separadores secundarios y, en último
         extremo, corte por longitud

    El nivel 2 existe porque el corpus contiene unidades que no son oraciones
    aunque lo parezcan: tablas presupuestales extraídas en línea recta e índices
    con puntos guía. En esos casos no hay ninguna oración que preservar.
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
    """Trocea una unidad que no cabe y que el divisor de oraciones no pudo partir.

    Prueba primero los separadores secundarios de la fragmentación —puntos guía,
    punto y coma, dos puntos, salto de línea—. Si ninguno reduce el tamaño,
    corta por longitud entre palabras, nunca dentro de una: partir una palabra
    la vuelve irrecuperable para la búsqueda.
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
# Elección del sub-fragmento que se reporta
# ══════════════════════════════════════════════════════════════════════════════

def _normalizar(texto: str) -> str:
    """Minúsculas y sin diacríticos, para que «órbita» y «orbita» equivalgan."""
    descompuesto = unicodedata.normalize("NFD", texto.lower())
    return "".join(c for c in descompuesto if unicodedata.category(c) != "Mn")


def _terminos(texto: str) -> set:
    return {p for p in re.findall(r"[^\W\d_]{3,}", _normalizar(texto))
            if p not in VACIAS}


def elegir_mejor(subfragmentos: Sequence[str], consulta: Optional[str] = None) -> str:
    """Devuelve el sub-fragmento más pertinente para la consulta.

    ────────────────────────────────────────────────────────────────────────
    POR QUÉ SE REPORTA UNO Y NO TODOS

    §9.2.1 permite ambas cosas: lo único que prohíbe es situar dos
    sub-fragmentos en una misma posición del ranking. Con el 90,6 % de los
    fragmentos por encima de 250 palabras, la diferencia es sustancial:

      - reportándolos todos, las diez posiciones procederían de unos cinco
        fragmentos distintos, y la mitad de un fragmento que no contiene la
        parte relevante ocuparía una posición alta con relevancia nula. NDCG@10
        penaliza precisamente los irrelevantes en las primeras posiciones.
      - reportando el mejor, las diez posiciones proceden de diez fragmentos
        distintos, cada uno con su porción pertinente.

    ────────────────────────────────────────────────────────────────────────
    POR QUÉ POR SOLAPAMIENTO LÉXICO Y NO POR SIMILITUD VECTORIAL

    Lo inmediato sería codificar cada sub-fragmento y quedarse con el de mayor
    similitud. Se descartó por coste: `generador.py` se ejecuta en el entorno de
    evaluación, y codificar diez sub-fragmentos por consulta en CPU supondría
    horas de cómputo trasladadas a quien reproduce los resultados.

    El solapamiento léxico es determinista, cuesta microsegundos y no constituye
    un modelo, de modo que no incurre en la restricción de §8.3, que se refiere
    a modelos de lenguaje generativos.

    Sin consulta devuelve el primero, que es el comienzo del fragmento y por
    tanto la porción con más contexto propio.
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
        # Desempate por el orden original, para que la elección sea
        # determinista: §1.4 exige que dos ejecuciones produzcan lo mismo.
        return (len(comunes), -subfragmentos.index(trozo))

    return max(subfragmentos, key=puntuar)


# ══════════════════════════════════════════════════════════════════════════════
# Deduplicación
# ══════════════════════════════════════════════════════════════════════════════

def clave_de_texto(texto: str) -> str:
    """Clave para determinar si dos fragmentos dicen lo mismo.

    Normaliza **únicamente el espacio en blanco**. No convierte a minúsculas ni
    elimina puntuación de forma deliberada: dos textos que difieren en esos
    aspectos difieren realmente, y agruparlos supondría descartar contenido
    distinto.

    El espacio sí se normaliza porque está medido: los pares de fragmentos del
    corpus que producen exactamente el mismo vector tienen textos idénticos
    salvo por unos pocos caracteres de espaciado. El tokenizador del encoder ya
    los trata como iguales, y la deduplicación debe hacer lo mismo.
    """
    return " ".join(texto.split())


def deduplicar_por_texto(items: Sequence, texto_de=lambda x: x,
                         limite: Optional[int] = None) -> List:
    """Elimina los elementos cuyo texto ya apareció antes. Conserva el orden.

    ────────────────────────────────────────────────────────────────────────
    POR QUÉ HACE FALTA, CON LA MEDICIÓN

    El índice contiene 91.021 fragmentos pero solo 87.424 textos únicos: 3.597
    (el 4,0 %) repiten contenido, y el grupo mayor tiene 11 copias. No es un
    defecto del procesamiento, es el corpus, y responde a dos patrones:

      - filas tabulares idénticas repartidas entre varios documentos, y
      - secciones repetidas dentro de un mismo informe.

    Si dos de ellos entran en el top-10 ocupan dos de las diez posiciones con el
    mismo contenido. §10.2.1 evalúa la relevancia sobre el campo `text`, de modo
    que el segundo no aporta nada y desplaza a un fragmento que sí podría sumar.

    Se aplica sobre el texto **final** que se va a reportar, ya dividido a 250
    palabras, porque es ese texto el que se evalúa. Y dado que elimina
    candidatos, la búsqueda debe solicitar a FAISS más de diez para que tras
    deduplicar sigan quedando diez.

    No emplea ningún modelo: es comparación de cadenas.
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
    """Devuelve el texto listo para reportar, con `limite` palabras o menos.

    El `chunk_id` que se reporte junto a este texto debe seguir siendo el del
    fragmento original del índice (§9.2.1).
    """
    return elegir_mejor(dividir_a_limite(texto, limite), consulta)
