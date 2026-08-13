"""Fragmentación del texto (§3): cascada híbrida de cuatro niveles.

────────────────────────────────────────────────────────────────────────────────
EL PROBLEMA

La fragmentación debe satisfacer dos exigencias que tiran en direcciones
opuestas:

  1. §4.3 — el encoder tiene un límite de tokens de entrada. Un fragmento que lo
     supere se trunca en silencio: se codifica su comienzo y el resto deja de
     existir para la recuperación, sin ningún mensaje de error.
  2. §3.3, «Requisito obligatorio» — «Ningún fragmento puede contener oraciones
     o frases incompletas… Una oración que comienza en un chunk debe terminar en
     ese mismo chunk». Y establece cómo cumplirlo: «si se fija un tamaño máximo
     de n tokens, el corte efectivo debe retroceder al final de la última
     oración completa que quepa dentro de ese límite».

La estrategia implementada es híbrida —agrupación por párrafo con retroceso a
límite de oración—, que §3.2 autoriza expresamente siempre que se justifique en
el documento técnico.

────────────────────────────────────────────────────────────────────────────────
LA CASCADA Y LA MEDICIÓN QUE LA JUSTIFICA

Medido sobre los 1.826 documentos normalizados:

    prosa    57.613 bloques de párrafo → 17.218 (29,89 %) no caben por sí solos
    tabular 276.925 filas              →      83 ( 0,03 %) no caben por sí solas

Casi tres de cada diez párrafos de prosa exceden el presupuesto por sí mismos.
Ese dato descarta la estrategia «solo por párrafo»: dejaría 17.218 fragmentos
excedidos para que el encoder los truncara. De ahí la cascada, que se aplica por
unidad y solo desciende de nivel cuando el anterior no consigue que quepa:

    nivel 1  agrupar bloques de párrafo o filas hasta el presupuesto
    nivel 2  dividir el bloque en ORACIONES  ................... §3.3 literal
    nivel 3  dividir por separadores secundarios (`;` `:` puntos guía, salto)
    nivel 4  corte por longitud — último recurso, se anota en la metadata

Los niveles 3 y 4 apenas intervienen, y está medido: las unidades que ni
siquiera un divisor de oraciones puede partir son unas 893 en 157 documentos
(3,02 % de la prosa), y casi ninguna es prosa. Son tablas presupuestales
extraídas en línea recta, un atlas comparativo y un documento en árabe con
índice de puntos guía. El nivel 4 no rompe ninguna oración porque en esos casos
no existe ninguna: son celdas de tabla.

────────────────────────────────────────────────────────────────────────────────
ALTERNATIVA EVALUADA Y DESCARTADA

Se evaluó el uso de un divisor recursivo de biblioteca. El concepto de cascada
es el mismo; la implementación no resulta aplicable aquí:

  - Su lista de separadores por defecto no incluye el punto, de modo que no
    intenta mantener las oraciones íntegras: tal cual, incumple §3.3.
  - Aunque se le añada el punto, sus últimos recursos siguen siendo el espacio y
    el carácter suelto, que parten palabras por la mitad.
  - Cuenta caracteres, y el límite de §4.3 se expresa en tokens.
  - No distingue prosa de contenido tabular, y las filas indivisibles son la
    mitad del índice.

────────────────────────────────────────────────────────────────────────────────
DOS CAMINOS SEGÚN EL FORMATO

  - **Prosa** (pdf, json, html, txt, imágenes): los saltos de párrafo del texto
    normalizado son fronteras reales, verificadas sobre los 1.826 documentos.
  - **Tabular** (csv, xlsx, pbf): los bloques son filas, y una fila no se parte
    nunca. Se agrupan filas completas hasta agotar el presupuesto. §2.1 indica
    que cada fila «puede» tratarse como unidad de fragmentación independiente, y
    agrupar reduce los fragmentos tabulares un 83,6 % sin perder una sola
    palabra. Aquí §3.3 se cumple por construcción: una fila es una unidad
    cerrada.

────────────────────────────────────────────────────────────────────────────────
EL PRESUPUESTO SE INYECTA, NO SE FIJA EN EL CÓDIGO

`num_tokens` depende del tokenizador del encoder. Por eso `chunk_document()`
recibe la función que cuenta tokens como parámetro, con un contador de palabras
como valor por defecto. Aproximar con un factor fijo palabras×k sería incorrecto:
la relación entre tokens y palabras no es la misma en español, inglés y
portugués.
"""

import re
from pathlib import Path
from typing import Callable, Iterable, List, NamedTuple

from core.chunk import Chunk
from core.document import Document
from core.store import read_documents

DATOS = Path(__file__).resolve().parents[1] / "data"
DOCUMENTOS_LIMPIOS = DATOS / "documentos_limpios.jsonl"
CHUNKS = DATOS / "chunks.jsonl"

# Formatos cuyo texto son filas y no prosa: 103 documentos del corpus
# (30 CSV/XLSX y 73 PBF).
FORMATOS_TABULARES = frozenset({"csv", "xlsx", "pbf"})

# Presupuesto por defecto, en palabras, coherente con el contador por defecto.
PRESUPUESTO_POR_DEFECTO = 350

# Un contador de tokens es cualquier función texto → entero. Declararlo como
# tipo hace explícito que el encoder se enchufa en este punto y en ningún otro.
ContadorTokens = Callable[[str], int]


def contar_palabras(texto: str) -> int:
    """Contador por defecto: palabras separadas por espacios en blanco.

    Permite ejecutar y verificar la fragmentación sin descargar el modelo. El
    contador definitivo es el tokenizador del encoder, que se inyecta.
    """
    return len(texto.split())


# ══════════════════════════════════════════════════════════════════════════════
# Nivel 2 — corte por oración
# ══════════════════════════════════════════════════════════════════════════════

# Caracteres que cierran una oración. Se incluyen los de escrituras no latinas
# porque el corpus contiene documentos en chino, árabe, japonés y coreano.
FIN_ORACION = ".!?…。！？؟।"

# Comillas y paréntesis que pueden seguir al punto y siguen formando parte de la
# misma oración: «dijo que sí.»  /  (véase la nota 3.)
CIERRES = '")]»”’\'›〉】'

# Abreviaturas tras las cuales un punto NO termina la oración. Sin esta lista,
# «Art. 31» o «et al. 2022» producirían fragmentos partidos a media frase, que
# es precisamente lo que §3.3 prohíbe. Incluye las de los tres idiomas del
# corpus y las de citación académica, abundantes en informes de centros de
# pensamiento.
ABREVIATURAS = frozenset("""
sr sra srta dr dra prof profa ing lic mr mrs ms jr sr dept univ inc ltd corp
gen col lt cap cmdr adm sgt art arts cap caps vol vols fig figs tab tabs
núm num nro ed eds pp pág págs pag pags ver vs etc aprox ej cf al
ee uu ss aa av avda apdo tel
""".split())

# Divisores del nivel 3, en orden de preferencia: primero el más semántico. Se
# corta después del separador para no perderlo.
#   - puntos guía: los índices y tablas de los PDF están llenos de ellos
#   - `;` y `:`  : fronteras naturales de cláusula
#   - salto de línea simple: en contenido tabular separa `columna: valor`
SECUNDARIOS = (
    re.compile(r"(?<=\.)\s*(?=\S)"),          # tras un punto sin espacio detrás
    re.compile(r"(?<=;)\s*(?=\S)"),
    re.compile(r"(?<=:)\s*(?=\S)"),
    re.compile(r"\n"),
)


def _inicia_oracion(caracter: str) -> bool:
    """¿Este carácter puede iniciar una oración nueva?

    El criterio es «no es minúscula», y no «es mayúscula». Con el segundo, el
    corte no funcionaría en árabe, chino, japonés ni coreano, escrituras sin
    distinción de caja presentes en 18 documentos del corpus. Con el primero
    esos idiomas se segmentan correctamente y se sigue rechazando el caso que
    importa: «e.g. foo», «art. 31», «vs. china».
    """
    return not caracter.islower()


def _termina_oracion(texto: str, posicion: int) -> bool:
    """¿El texto inmediatamente anterior a `posicion` cierra una oración?

    Retrocede saltando comillas y paréntesis de cierre, porque `dijo que sí.»`
    termina oración igual que `dijo que sí.`. A continuación comprueba que la
    última palabra no sea una abreviatura conocida.
    """
    indice = posicion - 1
    while indice >= 0 and texto[indice] in CIERRES:
        indice -= 1
    if indice < 0 or texto[indice] not in FIN_ORACION:
        return False

    # Un punto tras una abreviatura no cierra oración. Se examina la última
    # «palabra» previa al punto, en minúsculas y sin puntos internos, para que
    # «et al.» y «EE.UU.» se reconozcan igual que «etc.».
    if texto[indice] == ".":
        inicio = indice
        while inicio > 0 and (texto[inicio - 1].isalpha() or texto[inicio - 1] == "."):
            inicio -= 1
        palabra = texto[inicio:indice].replace(".", "").lower()
        if palabra in ABREVIATURAS:
            return False
        # Una sola letra antes del punto es casi siempre una inicial: «J. Smith».
        if len(palabra) == 1:
            return False
    return True


def dividir_en_oraciones(texto: str) -> List[str]:
    """Divide un texto en oraciones sin perder ningún carácter que no sea espacio.

    Solo corta en posiciones donde ya existe espacio en blanco, y ese espacio se
    descarta. Esa propiedad es la que permite verificar la cobertura carácter a
    carácter: concatenando todos los fragmentos de un documento y eliminando los
    espacios debe obtenerse exactamente el texto original sin espacios.

    No se emplea `re.split` con lookbehind porque el contexto que hay que
    examinar hacia atrás es de ancho variable —el punto puede ir seguido de una
    o varias comillas de cierre— y `re` solo admite lookbehind de ancho fijo.
    """
    cortes: List[int] = []
    for espacio in re.finditer(r"\s+", texto):
        inicio, fin = espacio.span()
        if fin >= len(texto):
            continue
        if _termina_oracion(texto, inicio) and _inicia_oracion(texto[fin]):
            cortes.append(inicio)

    if not cortes:
        return [texto]

    partes, anterior = [], 0
    for corte in cortes:
        parte = texto[anterior:corte].strip()
        if parte:
            partes.append(parte)
        anterior = corte
    ultima = texto[anterior:].strip()
    if ultima:
        partes.append(ultima)
    return partes


def unir_bloques_partidos(bloques: List[str]) -> List[str]:
    """Recompone las oraciones que el extractor dividió en dos bloques.

    Un salto de párrafo no siempre es una frontera de oración. Los extractores
    de PDF emiten un salto de bloque por cada región de texto que detectan, y
    esas regiones se cortan también en los cambios de columna y de página. El
    resultado es que una oración comienza en un bloque y termina en el
    siguiente, de modo que agrupar bloques completos la partiría, que es
    exactamente lo que §3.3 prohíbe.

    El criterio para unir es doble, y deben cumplirse ambas condiciones:

      1. el bloque no termina en puntuación de cierre, y
      2. el siguiente comienza en minúscula.

    Con las dos se distingue una oración realmente partida («…the AI mod» +
    «els were…») de un encabezado o pie de figura, que legítimamente carece de
    punto final pero va seguido de algo que empieza en mayúscula («Figure 1.2.1»
    + «Number of AI patents…»). Unir esos dos también sería un error: son
    unidades distintas del documento.

    Solo se aplica a prosa. En contenido tabular los bloques son filas y ninguna
    termina en punto: unirlas fundiría el documento en un único bloque.
    """
    if not bloques:
        return []
    unidos = [bloques[0]]
    for bloque in bloques[1:]:
        anterior = unidos[-1]
        continua = (not _termina_oracion(anterior, len(anterior))
                    and bloque[:1].islower())
        if continua:
            # Se une con un espacio y no con un salto de párrafo: es la misma
            # oración, y así el fragmento se lee como lo escribió el autor.
            unidos[-1] = f"{anterior} {bloque}"
        else:
            unidos.append(bloque)
    return unidos


def _dividir_por_patron(texto: str, patron: re.Pattern) -> List[str]:
    """Corta por un separador del nivel 3, conservándolo a la izquierda."""
    partes = [p.strip() for p in patron.split(texto)]
    return [p for p in partes if p]


def _corte_duro(texto: str, contar: ContadorTokens, presupuesto: int) -> List[str]:
    """Nivel 4: corte por longitud. Último recurso.

    Solo se alcanza cuando una unidad no cabe y ningún separador puede
    dividirla. Está medido: alrededor de 25 casos en todo el corpus, todos
    celdas de tabla. Este corte no rompe ninguna oración porque en esos casos no
    hay ninguna.

    Corta entre palabras, nunca dentro de una: partir una palabra la vuelve
    irrecuperable para la búsqueda.
    """
    partes: List[str] = []
    actual: List[str] = []
    suma = 0
    for palabra in texto.split():
        # Se cuenta cada palabra por separado y se acumula, en lugar de contar
        # el texto unido en cada iteración. Por dos motivos: concatenar cadenas
        # repetidamente es cuadrático, y con el tokenizador real serían miles de
        # llamadas sobre textos cada vez más largos. La suma de las partes
        # sobreestima el total —un tokenizador de subpalabras fusiona tokens en
        # los bordes—, de modo que el resultado nunca excede el presupuesto.
        coste = contar(palabra)
        if actual and suma + coste > presupuesto:
            partes.append(" ".join(actual))
            actual, suma = [], 0
        actual.append(palabra)
        suma += coste
    if actual:
        partes.append(" ".join(actual))
    return partes or [texto]


class Unidad(NamedTuple):
    """Un fragmento indivisible de texto, con lo necesario para reagruparlo."""

    texto: str
    separador: str   # lo que precede al unirlo con la unidad anterior
    nivel: int       # 1 párrafo/fila · 2 oración · 3 secundario · 4 corte duro
    tokens: int


def _atomizar(
    texto: str,
    separador: str,
    contar: ContadorTokens,
    presupuesto: int,
    nivel: int = 1,
) -> List[Unidad]:
    """La cascada, expresada como una recursión sobre los niveles.

    Si el texto cabe, constituye una unidad. Si no cabe, se desciende un nivel y
    se reintenta con cada porción resultante.
    """
    tokens = contar(texto)
    if tokens <= presupuesto:
        return [Unidad(texto, separador, nivel, tokens)]

    if nivel == 1:
        trozos = dividir_en_oraciones(texto)
        # Si el bloque era una sola oración no se gana nada repitiendo el
        # intento: se desciende directamente al nivel 3.
        if len(trozos) > 1:
            return _concatenar(trozos, separador, " ", contar, presupuesto, 2)
        return _atomizar(texto, separador, contar, presupuesto, 2)

    if nivel == 2:
        for patron in SECUNDARIOS:
            trozos = _dividir_por_patron(texto, patron)
            if len(trozos) > 1:
                return _concatenar(trozos, separador, " ", contar, presupuesto, 3)
        return _atomizar(texto, separador, contar, presupuesto, 3)

    # Nivel 3 agotado: corte duro. Es terminal, de aquí no se desciende más.
    # Si volviera a invocarse a sí misma, una «palabra» más larga que el
    # presupuesto haría que `_corte_duro` devolviera siempre un único trozo
    # idéntico al de entrada y la recursión no terminaría nunca.
    trozos = _corte_duro(texto, contar, presupuesto)
    if len(trozos) <= 1:
        # Irreducible: se acepta excedido y se marca como nivel 4 para que quede
        # registrado, en lugar de partir una palabra por la mitad y volverla
        # irrecuperable.
        return [Unidad(texto, separador, 4, tokens)]
    return [
        Unidad(trozo, separador if indice == 0 else " ", 4, contar(trozo))
        for indice, trozo in enumerate(trozos)
    ]


def _concatenar(
    trozos: Iterable[str],
    separador_inicial: str,
    separador_interno: str,
    contar: ContadorTokens,
    presupuesto: int,
    nivel: int,
) -> List[Unidad]:
    """Atomiza cada porción y devuelve la lista completa, en orden."""
    unidades: List[Unidad] = []
    for indice, trozo in enumerate(trozos):
        sep = separador_inicial if indice == 0 else separador_interno
        unidades.extend(_atomizar(trozo, sep, contar, presupuesto, nivel))
    return unidades


# ══════════════════════════════════════════════════════════════════════════════
# Nivel 1 — agrupación
# ══════════════════════════════════════════════════════════════════════════════

def _agrupar(unidades: List[Unidad], contar: ContadorTokens, presupuesto: int):
    """Agrupa unidades consecutivas mientras quepan y va emitiendo fragmentos.

    Es un generador: no construye la lista completa en memoria, lo que importa
    con documentos tabulares de más de cien mil filas.

    La suma de los tokens de las unidades sobreestima el total real —al unir dos
    textos, un tokenizador de subpalabras puede fusionar tokens del borde—, de
    modo que agrupar por esa suma nunca excede el presupuesto. El `num_tokens`
    definitivo se vuelve a medir sobre el texto ya unido, en `chunk_document()`.
    """
    actual: List[Unidad] = []
    suma = 0
    for unidad in unidades:
        if actual and suma + unidad.tokens > presupuesto:
            yield actual
            actual, suma = [], 0
        actual.append(unidad)
        suma += unidad.tokens
    if actual:
        yield actual


def chunk_document(
    documento: Document,
    *,
    contar_tokens: ContadorTokens = contar_palabras,
    presupuesto: int = PRESUPUESTO_POR_DEFECTO,
) -> List[Chunk]:
    """Divide un documento en fragmentos. Es la función principal del módulo.

    Devuelve una lista de `Chunk` con `position` consecutiva empezando en 0
    (Tabla 1) y `chunk_id` determinista (§1.4). Un documento sin texto devuelve
    una lista vacía: son 8 documentos del corpus —fotografías sin texto y
    manifiestos del proceso de descarga— y un vector vacío no recupera nada.
    """
    if not documento.text.strip():
        return []

    es_tabular = documento.format in FORMATOS_TABULARES

    # Los bloques separados por salto de párrafo son párrafos en prosa y filas
    # en contenido tabular. La frontera es la misma; lo que cambia es qué
    # significa y hasta dónde puede dividirse.
    bloques = [b.strip() for b in documento.text.split("\n\n") if b.strip()]
    if not es_tabular:
        # Recompone las oraciones partidas por saltos de columna o de página.
        bloques = unir_bloques_partidos(bloques)

    unidades: List[Unidad] = []
    for bloque in bloques:
        unidades.extend(_atomizar(bloque, "\n\n", contar_tokens, presupuesto))

    fragmentos: List[Chunk] = []
    for posicion, grupo in enumerate(_agrupar(unidades, contar_tokens, presupuesto)):
        # La primera unidad del grupo no lleva separador delante.
        texto = grupo[0].texto + "".join(u.separador + u.texto for u in grupo[1:])
        texto = texto.strip()
        if not texto:
            continue

        # Se recuenta sobre el texto final: es el valor que exige la Tabla 1 y
        # el que debe respetar el límite del encoder.
        tokens = contar_tokens(texto)
        nivel = max(u.nivel for u in grupo)

        fragmentos.append(Chunk.from_document(
            documento,
            text=texto,
            position=posicion,
            num_tokens=max(tokens, 1),      # Chunk exige num_tokens > 0
            estrategia="fila" if es_tabular else "parrafo",
            nivel_corte=nivel,
            unidades=len(grupo),
        ))
    return fragmentos


def chunk_corpus(
    ruta: Path = DOCUMENTOS_LIMPIOS,
    *,
    contar_tokens: ContadorTokens = contar_palabras,
    presupuesto: int = PRESUPUESTO_POR_DEFECTO,
    paso: int = 1,
):
    """Recorre el corpus normalizado y emite fragmentos, documento a documento.

    Generador, por el mismo motivo que `_agrupar`: no hay razón para mantener
    todos los fragmentos en memoria.
    """
    for indice, documento in enumerate(read_documents(ruta)):
        if indice % paso:
            continue
        yield from chunk_document(
            documento, contar_tokens=contar_tokens, presupuesto=presupuesto
        )
