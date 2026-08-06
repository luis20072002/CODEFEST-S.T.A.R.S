"""Limpieza y normalización del texto de los Document (§2.2).

Se ejecuta ENTRE la extracción (loaders) y el chunking. Entra un `Document`
recién leído de `documentos.jsonl`, sale el mismo `Document` con el `text`
limpio y listo para que el Chunker lo parta por oraciones.

QUÉ PIDE §2.2, y dónde está cada cosa aquí:

  1. Normalización Unicode y codificación UTF-8  → `normalize_unicode()`
  2. Eliminación de caracteres de control y
     espacios redundantes                        → `strip_control_chars()` y
                                                   `collapse_whitespace()`
  3. Eliminación de boilerplate (cabeceras,
     pies, numeración, menús de navegación)      → `strip_repeated_boilerplate()`

Los cuatro pasos que NO pide el PDF por su nombre pero que el corpus obligó a
añadir (`decode_cid()`, `dehyphenate()`, `strip_spaced_letter_runs()` y
`strip_anonymous_columns()`) están justificados en sus propios docstrings con
la medición que los motivó. El primero es el más importante: sin él, dos
documentos entran al índice como decenas de miles de vectores de
`(cid:48)(cid:76)(cid:81)`.

⚠️ **Regla del equipo que atraviesa todo este módulo: se quita ruido, nunca
información.** Cada paso que borra algo trae en su docstring la medición que
demuestra que lo borrado no era contenido del documento. Antes de añadir una
regla nueva, hay que poder escribir esa medición.

────────────────────────────────────────────────────────────────────────────
LO QUE ESTE MÓDULO **NO** HACE, Y POR QUÉ (léelo antes de "mejorarlo")

**No filtra por alfabeto.** La tentación es enorme: hay documentos en inglés
llenos de caracteres chinos y árabes. Pero se midió sobre los 1826 y son DOS
fenómenos distintos que se ven igual:

  - `SWF_2022-executive-summary-english.pdf` lleva el logo de la fundación con
    su nombre en seis idiomas TESELADO como fondo de todas las páginas. El
    6-grama «SECURE WORLD FOUNDATION ФОНД БЕЗОПАСНОГО МИРА» aparece 448 veces
    y se come el **69,3%** del documento.
  - Pero hay **22 documentos legítimamente** en árabe, ruso, chino, coreano y
    japonés: las versiones oficiales de la ONU (`UNOOSA_st-space-088a/r/c`),
    los resúmenes ejecutivos traducidos de SWF, el AI Index en chino. §1.3
    dice que TODOS los documentos provistos forman el corpus.

Una regla por alfabeto borraría los 22. La señal que separa los dos casos no
es el alfabeto sino la **repetición**, y por eso la marca de agua la mata
`strip_repeated_boilerplate()` sin mirar ni una vez de qué script es cada
carácter. El corpus es 98,99% latino; el 1% restante se queda como está.
────────────────────────────────────────────────────────────────────────────
"""

import re
import unicodedata
from collections import Counter
from math import ceil

from core.document import Document

# Formatos donde cada línea es un registro (una fila de tabla, un feature de un
# PBF) y no una línea de prosa partida por el ancho de la página. En ellos NO se
# unen las líneas sueltas ni se busca boilerplate: un CSV repite sus valores por
# naturaleza, y unir sus filas mezclaría registros distintos en una sola frase.
TABULAR_FORMATS = frozenset({"csv", "xlsx", "pbf"})

# "Palabra plausible": 3 a 20 letras. Es la métrica con la que se resuelven las
# dos decisiones que el corpus obliga a tomar por documento —qué hacer con los
# caracteres de control y con qué desplazamiento decodificar los (cid:N)—. No
# pretende ser un diccionario: solo distingue una palabra de una mega-palabra
# pegada o de un montón de letras sueltas.
PLAUSIBLE_WORD = re.compile(r"\b[^\W\d_]{3,20}\b")


# ═══════════════════════════════════════════════════════════════════════════
# Paso 0 — Marcadores (cid:N) de PDF con la fuente rota
# ═══════════════════════════════════════════════════════════════════════════

CID = re.compile(r"\(cid:(\d+)\)")

# Palabras muy frecuentes en los tres idiomas del corpus. Son el juez de si una
# decodificación es correcta: producir letras es fácil (cualquier
# desplazamiento lo hace), producir ARTÍCULOS Y PREPOSICIONES REALES no.
CID_TESTIGO = re.compile(
    r"\b(the|and|of|for|that|with|from|this|are|was|not|has|been|"
    r"que|los|las|del|para|por|con|una|como|est[aá]|"
    r"dos|das|uma|com|mais|pelo)\b", re.IGNORECASE
)

# Rango de desplazamientos que se prueban. Los dos que aparecen en el corpus
# son 0 y 28, pero el rango es amplio porque el desplazamiento depende de cómo
# la imprenta subconjuntó la fuente y no hay razón para que sea siempre el mismo.
CID_OFFSETS = range(-32, 128)

# Cuánto texto se usa para elegir el desplazamiento. Con 120.000 caracteres hay
# señal de sobra y se evita repetir 160 pasadas sobre un documento entero.
CID_MUESTRA = 120_000

# Para aceptar una decodificación se exigen las dos cosas: un mínimo absoluto de
# palabras testigo, y que supongan al menos 1 de cada 20 palabras. Medido sobre
# los 18 documentos afectados, los dos grupos quedan a distancias enormes del
# corte, así que el resultado no es sensible al valor exacto:
#     F2-CSIS-200   348 testigos /  1.207 palabras = 29%   → se acepta
#     F3-CEOBS-030 8.929 testigos / 42.153 palabras = 21%  → se acepta
#     F3-RESDAL-096     3 testigos /    685 palabras = 0,4% → se rechaza
CID_MIN_TESTIGOS = 10
CID_RATIO = 20


def _cid_decodificados(text: str, offset: int) -> str:
    """Devuelve SOLO los caracteres que saldrían de los (cid:N), concatenados.

    Puntuar el documento entero no sirve: en los archivos donde los (cid:N) son
    el 1% del texto, el otro 99% —que es prosa correcta y no depende del
    desplazamiento— domina la puntuación y hace que todos los desplazamientos
    empaten. Aislando lo decodificado, la señal queda limpia.
    """
    partes, ultimo = [], 0
    for m in CID.finditer(text):
        if m.start() > ultimo:
            partes.append(" ")        # había texto sano entremedias: separa palabras
        codigo = int(m.group(1)) + offset
        partes.append(chr(codigo) if 32 <= codigo <= 0x2FFF else " ")
        ultimo = m.end()
    return "".join(partes)


def decode_cid(text: str) -> str:
    """Recupera el texto de los PDF que salieron extraídos como `(cid:N)`.

    QUÉ ES ESTO. Cuando la fuente empotrada en un PDF no trae tabla `ToUnicode`,
    `pdfminer` no puede saber qué letra representa cada glifo y emite su índice
    interno: `(cid:80)(cid:85)(cid:66)(cid:76)(cid:73)(cid:67)`. Hay **18
    documentos así**, y en dos de ellos es el 99% del texto:

        F2-CSIS-200  (CSIS_plaw-105publ270.pdf)          99,5%     473 palabras
        F3-CEOBS-030 (CEOBS_minamata-…-sudan.pdf)        99,2%  22.896 palabras

    Sin este paso, esos dos entran al índice como decenas de miles de vectores
    de `(cid:48)(cid:76)(cid:81)`, que no pueden casar con ninguna consulta y
    además compiten en el ranking con documentos reales.

    CÓMO SE RECUPERA. En estas fuentes el índice del glifo suele ser el código
    del carácter más una constante, porque la imprenta subconjuntó el tipo de
    letra conservando el orden. Basta con encontrar esa constante, y se
    encuentra probándolas todas y quedándose con la que produce **texto en un
    idioma real** —medido con `CID_TESTIGO`, no con "produce letras", que lo
    hace cualquiera—. Los dos casos del corpus:

        (cid:80)(cid:85)(cid:66)(cid:76)(cid:73)(cid:67)  +0  → «PUBLIC»
        (cid:48)(cid:76)(cid:81)(cid:68)(cid:80)(cid:68)  +28 → «Minamat»

    ⚠️ **No siempre se puede, y por eso hay una prueba de aceptación.** Un PDF
    puede empotrar varias fuentes con mapeos distintos, y entonces ningún
    desplazamiento único sirve para todo el documento. Pasa en `F3-CEOBS-030`,
    donde el cuerpo decodifica perfecto con +28 (8.929 palabras testigo) pero
    los titulares usan otra fuente: se recupera el cuerpo, que es lo que
    importa, y los titulares quedan ilegibles —y como además son cabeceras
    repetidas, se los lleva `strip_repeated_boilerplate()` más adelante—.
    En `F3-RESDAL-096` no hay desplazamiento que funcione.

    **Cuando no se puede decodificar, los marcadores se borran.** Es lo correcto:
    dejar `(cid:212)` en el índice no aporta nada y sí estorba. Los 16
    documentos restantes tienen menos del 10% de su texto en marcadores —son
    ligaduras y símbolos sueltos— así que borrarlos no les quita contenido.
    """
    # Atajo por comparación de subcadena antes de tocar la expresión regular:
    # 1808 de los 1826 documentos no tienen ni un marcador y no deben pagar nada.
    if "(cid:" not in text:
        return text

    muestra = text[:CID_MUESTRA]
    mejor_offset, mejor_testigos, mejor_palabras = None, -1, -1
    for offset in CID_OFFSETS:
        candidato = _cid_decodificados(muestra, offset)
        testigos = sum(1 for _ in CID_TESTIGO.finditer(candidato))
        palabras = sum(1 for _ in PLAUSIBLE_WORD.finditer(candidato))
        # Se ordena por testigos y solo se desempata por palabras: un
        # desplazamiento equivocado puede producir muchas "palabras" (letras
        # seguidas), pero no produce artículos ni preposiciones.
        if (testigos, palabras) > (mejor_testigos, mejor_palabras):
            mejor_offset, mejor_testigos, mejor_palabras = offset, testigos, palabras

    aceptada = (mejor_testigos >= CID_MIN_TESTIGOS
                and mejor_testigos * CID_RATIO >= mejor_palabras)
    if not aceptada:
        return CID.sub(" ", text)

    def traducir(m: re.Match) -> str:
        codigo = int(m.group(1)) + mejor_offset
        return chr(codigo) if 32 <= codigo <= 0x2FFF else " "

    return CID.sub(traducir, text)


# ═══════════════════════════════════════════════════════════════════════════
# Paso 1 — Normalización Unicode (§2.2.1)
# ═══════════════════════════════════════════════════════════════════════════

def normalize_unicode(text: str) -> str:
    """Pasa el texto a NFC y unifica todos los separadores de línea a `\\n`.

    NFC (Normalization Form Composed) resuelve que «ó» se puede escribir de dos
    formas distintas en Unicode: como un solo carácter (U+00F3) o como «o» +
    tilde combinante (U+006F U+0301). Se ven idénticos en pantalla pero son
    strings DISTINTOS para Python y, lo que importa aquí, producen tokens
    distintos en el encoder. Sin este paso, «órbita» escrito de las dos formas
    generaría dos vectores diferentes para la misma palabra.

    ⚠️ **NFC no quita tildes: las COMPONE.** Quitar los diacríticos sería un
    error caro aquí — las 50 consultas están en español y con tilde («órbita»,
    «satélites»), y el encoder es multilingüe: darle texto sin tildes lo saca de
    su distribución de entrenamiento. Verificado sobre la salida: 0 documentos
    fuera de NFC y 456.890 letras acentuadas conservadas.

    Los separadores que unifica y por qué importan:

    - `U+2028` (LINE SEPARATOR) y `U+2029` (PARAGRAPH SEPARATOR): hay 26 y 6 en
      `documentos.jsonl`. Son la trampa del paso 2: pertenecen a las categorías
      Unicode `Zl`/`Zp`, **no** a `Cc`, así que un filtro de "caracteres de
      control" no los toca y sobreviven hasta el índice. Además `str.splitlines()`
      SÍ corta por ellos, así que cualquier código que lea el .jsonl con
      `splitlines()` en vez de iterando el archivo partiría líneas por la mitad.
    - `\\f` (FORM FEED, U+000C): es el salto de página que emiten los PDF. Se
      convierte a `\\n\\n` —y no se borra— porque un cambio de página es una
      frontera de párrafo de pleno derecho, justo lo que necesitan §3.2 y §3.3.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace(" ", "\n")      # LINE SEPARATOR      → salto simple
    text = text.replace(" ", "\n\n")    # PARAGRAPH SEPARATOR → salto doble
    text = text.replace("\f", "\n\n")        # salto de página     → salto doble
    text = text.replace("\v", "\n")
    return text


# ═══════════════════════════════════════════════════════════════════════════
# Paso 2 — Caracteres de control e invisibles (§2.2.2)
# ═══════════════════════════════════════════════════════════════════════════

# Se escriben con secuencias \uXXXX y no con el carácter literal a propósito:
# son todos invisibles, así que pegados tal cual harían este bloque imposible de
# revisar en un diff.
#
# GRUPO 1 — de ANCHURA CERO. Siempre se borran, porque no ocupan ninguna
# posición visual: el texto correcto es el que queda al quitarlos.
ZERO_WIDTH = re.compile(
    "["
    "­"                             # SOFT HYPHEN: guion invisible de corte
    "​-‏"                      # espacios de ancho cero y marcas RTL/LTR
    "‪-‮⁠-⁤⁪-⁯"   # controles bidireccionales
    "﻿"                             # BOM colado a mitad de texto
    "]"
)

# GRUPO 2 — los que SÍ ocupaban una posición en la página. Aquí no hay una
# respuesta única y por eso decide `strip_control_chars()` documento a
# documento; ver su docstring.
CONTROL_CHARS = re.compile(
    "["
    "\x00-\x08\x0b\x0e-\x1f\x7f-\x9f"    # Cc (control) menos \t y \n
    "�"                             # REPLACEMENT CHAR: decodificación rota
    "-"                      # Área de Uso Privado: fuentes PDF rotas
    "]"
)


def strip_control_chars(text: str) -> str:
    """Quita los caracteres de control e invisibles, **sin pegar palabras**.

    Los de anchura cero se borran siempre. Con el resto, la pregunta que hay que
    responder es otra y el corpus da las dos respuestas contrarias:

    - En los **21 PDF de DAIO**, el `\\x07` (BEL) **hace de espacio**: sus notas
      al pie salen extraídas como `36 \\x07Barany,\\x07Armies\\x07of\\x07Arabia`.
      Borrarlo produce `Barany,ArmiesofArabia`, una mega-palabra que no casa con
      ninguna consulta. **El término no se pierde: se vuelve inencontrable**,
      que a efectos de recuperación es lo mismo pero más difícil de detectar.
    - En los **6 PDF de ESA**, el control está DENTRO de las palabras (fuentes
      CID sin tabla ToUnicode: el PDF trae los índices internos del tipo de
      letra). Ahí sustituir por espacio parte palabras que estaban bien.

    Así que la decisión se toma **por documento**, probando las dos variantes y
    quedándose con la que produzca más palabras plausibles. Es determinista
    —función pura del texto, requisito de §1.4— y optimiza justo lo que
    importa. Medido sobre los 51 documentos afectados:

        borrar todo   1.035.133 palabras plausibles · 27.637 mega-palabras
        espacio todo  1.090.699 palabras plausibles · 16.159 mega-palabras
        por documento  se queda con el mejor de los dos en cada caso

    El desempate va a **borrar** (`>` y no `>=`) para conservar el
    comportamiento anterior cuando ninguna de las dos opciones gana, que es lo
    que pasa en los documentos donde solo hay un puñado de estos caracteres.
    """
    text = ZERO_WIDTH.sub("", text)

    # Atajo: la inmensa mayoría de los documentos no tiene ninguno de estos
    # caracteres, y sin él se pagarían dos pasadas extra sobre cada uno de
    # ellos —incluido el XLSX de seis millones de palabras— para nada.
    if not CONTROL_CHARS.search(text):
        return text

    borrado = CONTROL_CHARS.sub("", text)
    espaciado = CONTROL_CHARS.sub(" ", text)
    palabras_borrado = sum(1 for _ in PLAUSIBLE_WORD.finditer(borrado))
    palabras_espaciado = sum(1 for _ in PLAUSIBLE_WORD.finditer(espaciado))
    return espaciado if palabras_espaciado > palabras_borrado else borrado


# ═══════════════════════════════════════════════════════════════════════════
# Paso 3 — Palabras partidas por el salto de línea
# ═══════════════════════════════════════════════════════════════════════════

# Letra + guion + fin de línea + la letra con la que sigue la línea siguiente.
# `[^\W\d_]` es el idiom de Python para "letra Unicode": `\w` ya excluye espacios
# y puntuación, y de lo que queda se descartan los dígitos y el guion bajo. Hace
# falta esa forma porque el módulo `re` de la stdlib no soporta `\p{L}`.
HYPHEN_BREAK = re.compile(r"([^\W\d_])-[ \t]*\n[ \t]*([^\W\d_])")


def dehyphenate(text: str) -> str:
    """Vuelve a unir las palabras que el PDF partió al final de la línea.

    En un PDF maquetado a dos columnas, «información» puede salir extraída como
    «infor-\\nmación». Si eso llega al encoder son dos tokens rotos —«infor» y
    «mación»— y la palabra deja de coincidir con la consulta.

    La condición de que la letra siguiente sea **minúscula** es lo que evita el
    falso positivo obvio: «Latino-\\nAmericano» es un compuesto real y debe
    conservar su guion, mientras que «infor-\\nmación» es un corte tipográfico.
    La mayúscula después del guion delata al compuesto.

    La comprobación va en una función de reemplazo y no en la propia expresión
    regular porque `.islower()` entiende de Unicode (acepta «ó», «ç», «ñ») y una
    clase de caracteres escrita a mano siempre se dejaría alguna letra fuera.
    Cuando no se cumple la condición se devuelve `m.group(0)`, es decir, el
    trozo original sin tocar.
    """
    def join(m: re.Match) -> str:
        first, second = m.group(1), m.group(2)
        return first + second if second.islower() else m.group(0)

    return HYPHEN_BREAK.sub(join, text)


# ═══════════════════════════════════════════════════════════════════════════
# Paso 4 — Letras sueltas separadas por espacios
# ═══════════════════════════════════════════════════════════════════════════

# `\b[^\W\d_]\b` = una letra AISLADA (los \b garantizan que no forma parte de
# una palabra más larga). `[^\W\d_]` es el idiom de Python para "letra Unicode":
# \w quita puntuación y espacios, y de lo que queda descartamos dígitos y "_".
SPACED_LETTERS = re.compile(r"(?:\b[^\W\d_]\b[ \t]+){10,}(?:\b[^\W\d_]\b)?")


def strip_spaced_letter_runs(text: str) -> str:
    """Elimina las tiradas largas de letras sueltas separadas por espacios.

    Hay **188 PDF** con este patrón, y detrás se esconden DOS averías distintas
    de la extracción, las dos igual de inservibles:

    1. **Letter-spacing de títulos.** `G L O B A L  C O U N T E R S P A C E`. El
       PDF separa las letras del titular por estética. Al extraerlo se pierden
       las fronteras entre palabras, así que ni siquiera se puede reconstruir:
       uniéndolo saldría «GLOBALCOUNTERSPACECAPABILITIES», un token único que no
       casa con ninguna consulta.

    2. **Columnas fusionadas.** `i s m p a o t n e s c e h t a o n t g h e e`.
       Aquí `pdfplumber` leyó DOS columnas contiguas intercalando sus caracteres:
       las posiciones pares dicen «climate change» y las impares «response to
       the». Es un fallo de layout, no de limpieza — no se arregla aquí.

    Se borran en vez de intentar recomponerlas porque en los dos casos el texto
    ya está perdido, y dejarlo mete ruido en el índice. El coste está medido y
    es asumible: **234.834 caracteres sobre ~150 millones de letras, el 0,16%
    del corpus**. Solo dos documentos pierden algo apreciable (`F2-CSIS-113` un
    30,5% y `F2-UNOOSA-012` un 6,1%); en el resto es menos del 7%.

    El umbral de **10** letras seguidas es holgado a propósito: ni las iniciales
    («J. F. K.»), ni las enumeraciones («a b c»), ni las variables de una
    fórmula llegan a diez letras aisladas consecutivas, así que no hay falsos
    positivos que valga la pena temer.
    """
    return SPACED_LETTERS.sub(" ", text)


# ═══════════════════════════════════════════════════════════════════════════
# Paso 5 — Boilerplate por repetición (§2.2.3)
# ═══════════════════════════════════════════════════════════════════════════

NGRAM_SIZE = 6          # longitud en palabras del patrón que se busca repetido
MIN_REPEATS = 5         # suelo absoluto: por debajo de esto nunca se borra nada
WORDS_PER_PAGE = 400    # estimación de palabras por página, para el umbral adaptativo
EVERY_N_PAGES = 2       # exigimos que el patrón salga al menos 1 vez cada N páginas


def _repeat_threshold(n_words: int) -> int:
    """Cuántas repeticiones hacen falta para considerar un n-grama boilerplate.

    No puede ser un número fijo. El boilerplate es **mobiliario de página**
    —marca de agua, cabecera, pie, numeración— así que se repite más o menos
    una vez por página: cuanto más largo el documento, más veces sale. Un umbral
    fijo de 5 sería correcto en un informe de 20 páginas y absurdo en uno de
    1.200, donde borraría frases legítimas de la prosa que casualmente se repiten.

    Por eso el umbral escala con el tamaño: se estima el número de páginas y se
    exige que el patrón aparezca al menos una vez cada `EVERY_N_PAGES`. Sobre
    los casos reales del corpus el criterio acierta en todos:

        F2-SWF-035    14.883 palabras → umbral  19 · marca de agua ×448   ✔
        F2-ESA-031     9.931 palabras → umbral  13 · puntos guía  ×1125   ✔
        F2-CSIS-201  508.576 palabras → umbral 636 · pie de imprenta ×1120 ✔
        F1-ILIA-005   88.099 palabras → umbral 110 · título al pie ×120   ✔
    """
    pages = max(1, n_words / WORDS_PER_PAGE)
    return max(MIN_REPEATS, ceil(pages / EVERY_N_PAGES))


def strip_repeated_boilerplate(text: str, *, ngram_size: int = NGRAM_SIZE) -> str:
    """Borra los fragmentos que se repiten muchas veces DENTRO del documento.

    Es el paso que cumple §2.2 punto 3 («eliminación de boilerplate: cabeceras,
    pies de página, numeración, menús de navegación») **sin listas negras
    escritas a mano**. Una sola regla se lleva por delante todos estos casos
    reales, que a simple vista no se parecen en nada:

        marca de agua   «SECURE WORLD FOUNDATION ФОНД БЕЗОПАСНОГО МИРА» ×448
        pie de imprenta «2020 Jkt 099139 PO 00092 Frm»                 ×1120
        puntos guía     «. . . . . .» del índice de contenidos         ×1966
        título al pie   «Índice Latinoamericano de Inteligencia…»      ×120

    CÓMO FUNCIONA. Se recorren todos los n-gramas de `ngram_size` palabras y se
    cuenta cuántas veces aparece cada uno. Los que superan el umbral adaptativo
    se marcan, y se borran todas las palabras que cubren. La prosa real casi
    nunca repite seis palabras seguidas; el mobiliario de página lo hace en cada
    página, que es exactamente la diferencia que estamos explotando.

    Se usan las tuplas de palabras como clave del contador, y no su `hash()`,
    por **reproducibilidad (§1.4)**: dos tuplas distintas pueden compartir hash,
    y como Python aleatoriza el hash de los strings en cada proceso, la colisión
    caería en documentos distintos en cada corrida. Sería una diferencia
    minúscula y silenciosa en el índice — justo la clase de cosa que hace que
    `generador.py` no reproduzca `resultados.jsonl`.
    """
    # `finditer` en vez de `split()` porque necesitamos las POSICIONES de cada
    # palabra en el texto original: solo así podemos recortar las palabras
    # sobrantes dejando intactos los `\n\n` que hay entre medias.
    matches = list(re.finditer(r"\S+", text))
    if len(matches) < ngram_size * 2:
        return text

    words = [m.group() for m in matches]
    grams = [tuple(words[i:i + ngram_size]) for i in range(len(words) - ngram_size + 1)]
    counts = Counter(grams)
    threshold = _repeat_threshold(len(words))

    # `bytearray` como vector de banderas: una posición por palabra, 1 = se borra.
    # Es mucho más compacto que una lista de bool y aquí solo guardamos 0/1.
    #
    # ⚠️ Se CONSERVA la primera aparición de cada n-grama y se borran solo las
    # repeticiones. La diferencia importa: hay encabezados de sección y leyendas
    # de tabla que se repiten en cada página y por tanto disparan el umbral, pero
    # cuyo texto sí aporta al documento la primera vez. Sin esta salvedad,
    # `SWF_2025-executive-summary-arabic.pdf` perdía el 19% de su contenido —el
    # encabezado «armas antisatélite que se colocan en órbita terrestre» ×24— y
    # esos términos desaparecían del índice por completo. Con ella, 448 copias de
    # una marca de agua colapsan a una sola y ningún término se pierde.
    drop = bytearray(len(words))
    seen: set = set()
    for i, gram in enumerate(grams):
        if counts[gram] < threshold:
            continue
        if gram not in seen:
            seen.add(gram)
            continue
        drop[i:i + ngram_size] = b"\x01" * ngram_size

    if not any(drop):
        return text

    # Reconstrucción: se copia el texto original saltándose SOLO los tramos de
    # las palabras marcadas. Los espacios y saltos de línea entre palabras caen
    # dentro de los tramos que se copian, así que la estructura de párrafos
    # sobrevive; los huecos que queden los limpia `collapse_whitespace()`.
    out, last = [], 0
    for match, flag in zip(matches, drop):
        if flag:
            out.append(text[last:match.start()])
            last = match.end()
    out.append(text[last:])
    return "".join(out)


# ═══════════════════════════════════════════════════════════════════════════
# Paso 5 bis — Columnas sin encabezado en los tabulares
# ═══════════════════════════════════════════════════════════════════════════

# Una línea ENTERA cuyo nombre de columna es el inventado por
# `tabular_loader._clean_header()` y cuyo valor es un número pelado.
#
# ⚠️ El `$` no es decorativo: sin él, `columna_1: 2020-01-01` casaría hasta el
# `2020` y la sustitución dejaría un `-01-01` suelto, partiendo un valor real
# por la mitad. Con `$` la línea o casa entera o no casa. El `\n?` posterior se
# lleva además el salto, para no dejar una línea en blanco dentro de la fila.
ANON_COLUMN = re.compile(r"^columna_\d+: -?\d+(?:\.\d+)?[ \t]*$\n?", re.MULTILINE)


def strip_anonymous_columns(text: str) -> str:
    """Quita el contador de filas que los CSV traen en una columna sin nombre.

    QUÉ PASA. `TabularLoader` sigue §2.1 al pie de la letra y emite cada fila
    como pares «columna: valor». Cuando el CSV trae una columna **sin
    encabezado**, `_clean_header()` le inventa un nombre posicional
    (`columna_1`) para no desalinear el resto. En cuatro CSV del AI Index esa
    columna es el **número de fila del exporte**, y el resultado son **231.830
    líneas** de `columna_1: 0`, `columna_1: 1`, `columna_1: 2`…

    POR QUÉ SE PUEDE BORRAR SIN PERDER INFORMACIÓN — está medido, no supuesto.
    Se comprobó que los valores forman secuencias `+1` perfectas con reinicios
    a 0 (26 reinicios en total, todos desde valores cercanos a 10.000): son las
    descargas de PubMed, que salen en lotes de 10.000 registros y se
    concatenaron. Los cuatro documentos afectados:

        F1-AIINDEX-056  111.775 líneas   13 reinicios
        F1-AIINDEX-063   61.521 líneas    7 reinicios
        F1-AIINDEX-059   46.514 líneas    5 reinicios
        F1-AIINDEX-057   12.020 líneas    1 reinicio

    El argumento del PDF, además de la medición: §2.1 pide el formato
    «columna: valor» precisamente «de modo que **cada valor conserve el nombre
    de su columna como contexto**». Una columna sin encabezado no aporta ningún
    contexto — el nombre `columna_1` **lo inventamos nosotros**, no está en el
    archivo de ADL. Un entero suelto bajo una etiqueta inventada no es
    recuperable por ninguna consulta.

    ⚠️ **Solo se borra si el valor es un número pelado.** Si alguna columna sin
    encabezado llegara a traer texto, se conserva entera: ahí sí habría
    contenido que un fragmento podría necesitar. Es la diferencia entre quitar
    ruido y quitar información, y la regla del equipo es que lo segundo no se
    hace nunca.
    """
    # Atajo: solo cuatro documentos del corpus tienen estas líneas, y el resto
    # de los tabulares no debe pagar una pasada de expresión regular sobre
    # textos que llegan a los seis millones de palabras.
    if "columna_" not in text:
        return text
    return ANON_COLUMN.sub("", text)


# ═══════════════════════════════════════════════════════════════════════════
# Paso 6 — Espacios redundantes (§2.2.2)
# ═══════════════════════════════════════════════════════════════════════════

def collapse_whitespace(text: str, *, join_lines: bool = True) -> str:
    """Colapsa espacios redundantes **conservando las fronteras de párrafo**.

    ⚠️ Este es el paso donde es más fácil arruinar todo lo que viene después.
    Un `re.sub(r"\\s+", " ", text)` —que es lo primero que uno escribe— aplasta
    también los `\\n\\n`, y esos saltos dobles son la frontera de párrafo que
    necesitan §3.2 (chunking por párrafo) y §3.3 (completitud lingüística). Los
    seis loaders los emiten a propósito; si se pierden aquí, **no se recuperan
    nunca** y el chunker se queda sin señal para cortar.

    `join_lines` controla qué pasa con los saltos de línea SUELTOS. En prosa son
    el corte de línea por ancho de página —«…la seguridad\\nespacial…»— y unirlos
    con un espacio deja la oración entera en una sola línea, que es lo que
    facilita partir por oraciones después. En los formatos tabulares es al revés:
    cada `\\n` separa dos registros distintos y unirlos los fusionaría, así que
    ahí se llama con `join_lines=False`.
    """
    text = re.sub(r"[^\S\n]+", " ", text)     # espacios y tabs → un espacio (sin tocar \n)
    text = re.sub(r" *\n *", "\n", text)      # sin espacios pegados a los saltos
    if join_lines:
        # Un `\n` que no tiene otro `\n` ni delante ni detrás: es corte de línea,
        # no de párrafo. Los lookaround son lo que garantiza no tocar los dobles.
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)    # tres o más saltos → exactamente dos
    text = re.sub(r"[^\S\n]{2,}", " ", text)
    return text.strip()


# ═══════════════════════════════════════════════════════════════════════════
# Orquestación
# ═══════════════════════════════════════════════════════════════════════════

def clean_text(text: str, *, tabular: bool = False) -> str:
    """Aplica los pasos de limpieza, en orden.

    **El orden no es negociable**, porque cada paso depende de que el anterior
    ya haya corrido:

    0. `decode_cid`             el primero de todos, porque RECONSTRUYE el texto:
                                mientras las palabras están escritas como
                                `(cid:80)(cid:85)`, ningún paso posterior puede
                                reconocer una palabra, una oración ni un
                                n-grama repetido. Todo lo demás opera sobre el
                                texto ya reconstruido.
    1. `normalize_unicode`      justo después, para que pase también por NFC lo
                                que acaba de producir el paso 0. Y antes que
                                todo lo demás: si `U+2029` no se convierte aquí,
                                el paso 6 nunca lo verá como salto.
    2. `strip_control_chars`    después de normalizar y antes de tocar espacios:
                                borra el guion suave que estorbaría al paso 3.
    3. `dehyphenate`            ANTES del paso 6: necesita los `\\n` intactos
                                para saber dónde se cortó la línea.
    4. `strip_spaced_letter_runs` ANTES del paso 6: si se colapsan los espacios
                                primero, el patrón «G L O B A L» sigue igual,
                                pero si se unieran las líneas se partiría en dos.
    5. `strip_repeated_boilerplate` antes de colapsar, para que los n-gramas se
                                cuenten sobre el texto ya sin basura de control
                                (si no, dos copias de la misma cabecera con
                                distintos bytes nulos contarían como distintas).
    5b. `strip_anonymous_columns` **solo en tabulares**, y antes del paso 6:
                                necesita que cada par «columna: valor» siga
                                estando en su propia línea para poder anclar
                                con `^` y `$`.
    6. `collapse_whitespace`    el último: es el único que puede destruir
                                información que los otros necesitan.

    `tabular` intercambia los pasos que dependen del formato:

    - **desactiva** la detección de boilerplate (una tabla repite valores por
      naturaleza, y el umbral la masacraría) y la unión de líneas (cada línea
      es un registro, unirlas fusionaría registros distintos);
    - **activa** `strip_anonymous_columns()`, que solo tiene sentido aquí.
    """
    text = decode_cid(text)
    text = normalize_unicode(text)
    text = strip_control_chars(text)
    text = dehyphenate(text)
    text = strip_spaced_letter_runs(text)
    if tabular:
        # Solo tiene sentido en tabulares: `columna_N` lo genera
        # `tabular_loader._clean_header()` y no aparece en ningún otro formato.
        text = strip_anonymous_columns(text)
    else:
        text = strip_repeated_boilerplate(text)
    return collapse_whitespace(text, join_lines=not tabular)


def clean_document(document: Document) -> Document:
    """Devuelve una COPIA del Document con el `text` limpio.

    No modifica el original a propósito: así se puede comparar el antes y el
    después en el diagnóstico de abajo, y una corrida de limpieza nunca deja el
    `documentos.jsonl` en un estado a medias si algo revienta a la mitad.

    En `metadata` queda anotado cuánto se recortó. No es decorativo: es lo que
    permite detectar de un vistazo si el cleaner se pasó de agresivo con algún
    documento, sin tener que volver a leer los 1826.
    """
    tabular = document.format in TABULAR_FORMATS
    before = len(document.text)
    cleaned = clean_text(document.text, tabular=tabular)

    # `dataclasses.replace` no vale aquí porque queremos además tocar `metadata`;
    # se copia el dict para no compartirlo con el Document original.
    metadata = dict(document.metadata)
    metadata["chars_before_cleaning"] = before
    metadata["chars_removed"] = before - len(cleaned)

    # Se anota cuántos marcadores (cid:N) traía, para poder auditar después qué
    # documentos pasaron por la reconstrucción del paso 0 sin volver a leer el
    # corpus crudo. La comprobación de subcadena va primero para que los 1808
    # documentos sanos no paguen una pasada de expresión regular.
    if "(cid:" in document.text:
        metadata["cid_markers"] = sum(1 for _ in CID.finditer(document.text))

    return Document(
        doc_id=document.doc_id,
        source=document.source,
        format=document.format,
        text=cleaned,
        phenomenon=document.phenomenon,
        title=document.title,
        language=document.language,
        metadata=metadata,
    )


if __name__ == "__main__":
    # Diagnóstico: `py -m preprocess.cleaner` desde src/.
    # Limpia todo el corpus en memoria y resume qué se fue, SIN escribir nada.
    import sys
    from pathlib import Path

    from core.store import read_documents

    ruta = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).resolve().parents[1] / "data" / "documentos.jsonl"
    )

    total_before = total_after = 0
    por_formato: Counter = Counter()
    palabras_formato: Counter = Counter()
    peores: list = []
    sin_texto = 0

    for documento in read_documents(ruta):
        limpio = clean_document(documento)
        antes, despues = len(documento.text), len(limpio.text)
        total_before += antes
        total_after += despues
        por_formato[documento.format] += antes - despues
        palabras_formato[documento.format] += len(limpio.text.split())
        if not limpio.text.strip():
            sin_texto += 1
        if antes > 2000:
            peores.append(((antes - despues) / antes, antes, despues, documento.doc_id,
                           documento.source.split("/")[-1]))

    print(f"{ruta}\n")
    print(f"caracteres antes  : {total_before:,}")
    print(f"caracteres después: {total_after:,}")
    print(f"eliminado         : {total_before - total_after:,} "
          f"({100 * (total_before - total_after) / total_before:.2f}%)")
    print(f"documentos que quedan sin texto: {sin_texto}")

    print("\ncaracteres eliminados por formato:")
    for formato, n in por_formato.most_common():
        print(f"  {formato:6} {n:12,}")

    print("\nlos 20 documentos con mayor reducción (>2000 chars originales):")
    peores.sort(reverse=True)
    for ratio, antes, despues, doc_id, nombre in peores[:20]:
        print(f"  {ratio:6.1%}  {antes:9,} → {despues:9,}  {doc_id:18} {nombre[:50]}")
