"""Chunker (§3): parte los documentos limpios en fragmentos que se puedan codificar.

    py -m chunking.chunker              # los 1826 → data/chunks.jsonl
    py -m chunking.chunker 50           # muestra: 1 de cada 50, sin escribir

────────────────────────────────────────────────────────────────────────────────
QUÉ RESUELVE Y POR QUÉ ASÍ

El chunking tiene que satisfacer dos exigencias que tiran en direcciones
opuestas:

  1. **§4.3** — el encoder tiene un tope de tokens (comúnmente 512), así que
     ningún fragmento puede pasarse de ese presupuesto o el modelo lo trunca en
     silencio: se codifica el principio y el resto deja de existir para la
     recuperación, sin un solo mensaje de error.
  2. **§3.3, «Requisito obligatorio»** (enmarcado en el PDF, no es una
     sugerencia) — «Ningún fragmento puede contener oraciones o frases
     incompletas… Una oración que comienza en un chunk debe terminar en ese
     mismo chunk». Y dice cómo cumplirlo: «si se fija un tamaño máximo de n
     tokens, el corte efectivo debe **retroceder al final de la última oración
     completa** que quepa dentro de ese límite».

La estrategia elegida es **híbrida: agrupación por párrafo con retroceso a
límite de oración**, que §3.2 autoriza expresamente («los equipos pueden
diseñar estrategias híbridas que combinen elementos de las anteriores. Lo que
se exige es que la estrategia elegida se justifique explícitamente en el
documento técnico entregable»).

────────────────────────────────────────────────────────────────────────────────
LA CASCADA, Y LA MEDICIÓN QUE LA JUSTIFICA (2026-08-06, sobre los 1826 limpios)

Medido con un presupuesto de 350 palabras:

    prosa    57.613 bloques \\n\\n → 17.218 (29,89%) YA no caben
    tabular 276.925 filas        →      83 ( 0,03%) YA no caben

Casi tres de cada diez párrafos de prosa se pasan por sí solos. Es el dato que
descarta la estrategia «solo por párrafo»: dejaría 17.218 fragmentos pasados de
tamaño para que el encoder los truncara. De ahí la cascada, que se aplica por
unidad y solo baja de nivel cuando el nivel anterior no consigue que quepa:

    nivel 1  agrupar bloques `\\n\\n` hasta el presupuesto
    nivel 2  partir el bloque en ORACIONES  ..................... §3.3 literal
    nivel 3  partir por separadores secundarios (`;` `:` puntos guía, `\\n`)
    nivel 4  corte por longitud — último recurso, se anota en la metadata

**Los niveles 3 y 4 casi no se usan, y eso está medido, no supuesto.** Las
unidades que ni siquiera un cortador de oraciones puede partir son ~893 en 157
documentos (3,02% de la prosa) y **casi ninguna es prosa**: son tablas
presupuestales extraídas en línea recta (`F2-CSIS-156`, `F2-CSIS-173`), el
Atlas comparativo de RESDAL (`F3-RESDAL-032`) y un PDF en árabe con índice de
puntos guía (`F2-UNOOSA-025`). El nivel 4 no rompe ninguna oración porque en
esos casos no hay ninguna que romper — son celdas de tabla.

────────────────────────────────────────────────────────────────────────────────
POR QUÉ NO SE USA `RecursiveCharacterTextSplitter` DE LANGCHAIN

Se evaluó y se descartó (2026-08-06). El concepto de cascada es correcto; esa
implementación concreta no sirve aquí:

  - Su lista de separadores por defecto es `["\\n\\n", "\\n", " ", ""]`, **sin el
    punto**: no intenta mantener oraciones juntas, así que tal cual viene
    **incumple §3.3**, que es requisito obligatorio.
  - Aunque se le añada `"."`, sus últimos recursos siguen siendo `" "` y `""`,
    que cortan por espacio y por carácter suelto — falla justo en los casos
    donde hace falta criterio, y partiendo palabras por la mitad.
  - Cuenta **caracteres**, no tokens, y el límite de §4.3 son tokens.
  - No distingue prosa de tabular, y el 52,1% del corpus son filas que no se
    pueden partir (`ESTADO.md` §12).
  - Añade una dependencia a una entrega que §1.4 juzga por reproducibilidad,
    para resolver lo que aquí son unas decenas de líneas.

────────────────────────────────────────────────────────────────────────────────
DOS CAMINOS SEGÚN EL FORMATO

  - **Prosa** (pdf, json, html, txt, jpg…): los `\\n\\n` son fronteras de párrafo
    reales, verificadas sobre los 1826 en la prueba A del cleaner. Además se
    comprobó que no hay separadores Unicode invisibles (U+2028/U+2029: 0 en
    todo el corpus) que las falseen.
  - **Tabular** (csv, xlsx, pbf): los bloques `\\n\\n` son **filas**, y una fila
    no se parte nunca. Se agrupan filas enteras hasta el presupuesto — decisión
    de `ESTADO.md` §4 y §12: §2.1 dice que cada fila «**puede**» ser una unidad
    de fragmentación, no que «deba», y agrupar baja los chunks tabulares un
    83,6% sin perder una sola palabra. Aquí §3.3 se cumple sola: una fila es
    una unidad cerrada.

────────────────────────────────────────────────────────────────────────────────
EL PRESUPUESTO SE INYECTA, NO SE CABLEA

`num_tokens` depende del tokenizador del encoder, que se elige en la Fase 5.
Por eso `chunk_document()` recibe **la función que cuenta tokens** como
parámetro, con un contador de palabras como valor por defecto.

⚠️ Lo que NO se debe hacer es cablear un `palabras × 1,3`: la relación
token/palabra cambia entre español, inglés y portugués, y con un tope de 512
los fragmentos en español se pasarían. Re-chunkear cuesta minutos (no las ~4 h
de la extracción), así que se corre ahora con el contador de palabras para
validar la cascada y se vuelve a correr con el tokenizador real cuando exista.
"""

import re
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, List, NamedTuple

from core.chunk import Chunk
from core.document import Document
from core.store import read_documents, write_chunks

DATOS = Path(__file__).resolve().parents[1] / "data"
DOCUMENTOS_LIMPIOS = DATOS / "documentos_limpios.jsonl"
CHUNKS = DATOS / "chunks.jsonl"

# Formatos cuyo texto son filas y no prosa. Sale de `ESTADO.md` §12: 103
# documentos = 30 CSV/XLSX + 73 PBF, el 52,1% de las palabras del corpus.
FORMATOS_TABULARES = frozenset({"csv", "xlsx", "pbf"})

# Presupuesto por defecto, EN PALABRAS, porque el contador por defecto cuenta
# palabras. Es el mismo proxy de 350 que usó `ESTADO.md` §12 para estimar el
# tamaño del índice. Con el tokenizador real se pasa `presupuesto=512` (§4.3).
PRESUPUESTO_POR_DEFECTO = 350

# Un contador de tokens es cualquier función texto → entero. Declararlo como
# tipo hace explícito que el encoder se enchufa aquí y en ningún otro sitio.
ContadorTokens = Callable[[str], int]


def contar_palabras(texto: str) -> int:
    """Contador por defecto: palabras separadas por espacios.

    NO es el contador definitivo. Sirve para validar la cascada antes de que
    exista el encoder, y para que el chunker se pueda ejecutar y verificar sin
    descargar un modelo de 2 GB.
    """
    return len(texto.split())


# ══════════════════════════════════════════════════════════════════════════════
# Nivel 2 — corte por oración
# ══════════════════════════════════════════════════════════════════════════════

# Caracteres que cierran una oración. Se incluyen los de escritura no latina
# porque el corpus los tiene: 9 documentos en chino, 5 en árabe, 2 en japonés
# y 2 en coreano (`ESTADO.md` §11).
FIN_ORACION = ".!?…。！？؟।"

# Comillas y paréntesis que pueden ir DESPUÉS del punto y siguen siendo parte
# de la misma oración: «dijo que sí.»  /  (ver nota 3.)
CIERRES = '")]»”’\'›〉】'

# Abreviaturas tras las que un punto NO termina la oración. Sin esta lista,
# «Art. 31» o «et al. 2022» producirían fragmentos partidos a media frase, que
# es exactamente lo que §3.3 prohíbe. Están las de los tres idiomas del corpus
# más las de citación académica, que abundan en los informes de think tanks.
ABREVIATURAS = frozenset("""
sr sra srta dr dra prof profa ing lic mr mrs ms jr sr dept univ inc ltd corp
gen col lt cap cmdr adm sgt art arts cap caps vol vols fig figs tab tabs
núm num nro ed eds pp pág págs pag pags ver vs etc aprox ej cf al
ee uu ss aa av avda apdo tel
""".split())

# Divisores del nivel 3, en orden de preferencia: primero el más «semántico».
# Se cortan DESPUÉS del separador para no perderlo.
#   - puntos guía (`......`) → los índices y tablas de los PDF están llenos
#   - `;` y `:`              → fronteras naturales de cláusula
#   - salto de línea simple  → en tabular separa `columna: valor`
SECUNDARIOS = (
    re.compile(r"(?<=\.)\s*(?=\S)"),          # tras un punto sin espacio detrás
    re.compile(r"(?<=;)\s*(?=\S)"),
    re.compile(r"(?<=:)\s*(?=\S)"),
    re.compile(r"\n"),
)


def _inicia_oracion(caracter: str) -> bool:
    """¿Este carácter puede empezar una oración nueva?

    El criterio es «**no** es minúscula», y no «es mayúscula», a propósito. Con
    «es mayúscula» el corte no funcionaría en árabe, chino, japonés ni coreano,
    que no tienen caja — y el corpus tiene 18 documentos en esos idiomas. Con
    «no es minúscula» esos idiomas cortan bien y se sigue rechazando el caso
    que importa rechazar: «e.g. foo», «art. 31», «vs. china».
    """
    return not caracter.islower()


def _termina_oracion(texto: str, posicion: int) -> bool:
    """¿El texto justo antes de `posicion` cierra una oración?

    Retrocede saltando comillas y paréntesis de cierre, porque `dijo que sí.»`
    termina oración igual que `dijo que sí.`. Después comprueba que la última
    palabra no sea una abreviatura conocida.
    """
    indice = posicion - 1
    while indice >= 0 and texto[indice] in CIERRES:
        indice -= 1
    if indice < 0 or texto[indice] not in FIN_ORACION:
        return False

    # Un punto tras una abreviatura no cierra oración. Se mira la última
    # "palabra" antes del punto, en minúsculas y sin puntos internos, para que
    # «et al.» y «EE.UU.» caigan en la lista igual que «etc.».
    if texto[indice] == ".":
        inicio = indice
        while inicio > 0 and (texto[inicio - 1].isalpha() or texto[inicio - 1] == "."):
            inicio -= 1
        palabra = texto[inicio:indice].replace(".", "").lower()
        if palabra in ABREVIATURAS:
            return False
        # Una sola letra antes del punto es casi siempre una inicial («J. Smith»).
        if len(palabra) == 1:
            return False
    return True


def dividir_en_oraciones(texto: str) -> List[str]:
    """Parte un texto en oraciones sin perder ni un carácter que no sea espacio.

    Solo corta en posiciones donde ya hay espacio en blanco, y el espacio se
    descarta. Esa propiedad es la que permite que la prueba de cobertura del
    verificador compare carácter a carácter: si se concatenan todos los
    fragmentos de un documento y se quitan los espacios, tiene que salir
    exactamente el texto original sin espacios.

    No usa `re.split` con lookbehind porque el contexto que hay que mirar hacia
    atrás es de ancho variable (el punto puede venir seguido de una o varias
    comillas de cierre) y `re` solo admite lookbehind de ancho fijo.
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
    """Recompone las oraciones que el extractor partió en dos bloques.

    ⚠️ **Esta función corrige una suposición falsa que costó un fallo del
    verificador** (2026-08-06). Se había dado por hecho que un `\\n\\n` era
    siempre una frontera de párrafo y, por tanto, de oración. No lo es: los
    loaders emiten `\\n\\n` por cada bloque de texto que devuelve el extractor
    del PDF, y esos bloques se cortan también en los **saltos de columna y de
    página**. Resultado: una oración empieza en un bloque y termina en el
    siguiente, y agrupar bloques enteros la partía — justo lo que §3.3 prohíbe.

    El criterio para unir es doble, y las dos condiciones tienen que darse:

      1. el bloque **no** termina en puntuación de cierre, y
      2. el siguiente **empieza en minúscula**.

    Con las dos se distingue una oración partida de verdad («…the AI mod» +
    «els were…») de un encabezado o pie de figura, que legítimamente no lleva
    punto final pero va seguido de algo que empieza en mayúscula («Figure 1.2.1»
    + «Number of AI patents…»). Unir esos dos también habría sido un error: son
    unidades distintas del documento.

    Solo se aplica a prosa. En tabular los `\\n\\n` son filas y ninguna termina
    en punto: unirlas fundiría el corpus tabular en un solo bloque.
    """
    if not bloques:
        return []
    unidos = [bloques[0]]
    for bloque in bloques[1:]:
        anterior = unidos[-1]
        continua = (not _termina_oracion(anterior, len(anterior))
                    and bloque[:1].islower())
        if continua:
            # Se une con un espacio, no con `\n\n`: es la misma oración, y así
            # el texto del fragmento se lee como lo escribió el autor.
            unidos[-1] = f"{anterior} {bloque}"
        else:
            unidos.append(bloque)
    return unidos


def _dividir_por_patron(texto: str, patron: re.Pattern) -> List[str]:
    """Corta por un separador del nivel 3, conservando el separador a la izquierda."""
    partes = [p.strip() for p in patron.split(texto)]
    return [p for p in partes if p]


def _corte_duro(texto: str, contar: ContadorTokens, presupuesto: int) -> List[str]:
    """Nivel 4: corta por longitud. Último recurso.

    Solo se llega aquí cuando una unidad no cabe y ningún separador la puede
    partir. Medido: ~25 casos en todo el corpus, y son celdas de tabla, no
    prosa — así que este corte no rompe ninguna oración, porque no hay ninguna.

    Corta entre palabras, nunca dentro de una: partir una palabra la haría
    inencontrable, que es el mismo daño que ya causó el bug del `\\x07`
    (`ESTADO.md` §11).
    """
    partes: List[str] = []
    actual: List[str] = []
    suma = 0
    for palabra in texto.split():
        # Se cuenta cada palabra por separado y se acumula, en vez de contar el
        # texto unido en cada vuelta. Dos motivos: unir cadenas cada vez es
        # cuadrático, y con el tokenizador real serían miles de llamadas sobre
        # textos cada vez más largos. La suma de las partes sobreestima el total
        # (un tokenizador de subpalabras fusiona tokens en los bordes), así que
        # el resultado nunca se pasa del presupuesto.
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
    """Un trozo indivisible de texto, con lo que hace falta para reagruparlo."""

    texto: str
    separador: str   # lo que va delante al unirlo con la unidad anterior
    nivel: int       # 1 párrafo/fila · 2 oración · 3 secundario · 4 corte duro
    tokens: int


def _atomizar(
    texto: str,
    separador: str,
    contar: ContadorTokens,
    presupuesto: int,
    nivel: int = 1,
) -> List[Unidad]:
    """La cascada, escrita como una recursión sobre los niveles.

    Es literalmente «el método recursivo», pero con una cascada nuestra cuyo
    último recurso está medido y documentado, en vez de la de una librería que
    corta por espacio y por carácter suelto.

    Si el texto cabe, es una unidad y se acabó. Si no cabe, se baja un nivel y
    se vuelve a intentar con cada trozo.
    """
    tokens = contar(texto)
    if tokens <= presupuesto:
        return [Unidad(texto, separador, nivel, tokens)]

    if nivel == 1:
        trozos = dividir_en_oraciones(texto)
        # Si el bloque era una sola oración, no hay nada que ganar repitiendo:
        # se baja directo al nivel 3.
        if len(trozos) > 1:
            return _concatenar(trozos, separador, " ", contar, presupuesto, 2)
        return _atomizar(texto, separador, contar, presupuesto, 2)

    if nivel == 2:
        for patron in SECUNDARIOS:
            trozos = _dividir_por_patron(texto, patron)
            if len(trozos) > 1:
                return _concatenar(trozos, separador, " ", contar, presupuesto, 3)
        return _atomizar(texto, separador, contar, presupuesto, 3)

    # Nivel 3 agotado: corte duro. Es TERMINAL — de aquí no se baja más.
    # Si volviera a llamarse a sí misma, una sola «palabra» más larga que el
    # presupuesto (las mega-palabras del bug del `\x07`, `ESTADO.md` §11) haría
    # que `_corte_duro` devolviera siempre un único trozo idéntico al de
    # entrada, y la recursión no terminaría nunca.
    trozos = _corte_duro(texto, contar, presupuesto)
    if len(trozos) <= 1:
        # Irreducible: se acepta pasado de tamaño y se marca como nivel 4 para
        # que el verificador lo cuente y quede a la vista, en vez de partir una
        # palabra por la mitad y hacerla inencontrable.
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
    """Atomiza cada trozo y devuelve la lista completa, en orden."""
    unidades: List[Unidad] = []
    for indice, trozo in enumerate(trozos):
        sep = separador_inicial if indice == 0 else separador_interno
        unidades.extend(_atomizar(trozo, sep, contar, presupuesto, nivel))
    return unidades


# ══════════════════════════════════════════════════════════════════════════════
# Nivel 1 — agrupación
# ══════════════════════════════════════════════════════════════════════════════

def _agrupar(unidades: List[Unidad], contar: ContadorTokens, presupuesto: int):
    """Junta unidades consecutivas mientras quepan, y va soltando fragmentos.

    Es un generador (`yield`): no construye la lista entera en memoria, lo que
    importa con documentos como `F1-AIINDEX-056`, que tiene 111.775 filas.

    La suma de los tokens de las unidades es una **sobreestimación** del total
    real —al unir dos textos un tokenizador de subpalabras puede fusionar
    tokens del borde—, así que agrupar por esa suma nunca se pasa del
    presupuesto. El `num_tokens` definitivo sí se vuelve a medir sobre el texto
    ya unido, en `chunk_document()`.
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
    """Parte un documento en fragmentos. Es la función principal del módulo.

    Devuelve una lista de `Chunk` con `position` consecutiva empezando en 0
    (Tabla 1) y `chunk_id` determinista (§1.4). Un documento sin texto devuelve
    lista vacía: son los 8 legítimos de `ESTADO.md` §9 (fotos sin texto y
    manifiestos de scraping), y un vector vacío no recupera nada.

    Los argumentos van por nombre a propósito: `chunk_document(doc, tok, 512)`
    no dice cuál es cuál.
    """
    if not documento.text.strip():
        return []

    es_tabular = documento.format in FORMATOS_TABULARES

    # Los bloques `\n\n` son párrafos en prosa y filas en tabular. La frontera
    # es la misma; lo que cambia es qué significa y hasta dónde se puede partir.
    bloques = [b.strip() for b in documento.text.split("\n\n") if b.strip()]
    if not es_tabular:
        # Recompone las oraciones partidas por saltos de columna o de página.
        # Sin esto, agrupar bloques enteros viola §3.3 — está explicado en el
        # docstring de la función y lo detectó la prueba A del verificador.
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
        # el que tiene que respetar el tope del encoder.
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
    """Recorre el corpus limpio y va produciendo fragmentos, documento a documento.

    Generador, por lo mismo que `_agrupar`: el corpus limpio son 208 MB y los
    fragmentos estimados ~84.000. No hay razón para tenerlos todos en memoria.
    """
    for indice, documento in enumerate(read_documents(ruta)):
        if indice % paso:
            continue
        yield from chunk_document(
            documento, contar_tokens=contar_tokens, presupuesto=presupuesto
        )


if __name__ == "__main__":
    from collections import Counter

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    argumentos = sys.argv[1:]

    # `--tokens` cambia el contador de palabras por el tokenizador real del
    # encoder y el presupuesto a 512 (§4.3). Es la corrida definitiva: la de
    # por defecto solo sirve para validar la cascada sin descargar el modelo.
    contar = contar_palabras
    presupuesto = PRESUPUESTO_POR_DEFECTO
    etiqueta = "palabras — proxy, ver la cabecera del módulo"
    if "--tokens" in argumentos:
        argumentos.remove("--tokens")
        from embedding.encoder import MAX_TOKENS, cargar_modelo, contador_de_tokens

        print("cargando el tokenizador del encoder…")
        contar = contador_de_tokens(cargar_modelo())
        presupuesto = MAX_TOKENS
        etiqueta = "tokens reales del encoder"

    paso = int(argumentos[0]) if argumentos else 1
    escribir = paso == 1      # con muestreo no se pisa el archivo bueno

    if not DOCUMENTOS_LIMPIOS.is_file():
        print(f"No existe {DOCUMENTOS_LIMPIOS}. Corre antes `py -m preprocess.runner`.")
        raise SystemExit(1)

    print(f"presupuesto: {presupuesto} (contador: {etiqueta})")
    print(f"origen     : {DOCUMENTOS_LIMPIOS.name}"
          + (f"   (1 de cada {paso} documentos)" if paso > 1 else ""))

    inicio = time.perf_counter()
    por_nivel: Counter = Counter()
    por_grupo: Counter = Counter()
    tokens: List[int] = []
    docs_con_chunks = set()
    docs_sin_chunks = []
    pasados = 0

    def recorrer():
        """Envuelve el generador para ir midiendo sin duplicar el bucle."""
        for indice, documento in enumerate(read_documents(DOCUMENTOS_LIMPIOS)):
            if indice % paso:
                continue
            fragmentos = chunk_document(
                documento, contar_tokens=contar, presupuesto=presupuesto
            )
            if fragmentos:
                docs_con_chunks.add(documento.doc_id)
            elif documento.text.strip():
                docs_sin_chunks.append(documento.doc_id)
            for fragmento in fragmentos:
                por_nivel[fragmento.metadata["nivel_corte"]] += 1
                por_grupo[fragmento.metadata["estrategia"]] += 1
                tokens.append(fragmento.num_tokens)
                yield fragmento
            if len(docs_con_chunks) % 200 == 0 and fragmentos:
                print(f"  … {len(docs_con_chunks)} documentos, {len(tokens):,} fragmentos",
                      file=sys.stderr, flush=True)

    if escribir:
        total = write_chunks(CHUNKS, recorrer())
    else:
        total = sum(1 for _ in recorrer())

    pasados = sum(1 for t in tokens if t > presupuesto)
    segundos = time.perf_counter() - inicio
    tokens.sort()

    linea = "─" * 74
    print(f"\n{linea}")
    print(f"fragmentos          : {total:,}")
    print(f"documentos con chunk: {len(docs_con_chunks):,}")
    print(f"documentos con texto y SIN chunk: {len(docs_sin_chunks)}"
          + (f"  {docs_sin_chunks[:10]}" if docs_sin_chunks else "  ✔"))
    print(f"tiempo              : {segundos:,.1f} s")
    if tokens:
        print(f"tokens por fragmento: mediana {tokens[len(tokens)//2]:,}  "
              f"min {tokens[0]}  max {tokens[-1]:,}")
        print(f"fragmentos por encima del presupuesto: {pasados}"
              + ("  ✔" if not pasados else "  ✖ revisar"))
    print(f"por estrategia      : {dict(por_grupo)}")
    print("por nivel de la cascada:")
    nombres = {1: "1 párrafo/fila", 2: "2 oración", 3: "3 secundario", 4: "4 corte duro"}
    for nivel in sorted(por_nivel):
        n = por_nivel[nivel]
        print(f"   {nombres[nivel]:<16} {n:>9,}  ({100*n/total:5.2f}%)")
    if escribir:
        print(f"\nescrito en {CHUNKS}")
    print(linea)
