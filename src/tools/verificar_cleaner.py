"""Verificación del cleaner: ¿se fue la basura sin llevarse la información?

`cleaner.py` borra el 100% de lo que decide borrar y no avisa. Sus errores no
producen excepciones: producen un índice peor, y eso no se nota hasta que las
métricas de §10 ya están calculadas y es tarde. Esta herramienta es el
contrapeso: corre las cuatro comprobaciones que sí pueden fallar en silencio y
da un veredicto.

    py -m tools.verificar_cleaner            # sobre los 1826 documentos
    py -m tools.verificar_cleaner 200        # muestra rápida: 1 de cada 200

LAS CUATRO PRUEBAS, Y QUÉ DESASTRE PREVIENE CADA UNA:

  A. PÁRRAFOS — que los `\\n\\n` sobrevivan.
     §3.2 ofrece el chunking «Por párrafo»: «Se respetan los saltos de párrafo
     del texto original como unidades de segmentación». Si `collapse_whitespace()`
     aplasta los dobles saltos, esa estrategia deja de estar disponible **para
     siempre**: la frontera no se puede reconstruir sin volver a extraer los
     1826 archivos. Falla si un documento que tenía párrafos queda como un
     bloque único.

  B. VOLUMEN — que no se borre de más.
     Falla si algún documento con texto queda vacío. Y lista los que más
     porcentaje pierden: cada uno de esos tiene que tener una explicación
     conocida (marca de agua, puntos guía del índice, letter-spacing). Uno
     perdiendo el 60% sin explicación es un bug.

  C. TÉRMINOS — que ningún documento pierda un término de las consultas.
     La prueba que de verdad protege la recuperación. Se sacan los términos de
     contenido de las **50 preguntas reales** (más sus equivalentes en inglés y
     portugués, ver `TERMINOS_EXTRA`) y se comprueba, documento por documento,
     que ninguno que contenía un término lo pierda del todo.
     ⚠️ El criterio NO es que el número TOTAL de apariciones no baje: eso
     bajaría legítimamente, porque hay documentos con el título repetido en el
     pie de cada página (`F1-ILIA-005`, «Índice Latinoamericano de Inteligencia
     Artificial» ×120) y quitar 119 copias es justo lo que queremos. El
     criterio es la **cobertura**: si un documento hablaba de un término, tiene
     que seguir siendo encontrable por él. Es exactamente lo que garantiza la
     salvaguarda de `strip_repeated_boilerplate()` de conservar la primera
     aparición, y esta prueba es lo que comprueba que esa salvaguarda funciona.

  D. NORMALIZACIÓN — que §2.2 se cumpla de verdad en la salida.
     Comprueba sobre el texto YA limpio que no queda ni un carácter de control,
     ni un `U+FFFD`, ni nada del Área de Uso Privado, y que todo está en NFC.
     Es barata y cierra los puntos 1 y 2 de §2.2 con una medición en vez de con
     un «debería».

No escribe nada ni toca el corpus: lee `documentos.jsonl`, limpia en memoria y
compara. Para revisar con los ojos lo que se fue, la herramienta es otra:
`py -m tools.diff_cleaner`.
"""

import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from core.store import read_documents
from preprocess.cleaner import CONTROL_CHARS, ZERO_WIDTH, clean_document

DATOS = Path(__file__).resolve().parents[1] / "data"
DOCUMENTOS = DATOS / "documentos.jsonl"
PREGUNTAS = DATOS / "Extracto_Preguntas_50_v2.pdf"

# ── Prueba A ────────────────────────────────────────────────────────────────
# Un documento con al menos estos párrafos SÍ tiene estructura que perder. Por
# debajo del umbral no se juzga: un JSON de tres líneas puede quedar en un solo
# párrafo sin que eso indique nada malo.
MIN_PARRAFOS_PARA_JUZGAR = 10

# ── Prueba C ────────────────────────────────────────────────────────────────
# Cuántos términos de las consultas se vigilan, por frecuencia en las 50
# preguntas. Con más, la prueba tarda más y añade palabras cada vez menos
# discriminantes.
N_TERMINOS = 40

# Palabras vacías de los tres idiomas del corpus (§10.1: es, en y pt). No es
# una lista exhaustiva ni pretende serlo: solo tiene que quitar el ruido de la
# cabeza del ranking de frecuencias para que suban los términos con contenido.
VACIAS = frozenset("""
como esta estan que para los las del con una unos unas por sobre entre sus
cual cuales cuando donde quien quienes tiene tienen puede pueden hacia desde
ante bajo cabe contra hasta segun sin tras han haber sido ser estar mas menos
muy tambien pero porque sino aunque mientras cada todo toda todos todas otro
otra otros otras mismo misma cuanto cuanta cuantos cuantas qué cómo cuál
cuáles según están está más
the and for that with this from what which their they have has been are was
were will can could should would about into over under between within
of in on at to by as it its is be or not no if then than
como para por com uma dos das nos nas seu sua seus suas mais menos onde quem
quais quando porque tambem sao esta estao pelo pela pelos pelas
""".split())


# Términos añadidos a mano, y por qué hace falta añadirlos: **las 50 consultas
# están TODAS en español** (comprobado leyendo el PDF entero, pese a que §10.1
# dice que se reparten entre los tres idiomas del corpus), mientras que la mayor
# parte del corpus está en inglés y una porción en portugués. Sin esta lista, la
# prueba C solo vigilaría los documentos en español y quedaría ciega justo donde
# hay más texto.
#
# Son la traducción de los conceptos que las propias consultas repiten —IA,
# satélites, órbita, armas, grupos armados, territorio—, no una lista inventada.
TERMINOS_EXTRA = (
    # inglés
    "artificial", "intelligence", "machine", "learning", "autonomous", "drone",
    "drones", "military", "defense", "defence", "weapon", "weapons", "satellite",
    "satellites", "orbit", "orbital", "debris", "space", "counterspace",
    "jamming", "spoofing", "security", "territory", "territorial", "conflict",
    "armed", "governance", "deforestation", "mining", "trafficking",
    # portugués
    "inteligência", "artificial", "satélite", "satélites", "órbita", "espacial",
    "segurança", "militar", "território", "conflito", "armados", "mineração",
    "desmatamento", "amazônia",
)


def cargar_terminos(ruta_pdf: Path) -> list:
    """Saca los términos de contenido de las 50 consultas del PDF de preguntas.

    Se sacan de las preguntas reales y no de una lista escrita a mano porque el
    vocabulario que importa proteger es exactamente el que va a llegar en las
    consultas — cualquier otra lista sería una suposición nuestra.

    Devuelve los `N_TERMINOS` más frecuentes, en minúsculas y sin las palabras
    vacías. Las siglas en mayúsculas (IA, LEO, NBQR, DIH) se conservan aunque
    sean cortas: son los términos MÁS discriminantes del corpus y una regla de
    longitud mínima las tiraría todas.
    """
    from pypdf import PdfReader

    texto = "\n".join((pagina.extract_text() or "") for pagina in PdfReader(ruta_pdf).pages)

    # Las siglas se buscan ANTES de pasar a minúsculas: es la única forma de
    # distinguir «IA» (sigla) de «ia» (final de palabra en «vigencia»).
    siglas = {s.lower() for s in re.findall(r"\b[A-ZÁÉÍÓÚÑ]{2,6}\b", texto)}

    # `[^\W\d_]` = letra Unicode; el `{4,}` deja fuera artículos y preposiciones
    # que no estén en VACIAS.
    palabras = [p.lower() for p in re.findall(r"[^\W\d_]{4,}", texto)]
    frecuencias = Counter(p for p in palabras if p not in VACIAS)

    terminos = [t for t, _ in frecuencias.most_common(N_TERMINOS)]
    for sigla in sorted(siglas):
        if sigla not in terminos and sigla not in VACIAS:
            terminos.append(sigla)

    # `dict.fromkeys` como "set que conserva el orden": quita los duplicados
    # entre las tres fuentes sin desordenar la lista, que es lo que hace legible
    # la tabla del informe. Un `set()` la barajaría en cada corrida.
    return list(dict.fromkeys(terminos + list(TERMINOS_EXTRA)))


def compilar_buscador(terminos: list) -> re.Pattern:
    """Compila TODOS los términos en una sola expresión regular con alternativas.

    La alternativa ingenua —un `re.findall()` por término y por documento— sería
    40 pasadas sobre cada texto, y hay documentos de seis millones de palabras.
    Con una sola expresión se hace **una** pasada por texto y se identifica cada
    coincidencia por lo que casó.

    Los términos se ordenan de más largo a más corto porque `re` con `|` toma la
    PRIMERA alternativa que case, no la más larga: si «ia» fuera antes que
    «inteligencia» en la alternancia, nunca se contaría «inteligencia».
    """
    ordenados = sorted(terminos, key=len, reverse=True)
    patron = "|".join(re.escape(t) for t in ordenados)
    return re.compile(rf"\b(?:{patron})\b", re.IGNORECASE)


def contar_parrafos(texto: str) -> int:
    """Cuenta los párrafos, entendidos como bloques separados por línea en blanco."""
    return sum(1 for bloque in re.split(r"\n\s*\n", texto) if bloque.strip())


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # Argumento opcional: procesar 1 de cada N documentos, para una pasada
    # rápida mientras se itera sobre el cleaner. La muestra es sistemática y por
    # tanto determinista: dos corridas miran los mismos documentos.
    paso = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    if not DOCUMENTOS.is_file():
        print(f"No existe {DOCUMENTOS}. Corre antes el orquestador.")
        return 1

    terminos = cargar_terminos(PREGUNTAS)
    buscador = compilar_buscador(terminos)
    print(f"términos vigilados ({len(terminos)}), sacados de las 50 consultas:")
    print("  " + ", ".join(terminos) + "\n")

    # Acumuladores de las cuatro pruebas
    total = 0
    fallos_parrafos: list = []          # A
    vaciados: list = []                 # B
    reducciones: list = []              # B (informativo)
    docs_con_termino_antes: Counter = Counter()    # C
    docs_con_termino_despues: Counter = Counter()  # C
    apariciones_antes: Counter = Counter()         # C (informativo)
    apariciones_despues: Counter = Counter()       # C (informativo)
    perdidas: list = []                 # C
    restos_control: list = []           # D
    sin_nfc: list = []                  # D

    for indice, documento in enumerate(read_documents(DOCUMENTOS)):
        if indice % paso:
            continue
        total += 1
        limpio = clean_document(documento)
        antes, despues = documento.text, limpio.text

        # ── A. párrafos ────────────────────────────────────────────────────
        parrafos_antes = contar_parrafos(antes)
        parrafos_despues = contar_parrafos(despues)
        if parrafos_antes >= MIN_PARRAFOS_PARA_JUZGAR and parrafos_despues <= 1:
            fallos_parrafos.append((documento.doc_id, documento.format,
                                    parrafos_antes, parrafos_despues))

        # ── B. volumen ─────────────────────────────────────────────────────
        if antes.strip() and not despues.strip():
            vaciados.append((documento.doc_id, documento.format, len(antes)))
        if len(antes) > 2000:
            reducciones.append(((len(antes) - len(despues)) / len(antes),
                                len(antes), len(despues), documento.doc_id,
                                documento.source.split("/")[-1]))

        # ── C. términos ────────────────────────────────────────────────────
        # `set` para la cobertura (¿está o no está?) y `Counter` para el total
        # de apariciones, que es solo informativo.
        halladas_antes = [m.group().lower() for m in buscador.finditer(antes)]
        halladas_despues = [m.group().lower() for m in buscador.finditer(despues)]
        apariciones_antes.update(halladas_antes)
        apariciones_despues.update(halladas_despues)
        presentes_antes, presentes_despues = set(halladas_antes), set(halladas_despues)
        for termino in presentes_antes:
            docs_con_termino_antes[termino] += 1
        for termino in presentes_despues:
            docs_con_termino_despues[termino] += 1
        for termino in presentes_antes - presentes_despues:
            perdidas.append((documento.doc_id, documento.format, termino,
                             halladas_antes.count(termino)))

        # ── D. normalización ───────────────────────────────────────────────
        if CONTROL_CHARS.search(despues) or ZERO_WIDTH.search(despues):
            restos_control.append(documento.doc_id)
        if despues != unicodedata.normalize("NFC", despues):
            sin_nfc.append(documento.doc_id)

        if total % 200 == 0:
            print(f"  … {total} documentos", file=sys.stderr, flush=True)

    # ═══════════════════════════════════════════════════════════════════════
    # Informe
    # ═══════════════════════════════════════════════════════════════════════
    linea = "─" * 74
    print(f"\n{linea}\ndocumentos examinados: {total}"
          + (f"  (1 de cada {paso})" if paso > 1 else ""))

    print(f"\n{linea}\nA. PÁRRAFOS — que los saltos dobles sobrevivan (§3.2)")
    if fallos_parrafos:
        print(f"   ✖ FALLA: {len(fallos_parrafos)} documentos quedaron como bloque único")
        for doc_id, formato, pa, pd in fallos_parrafos[:20]:
            print(f"     {doc_id:20} {formato:5} {pa:6} párrafos → {pd}")
    else:
        print("   ✔ PASA: ningún documento con estructura la perdió")

    print(f"\n{linea}\nB. VOLUMEN — que no se borre de más")
    if vaciados:
        print(f"   ✖ FALLA: {len(vaciados)} documentos con texto quedaron vacíos")
        for doc_id, formato, n in vaciados[:20]:
            print(f"     {doc_id:20} {formato:5} tenía {n:,} caracteres")
    else:
        print("   ✔ PASA: ningún documento con texto quedó vacío")

    print("\n   los 15 con mayor reducción (revisar que cada uno tenga explicación):")
    reducciones.sort(reverse=True)
    for ratio, a, d, doc_id, nombre in reducciones[:15]:
        print(f"     {ratio:6.1%}  {a:10,} → {d:10,}  {doc_id:18} {nombre[:42]}")

    print(f"\n{linea}\nC. TÉRMINOS — que ningún documento pierda un término de las consultas")
    if perdidas:
        print(f"   ✖ FALLA: {len(perdidas)} casos de documento que pierde un término")
        for doc_id, formato, termino, veces in perdidas[:25]:
            print(f"     {doc_id:20} {formato:5} perdió {termino!r} (aparecía {veces}×)")
        if len(perdidas) > 25:
            print(f"     … y {len(perdidas) - 25} más")
    else:
        print("   ✔ PASA: todo documento que contenía un término sigue conteniéndolo")

    print("\n   cobertura y apariciones por término (docs antes→después | apariciones):")
    for termino in sorted(terminos, key=lambda t: -docs_con_termino_antes[t]):
        da, dd = docs_con_termino_antes[termino], docs_con_termino_despues[termino]
        if not da:
            continue      # término que no aparece en el corpus: no informa de nada
        aa, ad = apariciones_antes[termino], apariciones_despues[termino]
        marca = "  ⚠️" if dd < da else ""
        print(f"     {termino:22} {da:5} → {dd:5} docs  |  {aa:9,} → {ad:9,}{marca}")

    print(f"\n{linea}\nD. NORMALIZACIÓN — §2.2 puntos 1 y 2, medidos sobre la salida")
    if restos_control:
        print(f"   ✖ FALLA: quedan caracteres de control en {len(restos_control)} documentos")
        print("     " + ", ".join(restos_control[:15]))
    else:
        print("   ✔ PASA: no queda ningún carácter de control ni invisible")
    if sin_nfc:
        print(f"   ✖ FALLA: {len(sin_nfc)} documentos no están en NFC")
        print("     " + ", ".join(sin_nfc[:15]))
    else:
        print("   ✔ PASA: todo el texto está en NFC")

    fallo = bool(fallos_parrafos or vaciados or perdidas or restos_control or sin_nfc)
    print(f"\n{linea}")
    print("VEREDICTO: ✖ HAY FALLOS — no seguir a la Fase 4 sin resolverlos" if fallo
          else "VEREDICTO: ✔ TODAS LAS PRUEBAS PASAN")
    print(linea)

    # Código de salida distinto de 0 si algo falla: así la comprobación se
    # puede encadenar en un script sin tener que leer la salida a ojo.
    return 1 if fallo else 0


if __name__ == "__main__":
    raise SystemExit(main())
