"""Canonicalización de entidades para el grafo de conocimiento (§7).

    py -m graph.canonical           # autoprueba: 40 casos con veredicto

────────────────────────────────────────────────────────────────────────────────
EL PROBLEMA, Y POR QUÉ ES EL QUE DECIDE SI EL GRAFO SIRVE

GLiNER devuelve **formas superficiales**, tal como aparecen en el texto. Sobre un
corpus que es 55,0 % inglés, 35,1 % español y 5,9 % portugués (`ESTADO.md` §11),
un mismo país llega escrito de seis maneras:

    EE.UU.  ·  EEUU  ·  Estados Unidos  ·  United States  ·  U.S.  ·  USA

Sin canonicalización, el grafo tiene seis nodos donde debería tener uno, y las
relaciones se reparten entre ellos: `(EE.UU., desarrolla, X)` y
`(United States, desarrolla, X)` quedan como hechos distintos sobre entidades
distintas. El resultado no es un grafo malo, es un grafo **inservible** — y §7.1
pide explícitamente que los nodos representen entidades, no cadenas.

────────────────────────────────────────────────────────────────────────────────
DOS CAPAS, Y SOLO UNA SE PUEDE CERRAR SIN DATOS

**Capa 1 — la clave determinista.** Colapsa las variantes *ortográficas* de una
misma forma: mayúsculas, tildes, puntos de sigla, artículos, espaciado. Es una
función pura del texto, se verifica con casos y **no necesita el corpus**. Es lo
que resuelve `EE.UU.` = `EEUU` = `ee. uu.` y `México` = `Mexico`.

**Capa 2 — la tabla de alias.** Colapsa las variantes *translingües y por
sinonimia*, que ninguna regla ortográfica puede deducir: `Estados Unidos` =
`United States`, `OTAN` = `NATO`, `IA` = `artificial intelligence`. Esto **no se
puede derivar**, hay que enumerarlo.

⚠️ **La tabla de abajo es una SEMILLA, no la solución.** Está poblada con las
entidades que el corpus garantiza —los observatorios de origen, los países de los
tres fenómenos y los conceptos de sus temas— pero la tabla definitiva se curará
sobre **la lista de frecuencias de la primera pasada real de GLiNER**. Hasta
entonces, cualquier medición de calidad del grafo es provisional.

────────────────────────────────────────────────────────────────────────────────
POR QUÉ LA IDENTIDAD INCLUYE EL TIPO

La identidad de un nodo es el par `(tipo, clave)`, no la clave sola. El motivo es
concreto y sale del corpus: **«Amazon»** es una región en F3 y una empresa en F1.
Con la clave sola serían el mismo nodo y el grafo afirmaría que una selva
desarrolla servicios en la nube.

⚠️ Pero eso obliga a **normalizar también el tipo**: GLiNER recibe etiquetas de
grano fino (`country`, `location`, `weapon system`…) y dos etiquetas distintas
para la misma entidad la partirían en dos nodos. `TIPOS_CANONICOS` las reduce a
los cinco tipos que enumera §7.1 más los dos que los fenómenos exigen.

────────────────────────────────────────────────────────────────────────────────
LO QUE ESTE MÓDULO **NO** HACE, A PROPÓSITO

`sugerir_alias()` detecta candidatos por sigla —`OTAN` frente a `Organización del
Tratado del Atlántico Norte`— pero **no los aplica**. Solo la tabla curada se
aplica automáticamente.

La razón es que la heurística de siglas produce falsos positivos costosos: `AI`
es sigla válida de *Artificial Intelligence* y de *Amnesty International*, y las
dos aparecen en un corpus sobre IA militar y derechos humanos. Fusionarlas
inventaría hechos. Así que la heurística **propone y una persona decide**, que es
la misma línea que §10 de `ESTADO.md` traza para el OCR: ante la duda, no
indexar cuesta menos que indexar algo falso.
"""

import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Se reutiliza el normalizador de `retrieval/fragmentos.py` en vez de reescribir
# el paso de minúsculas y tildes. Es la misma regla que llevó a `fragmentos.py` a
# importar `dividir_en_oraciones()` del chunker (`ESTADO.md` §15): dos
# implementaciones de la misma normalización acaban divergiendo, y cuando lo
# hacen el síntoma es un grafo con nodos duplicados que nadie relaciona con esto.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval.fragmentos import _normalizar  # noqa: E402


# ── Tipos ────────────────────────────────────────────────────────────────────

# Las etiquetas de grano fino que se le piden a GLiNER, reducidas a los tipos que
# enumera §7.1 («personas, organizaciones, lugares, conceptos, eventos») más los
# dos que los fenómenos del reto hacen imprescindibles: normas y grupos armados.
TIPOS_CANONICOS: Dict[str, str] = {
    "person": "PER",
    "organization": "ORG",
    "company": "ORG",
    "institution": "ORG",
    "country": "LOC",
    "location": "LOC",
    "city": "LOC",
    "region": "LOC",
    "weapon system": "TEC",
    "military technology": "TEC",
    "technology": "TEC",
    "space object": "TEC",
    "treaty or regulation": "NORM",
    "treaty": "NORM",
    "regulation": "NORM",
    "law": "NORM",
    "armed group": "GRP",
    "event": "EVT",
}

# Tipo de salida cuando la etiqueta no está en el mapa. No se descarta la
# entidad: se marca, para que el verificador pueda contar cuántas caen aquí y
# decidir si falta una entrada. Silenciarlas ocultaría un mapa incompleto.
TIPO_DESCONOCIDO = "OTRO"


# ── Capa 1: la clave determinista ────────────────────────────────────────────

# Artículos y determinantes iniciales en los tres idiomas del corpus. Se quitan
# solo al PRINCIPIO: «el Salvador» no es «Salvador», pero «El Salvador» como país
# sí lleva artículo propio — por eso el artículo se retira para la CLAVE y el
# nombre canónico se conserva intacto (ver `canonizar`).
ARTICULOS = ("el ", "la ", "los ", "las ", "the ", "o ", "a ", "os ", "as ",
             "l'", "un ", "una ")

# Sufijos societarios que no distinguen entidades: «Airbus S.A.» y «Airbus» son
# la misma. Se quitan del final.
SUFIJOS_SOCIETARIOS = (" sa", " s a", " inc", " ltd", " llc", " gmbh", " plc",
                       " corp", " co", " sl", " srl", " ltda", " bv", " ag")

# Topónimos cuyo artículo **forma parte del nombre**. Sin esta excepción,
# «El Salvador» daría clave `salvador` y colisionaría con **Salvador de Bahía**,
# y las dos son plausibles en F3 (dinámicas territoriales en América Latina).
# Es una lista corta y curada; se compara ya en minúsculas y sin tildes.
NOMBRES_CON_ARTICULO = frozenset({
    "el salvador", "la haya", "la paz", "el cairo", "los angeles", "la habana",
    "el alto", "la plata", "el callao", "las vegas", "el chaco", "la guajira",
})

_PUNTUACION_BORDE = re.compile(r"^[\W_]+|[\W_]+$", re.UNICODE)
_ESPACIOS = re.compile(r"\s+")
_POSESIVO = re.compile(r"['’]s\b")


def clave_canonica(nombre: str) -> str:
    """La clave con la que dos formas superficiales se consideran la misma.

    Función **pura**: mismo texto, misma clave, siempre. Eso importa porque §1.4
    exige reproducibilidad y el grafo forma parte de la entrega.

    Los pasos, y cada uno con el caso del corpus que lo motiva:

    1. **NFC** — «ó» se puede escribir de dos formas en Unicode y el corpus trae
       las dos (`preprocess/cleaner.py` lo documenta con 456.890 acentos).
    2. **Minúsculas y sin tildes**, vía `_normalizar()`: `México` = `Mexico`, que
       es como aparece en las fuentes en inglés.
    3. **Posesivo sajón**: `NATO's` = `NATO`.
    4. **Puntos y espacios internos de sigla**: `EE.UU.` = `EE UU` = `EEUU`.
       Es el caso más frecuente y el que más nodos duplicados ahorra.
    5. **Artículo inicial**: `the United States` = `United States`.
    6. **Sufijo societario**: `Airbus S.A.` = `Airbus`.
    7. **Puntuación de los bordes y espaciado**.

    ⚠️ El paso 4 **solo** compacta la secuencia de grupos de una o dos letras
    separadas por puntos. Aplicarlo a todo el texto uniría palabras legítimas:
    `U.S. Army` debe dar `us army`, no `usarmy`.

    ⚠️ **Los grupos son de una O DOS letras, y eso es específico del español**:
    las siglas en plural duplican la letra —`EE.UU.` por *Estados Unidos*,
    `RR.HH.`, `JJ.OO.`—. La primera versión de esta función solo aceptaba letras
    sueltas, así que resolvía `U.S.` y **fallaba con `EE.UU.`**, que es la forma
    más frecuente del país más citado del corpus.

    🔴 **LO QUE ESTA CLAVE NO HACE: normalizar el número gramatical.**
    `arma antisatélite` y `armas antisatélite` dan claves distintas, o sea dos
    nodos. Es deliberado: quitar la `-s` final es destructivo en los tres idiomas
    del corpus —`Gales` daría `gale`, `Estados Unidos` daría `estado unido`,
    `París` daría `pari`— y las reglas de plural del español (`-s`, `-es`,
    `-ces`→`z`) y del inglés (`-ies`→`y`) no coinciden. Una regla automática
    fusionaría entidades distintas, que es el fallo caro.

    **Se trata por las otras dos vías:** las variantes en plural de las entidades
    conocidas van enumeradas en la tabla de alias, y `sugerir_alias()` propone
    los pares singular/plural que aparezcan en la pasada real para que una
    persona los añada. Es la misma división de trabajo que con las siglas.
    """
    if not nombre:
        return ""

    texto = unicodedata.normalize("NFC", nombre)
    texto = _normalizar(texto)             # minúsculas + sin tildes (reutilizado)
    texto = _POSESIVO.sub("", texto)

    # Paso 4: una sigla es una secuencia de grupos de una o dos letras separados
    # por puntos, con o sin espacios. El `{2,}` exige al menos dos grupos para no
    # tocar una inicial suelta como «J. Smith», donde el punto sí separa dos
    # cosas. El `{1,2}` de la letra cubre las siglas en plural del español
    # (`EE.UU.`, `RR.HH.`), que se escriben doblando.
    def _compactar(m: "re.Match") -> str:
        return m.group(0).replace(".", "").replace(" ", "") + " "

    texto = re.sub(r"(?:[a-z]{1,2}\.\s*){2,}", _compactar, texto)

    texto = _ESPACIOS.sub(" ", texto).strip()

    # Paso 5: solo el primero, y solo si queda algo detrás. `while` y no `if`
    # porque «the the» no existe pero «a los » sí puede aparecer troceado.
    cambiado = True
    while cambiado:
        # Excepción antes de tocar nada: si el nombre completo es uno de los
        # topónimos que llevan artículo propio, el artículo se queda.
        if texto in NOMBRES_CON_ARTICULO:
            break
        cambiado = False
        for art in ARTICULOS:
            if texto.startswith(art) and len(texto) > len(art):
                texto = texto[len(art):]
                cambiado = True
                break

    # Paso 6: sufijos societarios, tras quitarles ya los puntos en el paso 4.
    for suf in SUFIJOS_SOCIETARIOS:
        if texto.endswith(suf) and len(texto) > len(suf) + 2:
            texto = texto[: -len(suf)]
            break

    texto = _PUNTUACION_BORDE.sub("", texto)
    return _ESPACIOS.sub(" ", texto).strip()


# ── Capa 2: la tabla de alias ────────────────────────────────────────────────

# Cada entrada es `forma alternativa -> nombre canónico`. Se indexa por
# `clave_canonica()`, así que NO hace falta enumerar variantes ortográficas:
# poner «Estados Unidos» ya cubre «estados unidos» y «ESTADOS UNIDOS».
#
# ⚠️ SEMILLA. Poblada con lo que el corpus garantiza: los observatorios de origen
# de los `doc_id`, los países de los tres fenómenos y los conceptos centrales de
# sus temas. **Hay que extenderla desde la lista de frecuencias de la primera
# pasada de GLiNER**, y hasta entonces el grafo tendrá duplicados translingües
# que esta tabla no cubre.
#
# ⚠️ Cada entrada tiene que ser defensible: una entrada de más que nunca casa no
# cuesta nada, pero una entrada EQUIVOCADA fusiona dos entidades distintas y el
# grafo afirma un hecho falso. Ante la duda, no se añade.
ALIAS_SEMILLA: Dict[str, str] = {}

# Las mismas entradas pero conservando las formas superficiales, `nombre
# canónico -> [variantes]`. `ALIAS_SEMILLA` solo guarda claves normalizadas —sin
# tildes ni puntos— y con esas no se puede buscar en el texto original. Esto sí
# sirve para armar un gazetteer, que es lo que permite probar el extractor de
# relaciones sobre prosa real sin haber pasado GLiNER todavía.
VARIANTES_SEMILLA: Dict[str, List[str]] = {}


def _sembrar(canonico: str, *variantes: str) -> None:
    """Registra un nombre canónico y todas sus variantes conocidas."""
    for v in (canonico, *variantes):
        ALIAS_SEMILLA[clave_canonica(v)] = canonico
    VARIANTES_SEMILLA.setdefault(canonico, []).extend((canonico, *variantes))


# Países y bloques. Se elige la forma española como canónica porque las 50
# consultas están todas en español (`ESTADO.md` §5) y facilita leer el grafo.
_sembrar("Estados Unidos", "EE.UU.", "EEUU", "United States", "U.S.", "USA",
         "US", "United States of America")
_sembrar("Reino Unido", "United Kingdom", "UK", "Great Britain")
_sembrar("China", "People's Republic of China", "PRC", "República Popular China")
_sembrar("Rusia", "Russia", "Russian Federation", "Federación Rusa")
_sembrar("Corea del Norte", "North Korea", "DPRK", "RPDC")
_sembrar("Corea del Sur", "South Korea", "Republic of Korea", "ROK")
_sembrar("Brasil", "Brazil")
_sembrar("México", "Mexico")
_sembrar("Perú", "Peru")
_sembrar("Japón", "Japan")
_sembrar("Alemania", "Germany")
_sembrar("Francia", "France")
_sembrar("Países Bajos", "Netherlands", "Holanda")
_sembrar("Turquía", "Turkey", "Türkiye")
_sembrar("Irán", "Iran")
_sembrar("Ucrania", "Ukraine")
_sembrar("Unión Europea", "European Union", "EU", "UE")

# Organizaciones. Las que aparecen como observatorio de origen en los `doc_id`
# del corpus, más las que los tres temas obligan.
_sembrar("OTAN", "NATO", "North Atlantic Treaty Organization",
         "Organización del Tratado del Atlántico Norte")
_sembrar("Naciones Unidas", "ONU", "UN", "United Nations")
_sembrar("OEA", "Organization of American States",
         "Organización de los Estados Americanos")
_sembrar("UNOOSA", "United Nations Office for Outer Space Affairs",
         "Oficina de Asuntos del Espacio Ultraterrestre")
_sembrar("ESA", "European Space Agency", "Agencia Espacial Europea")
_sembrar("NASA", "National Aeronautics and Space Administration")
_sembrar("SIPRI", "Stockholm International Peace Research Institute")
_sembrar("CSIS", "Center for Strategic and International Studies")
_sembrar("Secure World Foundation", "SWF")
_sembrar("Atlantic Council", "Consejo Atlántico")
_sembrar("CICR", "ICRC", "International Committee of the Red Cross",
         "Comité Internacional de la Cruz Roja")
_sembrar("MAPP-OEA", "MAPP/OEA", "Misión de Apoyo al Proceso de Paz")
_sembrar("RESDAL", "Red de Seguridad y Defensa de América Latina")
_sembrar("INPE", "Instituto Nacional de Pesquisas Espaciais")

# Tecnologías y conceptos. Son los que §7.1 llama «conceptos» y que un NER
# clásico no devolvería; GLiNER sí, por ser zero-shot.
_sembrar("inteligencia artificial", "IA", "AI", "artificial intelligence")
_sembrar("aprendizaje automático", "machine learning", "ML")
# ⚠️ **`AWS` se retiró de esta entrada.** Es sigla de *autonomous weapon system*,
# pero en un corpus sobre IA significa abrumadoramente **Amazon Web Services**:
# se midió sobre `F1-CSET-017`, cuyos chunks hablan de «cloud stack», «Zone» y
# «firms deliver the core», y el grafo afirmaba que un sistema de armas autónomo
# opera infraestructura en la nube. Es el caso que ilustra por qué cada entrada de
# esta tabla tiene que ser defensible: una de más que nunca casa no cuesta nada,
# pero una equivocada fabrica hechos falsos.
_sembrar("sistema de armas autónomo", "LAWS", "sistemas de armas autónomas letales",
         "lethal autonomous weapons systems", "autonomous weapon system",
         # ⚠️ El plural va enumerado a mano: la clave NO normaliza el número
         # gramatical (ver la nota de `clave_canonica`), así que sin estas
         # entradas «sistemas de armas autónomos» sería un nodo aparte.
         "sistemas de armas autónomos", "sistemas de armas autónomas",
         "autonomous weapons systems", "autonomous weapon systems")
_sembrar("arma antisatélite", "ASAT", "anti-satellite weapon", "antisatélite",
         "armas antisatélite", "armas antisatélites", "anti-satellite weapons",
         "armas antisatelitales")
_sembrar("órbita baja terrestre", "LEO", "low earth orbit", "órbita terrestre baja")
_sembrar("vehículo aéreo no tripulado", "UAV", "dron", "drone", "UAS",
         "vehículos aéreos no tripulados", "drones", "UAVs")
_sembrar("desecho orbital", "space debris", "basura espacial", "orbital debris",
         "desechos orbitales", "desechos espaciales")
_sembrar("operaciones de proximidad", "RPO", "rendezvous and proximity operations")

# Normas y tratados. El ejemplo de §7.1 usa uno de estos.
_sembrar("Convenio de Ginebra", "Geneva Convention", "Convenios de Ginebra",
         "Geneva Conventions")
_sembrar("Tratado del Espacio Exterior", "Outer Space Treaty",
         "Tratado sobre el Espacio Ultraterrestre")
_sembrar("TNP", "NPT", "Treaty on the Non-Proliferation of Nuclear Weapons",
         "Tratado de No Proliferación")
_sembrar("derecho internacional humanitario", "DIH", "IHL",
         "international humanitarian law")


def cargar_alias(ruta: Optional[Path] = None) -> Dict[str, str]:
    """Devuelve la tabla de alias, con las curadas a mano si existe el archivo.

    El archivo es un JSON `{"nombre canónico": ["variante", …]}`. Vive fuera del
    código a propósito: extender la tabla tras cada pasada de GLiNER es trabajo
    de curación, no de programación, y no debería exigir tocar un `.py`.

    Las entradas del archivo **ganan** sobre la semilla, para poder corregir una
    entrada de aquí sin editar este módulo.
    """
    tabla = dict(ALIAS_SEMILLA)
    if ruta is None:
        ruta = Path(__file__).resolve().parents[1] / "data" / "alias_entidades.json"
    if ruta.is_file():
        for canonico, variantes in json.loads(
                ruta.read_text(encoding="utf-8")).items():
            for v in (canonico, *variantes):
                tabla[clave_canonica(v)] = canonico
    return tabla


# ── La operación que consume el constructor del grafo ────────────────────────

def canonizar(nombre: str, tipo: str,
              alias: Optional[Dict[str, str]] = None) -> Tuple[str, str, str]:
    """De una entidad de GLiNER a su identidad en el grafo.

    Devuelve `(nombre_canonico, tipo_canonico, clave)`, donde `clave` es la que
    identifica el nodo junto con el tipo.

    El **nombre canónico** es para leer el grafo; la **clave** es para
    identificarlo. No son lo mismo a propósito: el nodo se llama
    «Estados Unidos» y no «estados unidos», que es lo que un humano espera ver al
    abrir el `.graphml`, mientras que la clave sin tildes ni artículos es la que
    hace coincidir las seis formas del texto.
    """
    if alias is None:
        alias = ALIAS_SEMILLA

    clave = clave_canonica(nombre)
    tipo_canonico = TIPOS_CANONICOS.get(tipo.strip().lower(), TIPO_DESCONOCIDO)

    canonico = alias.get(clave)
    if canonico is not None:
        # Al resolver por alias, la clave pasa a ser la del nombre canónico: así
        # «NATO» y «OTAN» colapsan en el MISMO nodo y no en dos que se llaman
        # igual. Sin esta línea la tabla renombraría sin fusionar.
        return canonico, tipo_canonico, clave_canonica(canonico)

    # Sin alias, el nombre se conserva tal como vino —§7 no pide reescribirlo— y
    # solo la clave se normaliza.
    return nombre.strip(), tipo_canonico, clave


def agrupar_por_identidad(
    entidades: Iterable[Tuple[str, str]],
    alias: Optional[Dict[str, str]] = None,
) -> Dict[Tuple[str, str], Dict]:
    """Agrupa `(nombre, tipo)` en nodos, contando variantes y frecuencia.

    La salida es `{(tipo, clave): {"nombre", "variantes", "n"}}`. Es la entrada
    natural del constructor del grafo y también lo que se mira para curar la
    tabla de alias: `variantes` dice qué formas superficiales colapsaron, y una
    revisión rápida de las más frecuentes destapa las que faltan.
    """
    nodos: Dict[Tuple[str, str], Dict] = {}
    for nombre, tipo in entidades:
        canonico, tipo_canonico, clave = canonizar(nombre, tipo, alias)
        if not clave:
            continue                      # una entidad vacía no es un nodo
        id_nodo = (tipo_canonico, clave)
        nodo = nodos.setdefault(
            id_nodo, {"nombre": canonico, "variantes": {}, "n": 0})
        nodo["n"] += 1
        nodo["variantes"][nombre] = nodo["variantes"].get(nombre, 0) + 1
    return nodos


# ── La heurística que PROPONE, sin aplicar ───────────────────────────────────

# Palabras que no aportan inicial a una sigla: «Organización del Tratado del
# Atlántico Norte» da OTAN, no ODTDAN.
VACIAS_SIGLA = {"de", "del", "la", "las", "el", "los", "y", "e", "a", "en",
                "of", "the", "for", "and", "on", "in", "to", "da", "do", "dos",
                "das", "e"}


def es_sigla_de(sigla: str, nombre: str) -> bool:
    """¿`sigla` son las iniciales de las palabras significativas de `nombre`?

    Comparación sobre las claves canónicas, así que `O.T.A.N.` funciona igual que
    `OTAN`.
    """
    s = clave_canonica(sigla).replace(" ", "")
    if not (2 <= len(s) <= 8) or not s.isalpha():
        return False
    palabras = [p for p in clave_canonica(nombre).split()
                if p not in VACIAS_SIGLA]
    if len(palabras) < 2:
        return False
    return "".join(p[0] for p in palabras) == s


def _es_plural_de(a: str, b: str) -> bool:
    """¿`a` parece el plural de `b`, según las reglas de es/en/pt?

    Comparación conservadora: solo los sufijos regulares, y exigiendo que la raíz
    sea razonablemente larga para no proponer fusiones entre palabras cortas que
    coinciden por casualidad.
    """
    if len(b) < 4 or len(a) <= len(b):
        return False
    for sufijo in ("s", "es"):
        if a == b + sufijo:
            return True
    # español: -z → -ces  (lápiz/lápices);  inglés: -y → -ies  (entity/entities)
    if b.endswith("z") and a == b[:-1] + "ces":
        return True
    if b.endswith("y") and a == b[:-1] + "ies":
        return True
    return False


def sugerir_alias(nodos: Dict[Tuple[str, str], Dict],
                  minimo: int = 2) -> List[Tuple[str, str, str]]:
    """Propone fusiones por sigla y por número gramatical, para revisión humana.

    Devuelve `(sigla, nombre_largo, motivo)`. **No modifica nada.**

    ⚠️ **Por qué solo propone.** `AI` es sigla válida de *Artificial
    Intelligence* y de *Amnesty International*, y las dos caben en un corpus
    sobre IA militar y derechos humanos. Fusionarlas automáticamente inventaría
    un hecho. El criterio es el de `ESTADO.md` §10: ante la duda, no indexar
    cuesta menos que indexar algo falso.

    Solo compara nodos del **mismo tipo**: una sigla de organización no debería
    fusionarse con un topónimo aunque las iniciales coincidan.
    """
    sugerencias: List[Tuple[str, str, str]] = []
    por_tipo: Dict[str, List[Tuple[str, Dict]]] = {}
    for (tipo, clave), nodo in nodos.items():
        if nodo["n"] >= minimo:
            por_tipo.setdefault(tipo, []).append((clave, nodo))

    for tipo, lista in por_tipo.items():
        cortas = [(c, n) for c, n in lista if len(c.replace(" ", "")) <= 8
                  and " " not in c]
        largas = [(c, n) for c, n in lista if " " in c]
        for clave_corta, nodo_corto in cortas:
            for clave_larga, nodo_largo in largas:
                if es_sigla_de(clave_corta, clave_larga):
                    sugerencias.append((
                        nodo_corto["nombre"], nodo_largo["nombre"],
                        f"[{tipo}] iniciales coinciden · "
                        f"{nodo_corto['n']}+{nodo_largo['n']} menciones"))

        # Pares singular/plural. Es el hueco que `clave_canonica()` deja a
        # propósito, y a escala solo se puede cerrar proponiendo: en la pasada
        # real habrá cientos de estos y curarlos a mano uno por uno es el
        # trabajo, pero encontrarlos no debería serlo.
        for i, (clave_a, nodo_a) in enumerate(lista):
            for clave_b, nodo_b in lista[i + 1:]:
                if _es_plural_de(clave_a, clave_b) or _es_plural_de(clave_b, clave_a):
                    sugerencias.append((
                        nodo_b["nombre"], nodo_a["nombre"],
                        f"[{tipo}] singular/plural · "
                        f"{nodo_b['n']}+{nodo_a['n']} menciones"))
    return sorted(sugerencias)


# ── Autoprueba ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    fallos = 0
    linea = "─" * 74

    def igual(a: str, b: str, nota: str = "") -> None:
        """Dos formas que DEBEN dar la misma clave."""
        global fallos
        ka, kb = clave_canonica(a), clave_canonica(b)
        ok = ka == kb and ka != ""
        print(f"  {'✔' if ok else '✘'} {a!r:38s} == {b!r:34s} → {ka!r}"
              f"{'  ' + nota if nota else ''}")
        if not ok:
            fallos += 1

    def distinto(a: str, b: str, nota: str = "") -> None:
        """Dos formas que NO deben colapsar. Tan importante como lo anterior."""
        global fallos
        ka, kb = clave_canonica(a), clave_canonica(b)
        ok = ka != kb
        print(f"  {'✔' if ok else '✘'} {a!r:38s} != {b!r:34s} → "
              f"{ka!r} / {kb!r}{'  ' + nota if nota else ''}")
        if not ok:
            fallos += 1

    print(f"{linea}\nA. CLAVE — variantes ortográficas que deben colapsar\n{linea}")
    igual("EE.UU.", "EEUU", "sigla plural española: dobla la letra")
    igual("EE. UU.", "EEUU")
    igual("RR.HH.", "RRHH")
    igual("U.S.", "US")
    igual("México", "Mexico", "tildes")
    igual("PERÚ", "peru")
    igual("the United States", "United States", "artículo inicial")
    igual("La OTAN", "OTAN")
    igual("NATO's", "NATO", "posesivo sajón")
    igual("Airbus S.A.", "Airbus", "sufijo societario")
    igual("  Naciones   Unidas  ", "Naciones Unidas", "espaciado")
    igual("«SIPRI»", "SIPRI", "puntuación de borde")
    igual("Órbita Baja Terrestre", "orbita baja terrestre")

    print(f"\n{linea}\nB. CLAVE — lo que NO debe colapsar\n{linea}")
    distinto("U.S. Army", "US", "una sigla no absorbe la palabra siguiente")
    distinto("Corea del Norte", "Corea del Sur", "países distintos")
    # El caso que la versión anterior dejaba pasar: «El Salvador» daba clave
    # `salvador` y colisionaba con Salvador de Bahía. Las dos son plausibles en
    # F3, y fusionarlas afirmaría hechos de un país sobre una ciudad.
    distinto("El Salvador", "Salvador", "el artículo es parte del nombre")
    distinto("El Salvador", "Salvador de Bahía")
    igual("El Salvador", "el salvador", "pero sigue siendo insensible al caso")
    distinto("El Salvador", "Salvador Allende", "artículo vs nombre")
    igual("La Paz", "la paz", "otro topónimo con artículo propio")
    distinto("OTAN", "ONU")
    distinto("J. Smith", "JS", "inicial suelta no es sigla")

    print(f"\n{linea}\nC. ALIAS — variantes translingües y por sinonimia\n{linea}")
    alias = cargar_alias()
    for a, b in [("United States", "EE.UU."), ("NATO", "OTAN"),
                 ("artificial intelligence", "IA"), ("UN", "Naciones Unidas"),
                 ("LAWS", "sistemas de armas autónomas letales"),
                 ("low earth orbit", "órbita baja terrestre"),
                 ("Geneva Convention", "Convenio de Ginebra"),
                 ("Brazil", "Brasil"), ("SWF", "Secure World Foundation")]:
        na, ta, ka = canonizar(a, "organization", alias)
        nb, tb, kb = canonizar(b, "organization", alias)
        ok = ka == kb and na == nb
        print(f"  {'✔' if ok else '✘'} {a!r:34s} y {b!r:38s} → {na!r}")
        if not ok:
            fallos += 1

    print(f"\n{linea}\nD. TIPOS — el mapa de §7.1 y la identidad tipada\n{linea}")
    for etiqueta, esperado in [("country", "LOC"), ("location", "LOC"),
                               ("organization", "ORG"), ("weapon system", "TEC"),
                               ("treaty or regulation", "NORM"),
                               ("armed group", "GRP"), ("person", "PER"),
                               ("inventada", TIPO_DESCONOCIDO)]:
        _, t, _ = canonizar("X", etiqueta, alias)
        ok = t == esperado
        print(f"  {'✔' if ok else '✘'} {etiqueta!r:24s} → {t}")
        if not ok:
            fallos += 1

    # El caso que motiva la identidad tipada: «Amazon» región vs empresa.
    _, t1, k1 = canonizar("Amazon", "region", alias)
    _, t2, k2 = canonizar("Amazon", "company", alias)
    ok = (t1, k1) != (t2, k2)
    print(f"  {'✔' if ok else '✘'} 'Amazon' como region y como company son "
          f"nodos distintos → {(t1, k1)} / {(t2, k2)}")
    if not ok:
        fallos += 1

    print(f"\n{linea}\nE. AGRUPACIÓN — seis formas, un nodo\n{linea}")
    entrada = [("EE.UU.", "country"), ("Estados Unidos", "country"),
               ("United States", "country"), ("U.S.", "country"),
               ("the United States", "country"), ("USA", "country"),
               ("OTAN", "organization"), ("NATO", "organization"),
               ("Colombia", "country")]
    nodos = agrupar_por_identidad(entrada, alias)
    print(f"  {len(entrada)} entidades de entrada → {len(nodos)} nodos")
    for (tipo, clave), n in sorted(nodos.items(), key=lambda x: -x[1]["n"]):
        print(f"    [{tipo}] {n['nombre']:20s} n={n['n']}  "
              f"variantes={sorted(n['variantes'])}")
    ok = len(nodos) == 3
    print(f"  {'✔' if ok else '✘'} se esperaban 3 nodos "
          f"(Estados Unidos, OTAN, Colombia)")
    if not ok:
        fallos += 1

    print(f"\n{linea}\nF. SUGERENCIAS — propone, no aplica\n{linea}")
    nodos2 = agrupar_por_identidad(
        [("UNOOSA", "organization"), ("UNOOSA", "organization"),
         ("Oficina de Asuntos del Espacio Ultraterrestre", "organization"),
         ("Oficina de Asuntos del Espacio Ultraterrestre", "organization"),
         ("CTBT", "treaty"), ("CTBT", "treaty"),
         ("Comprehensive Test Ban Treaty", "treaty"),
         ("Comprehensive Test Ban Treaty", "treaty")], alias)
    for sigla, largo, motivo in sugerir_alias(nodos2):
        print(f"    {sigla!r} ←→ {largo!r}   {motivo}")
    print("  (se imprimen para revisión humana; NO se aplican)")

    print(f"\n{linea}")
    print(f"tabla de alias sembrada: {len(alias)} claves")
    if fallos:
        print(f"VEREDICTO: ✘ {fallos} comprobación(es) fallan")
    else:
        print("VEREDICTO: ✔ todas las comprobaciones pasan")
    print(linea)
    raise SystemExit(1 if fallos else 0)
