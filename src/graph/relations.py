"""Extracción de relaciones por patrón verbal, para el grafo (§7.2).

    py -m graph.relations           # autoprueba + tripletas de prosa real

────────────────────────────────────────────────────────────────────────────────
QUÉ RESUELVE, Y POR QUÉ NO VALE LA CO-OCURRENCIA

§7.1 pide `T ⊆ E × R × E`: tripletas con **relación tipada y dirección**. Su
ejemplo lo deja claro:

    (EE.UU., desarrolla, sistema-de-armas-autónomo)

Un grafo donde todas las aristas dicen `aparece_con` cumple la letra de §7 —hay
nodos y hay aristas— pero **no su ejemplo**, y no aporta nada que el índice
vectorial no dé ya: que dos términos aparezcan cerca es exactamente lo que la
proximidad semántica de los embeddings captura. §7.1 dice que el grafo
«complementa la base vectorial porque captura relaciones **explícitas**». Sin
tipo ni dirección no hay nada explícito.

**El patrón verbal es lo que convierte co-ocurrencia en conocimiento.**

────────────────────────────────────────────────────────────────────────────────
POR QUÉ HEURÍSTICAS Y NO UN MODELO

§7.2 autoriza tres vías para las relaciones: «modelos de clasificación de
relaciones, **heurísticas basadas en patrones lingüísticos**, o dependencias
sintácticas». Se elige la segunda por dos razones:

1. 🔴 **Un LLM está prohibido.** La vía obvia —un decoder lee el párrafo y
   devuelve la tripleta— viola §4.2, que prohíbe los decoders en la construcción
   del índice. La licencia del modelo no salva la arquitectura.
2. Un clasificador de relaciones supervisado exigiría un inventario de relaciones
   fijo y datos de entrenamiento en tres idiomas. No existe para este dominio.

Las heurísticas son además **deterministas**, que es lo que §1.4 exige del
entregable: el mismo texto produce las mismas tripletas siempre.

────────────────────────────────────────────────────────────────────────────────
LOS CUATRO PROBLEMAS QUE UNA VERSIÓN INGENUA SE COME

Buscar «un verbo entre dos entidades» produce basura. Estos son los casos que
hay que tratar, y cada uno está cubierto por la autoprueba:

**1. Voz pasiva invierte la dirección.** «Los sistemas autónomos son
desarrollados **por** EE.UU.» tiene el mismo orden superficial que la voz activa
pero la tripleta correcta es `(EE.UU., desarrolla, sistemas autónomos)`. Sin
detectarlo, el grafo afirma que unos sistemas desarrollan un país.

**2. La negación afirma lo contrario.** «China **no** ha firmado el tratado»
produciría `(China, firma, tratado)`. Es afirmar un hecho falso, que es lo mismo
que §10 de `ESTADO.md` prohíbe para el OCR: no se indexa lo que el documento no
dice.

**3. Una tercera entidad en medio rompe el vínculo.** En «EE.UU. vendió a
Colombia sistemas de vigilancia», el par (EE.UU., sistemas) tiene el verbo en
medio, pero también lo tiene (Colombia, sistemas), y solo una de las dos lecturas
es la del texto. Se resuelve **relacionando solo entidades adyacentes**.

**4. La coordinación esconde sujetos.** «EE.UU. **y** China desarrollan armas
autónomas» son dos tripletas, no una: con la regla de adyacencia estricta se
perdería la de EE.UU. Se resuelve agrupando las entidades que solo están
separadas por coordinadores.
"""

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graph.canonical import canonizar, cargar_alias  # noqa: E402
from retrieval.fragmentos import _normalizar  # noqa: E402


# ── Inventario de relaciones ─────────────────────────────────────────────────

# `relación canónica -> raíces de verbo en los tres idiomas del corpus`.
#
# Se guardan **raíces** y no formas completas: el patrón las ancla con `\w*` al
# final, así que `desarroll` cubre desarrolla, desarrollan, desarrollado,
# desarrollando y desenvolve. Es lo que evita enumerar la conjugación española,
# que son decenas de formas por verbo.
#
# ⚠️ Las relaciones se mantienen **separadas cuando significan cosas distintas**:
# `regula` y `prohibe` no se fusionan aunque ambas hablen de normativa, porque el
# grafo diría que un tratado permite lo que prohíbe. Es el mismo criterio por el
# que la canonicalización no fusiona entidades ante la duda.
RELACIONES: Dict[str, Tuple[str, ...]] = {
    # ⚠️ Sin `design`: casa con «designated» —vía la terminación inglesa `ated`— y
    # producía `(UAV, desarrolla, EE.UU.)` sobre «are designated as being
    # headquartered in». «Designar» no es «desarrollar». El español queda cubierto
    # por `diseñ`.
    "desarrolla":   ("desarroll", "desenvolv", "develop", "fabric", "manufactur",
                     "produc", "construy", "build", "diseñ"),
    # ⚠️ La raíz es `use` y no `us`: con `us` la terminación `e` casaba dentro de
    # «dual-use», porque el guion cuenta como frontera de palabra. Con `use` +
    # terminación obligatoria solo casan «used», «uses», «using», y «dual-use»
    # queda fuera porque tras `use` no hay terminación.
    "opera":        ("oper", "emple", "utiliz", "use", "despleg", "deploy",
                     "manej"),
    "lanza":        ("lanz", "launch", "pon en orbita", "orbit"),
    "regula":       ("regul", "govern", "rige", "establec", "norm"),
    "prohibe":      ("prohib", "ban", "veta", "forbid", "restring", "restrict"),
    # ⚠️ Sin `firm`: es **sustantivo en los dos idiomas** —«firms» en inglés es
    # empresas, «la firma» en español es empresa o rúbrica— y producía
    # `(EE.UU., firma, IA)` sobre «firms deliver the core». Se pierde «firmó», y
    # es una pérdida real; se compensa con `ratific`, `suscrib` y `refrend`.
    "firma":        ("sign", "ratific", "ratif", "adhier", "suscrib", "refrend"),
    "financia":     ("financ", "fund", "invierte en", "invest", "subvencion"),
    "coopera_con":  ("cooper", "colabor", "collaborat", "partner", "acuerd"),
    "ataca":        ("atac", "attack", "agred", "interfier", "interfer",
                     "inhib", "jam", "destruy", "destroy"),
    "controla":     ("control", "domin", "ocup", "occup", "administr"),
    "pertenece_a":  ("pertenec", "miembro de", "member of", "forma parte",
                     "part of", "integra"),
    "ubicado_en":   ("ubicad", "situad", "located", "based in", "con sede"),
}

# Relación de reserva para pares sin verbo. **Desactivada por defecto**: es
# co-ocurrencia, no conocimiento, y mezclarla con las tipadas sin poder
# distinguirlas degradaría el grafo. Se puede activar para medir qué proporción
# de pares queda sin tipar, que es un dato útil para decidir si el inventario de
# verbos está incompleto.
RELACION_COOCURRENCIA = "aparece_con"

# Terminaciones verbales admitidas tras una raíz, ya sin tildes y en minúsculas.
# La terminación es **obligatoria**, y eso es lo que impide dos clases de falso
# positivo que se midieron sobre prosa real del corpus:
#
#   · `sign` casaba dentro de «significantly» y emitía `(…, firma, …)`
#   · `based in` casaba dentro de «based institutions» y emitía `ubicado_en`
#
# Con terminación obligatoria, «significantly» no casa porque tras `sign` viene
# `i`, que no abre ninguna terminación, y el `\b` final falla. Exigirla descarta
# de paso el sustantivo homógrafo: «control» no dispara `controla`, pero
# «controla», «controls» y «controlled» sí.
TERMINACIONES = (r"(?:a|as|an|amos|ando|ad[oa]s?|ar|aron|ab[ae]n?|e|es|en|"
                 r"i[oó]|id[oa]s?|ir|ieron|ia|ian|o|s|ed|ing|"
                 r"at(?:e|es|ed|ing))\b")

# Cuántos caracteres del conector puede haber ANTES del verbo. El predicado de
# una oración simple empieza justo tras el sujeto; si el verbo aparece a 50
# caracteres, entre medias hay una subordinada y el verbo no rige a esta entidad.
MAX_ANTES_DEL_VERBO = 25

# Determinantes que delatan un **sustantivo deverbal**, no un verbo.
#
# ⚠️ El caso es específico del español y no tiene solución léxica: al quitar las
# tildes, `desarrolló` (verbo, 3ª persona) y `desarrollo` (sustantivo) son la
# MISMA cadena. Medido sobre prosa real, «el desarrollo del sector» y «equipo de
# desarrollo de 18 ingenieros» emitían tripletas `desarrolla` falsas.
#
# Lo que sí distingue es la palabra de delante: un verbo no va precedido de
# artículo. `el desarrollo` es sustantivo; `EE.UU. desarrolló` es verbo.
DETERMINANTE_PREVIO = re.compile(
    r"\b(el|la|los|las|un|una|unos|unas|su|sus|este|esta|ese|esa|dicho|dicha|"
    r"the|its|their|this|that|a|an|of|de|del|en|con|para|por|sobre|tras)\s+$",
    re.I)

# Distancia máxima en caracteres entre el final de una entidad y el principio de
# la siguiente. Por encima de esto el verbo que haya en medio pertenece a otra
# oración o a otra cláusula.
#
# ⚠️ El valor **hay que medirlo** sobre las tripletas reales, no fijarlo a ojo:
# es la misma trampa que los factores de bonificación de `ESTADO.md` §21, donde
# los valores «razonables» resultaron no hacer nada. Se bajó de 140 a 60 tras ver
# que a 140 entraban conectores como «models originated from U.S.-based
# institutions, far outpacing the», que no es un predicado.
MAX_DISTANCIA = 60

# Negadores. Si aparecen en el conector, la tripleta NO se emite.
#
# ⚠️ Incluye los verbos de **abstención**, no solo los adverbios de negación:
# «refraining from using» produce una tripleta afirmativa con `usar` si solo se
# buscan `no`/`not`, y el texto dice justamente lo contrario. Se detectó sobre
# prosa real del corpus.
NEGACIONES = re.compile(
    r"\b(no|not|never|nunca|jam[áa]s|sin|without|n[ãa]o|nem|ni|neither|nor|"
    r"refrain\w*|abstain\w*|absten\w*|abstien\w*|renunci\w*|cease[ds]?|"
    r"dej[óo] de|failed to|impid\w*)\b", re.I)

# Un conector no puede cruzar una frontera de oración: si lo hace, las dos
# entidades están en oraciones distintas y ningún verbo las relaciona. Se
# detectó con `(Estados Unidos, desarrolla, China)`, cuyo conector era
# «chip manufacturing.» — el punto separaba dos afirmaciones sin relación.
#
# El salto de párrafo `\n\n` cuenta también, y es una frontera más fuerte que el
# punto: se midió `(EE.UU., coopera_con, IA)` cuyo conector era «partners\n\nAs
# the global race for», o sea dos secciones distintas del documento.
FIN_DE_ORACION = re.compile(r"[.!?][\"'»)\]]?\s|\n\s*\n")

# Coordinadores puros: si dos entidades solo están separadas por esto, forman un
# grupo coordinado y comparten la relación.
#
# ⚠️ **El coordinador es OBLIGATORIO, y la coma cuenta como tal.** La primera
# versión lo marcaba opcional (`(…)?`), con lo que un conector de **solo espacio**
# casaba como coordinación: en «financia a Colombia programas de vigilancia»,
# `Colombia` y `programas` quedaban agrupados como si el texto dijera «Colombia y
# programas», y la relación se emitía dos veces. Dos entidades pegadas no están
# coordinadas — son aposición o complemento, y el texto no autoriza a repartir el
# verbo entre las dos.
COORDINADORES = re.compile(
    r"(?:[,;]|&|\by\b|\be\b|\band\b|\bo\b|\bu\b|\bor\b|\bou\b|"
    r"\bas well as\b|\bas[íi] como\b|\bjunto con\b|\btogether with\b|\s)+",
    re.I)


def _es_coordinacion(entre: str) -> bool:
    """¿El texto entre dos entidades es solo una coordinación?

    Exige que quede **algo** tras recortar: la cadena vacía no coordina nada.
    """
    limpio = entre.strip()
    if not limpio:
        return False
    return bool(COORDINADORES.fullmatch(limpio))

# Participio, que es lo que de verdad marca la pasiva. El auxiliar no se exige:
# ver el 🔴 de `_es_pasiva()` sobre las pasivas reducidas del inglés.
_PASIVA_PARTICIPIO = re.compile(
    r"\w+(ad[oa]s?|id[oa]s?|ed|to)\b", re.I)

# El agente de la pasiva. Se exige que el conector TERMINE con esto, o sea que la
# entidad de la derecha sea el agente.
_AGENTE = re.compile(r"\b(por|by|pelos?|pelas?)\s*$", re.I)


@dataclass
class Triple:
    """Una tripleta con su evidencia textual.

    Los campos van en inglés como en `Document` y `Chunk`, y por el mismo motivo:
    la traducción a los nombres que pida el entregable se hace una sola vez, al
    serializar (`ESTADO.md` §0).

    `doc_id` y `chunk_id` **no son opcionales en la práctica**: §7.2 exige que
    cada tripleta mantenga la referencia a su origen «lo que permite rastrear la
    evidencia textual de cada relación». Los rellena quien llama, que es el que
    sabe de qué chunk salió la unidad.
    """
    subject: str
    relation: str
    object: str
    subject_type: str = ""
    object_type: str = ""
    doc_id: str = ""
    chunk_id: str = ""
    evidence: str = ""          # el conector que disparó la relación
    passive: bool = False       # si se invirtió la dirección por voz pasiva

    def as_tuple(self) -> Tuple[str, str, str]:
        return (self.subject, self.relation, self.object)


# ── Detección de la relación en el conector ──────────────────────────────────

def _buscar_verbo(conector: str) -> Optional[str]:
    """Devuelve la relación canónica que aparece en el conector, o `None`.

    Recorre el inventario en orden de declaración, así que si un conector
    contiene dos verbos gana el primero del inventario. Es determinista, que es
    lo que importa para §1.4; que sea el «mejor» de los dos es una decisión que
    no se puede tomar sin análisis sintáctico.

    ⚠️ **Se quitan las tildes antes de comparar, y no es opcional.** Las raíces
    del inventario se escriben sin acentos, pero las conjugaciones españolas los
    llevan: `prohíbe`, `financió`, `atacó`, `lanzó`, `está desarrollando`. La
    primera versión solo pasaba a minúsculas, así que la raíz `prohib` **no
    casaba con «prohíbe»** — y el resultado era una tripleta menos, en silencio.
    """
    minusculas = _normalizar(conector)
    for relacion, raices in RELACIONES.items():
        for raiz in raices:
            # `\b` delante para no casar dentro de otra palabra —la raíz `us` de
            # «usar» no debe dispararse con «Rusia»— y TERMINACIONES detrás para
            # que lo que sigue sea una forma verbal y no cualquier palabra que
            # empiece igual.
            for m in re.finditer(r"\b" + re.escape(raiz) + TERMINACIONES,
                                 minusculas):
                # El verbo tiene que estar al principio del conector: es donde va
                # el predicado de una oración simple.
                if m.start() > MAX_ANTES_DEL_VERBO:
                    break
                # Precedido de determinante ⇒ es un sustantivo deverbal, no el
                # verbo. Se sigue buscando: puede haber otra ocurrencia válida
                # más adelante dentro del margen.
                if DETERMINANTE_PREVIO.search(minusculas[:m.start()]):
                    continue
                return relacion
    return None


def _es_pasiva(conector: str) -> bool:
    """¿El conector es una construcción pasiva con agente a la derecha?

    Requiere **participio y marcador de agente al final**. Solo el `por` no basta
    —«viajó por Europa» no es pasiva, y «viajo» no es participio— y solo el
    participio tampoco: «el informe publicado detalla» es activa y no termina en
    agente.

    🔴 **El auxiliar NO se exige, y la primera versión sí lo hacía.** Eso perdía
    las **pasivas reducidas**, que en inglés son la forma más común del giro: «the
    Process launched **by** Japan» no lleva `is`/`was`, y se emitía
    `(IA, lanza, Japón)` con la dirección invertida. Con el participio y el agente
    basta; el auxiliar solo añadía una condición que el idioma no siempre cumple.
    """
    return bool(_PASIVA_PARTICIPIO.search(conector) and _AGENTE.search(conector))


# ── Agrupación por coordinación ──────────────────────────────────────────────

def _agrupar_coordinadas(
    texto: str, entidades: Sequence[dict]
) -> List[List[dict]]:
    """Agrupa entidades consecutivas separadas solo por coordinadores.

    «EE.UU. y China desarrollan armas» da los grupos `[[EE.UU., China], [armas]]`,
    de modo que la relación se emite de cada miembro del grupo izquierdo a cada
    miembro del derecho. Sin esto, la regla de adyacencia perdería la tripleta de
    EE.UU., que es justamente la que el texto afirma primero.
    """
    if not entidades:
        return []
    grupos: List[List[dict]] = [[entidades[0]]]
    for anterior, actual in zip(entidades, entidades[1:]):
        entre = texto[anterior["end"]:actual["start"]]
        if _es_coordinacion(entre):
            grupos[-1].append(actual)      # misma coordinación
        else:
            grupos.append([actual])
    return grupos


# ── La función principal ─────────────────────────────────────────────────────

def extraer_relaciones(
    texto: str,
    entidades: Sequence[dict],
    *,
    doc_id: str = "",
    chunk_id: str = "",
    max_distancia: int = MAX_DISTANCIA,
    incluir_coocurrencia: bool = False,
    alias: Optional[Dict[str, str]] = None,
) -> List[Triple]:
    """Extrae tripletas de una unidad de texto y sus entidades.

    `entidades` es la lista que devuelve GLiNER: diccionarios con `start`, `end`,
    `text` y `label`. Se ordenan por posición aquí mismo, para no depender de que
    el modelo las devuelva ordenadas.

    Las entidades se canonizan antes de emitir la tripleta, de modo que
    `(NATO, firma, …)` y `(OTAN, firma, …)` sean el **mismo** hecho sobre el mismo
    nodo. Sin ese paso el grafo tendría el hecho duplicado en dos nodos.
    """
    if alias is None:
        alias = cargar_alias()

    ordenadas = sorted(entidades, key=lambda e: (e["start"], e["end"]))
    grupos = _agrupar_coordinadas(texto, ordenadas)

    tripletas: List[Triple] = []
    # Solo pares de grupos ADYACENTES: si hay un grupo en medio, el verbo
    # pertenece a alguno de los dos vínculos cortos y no al largo (problema 3 del
    # docstring del módulo).
    for izquierda, derecha in zip(grupos, grupos[1:]):
        fin_izquierda = max(e["end"] for e in izquierda)
        inicio_derecha = min(e["start"] for e in derecha)
        conector = texto[fin_izquierda:inicio_derecha]

        if len(conector) > max_distancia:
            continue
        # Frontera de oración: si el conector la cruza, no hay vínculo posible.
        if FIN_DE_ORACION.search(conector):
            continue
        # La negación se comprueba antes de buscar el verbo: da igual qué verbo
        # sea si la oración lo niega.
        if NEGACIONES.search(conector):
            continue

        relacion = _buscar_verbo(conector)
        if relacion is None:
            if not incluir_coocurrencia:
                continue
            relacion = RELACION_COOCURRENCIA

        pasiva = _es_pasiva(conector)
        origen, destino = (derecha, izquierda) if pasiva else (izquierda, derecha)

        for e_sujeto in origen:
            for e_objeto in destino:
                s_nombre, s_tipo, _ = canonizar(
                    e_sujeto["text"], e_sujeto.get("label", ""), alias)
                o_nombre, o_tipo, _ = canonizar(
                    e_objeto["text"], e_objeto.get("label", ""), alias)
                # Un nodo no se relaciona consigo mismo: «EE.UU. … Estados
                # Unidos» canonizan al mismo nodo y la arista sería un bucle sin
                # información.
                if (s_nombre, s_tipo) == (o_nombre, o_tipo):
                    continue
                tripletas.append(Triple(
                    subject=s_nombre, relation=relacion, object=o_nombre,
                    subject_type=s_tipo, object_type=o_tipo,
                    doc_id=doc_id, chunk_id=chunk_id,
                    evidence=conector.strip()[:120], passive=pasiva))
    return tripletas


# ── Gazetteer de prueba, para validar sin GLiNER ─────────────────────────────

def _es_sigla(forma: str) -> bool:
    """¿La forma es una sigla, o sea no tiene ni una minúscula?

    `EE.UU.`, `UN`, `MAPP-OEA` y `LAWS` sí; `Estados Unidos` no. Los puntos y
    guiones no cuentan, así que basta comparar con la versión en mayúsculas.
    """
    return any(c.isalpha() for c in forma) and forma == forma.upper()


def gazetteer(alias_variantes: Dict[str, List[str]]) -> "re.Pattern":
    """Compila un patrón que localiza las formas superficiales conocidas.

    ⚠️ **Esto NO sustituye a GLiNER.** Solo encuentra lo que ya está enumerado en
    la tabla de alias, así que no descubre entidades nuevas — que es precisamente
    para lo que sirve el NER. Existe para poder **probar el extractor de
    relaciones sobre prosa real del corpus** antes de tener la pasada de GLiNER,
    y para eso es suficiente: las relaciones se prueban con entidades conocidas.

    🔴 **Las siglas se buscan respetando las mayúsculas, y los nombres no.** La
    primera versión aplicaba `re.I` a todo, y entonces la variante `UN` de
    Naciones Unidas casaba con el **artículo español «un»**: el gazetteer
    encontraba «Naciones Unidas» en cada «un» del texto, y el extractor emitía
    tripletas sobre una entidad que no estaba ahí. Lo mismo habría pasado con
    `US` y el pronombre inglés «us», o con `IA`/`AI` dentro de texto en
    mayúsculas.

    El coste de la distinción es que «la otan» en minúsculas no se detecta. Es un
    coste aceptable: una sigla escrita en minúsculas es indistinguible de una
    palabra común, y **preferir el falso negativo al falso positivo** es la misma
    línea que `ESTADO.md` §10 fija para el OCR.

    Los nombres van primero en la alternancia y ordenados de más largo a más
    corto, para que en «Estados Unidos» gane el nombre completo y no una parte.
    """
    formas = {v for vs in alias_variantes.values() for v in vs}
    siglas = sorted((f for f in formas if _es_sigla(f)), key=len, reverse=True)
    nombres = sorted((f for f in formas if not _es_sigla(f)), key=len,
                     reverse=True)

    # Sin `re.I` global: por defecto todo distingue mayúsculas, y `(?i:…)` abre
    # la excepción solo para el grupo de los nombres.
    partes = []
    if nombres:
        partes.append(r"(?i:" + "|".join(re.escape(f) for f in nombres) + r")")
    if siglas:
        partes.append("|".join(re.escape(f) for f in siglas))

    # ⚠️ Los límites son lookarounds `(?<!\w)` y `(?!\w)`, **no** `\b`. Con `\b`
    # las siglas que acaban en punto no se detectan: `EE.UU.` termina en un
    # carácter que no es de palabra, así que tras él no hay frontera que
    # satisfacer y el patrón falla. Y `EE.UU.` es la forma más frecuente del país
    # más citado del corpus, así que el fallo se lo habría llevado casi todo.
    return re.compile(r"(?<!\w)(?:" + "|".join(partes) + r")(?!\w)")


def buscar_entidades(texto: str, patron: "re.Pattern",
                     etiqueta: str = "organization") -> List[dict]:
    """Encuentra las entidades del gazetteer, en el formato que da GLiNER."""
    return [{"start": m.start(), "end": m.end(), "text": m.group(0),
             "label": etiqueta}
            for m in patron.finditer(texto)]


# ── Autoprueba ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from graph.canonical import VARIANTES_SEMILLA

    alias = cargar_alias()
    fallos = 0
    linea = "─" * 74

    def caso(nombre: str, texto: str, ents: List[Tuple[int, int, str, str]],
             esperado: List[Tuple[str, str, str]], **kw) -> None:
        """Comprueba las tripletas que produce un texto con entidades dadas."""
        global fallos
        entidades = [{"start": a, "end": b, "text": t, "label": l}
                     for a, b, t, l in ents]
        obtenido = [t.as_tuple() for t in extraer_relaciones(
            texto, entidades, alias=alias, **kw)]
        ok = obtenido == esperado
        print(f"  {'✔' if ok else '✘'} {nombre}")
        print(f"      texto    : {texto}")
        print(f"      esperado : {esperado}")
        if not ok:
            print(f"      OBTENIDO : {obtenido}")
            fallos += 1

    print(f"{linea}\nA. DIRECCIÓN Y VOZ\n{linea}")
    t = "EE.UU. desarrolla sistemas de armas autónomos avanzados."
    caso("activa: sujeto → objeto", t,
         [(0, 6, "EE.UU.", "country"), (18, 45, "sistemas de armas autónomos", "weapon system")],
         [("Estados Unidos", "desarrolla", "sistema de armas autónomo")])

    t = "Los sistemas de armas autónomos son desarrollados por EE.UU."
    caso("pasiva: se invierte la dirección", t,
         [(4, 31, "sistemas de armas autónomos", "weapon system"), (53, 59, "EE.UU.", "country")],
         [("Estados Unidos", "desarrolla", "sistema de armas autónomo")])

    print(f"\n{linea}\nB. NEGACIÓN — no se afirma lo que el texto niega\n{linea}")
    t = "China no ha firmado el Convenio de Ginebra."
    caso("negación: no se emite tripleta", t,
         [(0, 5, "China", "country"), (23, 41, "Convenio de Ginebra", "treaty")],
         [])

    print(f"\n{linea}\nC. COORDINACIÓN — dos sujetos, dos tripletas\n{linea}")
    t = "EE.UU. y China desarrollan armas antisatélite."
    caso("coordinación con «y»", t,
         [(0, 6, "EE.UU.", "country"), (9, 14, "China", "country"),
          (27, 44, "armas antisatélite", "weapon system")],
         [("Estados Unidos", "desarrolla", "arma antisatélite"),
          ("China", "desarrolla", "arma antisatélite")])

    print(f"\n{linea}\nD. ADYACENCIA Y DISTANCIA\n{linea}")
    # El verbo SÍ está en el inventario, para que la prueba mida la adyacencia y
    # no el hecho accidental de que el verbo falte. Debe salir la tripleta
    # EE.UU.→Colombia y **no** la EE.UU.→programas, que salta una entidad.
    t = "EE.UU. financia a Colombia programas de vigilancia."
    caso("entidad en medio: no se salta ningún grupo", t,
         [(0, 6, "EE.UU.", "country"), (18, 26, "Colombia", "country"),
          (27, 50, "programas de vigilancia", "technology")],
         [("Estados Unidos", "financia", "Colombia")])

    t = ("La OTAN publicó un informe extenso sobre doctrina, presupuesto, "
         "capacidades y adiestramiento, y por separado se mencionó que "
         "Colombia opera aeronaves.")
    caso("distancia excesiva: no se relaciona", t,
         [(3, 7, "OTAN", "organization"), (120, 128, "Colombia", "country")],
         [])

    print(f"\n{linea}\nE. RELACIONES QUE NO SE FUSIONAN\n{linea}")
    t = "El Tratado del Espacio Exterior prohíbe las armas nucleares."
    caso("prohibe ≠ regula", t,
         [(3, 31, "Tratado del Espacio Exterior", "treaty"),
          (43, 58, "armas nucleares", "weapon system")],
         [("Tratado del Espacio Exterior", "prohibe", "armas nucleares")])

    print(f"\n{linea}\nF. PROSA REAL DEL CORPUS (gazetteer, sin GLiNER)\n{linea}")
    patron = gazetteer(VARIANTES_SEMILLA)
    ruta = Path(__file__).resolve().parents[1] / "data" / "chunks.jsonl"
    TAB = {"csv", "xlsx", "pbf"}
    encontradas: List[Triple] = []
    revisados = 0
    if ruta.is_file():
        with open(ruta, encoding="utf-8") as f:
            for linea_json in f:
                r = json.loads(linea_json)
                if r["format"] in TAB:
                    continue
                # ⚠️ Se salta el AI Index a propósito. Sus chunks de «prosa» son
                # pies de figura y volcados de ejes —«Number of notable models»,
                # «(1.9). In the members utilize»— y no contienen predicados. Una
                # muestra sacada de ahí no dice nada sobre la precisión del
                # extractor: dice que ese material no tiene relaciones que
                # extraer. Para juzgarlo hace falta prosa argumentativa.
                if "AIINDEX" in r["doc_id"]:
                    continue
                revisados += 1
                ents = buscar_entidades(r["text"], patron)
                if len(ents) >= 2:
                    encontradas += extraer_relaciones(
                        r["text"], ents, doc_id=r["doc_id"],
                        chunk_id=r["chunk_id"], alias=alias)
                if len(encontradas) >= 12 or revisados >= 4000:
                    break
        print(f"  {revisados:,} chunks de prosa revisados → "
              f"{len(encontradas)} tripletas")
        for t3 in encontradas[:12]:
            flecha = "⇐ pasiva" if t3.passive else ""
            print(f"    ({t3.subject}, {t3.relation}, {t3.object}) {flecha}")
            print(f"        {t3.chunk_id}   «…{t3.evidence}…»")
    else:
        print(f"  · no está {ruta.name}; se omite")

    print(f"\n{linea}")
    print(f"relaciones en el inventario: {len(RELACIONES)}")
    if fallos:
        print(f"VEREDICTO: ✘ {fallos} caso(s) fallan")
    else:
        print("VEREDICTO: ✔ todos los casos pasan")
    print(linea)
    raise SystemExit(1 if fallos else 0)
