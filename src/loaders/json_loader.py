"""Loader de documentos JSON del corpus.

Sigue §2.1: "interpretar el objeto y seleccionar explícitamente los campos que
contienen el texto de cada artículo (por ejemplo title, body_text,
body_paragraphs)". Es decir: selección explícita de campos, NO un recorrido
genérico del árbol, porque eso metería url/authors/pdf_links dentro del texto,
que es justo lo que el PDF dice que no se haga.

El loader SOLO lee y estructura. No limpia ni normaliza (eso es §2.2, y va en el
Preprocessor), ni detecta idioma, ni fragmenta.

CONTRATO: `load()` devuelve `list[Document]`, igual que el resto de loaders
del pipeline (ver nota en pdf_loader.py) — aquí siempre es una lista de un
solo elemento, pero se mantiene la misma interfaz para que el orquestador no
tenga que distinguir de dónde vino cada Document.
"""

import json
from pathlib import Path
from typing import Any

from core.catalog import CatalogEntry
from core.document import Document
from loaders.base_loader import BaseLoader

# Orden de prioridad para encontrar el cuerpo del documento. Se usa el PRIMER
# campo que exista y traiga contenido; los demás se ignoran como texto.
#
# El orden importa por una razón medida: body_text es idéntico a unir
# body_paragraphs (comprobado en 229/229 archivos de F2), así que si se usaran
# los dos se duplicaría todo el corpus. body_paragraphs va primero porque
# conserva los límites de párrafo que necesita el chunking (§3.2, §3.3).
BODY_FIELDS = (
    "body_paragraphs",   # forma estándar: CSIS, ESA, INPE, SWF, SIPRI, CEOBS, Alertas...
    "sections",          # CENIA: lista de {"heading": ..., "paragraphs": [...]}
    "content",           # el informe SWF 2026
    "abstract",          # CEEEP: revista académica, solo resumen
    "body_text",         # respaldo por si algún archivo no trae body_paragraphs
    "excerpt",           # último recurso: mejor un resumen corto que nada
)

# Campos con texto secundario que SÍ aporta significado y se anexa después del
# cuerpo. `lists` son viñetas de contenido real en CENIA (títulos de charlas).
EXTRA_TEXT_FIELDS = ("lists",)

# Umbral (en palabras) por debajo del cual el cuerpo se considera un "stub".
# Hay páginas cuyo body_paragraphs es solo "Read the full article at CSIS.org"
# mientras que su `excerpt` sí resume el contenido. Por debajo de este umbral se
# anexa el excerpt; por encima no, para no duplicar el resumen en cada documento.
SHORT_BODY_WORDS = 40

# Campos que nunca son cuerpo: son descriptivos y van a metadata (§2.1: "conviene
# conservarlos como metadata del documento en lugar de mezclarlos con el texto").
TITLE_FIELD = "title"

# En los JSON que son listas (catálogos de descarga, índices de tiles) no hay
# prosa. Se emiten como pares "clave: valor" por registro, que es el tratamiento
# que §2.1 prescribe para datos tabulares (CSV/XLSX): "de modo que cada valor
# conserve el nombre de su columna como contexto".
RECORD_SKIP_KEYS = frozenset({
    "status", "size_bytes", "size_mb", "from_cache", "error",
    "content_type", "scraped_at", "local_path",
})


class JSONLoadError(Exception):
    pass


class JSONLoader(BaseLoader):
    """Convierte un archivo .json del corpus en un Document."""

    # `entry` trae doc_id / source / phenomenon / format ya resueltos por el
    # catálogo. El loader no lee el Excel: recibe la metadata ya masticada.
    def load(self, path: str | Path, entry: CatalogEntry) -> list[Document]:
        raw = self._read(path)

        # Los JSON del corpus son objetos en su mayoría, pero 8 son listas.
        # isinstance comprueba el tipo ANTES de llamar .get(), que solo existe
        # en diccionarios y reventaría con AttributeError en esos 8.
        if isinstance(raw, dict):
            text, metadata = self._from_object(raw)
        elif isinstance(raw, list):
            text, metadata = self._from_records(raw)
        else:
            # Un JSON que en su raíz es un número o una cadena suelta.
            raise JSONLoadError(f"Estructura JSON no soportada ({type(raw).__name__}): {path}")

        # `titulo` es un campo propio de Document, no debe quedar duplicado
        # dentro de metadata_adicional. El título ya se agregó como primer
        # bloque de `text` en _from_object (señal fuerte para recuperación);
        # aquí se saca también como campo dedicado, sin quitarlo del texto.
        titulo = _as_text(metadata.pop(TITLE_FIELD, None)) or None

        return [Document(
            doc_id=entry.doc_id,
            fuente=entry.source,
            formato=entry.format,
            fenomeno=entry.phenomenon,
            idioma=None,           # lo llena el Preprocessor (§2.2), no el loader
            titulo=titulo,
            texto=text,
            metadata_adicional=metadata,
        )]

    # ---------- lectura ----------

    @staticmethod
    def _read(path: str | Path) -> Any:
        # encoding="utf-8" EXPLÍCITO: en Windows el valor por defecto es la
        # codificación del sistema, que corrompe los acentos del corpus en
        # portugués y español sin lanzar ningún error. Fallo silencioso.
        try:
            texto = Path(path).read_text(encoding="utf-8")
        except OSError as error:
            # Archivo inexistente, sin permisos, ruta rota, etc. Antes solo
            # se capturaba JSONDecodeError; un archivo faltante tumbaba todo
            # el orquestador en vez de quedar como un fallo de esa entrada.
            raise JSONLoadError(f"No se pudo leer el archivo en {path}: {error}") from error

        try:
            return json.loads(texto)
        except json.JSONDecodeError as error:
            raise JSONLoadError(f"JSON inválido en {path}: {error}") from error

    # ---------- caso 1: el JSON es un objeto (la gran mayoría) ----------

    def _from_object(self, obj: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        bloques: list[str] = []          # trozos de texto que luego se unen
        consumidos: set[str] = set()     # claves ya usadas como texto

        # El título va primero: es señal fortísima para la recuperación y §2.1
        # lo nombra explícitamente entre los campos de texto. Nota: el título
        # queda tanto en `text` (para el chunking/embeddings) como, más abajo
        # en load(), en el campo dedicado `titulo` de Document — no se quita
        # de aquí, solo se copia también hacia el campo dedicado.
        titulo = _as_text(obj.get(TITLE_FIELD))
        if titulo:
            bloques.append(titulo)
            consumidos.add(TITLE_FIELD)

        # Se recorre BODY_FIELDS en orden y se toma SOLO el primero con contenido.
        for campo in BODY_FIELDS:
            parrafos = _paragraphs(obj.get(campo))
            if parrafos:
                bloques.extend(parrafos)
                consumidos.add(campo)
                break   # <- la clave de no duplicar: se corta al primer acierto

        # Texto secundario que sí aporta (viñetas de CENIA), después del cuerpo.
        for campo in EXTRA_TEXT_FIELDS:
            parrafos = _paragraphs(obj.get(campo))
            if parrafos:
                bloques.extend(parrafos)
                consumidos.add(campo)

        # Rescate para páginas cuyo cuerpo es un stub ("lee el artículo completo
        # en...") pero cuyo excerpt sí describe el contenido. Se comprueba que no
        # esté ya incluido para no repetirlo.
        if "excerpt" not in consumidos and _word_count(bloques) < SHORT_BODY_WORDS:
            resumen = _as_text(obj.get("excerpt"))
            if resumen and resumen not in bloques:
                bloques.append(resumen)
                consumidos.add("excerpt")

        # Todo lo que no se usó como texto se conserva como metadata. OJO: el
        # título SÍ se conserva aquí también (no está en `consumidos` a
        # propósito para metadata — se quita explícitamente en load() para
        # pasarlo al campo `titulo` dedicado, no aquí, así esta función no
        # cambia de comportamiento respecto al original).
        metadata = {k: v for k, v in obj.items() if k not in consumidos or k == TITLE_FIELD}
        # "\n\n" separa párrafos. Se conserva a propósito: es la frontera que el
        # chunking por párrafo (§3.2) y la completitud lingüística (§3.3) usan.
        return "\n\n".join(bloques), metadata

    # ---------- caso 2: el JSON es una lista de registros ----------

    def _from_records(self, records: list[Any]) -> tuple[str, dict[str, Any]]:
        bloques: list[str] = []
        for record in records:
            if not isinstance(record, dict):
                # Una lista de cadenas sueltas: se toma cada una como párrafo.
                texto = _as_text(record)
                if texto:
                    bloques.append(texto)
                continue

            # Cada registro se vuelve un bloque de líneas "clave: valor",
            # saltando los campos puramente técnicos del scraping.
            lineas = [
                f"{clave}: {_as_text(valor)}"
                for clave, valor in record.items()
                if clave not in RECORD_SKIP_KEYS and _as_text(valor)
            ]
            if lineas:
                bloques.append("\n".join(lineas))

        # n_registros permite detectar después los archivos sin contenido útil
        # (p. ej. DEFENSA21_articulos-2.json, que es una lista vacía).
        return "\n\n".join(bloques), {"n_registros": len(records)}


# ---------- utilidades ----------


def _paragraphs(value: Any) -> list[str]:
    """Convierte cualquiera de las formas de cuerpo del corpus en una lista de párrafos."""
    if value is None:
        return []

    # Caso simple: ya es un texto suelto (body_text, abstract, content, excerpt).
    if isinstance(value, str):
        texto = value.strip()
        return [texto] if texto else []

    if isinstance(value, list):
        parrafos: list[str] = []
        for item in value:
            if isinstance(item, str):
                # body_paragraphs y lists: lista de cadenas.
                texto = item.strip()
                if texto:
                    parrafos.append(texto)
            elif isinstance(item, dict):
                # sections de CENIA: {"heading": "...", "paragraphs": [...]}.
                # El heading se conserva porque es una señal estructural útil
                # para el chunking jerárquico (§3.2).
                encabezado = _as_text(item.get("heading"))
                if encabezado:
                    parrafos.append(encabezado)
                parrafos.extend(_paragraphs(item.get("paragraphs")))
        return parrafos

    # El `content` del informe SWF 2026 es un diccionario anidado:
    # {"sections": {"Titulo de la seccion": "texto...", ...}}. Sin esta rama se
    # perdía el documento entero.
    if isinstance(value, dict):
        parrafos: list[str] = []
        for clave, contenido in value.items():
            if isinstance(contenido, str):
                texto = contenido.strip()
                if texto:
                    # La clave es el encabezado de la sección; se emite aparte
                    # para que el chunking estructural pueda aprovecharla.
                    parrafos.append(str(clave).strip())
                    parrafos.append(texto)
            else:
                # Contenedor intermedio (p. ej. la clave "sections"): se baja un
                # nivel sin emitir su nombre, que no es contenido.
                parrafos.extend(_paragraphs(contenido))
        return parrafos

    return []


def _word_count(bloques: list[str]) -> int:
    # Cuenta palabras de todos los bloques juntos, para decidir si el cuerpo es un stub.
    return sum(len(bloque.split()) for bloque in bloques)


def _as_text(value: Any) -> str:
    """Aplana un valor a texto plano, sin inventar formato."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        # Une listas de valores simples con "; " (p. ej. authors, keywords).
        return "; ".join(t for t in (_as_text(v) for v in value) if t)
    if isinstance(value, dict):
        return "; ".join(f"{k}: {t}" for k, v in value.items() if (t := _as_text(v)))
    return str(value)


if __name__ == "__main__":
    import sys
    from collections import Counter

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from core.catalog import Catalog

    root = Path(__file__).resolve().parents[1] / "data" / "data_raw"
    catalog = Catalog.from_excel(root / "Indice_Datos_Codefest.xlsx")
    loader = JSONLoader()

    documentos, fallos, vacios = [], [], []
    for entrada in catalog.entries(format="json"):
        try:
            documentos.extend(loader.load(root / entrada.source, entrada))
        except JSONLoadError as error:
            fallos.append((entrada.source, str(error)))

    # Un documento con muy poco texto casi siempre significa que el loader no
    # encontró el campo de cuerpo de ese esquema. Es la señal de alarma.
    for doc in documentos:
        if len(doc.texto.split()) < 20:
            vacios.append(doc)

    print(f"documentos JSON cargados : {len(documentos)}")
    print(f"fallos de lectura        : {len(fallos)}")
    print(f"con menos de 20 palabras : {len(vacios)}")

    palabras = sorted(len(d.texto.split()) for d in documentos)
    if palabras:
        print(f"palabras  min={palabras[0]}  mediana={palabras[len(palabras) // 2]}  "
              f"max={palabras[-1]}  total={sum(palabras):,}")

    print("por fenómeno :", dict(sorted(Counter(d.fenomeno for d in documentos).items())))

    for source, error in fallos[:10]:
        print(f"  FALLO: {source} -> {error}")
    for doc in vacios[:10]:
        print(f"  CORTO ({len(doc.texto.split())} pal.): {doc.fuente}")