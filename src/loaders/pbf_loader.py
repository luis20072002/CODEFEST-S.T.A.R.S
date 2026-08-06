"""Loader de documentos PBF (vector tiles) del corpus.

Sigue la especificación: "para extraer su contenido se usa una herramienta
que lea ese formato y lo decodifique. Una vez abierto, se recorren las capas
y, dentro de cada una, los elementos del mapa (municipios, zonas), leyendo
los atributos de cada uno. Esos atributos se pasan a texto como pares
atributo: valor. Como el mismo elemento se repite en varios niveles de zoom
por cada archivo, es conveniente quedarse con una sola versión para no
duplicar la data".

FORMATO CONFIRMADO contra el corpus real: cada .pbf es una TESELA SUELTA (un
único zoom/x/y), guardada en la estructura de carpetas típica de un tile
cache: `tiles/{zoom}/{x}/{nombre}.pbf` (comprobado con las rutas reales de
Amazon_Underworld, p. ej. `tiles/5/10/AMAZONUW_15.pbf`). NO es un contenedor
MBTiles/SQLite — esa fue la hipótesis inicial y no aplicó a este corpus, así
que se decodifica el archivo directo como vector tile, sin SQLite de por
medio.

IMPORTANTE — alcance de la deduplicación con esta granularidad: cada llamada
a `load()` procesa UN archivo = UNA tesela = UN nivel de zoom. La
deduplicación de abajo por lo tanto solo puede operar DENTRO de esa tesela
(entre las distintas capas de un mismo archivo, si un elemento aparece en más
de una). La deduplicación ENTRE niveles de zoom que pide la especificación
—el mismo municipio repetido en tiles/5/... y tiles/12/...— vive en archivos
.pbf *distintos* y por lo tanto en `CatalogEntry` distintas: no se puede
resolver dentro de un único `load()` sin cambiar qué cuenta como "un
documento" (agrupar varias teselas del mismo dataset/zona en un solo
Document en vez de una por archivo). Esa es una decisión de catálogo/
orquestador, no del loader — ver conversación.

El loader SOLO lee y estructura. No limpia ni normaliza (eso es del
Preprocessor, §2.2), no detecta idioma, no fragmenta, y no toca la geometría:
el pipeline de texto no la necesita, así que ni siquiera se decodifica más
allá de lo que exige la librería.
"""

import gzip
from pathlib import Path
from typing import Any

import mapbox_vector_tile

from core.catalog import CatalogEntry
from core.document import Document
from loaders.base_loader import BaseLoader


class PBFLoadError(Exception):
    pass


class PBFLoader(BaseLoader):
    """Convierte un archivo .pbf (una tesela de vector tile) del corpus en un Document."""

    # `entry` trae doc_id / source / phenomenon / format ya resueltos por el
    # catálogo, igual que en el resto de loaders.
    #
    # CONTRATO: devuelve list[Document], igual que el resto del pipeline
    # (ver nota de contrato en pdf_loader.py).
    def load(self, path: str | Path, entry: CatalogEntry) -> list[Document]:
        layers = self._read(path)

        # Deduplicación: la CLAVE es el propio conjunto de atributos, no un
        # id de geometría. Dentro de una misma tesela, un elemento podría
        # aparecer repetido si más de una capa lo incluye (p. ej. un
        # municipio presente tanto en "municipios" como en una capa de
        # "limites"); dos elementos con exactamente los mismos pares
        # atributo:valor se tratan como el mismo elemento, sin asumir el
        # nombre de un campo "id" que puede no existir o llamarse distinto
        # según la capa. (La deduplicación ENTRE zooms/archivos distintos no
        # aplica aquí — ver nota al inicio del archivo.)
        seen: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
        blocks: list[str] = []
        total_features = 0

        for layer_name, features in layers.items():
            for attributes in features:
                total_features += 1
                key = (layer_name, _attributes_key(attributes))
                if key in seen:
                    continue
                seen.add(key)
                block = _element_block(layer_name, attributes)
                if block:
                    blocks.append(block)

        text = "\n\n".join(blocks)
        metadata = {
            "capas": sorted(layers.keys()),
            "n_elementos_total": total_features,
            "n_elementos_unicos": len(blocks),
        }

        return [Document(
            doc_id=entry.doc_id,
            source=entry.source,
            format=entry.format,
            phenomenon=entry.phenomenon,
            language=None,         # lo llena el Preprocessor (§2.2), no el loader
            text=text,
            metadata=metadata,
        )]

    # ---------- lectura ----------

    @staticmethod
    def _read(path: str | Path) -> dict[str, list[dict[str, Any]]]:
        try:
            raw = _decompress(Path(path).read_bytes())
            decoded = mapbox_vector_tile.decode(raw)
        except Exception as error:
            raise PBFLoadError(f"No se pudo decodificar la tesela PBF en {path}: {error}") from error

        layers = {}
        for layer_name, content in decoded.items():
            features = [
                (feature.get("properties") or {})
                for feature in content.get("features", [])
            ]
            layers[layer_name] = features
        return layers


# ---------- utilidades ----------


def _decompress(tile_data: bytes) -> bytes:
    # La mayoría de MBTiles comprimen cada tesela con gzip (cabecera 0x1f8b);
    # algunos exportes la dejan sin comprimir. Se detecta por la cabecera en
    # vez de intentar-y-capturar-excepción, para no depender de que gzip
    # lance siempre el mismo tipo de error ante datos no comprimidos.
    if tile_data[:2] == b"\x1f\x8b":
        return gzip.decompress(tile_data)
    return tile_data


def _attributes_key(attributes: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    # Orden estable (sorted) para que el mismo conjunto de pares, sin importar
    # el orden en que la librería los entregue, produzca la misma clave.
    return tuple(sorted((str(k), _as_text(v)) for k, v in attributes.items()))


def _element_block(layer_name: str, attributes: dict[str, Any]) -> str:
    lines = [f"{key}: {text}" for key, value in attributes.items() if (text := _as_text(value))]
    if not lines:
        return ""
    # La capa se antepone como primera línea: es la señal de qué tipo de
    # elemento es (municipio, zona...), equivalente al "heading" que el
    # JSONLoader conserva para las secciones de CENIA.
    return "\n".join([f"capa: {layer_name}", *lines])


def _as_text(value: Any) -> str:
    """Aplana un valor de atributo de vector tile a texto plano."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else str(value)
    if isinstance(value, int):
        return str(value)
    return str(value).strip()


if __name__ == "__main__":
    import sys
    from collections import Counter

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from core.catalog import Catalog

    root = Path(__file__).resolve().parents[1] / "data" / "data_raw"
    catalog = Catalog.from_excel(root / "Indice_Datos_Codefest.xlsx")
    loader = PBFLoader()

    documents, failures, empty = [], [], []
    for entry in catalog.entries(format="pbf"):
        try:
            documents.extend(loader.load(root / entry.source, entry))
        except PBFLoadError as error:
            failures.append((entry.source, str(error)))

    for doc in documents:
        if doc.metadata["n_elementos_unicos"] == 0:
            empty.append(doc)

    print(f"documentos PBF cargados   : {len(documents)}")
    print(f"fallos de lectura         : {len(failures)}")
    print(f"sin elementos únicos      : {len(empty)}")

    # Cuánto se redujo por la deduplicación DENTRO de cada tesela (entre sus
    # capas). No mide duplicación entre zooms: eso queda entre archivos
    # distintos, fuera del alcance de un único load() — ver nota al inicio
    # del archivo.
    total = sum(d.metadata["n_elementos_total"] for d in documents)
    unique = sum(d.metadata["n_elementos_unicos"] for d in documents)
    if total:
        print(f"elementos totales antes de deduplicar (por tesela) : {total:,}")
        print(f"elementos únicos después de deduplicar (por tesela): {unique:,} "
              f"({unique / total:.1%})")

    print("por fenómeno :", dict(sorted(Counter(d.phenomenon for d in documents).items())))

    for source, error in failures[:10]:
        print(f"  FALLO: {source} -> {error}")
