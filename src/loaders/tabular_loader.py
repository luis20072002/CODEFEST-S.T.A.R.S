"""Loader de documentos XLSX/CSV del corpus.

Sigue §2.1: "se recomienda leer primero la fila de cabecera y luego recorrer
los registros uno a uno, obteniendo cada fila como una secuencia de pares
columna: valor separados por un delimitador, de modo que cada valor conserve
el nombre de su columna como contexto. Las celdas vacías pueden omitirse o
limpiarse. Cada fila puede tratarse como una unidad de fragmentación
independiente".

Es literalmente el mismo tratamiento que el JSONLoader ya aplica a los JSON
que son listas de registros (ver `_from_records` / `RECORD_SKIP_KEYS` en
json_loader.py): no hay prosa que extraer de una tabla, así que cada fila se
convierte en un bloque de pares "columna: valor" (uno por línea, para que el
nombre de columna quede pegado a su valor) y los bloques se separan entre sí
con la misma frontera "\n\n" que usa el resto del pipeline para marcar
unidades fragmentables — aquí la unidad es la fila, tal como pide §2.1.

El loader SOLO lee y estructura. Omitir celdas vacías es un requisito
explícito del enunciado, no limpieza discrecional (eso seguiría siendo
trabajo del Preprocessor, §2.2). No detecta idioma, no fragmenta.
"""

import csv
from datetime import date, datetime
from pathlib import Path
from typing import Any

from openpyxl import load_workbook

from core.catalog import CatalogEntry
from core.document import Document
from loaders.base_loader import BaseLoader

# Delimitadores candidatos para el Sniffer de CSV. El corpus mezcla archivos
# exportados con distintas configuraciones regionales (coma vs. punto y coma
# es el caso típico en exportes de Excel en español), así que no se asume uno
# fijo.
CSV_DELIMITERS = ",;\t"


class TabularLoadError(Exception):
    pass


class TabularLoader(BaseLoader):
    """Convierte un archivo .xlsx o .csv del corpus en un Document."""

    # `entry` trae doc_id / source / phenomenon / format ya resueltos por el
    # catálogo, igual que en JSONLoader y PDFLoader.
    def load(self, path: str | Path, entry: CatalogEntry) -> Document:
        if entry.format == "csv":
            encabezado, filas = self._read_csv(path)
        elif entry.format == "xlsx":
            encabezado, filas = self._read_xlsx(path)
        else:
            raise TabularLoadError(
                f"Formato no soportado por TabularLoader: {entry.format!r}"
            )

        # Cada fila se convierte en un bloque; las filas totalmente vacías
        # (comunes al final de exportes de Excel) no producen bloque y se
        # descartan aquí en vez de arrastrarlas como ruido hasta el chunking.
        bloques = [bloque for fila in filas if (bloque := _row_block(encabezado, fila))]
        text = "\n\n".join(bloques)

        metadata = {
            "columnas": encabezado,
            "n_filas": len(filas),
            "n_filas_con_datos": len(bloques),
        }

        return Document(
            doc_id=entry.doc_id,
            source=entry.source,
            format=entry.format,
            phenomenon=entry.phenomenon,
            language=None,        # lo llena el Preprocessor (§2.2), no el loader
            text=text,
            metadata=metadata,
        )

    # ---------- lectura ----------

    @staticmethod
    def _read_csv(path: str | Path) -> tuple[list[str], list[list[Any]]]:
        try:
            # utf-8-sig: absorbe el BOM que agregan Excel y muchos exportes en
            # Windows; con utf-8 a secas ese BOM queda pegado al nombre de la
            # primera columna ("\ufeffid" en vez de "id"), silenciosamente.
            with open(path, newline="", encoding="utf-8-sig") as f:
                muestra = f.read(4096)
                f.seek(0)
                try:
                    dialecto = csv.Sniffer().sniff(muestra, delimiters=CSV_DELIMITERS)
                except csv.Error:
                    # Muestra demasiado corta o ambigua (p. ej. una sola
                    # columna): coma por defecto en vez de fallar el archivo.
                    dialecto = csv.excel
                filas = list(csv.reader(f, dialecto))
        except OSError as error:
            raise TabularLoadError(f"No se pudo leer el CSV en {path}: {error}") from error

        if not filas:
            return [], []

        encabezado = _clean_header(filas[0])
        return encabezado, filas[1:]

    @staticmethod
    def _read_xlsx(path: str | Path) -> tuple[list[str], list[list[Any]]]:
        try:
            # read_only + data_only: igual que en core/catalog.py — se
            # necesitan los VALORES calculados, no las fórmulas, y read_only
            # evita cargar todo el árbol de estilos que no se va a usar aquí.
            libro = load_workbook(path, read_only=True, data_only=True)
        except Exception as error:
            raise TabularLoadError(f"No se pudo leer el XLSX en {path}: {error}") from error

        # Se lee solo la hoja activa. Los archivos tabulares del corpus son
        # catálogos de descarga / índices de una sola hoja; si en el futuro
        # aparece un XLSX con varias hojas relevantes, hay que decidir
        # explícitamente cómo repartirlas (¿un Document por hoja? ¿se
        # concatenan?) en vez de asumirlo aquí en silencio.
        hoja = libro.active
        filas_crudas = list(hoja.iter_rows(values_only=True))
        libro.close()

        if not filas_crudas:
            return [], []

        encabezado = _clean_header(list(filas_crudas[0]))
        filas = [list(fila) for fila in filas_crudas[1:]]
        return encabezado, filas


# ---------- utilidades ----------


def _clean_header(fila_cruda: list[Any]) -> list[str]:
    # Columnas sin nombre (encabezado vacío, común en columnas auxiliares de
    # Excel) reciben un nombre posicional para no perder el valor de esa
    # celda ni desalinear el resto de columnas.
    encabezado = []
    for i, valor in enumerate(fila_cruda):
        nombre = _as_text(valor)
        encabezado.append(nombre if nombre else f"columna_{i + 1}")
    return encabezado


def _row_block(encabezado: list[str], fila: list[Any]) -> str:
    # zip corta a la longitud más corta: si una fila de CSV trae menos
    # celdas que columnas de encabezado (líneas truncadas, error de exporte),
    # se toma lo que hay en vez de fallar todo el archivo por una fila.
    lineas = [
        f"{nombre}: {texto}"
        for nombre, valor in zip(encabezado, fila)
        if (texto := _as_text(valor))  # celdas vacías se omiten (requisito explícito)
    ]
    return "\n".join(lineas)


def _as_text(value: Any) -> str:
    """Aplana un valor de celda a texto plano, sin inventar formato."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        # openpyxl devuelve enteros como float (3 -> 3.0); sin esto, cada
        # cantidad entera del corpus saldría con un ".0" que no está en la
        # celda original.
        return str(int(value)) if value.is_integer() else str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value).strip()


if __name__ == "__main__":
    import sys
    from collections import Counter

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from core.catalog import Catalog

    root = Path(__file__).resolve().parents[1] / "data" / "data_raw"
    catalog = Catalog.from_excel(root / "Indice_Datos_Codefest.xlsx")
    loader = TabularLoader()

    documentos, fallos, vacios = [], [], []
    for entrada in list(catalog.entries(format="csv")) + list(catalog.entries(format="xlsx")):
        try:
            documentos.append(loader.load(root / entrada.source, entrada))
        except TabularLoadError as error:
            fallos.append((entrada.source, str(error)))

    # Un documento sin filas con datos casi siempre significa: archivo vacío
    # más allá del encabezado (como el caso DEFENSA21_articulos-2.json del
    # JSONLoader, pero en versión tabular) o encabezado mal detectado.
    for doc in documentos:
        if doc.metadata["n_filas_con_datos"] == 0:
            vacios.append(doc)

    print(f"documentos XLSX/CSV cargados : {len(documentos)}")
    print(f"fallos de lectura            : {len(fallos)}")
    print(f"sin filas con datos          : {len(vacios)}")

    print("por fenómeno :", dict(sorted(Counter(d.phenomenon for d in documentos).items())))

    for source, error in fallos[:10]:
        print(f"  FALLO: {source} -> {error}")
    for doc in vacios[:10]:
        print(f"  VACIO: {doc.source}")