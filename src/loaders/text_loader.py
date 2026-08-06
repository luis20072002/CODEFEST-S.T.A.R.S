"""Loader de archivos de texto plano (.txt).

§2.1: «Markdown/TXT: el contenido es mayoritariamente texto plano con marcado
ligero. Los encabezados (#, ##) y separadores son especialmente útiles como
señales de segmentación.»

Es el loader más simple del pipeline: leer y devolver. Lo único que no es
trivial es la codificación — ver `_leer`.

Hoy el catálogo tiene **un** archivo .txt, pero no es un archivo cualquiera:
`SWF_full-text.txt`, que por el nombre es el texto completo de un informe de
Secure World Foundation. Estaba perdiéndose entero porque `LOADERS` no tenía
la clave "txt".

`.md` se registra al mismo loader: la Tabla 1 lo admite como valor de `formato`
y §2.1 los trata juntos. Hoy no hay ninguno en el corpus.
"""

import re
from pathlib import Path
from typing import Any

from core.catalog import CatalogEntry
from core.document import Document
from loaders.base_loader import BaseLoader

# Codificaciones a probar, en orden. UTF-8 primero porque es lo que declara
# §2.2 y lo que trae la inmensa mayoría del corpus; cp1252 después porque es
# el defecto de Windows y es de donde salen los .txt guardados a mano.
CODIFICACIONES = ("utf-8-sig", "utf-8", "cp1252")

# Dos o más saltos de línea = frontera de párrafo, igual que en los otros
# loaders. Los saltos simples dentro de un párrafo suelen ser ajuste de ancho
# de línea, no cambio de contenido, así que NO se tratan como frontera:
# partirlos ahí cortaría oraciones, y §3.3 lo prohíbe.
LINEA_EN_BLANCO = re.compile(r"\n\s*\n")


class TextLoadError(Exception):
    pass


class TextLoader(BaseLoader):
    """Convierte un .txt (o .md) del corpus en un Document."""

    def load(self, path: str | Path, entry: CatalogEntry) -> list[Document]:
        crudo = self._leer(path)

        # Se normalizan los finales de línea de Windows y Mac clásico ANTES de
        # buscar párrafos: si no, "\r\n\r\n" no casaría con el patrón y el
        # documento entero saldría como un solo bloque sin fronteras.
        crudo = crudo.replace("\r\n", "\n").replace("\r", "\n")

        parrafos = [p.strip() for p in LINEA_EN_BLANCO.split(crudo)]
        texto = "\n\n".join(p for p in parrafos if p)

        metadata: dict[str, Any] = {
            "n_parrafos": len(parrafos),
            "bytes_origen": Path(path).stat().st_size,
        }

        return [Document(
            doc_id=entry.doc_id,
            source=entry.source,
            format=entry.format,
            phenomenon=entry.phenomenon,
            language=None,        # lo llena el Preprocessor (§2.2)
            text=texto,
            metadata=metadata,
        )]

    @staticmethod
    def _leer(path: str | Path) -> str:
        """Lee el archivo probando varias codificaciones.

        No se puede asumir UTF-8 a secas: los .txt del corpus son archivos
        sueltos de procedencia desconocida. Se prueba en orden y se devuelve
        la primera que decodifique limpio; si ninguna lo consigue, se fuerza
        UTF-8 con reemplazo — perder unos caracteres es mucho mejor que perder
        el documento entero por un byte suelto.

        `utf-8-sig` va primero porque es UTF-8 que además se come el BOM si
        está; con `utf-8` a secas, un BOM aparecería como "\\ufeff" pegado a la
        primera palabra del texto.
        """
        try:
            crudo = Path(path).read_bytes()
        except OSError as error:
            raise TextLoadError(f"No se pudo abrir {path}: {error}") from error

        for codificacion in CODIFICACIONES:
            try:
                return crudo.decode(codificacion)
            except UnicodeDecodeError:
                continue
        return crudo.decode("utf-8", errors="replace")


if __name__ == "__main__":
    # Diagnóstico: `py -m loaders.text_loader` desde src/.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from core.catalog import Catalog

    raiz = Path(__file__).resolve().parents[1] / "data" / "data_raw"
    catalogo = Catalog.from_excel(raiz / "Indice_Datos_Codefest.xlsx")
    loader = TextLoader()

    entradas = [e for e in catalogo.entries() if e.format in ("txt", "md")]
    print(f"{len(entradas)} archivo(s) de texto en el catálogo\n")

    for entrada in entradas:
        documento = loader.load(raiz / entrada.source, entrada)[0]
        print(f"{entrada.doc_id}  ({Path(entrada.source).name})")
        print(f"  palabras : {len(documento.text.split())}")
        print(f"  párrafos : {documento.metadata['n_parrafos']}")
        print(f"  extracto : {documento.text[:200]!r}\n")
