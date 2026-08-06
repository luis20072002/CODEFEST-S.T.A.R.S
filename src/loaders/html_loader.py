"""Loader de documentos HTML.

§2.1 lo describe así: «se elimina el marcado (etiquetas, atributos, scripts,
estilos) y se conserva únicamente el texto visible. Los elementos estructurales
del HTML (encabezados, párrafos, listas) pueden usarse como señales para la
estrategia de chunking».

Eso es exactamente lo que hace: tira el marcado y emite el texto visible con
los bloques separados por `\\n\\n`. Esa frontera de párrafo es la que necesitan
§3.2 y §3.3, igual que en los otros cinco loaders — el Preprocessor debe
conservarla al colapsar espacios.

POR QUÉ EXISTE, si el catálogo no tiene ni un HTML:

1. §1.3 lista HTML entre los formatos entregados por ADL, y la Tabla 1 lo
   admite como valor de `formato` («pdf, html o md»). Si ADL corrige el corpus,
   ya está cubierto.
2. Hay 2 archivos con extensión `.pdf` cuyo contenido es HTML
   (`SIPRI_22136.pdf` y `SIPRI_hsrc20lmip...-1.pdf`): descargas fallidas que
   capturaron la página web en vez del documento.
   ⚠️ Ojo con las expectativas: se comprobó que son **páginas de navegación**
   (menús, «jump to main content», listados), no documentos. §2.2 manda quitar
   el boilerplate de sitios web, así que tras limpiarlos no queda casi nada.
   Este loader los lee bien; lo que no puede es inventar contenido que el
   archivo no tiene.

NO usa BeautifulSoup a propósito: `html.parser` es de la librería estándar y
basta para «quitar marcado y quedarse con el texto». Una dependencia menos que
declarar en requirements.txt y que instalar en la máquina donde se reproduzca
la entrega (§1.4).
"""

import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from core.catalog import CatalogEntry
from core.document import Document
from loaders.base_loader import BaseLoader

# Etiquetas cuyo contenido NO es texto visible: se descarta entero, no solo la
# etiqueta. Un <script> lleva código dentro; si solo se quitara la etiqueta,
# el JavaScript acabaría dentro del texto del documento.
#
# ⚠️ Aquí SOLO pueden ir etiquetas que siempre se cierran. Se lleva un contador
# de profundidad, y una etiqueta vacía (<meta>, <link>, <br>) nunca dispara
# handle_endtag: si entrara en este conjunto, el contador jamás volvería a cero
# y el resto del documento se descartaría en silencio. Pasó: la primera versión
# incluía "meta" y "link" y devolvía 0 palabras en los dos archivos de prueba.
# Tampoco va "head": hay páginas que omiten </head> y provocarían lo mismo.
# El <title> se captura aparte, así que no hace falta.
INVISIBLES = {"script", "style", "noscript"}

# Etiquetas de bloque: al cerrarlas se emite una frontera de párrafo. Es la
# señal estructural que §2.1 dice que se puede aprovechar para el chunking.
BLOQUES = {
    "p", "div", "section", "article", "header", "footer", "main", "aside",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "li", "tr", "blockquote", "pre", "figcaption", "dt", "dd", "table", "ul", "ol",
}


class HTMLLoadError(Exception):
    pass


class _ExtractorDeTexto(HTMLParser):
    """Recorre el HTML y va acumulando solo el texto visible.

    HTMLParser funciona por eventos: llama a handle_starttag cuando encuentra
    una etiqueta de apertura, handle_data con el texto suelto, etc. Aquí se
    lleva un contador de profundidad dentro de etiquetas invisibles en vez de
    un simple booleano, porque pueden anidarse (<head> con <style> dentro).
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)   # convierte &amp; -> & solo
        self.partes: list[str] = []
        self._profundidad_invisible = 0
        self.titulo: str | None = None
        self._en_titulo = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in INVISIBLES:
            self._profundidad_invisible += 1
        elif tag == "title":
            self._en_titulo = True
        elif tag == "br":
            self.partes.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in INVISIBLES:
            # max(0, ...) por si el HTML viene mal formado y cierra una
            # etiqueta que nunca abrió. El corpus trae páginas reales, no
            # HTML validado.
            self._profundidad_invisible = max(0, self._profundidad_invisible - 1)
        elif tag == "title":
            self._en_titulo = False
        elif tag in BLOQUES:
            self.partes.append("\n\n")

    def handle_data(self, data: str) -> None:
        if self._en_titulo:
            self.titulo = (self.titulo or "") + data
            return
        if self._profundidad_invisible == 0:
            self.partes.append(data)

    def texto(self) -> str:
        crudo = "".join(self.partes)
        # Espacios y tabulaciones se colapsan; los saltos de línea NO se tocan
        # aquí, para no destruir las fronteras de párrafo que se acaban de
        # emitir. [^\S\n] = "espacio en blanco que no sea salto de línea".
        crudo = re.sub(r"[^\S\n]+", " ", crudo)
        # Tres o más saltos seguidos se reducen a dos: un <div><p>…</p></div>
        # emite dos fronteras seguidas y quedarían huecos enormes.
        crudo = re.sub(r"\n{3,}", "\n\n", crudo)
        # Se limpia cada línea por separado y se descartan las vacías.
        lineas = [linea.strip() for linea in crudo.split("\n")]
        salida, anterior_vacia = [], True
        for linea in lineas:
            if linea:
                salida.append(linea)
                anterior_vacia = False
            elif not anterior_vacia:
                salida.append("")        # conserva UNA línea en blanco = frontera
                anterior_vacia = True
        return "\n".join(salida).strip()


class HTMLLoader(BaseLoader):
    """Convierte un archivo HTML del corpus en un Document con su texto visible."""

    def load(self, path: str | Path, entry: CatalogEntry) -> list[Document]:
        crudo = self._leer(path)

        parser = _ExtractorDeTexto()
        try:
            parser.feed(crudo)
            parser.close()
        except Exception as error:
            raise HTMLLoadError(f"No se pudo interpretar el HTML en {path}: {error}") from error

        texto = parser.texto()
        titulo = (parser.titulo or "").strip() or None

        metadata: dict[str, Any] = {
            "n_parrafos": texto.count("\n\n") + 1 if texto else 0,
            # Se dejan anotados los bytes originales: para las descargas
            # fallidas (páginas de menú) el tamaño es una pista útil al
            # revisar por qué un documento salió casi vacío.
            "bytes_origen": Path(path).stat().st_size,
        }

        return [Document(
            doc_id=entry.doc_id,
            source=entry.source,
            # `format` viene del catálogo, que lo deriva de la extensión. Para
            # los 2 archivos .pdf-que-son-HTML el orquestador puede pasar un
            # entry con format="html"; este loader no lo decide.
            format=entry.format,
            phenomenon=entry.phenomenon,
            language=None,        # lo llena el Preprocessor (§2.2)
            title=titulo,
            text=texto,
            metadata=metadata,
        )]

    @staticmethod
    def _leer(path: str | Path) -> str:
        """Lee el archivo respetando la codificación que declare la página.

        No se puede asumir UTF-8: las páginas del corpus son capturas reales de
        sitios web y algunas declaran ISO-8859-1. Se mira la declaración del
        propio HTML antes de decidir, y si no hay o falla, se cae a UTF-8 con
        reemplazo — nunca se propaga un UnicodeDecodeError, porque perder el
        documento entero por un byte suelto sería peor.
        """
        try:
            crudo = Path(path).read_bytes()
        except OSError as error:
            raise HTMLLoadError(f"No se pudo abrir {path}: {error}") from error

        # <meta charset="..."> o <meta http-equiv content="...; charset=...">
        declarada = re.search(rb'charset=["\']?\s*([\w-]+)', crudo[:4096], re.IGNORECASE)
        if declarada:
            try:
                return crudo.decode(declarada.group(1).decode("ascii"), errors="replace")
            except LookupError:
                pass      # códec desconocido: se ignora y se usa UTF-8
        return crudo.decode("utf-8", errors="replace")


if __name__ == "__main__":
    # Diagnóstico: `py -m loaders.html_loader` desde src/.
    # Sin HTML en el catálogo, se prueba contra los 2 .pdf que son HTML.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from core.catalog import Catalog

    raiz = Path(__file__).resolve().parents[1] / "data" / "data_raw"
    catalogo = Catalog.from_excel(raiz / "Indice_Datos_Codefest.xlsx")
    loader = HTMLLoader()

    sospechosos = [
        "F3_Dinamicas_Territoriales/SIPRI/sipri_data/pdfs/SIPRI_22136.pdf",
        "F3_Dinamicas_Territoriales/SIPRI/sipri_data/pdfs/"
        "SIPRI_hsrc20lmip20report20320growth20employment20and20skills-1.pdf",
    ]

    # Catalog no expone acceso por clave, así que se indexa por `source`.
    por_ruta = {entrada.source: entrada for entrada in catalogo.entries()}

    for ruta in sospechosos:
        entrada = por_ruta[ruta]
        documentos = loader.load(raiz / ruta, entrada)
        documento = documentos[0]
        print("=" * 66)
        print(f"{entrada.doc_id}  ({Path(ruta).name})")
        print(f"  título   : {documento.title}")
        print(f"  palabras : {len(documento.text.split())}")
        print(f"  párrafos : {documento.metadata['n_parrafos']}")
        print(f"  extracto :\n{documento.text[:400]}")
        print()
