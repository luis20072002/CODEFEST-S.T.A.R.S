"""Convierte `entrega/informe_tecnico.md` en `entrega/informe_tecnico.pdf`.

    py -m tools.informe_a_pdf              # convierte y cuenta las paginas
    py -m tools.informe_a_pdf --abrir      # ademas lo abre al terminar

§1.4 exige un «Documento tecnico en PDF (maximo 8 paginas)». Este script hace
las dos cosas: lo genera y **cuenta las paginas reales**, devolviendo codigo de
salida 1 si se pasa del limite. El tope no se estima mirando el .md: se mide
sobre el PDF renderizado.

POR QUE PyMuPDF Y NO UN CONVERSOR EXTERNO. No hay pandoc, wkhtmltopdf ni
LibreOffice en la maquina de trabajo, y anadir una dependencia mas a una entrega
que §1.4 juzga por reproducibilidad no compensa. PyMuPDF ya estaba instalado
para el rasterizado de paginas del OCR, y su API `Story` renderiza HTML a PDF y
permite paginar contando.

⚠️ LIMITACION DE CARACTERES. `Story` usa las fuentes base-14 del PDF (Times,
Helvetica), que solo cubren Latin-1. Las vocales acentuadas, la «n» con tilde,
el simbolo § y las comillas angulares entran; la raya larga, las comillas
tipograficas, las flechas y los signos <= no. `_sanear()` los sustituye por
equivalentes ASCII antes de renderizar, para que no salgan huecos en blanco.
"""

import html as _html
import re
import sys
from pathlib import Path

import pymupdf

RAIZ = Path(__file__).resolve().parents[2]
ORIGEN = RAIZ / "entrega" / "informe_tecnico.md"
DESTINO = RAIZ / "entrega" / "informe_tecnico.pdf"
MAX_PAGINAS = 8

# Caracteres fuera de Latin-1 que aparecen al escribir en espanol, con su
# equivalente representable. Se aplican al texto ya leido, antes del HTML.
SUSTITUCIONES = {
    "—": "-", "–": "-", "‘": "'", "’": "'",
    "“": '"', "”": '"', "…": "...", "→": "->",
    "≤": "<=", "≥": ">=", "×": "x", "≈": "~",
    "✓": "", "✗": "", "‑": "-", " ": " ",
    "─": "-", "•": "-",
}


def _sanear(texto: str) -> str:
    for malo, bueno in SUSTITUCIONES.items():
        texto = texto.replace(malo, bueno)
    # Cualquier resto fuera de Latin-1 se elimina en vez de dejar un hueco.
    return texto.encode("latin-1", "ignore").decode("latin-1")


def _en_linea(texto: str) -> str:
    """Marcado de nivel de linea: negrita, cursiva y codigo.

    Se escapa primero el HTML y despues se aplican los patrones, para que un
    `<` del texto no se interprete como etiqueta.
    """
    texto = _html.escape(texto)
    texto = re.sub(r"`([^`]+)`", r"<code>\1</code>", texto)
    texto = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", texto)
    texto = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", texto)
    return texto


def markdown_a_html(md: str) -> str:
    """Convierte el subconjunto de Markdown que usa el informe.

    Cubre encabezados, parrafos, listas, tablas y reglas horizontales. No
    pretende ser un conversor general: solo lo que el documento emplea.
    """
    lineas = _sanear(md).split("\n")
    salida: list[str] = []
    i = 0
    en_lista = False

    def cerrar_lista() -> None:
        nonlocal en_lista
        if en_lista:
            salida.append("</ul>")
            en_lista = False

    while i < len(lineas):
        linea = lineas[i].rstrip()

        if not linea.strip():
            cerrar_lista()
            i += 1
            continue

        # Bloque de codigo delimitado por vallas. `Story` no respeta
        # `white-space: pre`, asi que cada linea se emite como su propio div y
        # la sangria se conserva con espacios duros; de lo contrario el arbol de
        # directorios se colapsaria en un parrafo corrido.
        if linea.lstrip().startswith("```"):
            cerrar_lista()
            i += 1
            salida.append('<div class="bloque">')
            while i < len(lineas) and not lineas[i].lstrip().startswith("```"):
                cruda = lineas[i].rstrip()
                # Todos los espacios pasan a duros, no solo la sangria: asi se
                # conserva tambien la alineacion en columnas del arbol.
                contenido = _html.escape(cruda).replace(" ", "&nbsp;")
                salida.append(f"<div>{contenido or '&nbsp;'}</div>")
                i += 1
            salida.append("</div>")
            i += 1                                   # se salta la valla de cierre
            continue

        # Tabla: una fila de cabecera seguida de la fila de guiones.
        if linea.startswith("|") and i + 1 < len(lineas) and \
                re.match(r"^\|[\s:|-]+\|$", lineas[i + 1].strip()):
            cerrar_lista()
            cabecera = [c.strip() for c in linea.strip("|").split("|")]
            salida.append("<table><thead><tr>")
            salida.extend(f"<th>{_en_linea(c)}</th>" for c in cabecera)
            salida.append("</tr></thead><tbody>")
            i += 2
            while i < len(lineas) and lineas[i].strip().startswith("|"):
                celdas = [c.strip() for c in lineas[i].strip().strip("|").split("|")]
                salida.append("<tr>")
                salida.extend(f"<td>{_en_linea(c)}</td>" for c in celdas)
                salida.append("</tr>")
                i += 1
            salida.append("</tbody></table>")
            continue

        if linea.startswith("#"):
            cerrar_lista()
            nivel = len(linea) - len(linea.lstrip("#"))
            salida.append(f"<h{nivel}>{_en_linea(linea.lstrip('#').strip())}</h{nivel}>")
            i += 1
            continue

        if linea.strip() in ("---", "***", "___"):
            cerrar_lista()
            salida.append("<hr/>")
            i += 1
            continue

        if re.match(r"^\s*[-*]\s+", linea):
            if not en_lista:
                salida.append("<ul>")
                en_lista = True
            salida.append(f"<li>{_en_linea(re.sub(r'^\s*[-*]\s+', '', linea))}</li>")
            i += 1
            continue

        # Parrafo: se acumulan las lineas seguidas hasta un blanco.
        cerrar_lista()
        parrafo = [linea]
        i += 1
        while i < len(lineas) and lineas[i].strip() and \
                not lineas[i].lstrip().startswith(("#", "|", "-", "*", "```")):
            parrafo.append(lineas[i].strip())
            i += 1
        salida.append(f"<p>{_en_linea(' '.join(parrafo))}</p>")

    cerrar_lista()
    return "\n".join(salida)


# Hoja de estilo. Sobria a proposito: sin color en los titulos, jerarquia por
# tamano y peso, y una unica linea de separacion bajo los de primer nivel.
CSS = """
body   { font-family: serif; font-size: 9.4pt; line-height: 1.32; color: #000; }
h1     { font-size: 16pt; margin: 0 0 2pt 0; }
h2     { font-size: 11pt; margin: 11pt 0 3pt 0;
         border-bottom: 0.6px solid #000; padding-bottom: 1.5pt; }
h3     { font-size: 9.8pt; margin: 7pt 0 2pt 0; }
p      { margin: 0 0 4pt 0; text-align: justify; }
ul     { margin: 0 0 4pt 0; padding-left: 11pt; }
li     { margin: 0 0 1.5pt 0; text-align: justify; }
code   { font-family: monospace; font-size: 8.6pt; }
div.bloque { font-family: monospace; font-size: 8.1pt; line-height: 1.18;
             margin: 3pt 0 6pt 0; }
table  { width: 100%; margin: 3pt 0 6pt 0; border-collapse: collapse;
         font-size: 8.4pt; }
th     { text-align: left; font-weight: bold; border-bottom: 0.6px solid #000;
         padding: 1.5pt 4pt 1.5pt 0; }
td     { padding: 1.2pt 4pt 1.2pt 0; border-bottom: 0.3px solid #999;
         vertical-align: top; }
hr     { margin: 5pt 0; }
"""


def convertir(origen: Path = ORIGEN, destino: Path = DESTINO,
              maximo: int = MAX_PAGINAS) -> int:
    """Renderiza el informe y devuelve el numero de paginas."""
    if not origen.is_file():
        raise SystemExit(f"No existe {origen}")

    cuerpo = markdown_a_html(origen.read_text(encoding="utf-8"))
    documento = f"<html><head><style>{CSS}</style></head><body>{cuerpo}</body></html>"

    historia = pymupdf.Story(html=documento)
    escritor = pymupdf.DocumentWriter(str(destino))
    hoja = pymupdf.paper_rect("a4")
    # Margenes de 2 cm a los lados y 1,8 cm arriba y abajo.
    marco = hoja + (56.7, 51, -56.7, -51)

    paginas = 0
    quedan = True
    while quedan:
        dispositivo = escritor.begin_page(hoja)
        quedan, _ = historia.place(marco)
        historia.draw(dispositivo)
        escritor.end_page()
        paginas += 1
        if paginas > 40:                       # freno ante un bucle infinito
            break
    escritor.close()
    return paginas


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    paginas = convertir()
    tamano = DESTINO.stat().st_size

    print(f"origen  : {ORIGEN.relative_to(RAIZ)}")
    print(f"destino : {DESTINO.relative_to(RAIZ)}  ({tamano:,} bytes)")
    print(f"paginas : {paginas}   (§1.4 admite un maximo de {MAX_PAGINAS})")

    if "--abrir" in sys.argv:
        import os
        os.startfile(DESTINO)                  # noqa: S606

    if paginas > MAX_PAGINAS:
        print(f"\nFALLA: el informe se pasa en {paginas - MAX_PAGINAS} pagina(s). "
              f"Hay que recortar el .md y volver a convertir.")
        return 1
    print("\nPASA: el informe cabe en el limite de §1.4.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
