"""Convierte `entrega/informe_tecnico.md` a PDF y cuenta las páginas.

    py -m tools.informe_a_pdf                          # md → pdf, rutas por defecto
    py -m tools.informe_a_pdf otro.md otro.pdf

**Devuelve código 1 si el PDF pasa de 8 páginas**, que es el tope de §1.4, para que
pueda encadenarse como los demás verificadores del proyecto.

No hay pandoc ni LaTeX en esta máquina, pero **PyMuPDF ya está instalado** —lo usan
los loaders— y su API `fitz.Story` pagina HTML+CSS. Cubre el subconjunto de Markdown
que usa el informe: encabezados, negrita, cursiva, código en línea, bloques de
código, tablas, listas y reglas horizontales.

⚠️ **El límite de 8 páginas depende de este conversor.** La tipografía de `CSS` y los
márgenes son los que producen el resultado medido; convertir el mismo `.md` con otra
herramienta pagina distinto y hay que volver a medir. Si hay que recortar, mide
primero qué sección cae en qué página: en la primera pasada la prosa no era lo que
ocupaba, sino los bloques preformateados.
"""
import html as _html
import re
import sys
from pathlib import Path

import fitz

A4 = fitz.paper_rect("a4")
MARGEN = 46          # ~1,6 cm

CSS = """
body { font-family: sans-serif; font-size: 8.8pt; line-height: 1.28; }
h1 { font-size: 16pt; margin: 0 0 2pt 0; }
h2 { font-size: 11.5pt; margin: 9pt 0 3pt 0; }
h3 { font-size: 9.6pt; margin: 7pt 0 2pt 0; }
p  { margin: 0 0 4pt 0; text-align: justify; }
li { margin: 0 0 2pt 0; }
pre { font-family: monospace; font-size: 7.2pt; line-height: 1.18;
      margin: 3pt 0 5pt 0; }
code { font-family: monospace; font-size: 8pt; }
table { font-size: 7.9pt; margin: 3pt 0 5pt 0; }
th { text-align: left; font-weight: bold; }
td { padding-right: 6pt; }
hr { margin: 6pt 0; }
"""


def en_linea(t: str) -> str:
    """Negrita, cursiva y codigo en linea. El escapado va antes que el marcado."""
    t = _html.escape(t)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", t)
    return t


def md_a_html(md: str) -> str:
    # Los emoji no existen en las fuentes base del renderizador y saldrian como
    # hueco; en un informe formal no aportan nada.
    md = md.replace("⚠️", "").replace("🔴", "").replace("💡", "").replace("✅", "")
    lineas = md.splitlines()
    salida, i = [], 0
    while i < len(lineas):
        ln = lineas[i]

        if ln.startswith("```"):                      # bloque de codigo
            i += 1
            buf = []
            while i < len(lineas) and not lineas[i].startswith("```"):
                buf.append(_html.escape(lineas[i]))
                i += 1
            i += 1
            salida.append("<pre>" + "<br/>".join(buf) + "</pre>")
            continue

        if ln.startswith("|"):                        # tabla
            filas = []
            while i < len(lineas) and lineas[i].startswith("|"):
                filas.append(lineas[i])
                i += 1
            def celdas(f):
                return [c.strip() for c in f.strip().strip("|").split("|")]
            cab = celdas(filas[0])
            cuerpo = [celdas(f) for f in filas[2:]]    # [1] es el separador
            t = ["<table>", "<tr>"]
            t += [f"<th>{en_linea(c)}</th>" for c in cab]
            t.append("</tr>")
            for fila in cuerpo:
                t.append("<tr>")
                t += [f"<td>{en_linea(c)}</td>" for c in fila]
                t.append("</tr>")
            t.append("</table>")
            salida.append("".join(t))
            continue

        if re.match(r"^#{1,3} ", ln):                 # encabezado
            n = len(ln) - len(ln.lstrip("#"))
            salida.append(f"<h{n}>{en_linea(ln[n:].strip())}</h{n}>")
            i += 1
            continue

        if ln.strip() == "---":
            salida.append("<hr/>")
            i += 1
            continue

        if re.match(r"^\s*(\d+\.|-) ", ln):           # lista
            etiqueta = "ol" if re.match(r"^\s*\d+\.", ln) else "ul"
            items = []
            while i < len(lineas) and (re.match(r"^\s*(\d+\.|-) ", lineas[i])
                                       or (lineas[i].startswith("   ")
                                           and lineas[i].strip() and items)):
                if re.match(r"^\s*(\d+\.|-) ", lineas[i]):
                    items.append(re.sub(r"^\s*(\d+\.|-) ", "", lineas[i]))
                else:
                    items[-1] += " " + lineas[i].strip()
                i += 1
            salida.append(f"<{etiqueta}>"
                          + "".join(f"<li>{en_linea(x)}</li>" for x in items)
                          + f"</{etiqueta}>")
            continue

        if not ln.strip():
            i += 1
            continue

        parrafo = []                                   # parrafo normal
        while i < len(lineas) and lineas[i].strip() \
                and not re.match(r"^(#{1,3} |\||```|---$|\s*(\d+\.|-) )", lineas[i]):
            parrafo.append(lineas[i].strip())
            i += 1
        salida.append(f"<p>{en_linea(' '.join(parrafo))}</p>")

    return "<html><body>" + "".join(salida) + "</body></html>"


RAIZ = Path(__file__).resolve().parents[2]
ENTRADA = RAIZ / "entrega" / "informe_tecnico.md"
SALIDA = RAIZ / "entrega" / "informe_tecnico.pdf"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    entrada = Path(sys.argv[1]) if len(sys.argv) > 1 else ENTRADA
    salida = Path(sys.argv[2]) if len(sys.argv) > 2 else SALIDA
    doc_html = md_a_html(entrada.read_text(encoding="utf-8"))

    story = fitz.Story(html=doc_html, user_css=CSS)
    escritor = fitz.DocumentWriter(salida)
    marco = A4 + (MARGEN, MARGEN, -MARGEN, -MARGEN)
    paginas = 0
    mas = True
    while mas:
        dispositivo = escritor.begin_page(A4)
        mas, _ = story.place(marco)
        story.draw(dispositivo)
        escritor.end_page()
        paginas += 1
    escritor.close()

    print(f"{entrada.name} -> {salida.name}")
    print(f"PAGINAS: {paginas}   (limite de §1.4: 8)")
    print(f"tamaño : {salida.stat().st_size:,} B")
    return 0 if paginas <= 8 else 1


if __name__ == "__main__":
    raise SystemExit(main())
