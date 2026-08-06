"""Inspección a ojo de lo que el cleaner borra: un HTML con antes y después.

`tools/verificar_cleaner.py` responde preguntas agregadas («¿se perdió algún
término?», «¿sobrevivieron los párrafos?»). Esta herramienta responde la que
ninguna métrica puede contestar sola: **mirando el texto, ¿lo que se fue era
basura?** Es el mismo patrón que `tools/inspeccionar_ocr.py`, que ya sirvió para
decidir lo de las figuras.

    py -m tools.diff_cleaner            # 20 documentos repartidos por formato
    py -m tools.diff_cleaner 40         # una muestra más grande
    py -m tools.diff_cleaner 20 pdf     # solo de un formato

Deja `src/data/diagnostico_cleaner/index.html`: cada documento en dos columnas,
crudo a la izquierda y limpio a la derecha, con lo eliminado resaltado.

La muestra está **estratificada por formato** y es determinista: se ordenan los
documentos de cada formato por `doc_id` y se toman a intervalos regulares. Dos
corridas con los mismos argumentos miran los mismos documentos, así que se
puede comparar antes y después de tocar una regla del cleaner. Se estratifica
porque los formatos se rompen de formas distintas —un PDF por la maquetación,
un CSV por la repetición de valores— y una muestra al azar se llenaría de JSON,
que son el 52% del corpus.

No modifica nada: solo lee `documentos.jsonl` y escribe en su carpeta de
salida, que está bajo `data/` y por tanto fuera de git.
"""

import html
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from core.store import read_documents
from preprocess.cleaner import clean_document
from preprocess.language import detect_language

DATOS = Path(__file__).resolve().parents[1] / "data"
DOCUMENTOS = DATOS / "documentos.jsonl"
SALIDA = DATOS / "diagnostico_cleaner"

# Cuánto texto se muestra de cada documento. El diff carácter a carácter de
# `SequenceMatcher` es cuadrático en el peor caso, así que sobre un documento de
# seis millones de palabras no terminaría nunca. Con una ventana desde el
# principio se ve lo que importa: las portadas, cabeceras y marcas de agua —que
# es donde vive el boilerplate— salen todas ahí.
VENTANA = 4000

ESTILO = """
body { font-family: -apple-system, Segoe UI, sans-serif; margin: 0; padding: 24px;
       background: #f6f7f9; color: #1a1a1a; }
h1 { font-size: 20px; } h2 { font-size: 15px; margin: 32px 0 6px; }
.meta { font-size: 12px; color: #555; margin-bottom: 8px; font-family: monospace; }
.par { display: flex; gap: 16px; align-items: flex-start; }
.col { flex: 1; background: #fff; border: 1px solid #dcdfe4; border-radius: 6px;
       padding: 12px; font-family: Consolas, monospace; font-size: 11.5px;
       line-height: 1.5; white-space: pre-wrap; word-break: break-word;
       max-height: 460px; overflow-y: auto; }
.tit { font-family: -apple-system, sans-serif; font-size: 11px; font-weight: 600;
       text-transform: uppercase; color: #666; margin-bottom: 6px; }
del { background: #ffd7d5; text-decoration: none; }
ins { background: #d7f5dd; text-decoration: none; }
.aviso { background: #fff8e1; border-left: 3px solid #f0b400; padding: 10px 14px;
         margin-bottom: 20px; font-size: 13px; }
"""


def marcar_diferencias(antes: str, despues: str) -> tuple:
    """Devuelve los dos textos en HTML, con lo eliminado y lo añadido resaltado.

    Se compara a nivel de **carácter** y no de palabra a propósito: buena parte
    de lo que hace el cleaner son cambios de un solo carácter —un guion suave
    que desaparece, un salto de línea que se vuelve espacio, un byte nulo que se
    va— y un diff por palabras los mostraría como "palabra entera sustituida",
    que oculta justo lo que se quiere ver.

    `autojunk=False` desactiva la heurística de SequenceMatcher que ignora los
    elementos muy frecuentes: en un texto, el carácter más frecuente es el
    espacio, y con la heurística activa el diff sale inservible.
    """
    matcher = SequenceMatcher(None, antes, despues, autojunk=False)
    izquierda, derecha = [], []
    for etiqueta, i1, i2, j1, j2 in matcher.get_opcodes():
        trozo_a = html.escape(antes[i1:i2])
        trozo_d = html.escape(despues[j1:j2])
        if etiqueta == "equal":
            izquierda.append(trozo_a)
            derecha.append(trozo_d)
        else:
            # 'replace' aparece en los dos lados; 'delete' solo a la izquierda y
            # 'insert' solo a la derecha.
            if trozo_a:
                izquierda.append(f"<del>{trozo_a}</del>")
            if trozo_d:
                derecha.append(f"<ins>{trozo_d}</ins>")
    return "".join(izquierda), "".join(derecha)


def muestrear(n: int, formato: str | None) -> list:
    """Elige `n` documentos repartidos entre los formatos, de forma determinista."""
    por_formato = defaultdict(list)
    for documento in read_documents(DOCUMENTOS):
        if formato and documento.format != formato:
            continue
        por_formato[documento.format].append(documento)

    if not por_formato:
        return []

    # Se reparten las plazas entre los formatos presentes; al menos una a cada
    # uno, para que ningún formato quede sin representar aunque tenga un solo
    # documento (el corpus tiene un TXT y un AVIF).
    formatos = sorted(por_formato)
    por_cada = max(1, n // len(formatos))

    muestra = []
    for nombre in formatos:
        documentos = sorted(por_formato[nombre], key=lambda d: d.doc_id)
        cuantos = min(por_cada, len(documentos))
        # `//` para tomarlos a intervalos regulares en lugar de los primeros:
        # los primeros de cada formato suelen ser del mismo observatorio.
        salto = max(1, len(documentos) // cuantos)
        muestra.extend(documentos[::salto][:cuantos])
    return muestra


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    formato = sys.argv[2] if len(sys.argv) > 2 else None

    if not DOCUMENTOS.is_file():
        print(f"No existe {DOCUMENTOS}. Corre antes el orquestador.")
        return 1

    muestra = muestrear(n, formato)
    if not muestra:
        print(f"La muestra salió vacía (¿formato {formato!r} inexistente?).")
        return 1

    SALIDA.mkdir(parents=True, exist_ok=True)
    partes = [
        "<meta charset='utf-8'><title>Diagnóstico del cleaner</title>",
        f"<style>{ESTILO}</style>",
        "<h1>Diagnóstico del cleaner — antes y después</h1>",
        "<div class='aviso'><b>Cómo leerlo:</b> en rojo lo que el cleaner "
        "<b>eliminó</b>, en verde lo que <b>quedó en su lugar</b> (casi siempre "
        "un espacio, al colapsar). Lo que buscas: que lo rojo sea mobiliario de "
        "página —marcas de agua, cabeceras, numeración, puntos guía del índice— "
        "y nunca prosa con contenido. Se muestran los primeros "
        f"{VENTANA:,} caracteres de cada documento.</div>",
        f"<p class='meta'>{len(muestra)} documentos · fuente: {DOCUMENTOS.name}</p>",
    ]

    for documento in muestra:
        limpio = clean_document(documento)
        idioma, confianza = detect_language(limpio.text)
        antes, despues = documento.text[:VENTANA], limpio.text[:VENTANA]
        izquierda, derecha = marcar_diferencias(antes, despues)

        total_antes, total_despues = len(documento.text), len(limpio.text)
        reduccion = (total_antes - total_despues) / total_antes if total_antes else 0

        partes.append(f"<h2>{html.escape(documento.doc_id)} — "
                      f"{html.escape(documento.source.split('/')[-1])}</h2>")
        partes.append(
            f"<p class='meta'>formato={documento.format} · fenómeno={documento.phenomenon} · "
            f"idioma={idioma} ({confianza:.2f}) · "
            f"{total_antes:,} → {total_despues:,} caracteres "
            f"({reduccion:.1%} eliminado)</p>"
        )
        partes.append(
            "<div class='par'>"
            f"<div class='col'><div class='tit'>crudo</div>{izquierda}</div>"
            f"<div class='col'><div class='tit'>limpio</div>{derecha}</div>"
            "</div>"
        )

    destino = SALIDA / "index.html"
    destino.write_text("\n".join(partes), encoding="utf-8")
    print(f"Escrito: {destino}")
    print("Ábrelo en el navegador y revisa que lo rojo sea siempre basura.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
