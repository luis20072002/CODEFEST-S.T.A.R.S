"""Herramienta de inspección del OCR de figuras.

Responde a mano la pregunta que ninguna métrica automática puede contestar:
**¿el texto que Tesseract saca de las figuras es información o es basura?**

La corrida del orquestador generó 1755 Document de figura, pero no persistió
nada, así que no hay forma de mirarlos. Esta herramienta vuelve a hacer el OCR
sobre una MUESTRA pequeña de PDFs (no sobre los 759) y deja el resultado en
disco para revisarlo con los ojos:

    src/data/diagnostico_ocr/
      index.html        <- ábrelo en el navegador: imagen y texto lado a lado
      figuras/*.png     <- cada figura tal como se la pasamos a Tesseract
      figuras/*.txt     <- lo que Tesseract leyó en ella

Uso, desde `src/`:

    py -m tools.inspeccionar_ocr            # 12 PDFs repartidos por el corpus
    py -m tools.inspeccionar_ocr 30         # una muestra más grande
    py -m tools.inspeccionar_ocr 6 2        # 6 PDFs, solo del fenómeno 2

La muestra es DETERMINISTA (se toma a intervalos regulares sobre los PDFs
ordenados por doc_id): dos corridas con los mismos argumentos miran los mismos
archivos, así que se puede comparar antes/después de un cambio.

NO modifica nada del pipeline ni del corpus: solo lee y escribe en su carpeta
de salida, que está bajo `data/` y por tanto fuera de git.
"""

import html
import re
import statistics
import sys
from pathlib import Path

from core.catalog import Catalog
from loaders.ocr_loader import OCRLoader, OCR_LANG, ensure_tesseract
from loaders.pdf_loader import PDFLoader

RAIZ_DATOS = Path(__file__).resolve().parents[1] / "data"
CORPUS = RAIZ_DATOS / "data_raw"
SALIDA = RAIZ_DATOS / "diagnostico_ocr"

# Una "palabra plausible" es una secuencia de 3+ letras (con acentos). Sirve de
# señal de ruido: el OCR sobre un gráfico de barras devuelve cosas como
# "|| 1l" o "8O 4O 2O", que no producen ninguna. NO es un diccionario — no
# valida que la palabra exista, solo que tenga forma de palabra.
PALABRA_PLAUSIBLE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]{3,}")


def _muestra_de_pdfs(catalogo: Catalog, cuantos: int, fenomeno: int | None) -> list:
    """Toma `cuantos` PDFs repartidos regularmente por todo el corpus.

    A intervalos regulares y no los primeros N: los primeros serían todos del
    mismo observatorio (van ordenados por doc_id) y el diagnóstico saldría
    sesgado hacia el estilo de figuras de esa sola fuente.
    """
    pdfs = catalogo.entries(phenomenon=fenomeno, format="pdf")
    if cuantos >= len(pdfs):
        return pdfs
    paso = len(pdfs) / cuantos
    return [pdfs[int(i * paso)] for i in range(cuantos)]


def _metricas(texto: str) -> dict:
    """Cifras que ayudan a separar señal de ruido sin abrir cada figura."""
    palabras = texto.split()
    plausibles = PALABRA_PLAUSIBLE.findall(texto)
    return {
        "n_palabras": len(palabras),
        "n_plausibles": len(plausibles),
        # Proporción de la salida que tiene forma de palabra. Cerca de 1 =
        # prosa legible; cerca de 0 = cifras sueltas y símbolos = ruido.
        "ratio": len(plausibles) / len(palabras) if palabras else 0.0,
    }


def main(cuantos: int = 12, fenomeno: int | None = None) -> None:
    entorno = ensure_tesseract()
    print(f"Tesseract {entorno['version']} · lang={OCR_LANG}\n")

    catalogo = Catalog.from_excel(CORPUS / "Indice_Datos_Codefest.xlsx")
    entradas = _muestra_de_pdfs(catalogo, cuantos, fenomeno)
    print(f"Muestra: {len(entradas)} PDFs"
          f"{'' if fenomeno is None else f' del fenómeno {fenomeno}'}\n")

    carpeta_figuras = SALIDA / "figuras"
    carpeta_figuras.mkdir(parents=True, exist_ok=True)

    pdf_loader = PDFLoader()
    ocr_loader = OCRLoader()
    registros = []

    for n, entrada in enumerate(entradas, start=1):
        ruta = CORPUS / entrada.source
        print(f"[{n}/{len(entradas)}] {entrada.doc_id}  {Path(entrada.source).name}", flush=True)
        try:
            # Se reutiliza la MISMA función que usa el pipeline real, para que
            # lo que se inspecciona sea exactamente lo que se indexaría.
            figuras = pdf_loader._extract_figures_with_caption(ruta)
        except Exception as error:
            print(f"    no se pudo abrir: {error}")
            continue

        print(f"    figuras con leyenda: {len(figuras)}")
        for i, figura in enumerate(figuras, start=1):
            nombre = f"{entrada.doc_id}_fig{i:02d}"
            try:
                documento = ocr_loader.load_from_bytes(
                    figura["bytes"],
                    doc_id=nombre,
                    source=entrada.source,
                    format="pdf_figura",
                    phenomenon=entrada.phenomenon,
                )
            except Exception as error:
                print(f"    {nombre}: OCR falló -> {error}")
                continue

            # La imagen se guarda con su extensión original (la que traía
            # dentro del PDF), para verla igual que la vio Tesseract.
            archivo_img = carpeta_figuras / f"{nombre}.{figura['extension']}"
            archivo_img.write_bytes(figura["bytes"])
            # encoding explícito: en Windows el defecto es cp1252 y el OCR
            # puede traer caracteres que revientan al escribir.
            (carpeta_figuras / f"{nombre}.txt").write_text(documento.text, encoding="utf-8")

            registros.append({
                "nombre": nombre,
                "doc_id": entrada.doc_id,
                "fenomeno": entrada.phenomenon,
                "fuente": entrada.source,
                "pagina": figura["pagina"],
                "leyenda": figura["leyenda"],
                "imagen": archivo_img.name,
                "texto": documento.text,
                **_metricas(documento.text),
            })

    _resumen(registros)
    _escribir_html(registros)


def _resumen(registros: list[dict]) -> None:
    print("\n" + "=" * 62)
    print("RESUMEN")
    print("=" * 62)
    if not registros:
        print("No se extrajo ninguna figura en esta muestra.")
        return

    vacias = [r for r in registros if r["n_palabras"] == 0]
    ruido = [r for r in registros if r["n_palabras"] > 0 and r["ratio"] < 0.5]
    utiles = [r for r in registros if r["ratio"] >= 0.5 and r["n_palabras"] >= 5]

    total = len(registros)
    print(f"figuras procesadas          : {total}")
    print(f"  sin texto (OCR vacío)     : {len(vacias):4d}  ({len(vacias)/total:.0%})")
    print(f"  ruido (<50% con forma de palabra) : {len(ruido):4d}  ({len(ruido)/total:.0%})")
    print(f"  con texto aprovechable    : {len(utiles):4d}  ({len(utiles)/total:.0%})")

    con_texto = [r["n_palabras"] for r in registros if r["n_palabras"]]
    if con_texto:
        print(f"\npalabras por figura (de las que dieron texto):")
        print(f"  mediana {statistics.median(con_texto):.0f} · "
              f"min {min(con_texto)} · max {max(con_texto)}")

    print("\nEjemplos de lo que consideraría RUIDO (revísalos tú):")
    for r in ruido[:5]:
        print(f"  {r['nombre']}: {r['texto'][:70]!r}")
    print("\nEjemplos de lo que consideraría ÚTIL:")
    for r in utiles[:5]:
        print(f"  {r['nombre']}: {r['texto'][:70]!r}")


def _escribir_html(registros: list[dict]) -> None:
    """Página estática con imagen y texto lado a lado.

    Es el entregable de verdad de esta herramienta: las cifras orientan, pero
    la decisión de si el OCR aporta se toma mirando.
    """
    filas = []
    for r in sorted(registros, key=lambda x: x["ratio"]):   # lo peor primero
        color = "#c0392b" if r["ratio"] < 0.5 else "#27ae60"
        filas.append(f"""
        <tr>
          <td><img src="figuras/{html.escape(r['imagen'])}" alt=""></td>
          <td>
            <div class="meta">{html.escape(r['nombre'])} · F{r['fenomeno']} ·
              pág {r['pagina']}</div>
            <div class="meta">leyenda: {html.escape(str(r['leyenda'])[:120])}</div>
            <div class="meta" style="color:{color}">
              {r['n_palabras']} palabras · {r['ratio']:.0%} con forma de palabra</div>
            <pre>{html.escape(r['texto']) or '(vacío)'}</pre>
          </td>
        </tr>""")

    documento = f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<title>Diagnóstico OCR de figuras</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; }}
  table {{ border-collapse: collapse; width: 100%; }}
  td {{ border-top: 1px solid #ddd; padding: 1rem; vertical-align: top; }}
  td:first-child {{ width: 45%; }}
  img {{ max-width: 100%; border: 1px solid #ccc; }}
  pre {{ white-space: pre-wrap; background: #f6f6f6; padding: .75rem;
         border-radius: 4px; font-size: .85rem; }}
  .meta {{ font-size: .8rem; color: #555; margin-bottom: .25rem; }}
</style></head><body>
<h1>Diagnóstico OCR de figuras</h1>
<p>{len(registros)} figuras, ordenadas de <b>peor a mejor</b> según qué
proporción de la salida tiene forma de palabra. Las de arriba son las
candidatas a ser ruido.</p>
<table>{''.join(filas)}</table>
</body></html>"""

    salida = SALIDA / "index.html"
    salida.write_text(documento, encoding="utf-8")
    print(f"\n→ Abre esto en el navegador:\n  {salida}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    f = int(sys.argv[2]) if len(sys.argv) > 2 else None
    main(n, f)
