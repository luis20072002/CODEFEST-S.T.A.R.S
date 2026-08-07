"""Lectura del archivo de consultas de ADL.

    py -m retrieval.consultas           # diagnóstico: cuántas lee y cómo salen

Vive aquí, y no dentro de `generador.py`, porque lo usan dos sitios: el
generador de la entrega y `tools/prueba_crosslingual.py`. Tener dos copias del
mismo parser es garantizar que un día divergen — es el mismo criterio que hizo
que `retrieval/fragmentos.py` reutilice el cortador de oraciones del chunker.

FORMATO DEL ARCHIVO. `Extracto_Preguntas_50_v2.pdf` trae las 50 consultas
identificadas como `q001`…`q050`. §9.3 exige que `resultados.jsonl` tenga
«exactamente 50 líneas, una por cada consulta del conjunto de evaluación», y
§10.3 que vayan en orden, así que el identificador **se lee del archivo**, no se
genera contando: si ADL entregara una versión con otros identificadores o en
otro orden, contarlos produciría una salida mal emparejada sin dar ningún error.

⚠️ **Las 50 están TODAS en español**, pese a que §10.1 dice que se reparten
entre los tres idiomas del corpus. Verificado leyendo el PDF entero; ver
`ESTADO.md` §5. Es lo que convierte el reto en cross-lingual.
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple

DATOS = Path(__file__).resolve().parents[1] / "data"
CONSULTAS = DATOS / "Extracto_Preguntas_50_v2.pdf"

# `q001`, `q17`, `Q050`… Se captura el número para normalizar el identificador.
RE_ID = re.compile(r"\bq0*(\d{1,3})\b", re.IGNORECASE)

# Por debajo de esto no es una consulta, es un resto de maquetación.
MIN_CARACTERES = 15


def cargar_consultas(ruta: Path = CONSULTAS) -> List[Tuple[str, str]]:
    """Devuelve `[(query_id, texto), …]` en el orden del archivo.

    Acepta el PDF de ADL o un `.txt` con una consulta por línea, para poder
    probar sin depender del PDF. El identificador se normaliza a `qNNN` con
    tres dígitos, que es el formato del esquema de §9.3.1.
    """
    ruta = Path(ruta)
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe el archivo de consultas: {ruta}")

    if ruta.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        texto = "\n".join((p.extract_text() or "") for p in PdfReader(ruta).pages)
    else:
        texto = ruta.read_text(encoding="utf-8")

    # re.split con un grupo de captura intercala los identificadores entre los
    # trozos de texto: [antes, "1", cuerpo1, "2", cuerpo2, …]. Por eso se
    # recorre de dos en dos empezando en 1.
    partes = RE_ID.split(texto)
    consultas: List[Tuple[str, str]] = []
    for i in range(1, len(partes) - 1, 2):
        numero = int(partes[i])
        cuerpo = " ".join(partes[i + 1].split())
        if len(cuerpo) >= MIN_CARACTERES:
            consultas.append((f"q{numero:03d}", cuerpo))
    return consultas


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ruta = Path(sys.argv[1]) if len(sys.argv) > 1 else CONSULTAS
    consultas = cargar_consultas(ruta)

    print(f"archivo   : {ruta}")
    print(f"consultas : {len(consultas)}\n")
    for query_id, texto in consultas[:3]:
        print(f"  {query_id}  {texto[:110]}")
    print("  …")
    for query_id, texto in consultas[-2:]:
        print(f"  {query_id}  {texto[:110]}")

    # Comprobaciones que importan para §9.3 y §10.3.
    ids = [q for q, _ in consultas]
    esperados = [f"q{i:03d}" for i in range(1, len(consultas) + 1)]
    print(f"\n50 consultas        : {'✔' if len(consultas) == 50 else '✖ ' + str(len(consultas))}")
    print(f"ids sin repetir     : {'✔' if len(set(ids)) == len(ids) else '✖'}")
    print(f"orden q001…qNNN     : {'✔' if ids == esperados else '✖ ' + str(ids[:5])}")
    largos = [len(t) for _, t in consultas]
    print(f"longitud del texto  : min {min(largos)}  mediana "
          f"{sorted(largos)[len(largos)//2]}  max {max(largos)}")
