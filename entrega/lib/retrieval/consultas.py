"""Lectura del archivo de consultas.

`Extracto_Preguntas_50_v2.pdf` contiene las 50 consultas de evaluación,
identificadas como `q001`–`q050`. §9.3 exige que `resultados.jsonl` tenga
«exactamente 50 líneas, una por cada consulta del conjunto de evaluación» y
§10.3 que aparezcan en el orden `q001, q002, …, q050`.

EL IDENTIFICADOR SE LEE DEL ARCHIVO, NO SE GENERA CONTANDO. Si se numeraran las
consultas por su posición, un archivo con identificadores distintos o en otro
orden produciría una salida mal emparejada sin emitir ningún error. Leerlos hace
que cualquier discrepancia sea visible.

NOTA SOBRE EL IDIOMA DE LAS CONSULTAS. §10.1 indica que las consultas se
distribuyen «en los tres idiomas del corpus (español, inglés y portugués)». En
el archivo entregado las 50 están en español, comprobado consulta por consulta.
Esa constatación es la que determina el criterio de selección del encoder: la
recuperación real es translingüe español→inglés contra un corpus 55 % en inglés,
y no una recuperación multilingüe simétrica.
"""

# El comité fijó Python >= 3.9.5 como entorno de evaluación, y este módulo anota
# `int | None` (PEP 604), que no existe hasta 3.10. Las anotaciones se evalúan al
# definir la función, así que sin esta línea el import falla con TypeError y
# `generador.py` muere antes de hacer nada. Con ella quedan como cadenas y no se
# evalúan nunca. No cambia el comportamiento en 3.10+.
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

NOMBRE_ARCHIVO = "Extracto_Preguntas_50_v2.pdf"


def _localizar_consultas() -> Path:
    """Localiza el archivo de consultas junto a esta biblioteca.

    Se prueban varias ubicaciones para que el script funcione tanto si se
    ejecuta desde el directorio de entrega como desde el repositorio completo.
    Devuelve la primera que exista; si ninguna existe devuelve la preferente, de
    modo que el error que se produzca al abrirla nombre la ruta esperada.
    """
    aqui = Path(__file__).resolve()
    candidatas = [
        aqui.parents[1] / "data" / NOMBRE_ARCHIVO,   # lib/data/  (entrega)
        aqui.parents[2] / "data" / NOMBRE_ARCHIVO,
        aqui.parents[3] / "src" / "data" / NOMBRE_ARCHIVO,
    ]
    for candidata in candidatas:
        if candidata.is_file():
            return candidata
    return candidatas[0]


CONSULTAS = _localizar_consultas()

# Reconoce `q001`, `q17`, `Q050`… Se captura el número para poder normalizar el
# identificador al formato de tres dígitos que usa el esquema de §9.3.1.
RE_ID = re.compile(r"\bq0*(\d{1,3})\b", re.IGNORECASE)

# Por debajo de esta longitud no es una consulta, sino un resto de maquetación
# del PDF.
MIN_CARACTERES = 15


def cargar_consultas(ruta: Path = CONSULTAS) -> List[Tuple[str, str]]:
    """Devuelve `[(query_id, texto), …]` en el orden en que aparecen.

    Acepta el PDF entregado o un archivo `.txt` con una consulta por línea. El
    identificador se normaliza a `qNNN` con tres dígitos, que es el formato del
    esquema de la Tabla 2.
    """
    ruta = Path(ruta)
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe el archivo de consultas: {ruta}")

    if ruta.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        texto = "\n".join((p.extract_text() or "") for p in PdfReader(ruta).pages)
    else:
        texto = ruta.read_text(encoding="utf-8")

    # `re.split` con un grupo de captura intercala los identificadores entre los
    # fragmentos de texto: [previo, "1", cuerpo1, "2", cuerpo2, …]. De ahí que
    # se recorra de dos en dos empezando en el índice 1.
    partes = RE_ID.split(texto)
    consultas: List[Tuple[str, str]] = []
    for i in range(1, len(partes) - 1, 2):
        numero = int(partes[i])
        cuerpo = " ".join(partes[i + 1].split())
        if len(cuerpo) >= MIN_CARACTERES:
            consultas.append((f"q{numero:03d}", cuerpo))
    return consultas


def fenomeno_de_consulta(query_id: str) -> int | None:
    """Fenómeno temático de una consulta, o `None` si no se conoce.

    El archivo de consultas no incluye este dato. La correspondencia se
    estableció revisando las 50 consultas una a una contra la definición de los
    tres fenómenos de §1.1:

        q001–q016 → F1    q017–q032 → F2    q033–q050 → F3

    Fuera de ese rango devuelve `None` en lugar de inferir un valor. El módulo
    de recuperación utiliza este dato para una bonificación de puntuación, de
    modo que un valor inventado empujaría los resultados hacia el fenómeno
    equivocado sin producir ningún error. Ante otro conjunto de consultas, la
    correspondencia debe volver a verificarse.
    """
    if not query_id or not query_id[1:].isdigit():
        return None
    numero = int(query_id[1:])
    if 1 <= numero <= 16:
        return 1
    if 17 <= numero <= 32:
        return 2
    if 33 <= numero <= 50:
        return 3
    return None
