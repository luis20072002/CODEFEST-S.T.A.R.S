"""Etapa de preprocesamiento completa: limpieza + idioma (§2.2).

Es el puente que faltaba. `cleaner.py` y `language.py` saben transformar UN
documento, pero hasta ahora nadie los llamaba: el orquestador termina su
trabajo al escribir `documentos.jsonl` con el texto **crudo** de los loaders.
Este módulo lee ese archivo, aplica las dos transformaciones a cada documento y
escribe el resultado en `documentos_limpios.jsonl`.

    data_raw/  ──(loaders, Fase 1)──▶  documentos.jsonl
                                            │
                                            ▼  ESTE MÓDULO
                                     documentos_limpios.jsonl
                                            │
                                            ▼  (chunker, Fase 4)

POR QUÉ UN ARCHIVO APARTE Y NO LIMPIAR AL VUELO DENTRO DEL CHUNKER.
El motivo NO es el tiempo: limpiar el corpus entero cuesta del orden de un
minuto, nada comparado con las 10 h de la extracción. Son otras tres razones:

  1. **Sin el "después" en disco no se puede verificar nada.** Los tests de
     `tools/verificar_cleaner.py` comparan el texto antes y después; si la
     limpieza ocurre dentro del chunker, el "después" no existe en ningún sitio
     y los errores solo se ven cuando ya son chunks, que es demasiado tarde.
  2. **Auditabilidad.** Un archivo intermedio se abre, se lee y se le pasa un
     diff cuando se toca una regla del cleaner. Un paso al vuelo no.
  3. **El idioma se calcula una vez.** Al vuelo se recalcularía en cada corrida
     del chunker, y con él la posibilidad de que cambie sin que nadie se entere.

⚠️ Igual que `documentos.jsonl`, este archivo es **de trabajo interno**: no se
entrega, no aparece en §1.4 y no debe confundirse con el `metadata.jsonl` de la
entrega, que lleva fragmentos con los nombres de campo en español de la Tabla 1.

ORDEN DE LAS DOS OPERACIONES: primero limpiar, después detectar el idioma.
No es indiferente. `F2-SWF-035` lleva la marca de agua «SECURE WORLD FOUNDATION
ФОНД БЕЗОПАСНОГО МИРА» repetida 448 veces, que ocupa el 69,3% del documento; si
el clasificador ve ese texto sin limpiar, etiqueta como ruso un documento
escrito en inglés. El cleaner la elimina, pero solo si corre antes.

Uso, desde `src/`:

    py -m preprocess.runner                     # documentos.jsonl → documentos_limpios.jsonl
    py -m preprocess.runner otro.jsonl salida.jsonl
"""

import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterator

from core.document import Document
from core.store import read_documents, write_documents
from preprocess.cleaner import clean_document
from preprocess.language import tag_document

DATOS = Path(__file__).resolve().parents[1] / "data"
ENTRADA = DATOS / "documentos.jsonl"
SALIDA = DATOS / "documentos_limpios.jsonl"

AVISO_CADA = 200      # cada cuántos documentos se informa del avance


class Estadisticas:
    """Contadores del proceso, para el resumen final.

    Van en una clase y no en variables sueltas porque `preprocess_documents()`
    es un **generador**: `write_documents()` lo consume perezosamente, así que
    los contadores tienen que vivir en un objeto que siga existiendo después de
    que el generador se agote y que el llamante pueda leer al terminar.
    """

    def __init__(self) -> None:
        self.total = 0
        self.chars_antes = 0
        self.chars_despues = 0
        self.vacios_antes = 0          # ya venían sin texto de la extracción
        self.vaciados = 0              # ⚠️ tenían texto y la limpieza los dejó en nada
        self.idiomas: Counter = Counter()
        self.sin_idioma: list = []


def preprocess_documents(documentos, estadisticas: Estadisticas) -> Iterator[Document]:
    """Limpia y etiqueta cada documento, devolviéndolos de uno en uno.

    Es un generador para que `write_documents()` pueda ir escribiendo a medida
    que se procesa: así nunca hay más de un documento en memoria. Importa más
    de lo que parece — el corpus limpio pesa unos 200 MB y hay un solo XLSX que
    aporta seis millones de palabras.
    """
    for documento in documentos:
        estadisticas.total += 1
        estadisticas.chars_antes += len(documento.text)
        tenia_texto = bool(documento.text.strip())

        limpio = tag_document(clean_document(documento))

        estadisticas.chars_despues += len(limpio.text)
        if not tenia_texto:
            estadisticas.vacios_antes += 1
        elif not limpio.text.strip():
            estadisticas.vaciados += 1

        estadisticas.idiomas[limpio.language or "SIN_IDIOMA"] += 1
        if limpio.language is None and tenia_texto:
            estadisticas.sin_idioma.append(
                (limpio.doc_id, limpio.format, len(limpio.text.split()))
            )

        # El avance va a stderr y no a stdout para no ensuciar la salida si
        # alguien redirige el resumen a un archivo.
        if estadisticas.total % AVISO_CADA == 0:
            print(f"  … {estadisticas.total} documentos", file=sys.stderr, flush=True)

        yield limpio


def run(entrada: Path = ENTRADA, salida: Path = SALIDA) -> Estadisticas:
    """Corre la etapa entera y devuelve las estadísticas."""
    estadisticas = Estadisticas()
    documentos = read_documents(entrada)
    escritos = write_documents(salida, preprocess_documents(documentos, estadisticas))

    # Comprobación barata pero que vale la pena: si el generador se cortara a
    # la mitad por cualquier motivo, esto lo delata en vez de dejar un archivo
    # incompleto que parece bueno.
    if escritos != estadisticas.total:
        raise RuntimeError(
            f"Se procesaron {estadisticas.total} documentos pero se escribieron "
            f"{escritos}. El archivo {salida} está incompleto."
        )
    return estadisticas


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    entrada = Path(sys.argv[1]) if len(sys.argv) > 1 else ENTRADA
    salida = Path(sys.argv[2]) if len(sys.argv) > 2 else SALIDA

    if not entrada.is_file():
        raise SystemExit(f"No existe la entrada: {entrada}")

    print(f"entrada: {entrada}")
    print(f"salida : {salida}\n")

    inicio = time.perf_counter()
    est = run(entrada, salida)
    duracion = time.perf_counter() - inicio

    eliminado = est.chars_antes - est.chars_despues
    porcentaje = 100 * eliminado / est.chars_antes if est.chars_antes else 0.0

    print(f"\ndocumentos procesados : {est.total}")
    print(f"tiempo                : {duracion:.1f}s")
    print(f"caracteres antes      : {est.chars_antes:,}")
    print(f"caracteres después    : {est.chars_despues:,}")
    print(f"eliminado             : {eliminado:,} ({porcentaje:.2f}%)")
    print(f"ya venían vacíos      : {est.vacios_antes}")
    print(f"vaciados por limpieza : {est.vaciados}"
          + ("   ⚠️ REVISAR" if est.vaciados else ""))

    print("\nreparto de idiomas:")
    for codigo, n in est.idiomas.most_common():
        print(f"  {codigo:12} {n:5}  ({100 * n / est.total:5.1f}%)")

    if est.sin_idioma:
        print(f"\ndocumentos con texto pero sin idioma asignado: {len(est.sin_idioma)}")
        for doc_id, formato, palabras in est.sin_idioma[:20]:
            print(f"  {doc_id:20} {formato:5} {palabras:8,} palabras")
        if len(est.sin_idioma) > 20:
            print(f"  … y {len(est.sin_idioma) - 20} más")

    print(f"\nListo. Verifica ahora con:  py -m tools.verificar_cleaner")
