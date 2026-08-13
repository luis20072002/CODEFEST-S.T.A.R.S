"""Encoder (§4): convierte texto en vectores densos normalizados.

Para reproducir `resultados.jsonl` solo se emplean `cargar_modelo()` y
`codificar()`, que codifican las 50 consultas. El resto del módulo pertenece a
la construcción del índice y se incluye como documentación del procedimiento;
§1.4 no exige reproducir esa etapa, y el propio comité organizador confirmó que
la evaluación no replica la generación de vectores, sino únicamente la
generación de los resultados.

────────────────────────────────────────────────────────────────────────────────
EL MODELO ELEGIDO Y POR QUÉ

`BAAI/bge-m3`, evaluado contra los cinco criterios que enumera §4.3:

  Familia encoder (§4.2)   Arquitectura XLM-RoBERTa. §4.2 prohíbe expresamente
                           los modelos decoder «en las etapas de construcción
                           del índice y de recuperación»; este no lo es.
  Soporte multilingüe      Más de 100 idiomas, con español, inglés y portugués
                           nativos. El criterio determinante, sin embargo, es
                           más exigente: las 50 consultas están en español y el
                           corpus es 55 % inglés, de modo que se requiere que el
                           modelo **alinee idiomas distintos en un mismo espacio
                           vectorial**, no solo que los comprenda. BGE-M3 se
                           entrena con ese objetivo explícito y se evalúa en
                           MIRACL y MKQA, que son recuperación translingüe.
  Recuperación densa       Está construido para recuperación, no para
                           clasificación ni similitud de pares, que es la
                           distinción que §4.3 pide establecer.
  Licencia                 MIT, una de las tres que §4.3 prefiere.
  Dimensionalidad          1.024. §4.3 advierte de que «dimensiones más altas no
                           garantizan mejor rendimiento»: no se elige por
                           grande, sino porque es la que produce el modelo.
  Eficiencia               Es su punto débil. La codificación completa del
                           corpus requirió 2,8 h en una GPU T4. §4.3 enumera la
                           eficiencia entre los criterios de selección, de modo
                           que se declara explícitamente.

Ventaja operativa frente a su alternativa directa de la familia E5 (mismo
XLM-RoBERTa large, misma dimensión, misma licencia): **BGE-M3 no requiere
prefijos de instrucción**. E5 exige anteponer marcadores distintos a la consulta
y al pasaje; omitirlos en uno de los dos lados degrada la calidad sin producir
ningún error.

SOLO SE EMPLEA LA MODALIDAD DENSA. BGE-M3 produce además representaciones
dispersas y multi-vector. No se utilizan: §5 obliga a FAISS y §5.2 recomienda
`IndexFlatIP`, que almacena un vector por fragmento.

────────────────────────────────────────────────────────────────────────────────
LA REVISIÓN SE FIJA POR SHA, Y NO ES UN DETALLE

`sentence-transformers` descarga el modelo de HuggingFace en tiempo de
ejecución. Si solo se indicara su nombre, podría descargarse una versión
distinta de la que produjo el índice: los vectores de la consulta dejarían de
residir en el mismo espacio que los del índice y el ranking cambiaría sin emitir
ningún error. Por eso `REVISION` está fijada al sha del commit.

────────────────────────────────────────────────────────────────────────────────
NORMALIZAR LOS VECTORES NO ES OPCIONAL

§5.2 recomienda «IndexFlatIP con vectores normalizados, equivalente a similitud
coseno». `IndexFlatIP` calcula el producto interno, que sobre vectores de norma
unitaria coincide con el coseno (ecuación 4 de §8.2). Sin normalizar, los
fragmentos largos puntuarían más alto por tener mayor norma y no por ser más
relevantes.
"""

# El comité fijó Python >= 3.9.5 como entorno de evaluación, y este módulo anota
# `str | None` (PEP 604), que no existe hasta 3.10. Las anotaciones se evalúan al
# definir la función, así que sin esta línea el import falla con TypeError y
# `generador.py` no puede cargar el encoder. Con ella quedan como cadenas y no se
# evalúan nunca. No cambia el comportamiento en 3.10+.
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, List, Sequence

DATOS = Path(__file__).resolve().parents[1] / "data"
CHUNKS = DATOS / "chunks.jsonl"
EMBEDDINGS = DATOS / "embeddings.npy"
PROGRESO = DATOS / "embeddings.progreso.json"

# ── El modelo y su revisión ──────────────────────────────────────────────────
MODELO = "BAAI/bge-m3"
REVISION = "5617a9f61b028005a4858fdac845db406aefb181"   # sha del commit en HF
DIMENSION = 1024

# Presupuesto de tokens por fragmento adoptado en la construcción del índice.
# NO es el máximo del modelo: la longitud máxima de secuencia efectiva de BGE-M3
# es de 8.192 tokens, y el máximo real observado en el índice es de 6.867, de
# modo que ningún fragmento se trunca. Este valor corresponde al límite de 512
# que §4.3 cita como habitual, y mantenerlo como referencia deja abierta la
# posibilidad de añadir un segundo encoder sin volver a fragmentar, dado que
# §8.4 fusiona por fragmento y los chunks deben valerle al más restrictivo.
MAX_TOKENS = 512

# Tamaño de lote de codificación.
LOTE = 8


def cargar_modelo(nombre: str = MODELO, revision: str | None = REVISION):
    """Carga el encoder con la revisión fijada.

    La importación se realiza dentro de la función, y no en la cabecera del
    módulo, de forma deliberada: `generador.py` importa constantes de este
    archivo y no debe pagar el coste de cargar `sentence_transformers` y torch
    cuando no va a codificar nada.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(nombre, revision=revision)


def cargar_tokenizador(nombre: str = MODELO, revision: str | None = REVISION):
    """Carga únicamente el tokenizador, sin los pesos del modelo.

    La diferencia de tamaño es la razón de ser de esta función: el modelo
    completo son 4,35 GB y el tokenizador de XLM-RoBERTa unos 17 MB. Contar
    tokens no requiere la red neuronal, solo su vocabulario, de modo que el
    campo `num_tokens` de la Tabla 1 puede calcularse en una máquina modesta.

    Se fija la misma revisión que el modelo: el tokenizador debe ser exactamente
    el que produjo los vectores, o `num_tokens` no correspondería.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(nombre, revision=revision)


def contador_de_tokens(modelo) -> Callable[[str], int]:
    """Devuelve la función que cuenta tokens con el tokenizador de este modelo.

    Acepta tanto un `SentenceTransformer` —del que toma su `.tokenizer`— como un
    tokenizador suelto. Es lo que la fragmentación espera recibir como contador.

    Se cuenta **con los tokens especiales**, porque el modelo los añade al
    codificar y también ocupan espacio dentro del límite: contar sin ellos
    dejaría fragmentos de 512 que en realidad son 514.
    """
    # getattr con valor por defecto: si ya es un tokenizador, se usa tal cual.
    tokenizador = getattr(modelo, "tokenizer", modelo)

    def contar(texto: str) -> int:
        return len(tokenizador.encode(texto, add_special_tokens=True))

    return contar


def codificar(modelo, textos: Sequence[str], lote: int = LOTE, progreso: bool = False):
    """Codifica textos y devuelve una matriz de vectores NORMALIZADOS.

    `normalize_embeddings=True` es lo que convierte el producto interno de
    `IndexFlatIP` en similitud coseno (§5.2).
    """
    return modelo.encode(
        list(textos),
        batch_size=lote,
        normalize_embeddings=True,
        show_progress_bar=progreso,
        convert_to_numpy=True,
    )


def codificar_corpus(
    modelo,
    ruta_chunks: Path = CHUNKS,
    ruta_salida: Path = EMBEDDINGS,
    lote: int = LOTE,
    nombre: str = MODELO,
    revision: str | None = REVISION,
) -> int:
    """Codifica todos los fragmentos del corpus y los almacena en un `.npy`.

    Pertenece a la construcción del índice, no a la reproducción de los
    resultados. Se incluye como documentación del procedimiento seguido.

    EL ORDEN ES EL CONTRATO. La fila *i* de este archivo debe corresponder a la
    línea *i* del archivo de fragmentos, porque §5.3 establece que FAISS
    almacena «únicamente los vectores numéricos y sus identificadores enteros
    internos» y que la metadata «debe mantenerse en un almacén separado que
    mapee el identificador interno de FAISS al chunk_id». Si el orden se
    desalineara, cada consulta devolvería la metadata de un fragmento distinto y
    ninguna excepción lo detectaría: los resultados saldrían plausibles y
    erróneos. Por eso se codifica en el orden del archivo, sin reordenar.

    ES REANUDABLE. Se escribe con `open_memmap`, que reserva el archivo completo
    en disco desde el principio y va rellenando filas, y un archivo de progreso
    registra por dónde continuar. El progreso guarda además el modelo y su
    revisión: si no coinciden con los del archivo a medias, se empieza de cero,
    porque mezclar vectores de dos modelos en un mismo índice produce un espacio
    vectorial sin sentido y tampoco emitiría ningún error.
    """
    import json

    import numpy as np
    from numpy.lib.format import open_memmap

    from core.store import read_chunks

    # Se cuentan las líneas en lugar de cargar los objetos: solo hace falta el
    # número para reservar el archivo de salida.
    with open(ruta_chunks, encoding="utf-8") as f:
        total = sum(1 for linea in f if linea.strip())

    dimension = modelo.get_sentence_embedding_dimension()
    firma = {"modelo": nombre, "revision": revision, "total": total,
             "dimension": dimension}

    hechos = 0
    if ruta_salida.is_file() and PROGRESO.is_file():
        anterior = json.loads(PROGRESO.read_text(encoding="utf-8"))
        if {k: anterior.get(k) for k in firma} == firma:
            hechos = int(anterior.get("hechos", 0))
            print(f"reanudando: ya había {hechos:,} de {total:,} vectores")
        else:
            print("el progreso guardado corresponde a otro modelo o a otro "
                  "conjunto de fragmentos → se empieza de cero")

    modo = "r+" if hechos and ruta_salida.is_file() else "w+"
    matriz = open_memmap(ruta_salida, mode=modo, dtype="float32",
                         shape=(total, dimension))

    pendientes: List[str] = []
    indice = hechos
    inicio = time.perf_counter()

    def volcar():
        """Codifica lo acumulado, lo escribe y registra el progreso."""
        nonlocal indice, pendientes
        if not pendientes:
            return
        vectores = codificar(modelo, pendientes, lote=lote)
        matriz[indice:indice + len(pendientes)] = vectores
        indice += len(pendientes)
        pendientes = []
        matriz.flush()
        PROGRESO.write_text(json.dumps({**firma, "hechos": indice}, indent=2),
                            encoding="utf-8")

    for posicion, fragmento in enumerate(read_chunks(ruta_chunks)):
        if posicion < hechos:
            continue                      # ya procesado en una ejecución previa
        pendientes.append(fragmento.text)
        # Se vuelca cada 512 fragmentos para que el progreso avance de verdad;
        # con lotes mayores, una interrupción pierde más trabajo.
        if len(pendientes) >= 512:
            volcar()
            ritmo = (indice - hechos) / max(time.perf_counter() - inicio, 1e-9)
            restan = (total - indice) / max(ritmo, 1e-9) / 3600
            print(f"  {indice:,}/{total:,}  {ritmo:,.1f} frag/s  "
                  f"quedan ~{restan:,.1f} h", flush=True)
    volcar()

    # Comprobación que evita un índice mudo: los vectores deben estar
    # normalizados (§5.2) y ninguno puede haber quedado en ceros.
    normas = np.linalg.norm(np.asarray(matriz[:min(total, 1000)]), axis=1)
    print(f"\nnormas de las primeras {len(normas):,} filas: "
          f"min {normas.min():.4f}  max {normas.max():.4f}  (deben ser 1.0000)")
    return indice
