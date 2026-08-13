"""Encoder (§4): convierte fragmentos en vectores densos.

    py -m embedding.encoder              # diagnóstico + banco de pruebas (64 chunks)
    py -m embedding.encoder 256          # banco de pruebas con 256 chunks
    py -m embedding.encoder --modelo intfloat/multilingual-e5-small 128
    py -m embedding.encoder --todo       # LA CORRIDA DE VERDAD → data/embeddings.npy

⚠️ **El modelo NO está descargado en la máquina de desarrollo** (se borró a propósito, ver
`ESTADO.md` §14). La primera llamada lo baja de HuggingFace. `--todo` es la corrida que hay
que ejecutar **en la máquina con GPU**, y es **reanudable**: si se corta, se vuelve a lanzar
el mismo comando y sigue por donde iba.

────────────────────────────────────────────────────────────────────────────────
EL MODELO ELEGIDO Y POR QUÉ (decidido el 2026-08-06)

`BAAI/bge-m3`, contra los cinco criterios que enumera §4.3:

  Familia encoder (§4.2)   ✔ XLM-RoBERTa. §4.2 prohíbe explícitamente los
                             decoders «en las etapas de construcción del índice
                             y de recuperación», y esto no es uno.
  Soporte multilingüe      ✔ 100+ idiomas, con es/en/pt nativos. Pero el
                             criterio que de verdad manda aquí es más exigente:
                             las 50 consultas están TODAS en español contra un
                             corpus 55% inglés (`ESTADO.md` §5), así que hace
                             falta que **alinee idiomas distintos en el mismo
                             espacio** (cross-lingual), no solo que los
                             entienda. BGE-M3 se entrena con ese objetivo y se
                             evalúa en MIRACL y MKQA, que son recuperación
                             entre idiomas.
  Recuperación densa       ✔ Está construido para recuperación, no para
                             clasificación ni similitud de pares — que es la
                             distinción que §4.3 pide hacer, y la que descarta
                             LaBSE (bitext mining) y paraphrase-multilingual.
  Licencia                 ✔ MIT, una de las tres que §4.3 prefiere.
  Dimensionalidad          1024. §4.3 avisa de que «dimensiones más altas no
                             garantizan mejor rendimiento»: 1024 no se elige
                             por grande, sino porque es la que trae el modelo.
  Eficiencia               ⚠️ Es el punto débil y hay que medirlo en ESTA
                             máquina, que no tiene GPU y tiene 2 hilos de CPU.
                             Para eso existe el banco de pruebas de abajo.

Ventaja operativa sobre `intfloat/multilingual-e5-large`, su alternativa
directa (mismo XLM-R large, misma dimensión, también MIT): **BGE-M3 no necesita
prefijos**. E5 exige anteponer `"query: "` y `"passage: "` a cada texto y, si se
olvidan en un lado, la calidad cae sin dar ningún error. En una entrega que
§1.4 excluye si no reproduce, un paso menos que se puede olvidar es un riesgo
menos.

⚠️ **SOLO SE USA LA MODALIDAD DENSA.** BGE-M3 también produce representaciones
dispersas y multi-vector. No se usan: §5 obliga a FAISS y §5.2 recomienda
`IndexFlatIP`, que guarda **un** vector por fragmento. La vía que el PDF sí
contempla para combinar señales es §4.4 + §8.4 (varios encoders, cada uno con
su índice, fusionados con CombSUM/CombMNZ), y eso se deja para el final.

────────────────────────────────────────────────────────────────────────────────
LA REVISIÓN SE FIJA, Y NO ES UN DETALLE

`sentence-transformers` descarga el modelo de HuggingFace en tiempo de
ejecución. Si solo se dijera su nombre, el jurado podría bajar una versión
distinta a la que produjo el índice: los vectores de la consulta dejarían de
vivir en el mismo espacio que los del índice y **el ranking cambiaría**. Es
silencioso y §1.4 excluye la entrega por ello. Por eso `REVISION` está fijada
al sha del commit y hay que repetirlo en el informe técnico.

────────────────────────────────────────────────────────────────────────────────
NORMALIZAR LOS VECTORES NO ES OPCIONAL

§5.2 recomienda «IndexFlatIP con vectores normalizados, equivalente a similitud
coseno». `IndexFlatIP` calcula producto interno; sobre vectores de norma 1 el
producto interno **es** el coseno. Si se indexan sin normalizar, los fragmentos
largos ganan por tener norma mayor, no por ser más relevantes.
"""

import sys
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

# Tope de tokens por fragmento. NO es el máximo del modelo (BGE-M3 admite 8192):
# es el nuestro. §4.3 habla de 512 como límite típico, y mantenerse ahí deja
# abierta la puerta a añadir un segundo encoder sin re-chunkear, porque §8.4
# fusiona **por fragmento** y los chunks tienen que valerle al más restrictivo.
MAX_TOKENS = 512

# Cuántos fragmentos se codifican de una vez. Con 2 hilos de CPU y sin GPU,
# lotes grandes no ayudan y disparan la memoria.
LOTE = 8


def cargar_modelo(nombre: str = MODELO, revision: str | None = REVISION):
    """Carga el encoder con la revisión fijada.

    Se importa aquí dentro, y no arriba del módulo, a propósito:
    `generador.py` (§1.4) tiene que poder importar constantes de este archivo
    sin pagar los segundos que tarda `sentence_transformers` en cargar torch.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(nombre, revision=revision)


def cargar_tokenizador(nombre: str = MODELO, revision: str | None = REVISION):
    """Carga SOLO el tokenizador, sin los pesos del modelo.

    ⚠️ **La diferencia de tamaño es enorme y por eso existe esta función**:
    `cargar_modelo()` descarga **4,35 GB**; el tokenizador de XLM-RoBERTa son
    unos **17 MB**. Contar tokens no necesita la red neuronal, solo su
    vocabulario.

    Gracias a esto, `py -m chunking.chunker --tokens` y
    `py -m indexing.metadata --tokens` se pueden ejecutar en una máquina
    modesta —incluida esta, donde el modelo se borró a propósito
    (`ESTADO.md` §14)— sin descargar los pesos.

    Se fija la misma `revision` que el modelo: el tokenizador tiene que ser
    exactamente el que produjo los vectores, o `num_tokens` mentiría.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(nombre, revision=revision)


def contador_de_tokens(modelo) -> Callable[[str], int]:
    """Devuelve la función que cuenta tokens con el tokenizador de ESTE modelo.

    Acepta tanto un `SentenceTransformer` (usa su `.tokenizer`) como un
    tokenizador suelto de `cargar_tokenizador()`, que es la vía barata.

    Es lo que `chunking.chunker` espera recibir en `contar_tokens`. Se cuenta
    **con los tokens especiales** (`<s>` y `</s>`) porque el modelo los añade al
    codificar y también ocupan sitio dentro del tope: contar sin ellos dejaría
    fragmentos de 512 que en realidad son 514 y se truncarían por detrás.

    ⚠️ Por esto el chunker recibe el contador como parámetro en vez de cablear
    un `palabras × 1,3`: la relación token/palabra no es la misma en español,
    inglés y portugués, y con un tope de 512 los fragmentos en español se
    pasarían.
    """
    # getattr con defecto: si ya es un tokenizador, se usa tal cual.
    tokenizador = getattr(modelo, "tokenizer", modelo)

    def contar(texto: str) -> int:
        return len(tokenizador.encode(texto, add_special_tokens=True))

    return contar


def codificar(modelo, textos: Sequence[str], lote: int = LOTE, progreso: bool = False):
    """Codifica textos y devuelve una matriz de vectores NORMALIZADOS.

    `normalize_embeddings=True` es lo que convierte el producto interno de
    `IndexFlatIP` en similitud coseno (§5.2). No quitarlo.
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
    """Codifica TODOS los fragmentos y los deja en un `.npy`. Es reanudable.

    Devuelve cuántos vectores hay en el archivo al terminar.

    ────────────────────────────────────────────────────────────────────────
    POR QUÉ ES REANUDABLE, Y POR QUÉ ESO NO ES UN LUJO

    Esta es la etapa más cara del proyecto: medida en horas, no en minutos
    (`ESTADO.md` §14). Una corrida que se cae a las seis horas y hay que
    empezar de cero es la diferencia entre entregar y no entregar. Se escribe
    con `open_memmap`, que reserva el archivo entero en disco desde el
    principio y va rellenando filas: si el proceso muere, lo hecho está
    guardado y el contador de `embeddings.progreso.json` dice por dónde
    seguir.

    ⚠️ **EL ORDEN ES EL CONTRATO.** La fila *i* de este `.npy` tiene que
    corresponder a la línea *i* de `chunks.jsonl`, porque §5.3 dice que FAISS
    solo guarda «los vectores numéricos y sus identificadores enteros
    internos» y que la metadata «debe mantenerse en un almacén separado que
    mapee el identificador interno de FAISS al chunk_id». Si el orden se
    desalinea, cada consulta devuelve la metadata de otro fragmento y **no lo
    detecta ninguna excepción**: los resultados salen plausibles y mal.
    Por eso se codifica en el orden del archivo, sin ordenar ni barajar nada.

    ⚠️ El progreso guarda **modelo y revisión**. Si no coinciden con los del
    archivo a medias, se empieza de cero: mezclar vectores de dos modelos en
    un mismo índice produce un espacio vectorial sin sentido, y tampoco daría
    ningún error.
    """
    import json

    import numpy as np
    from numpy.lib.format import open_memmap

    from core.store import read_chunks

    # Cuántos fragmentos hay. Se cuentan las líneas en vez de cargar los
    # objetos: es un archivo de cientos de miles de líneas y solo hace falta
    # el número para reservar el .npy.
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
            print("el progreso guardado es de otro modelo o de otro chunks.jsonl "
                  "→ se empieza de cero")

    modo = "r+" if hechos and ruta_salida.is_file() else "w+"
    matriz = open_memmap(ruta_salida, mode=modo, dtype="float32",
                         shape=(total, dimension))

    pendientes: List[str] = []
    indice = hechos
    inicio = time.perf_counter()

    def volcar():
        """Codifica lo acumulado, lo escribe y anota el progreso."""
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
            continue                      # ya estaba hecho de una corrida anterior
        pendientes.append(fragmento.text)
        # Se vuelca cada 512 para que el progreso avance de verdad; con lotes
        # más grandes, una caída pierde más trabajo.
        if len(pendientes) >= 512:
            volcar()
            ritmo = (indice - hechos) / max(time.perf_counter() - inicio, 1e-9)
            restan = (total - indice) / max(ritmo, 1e-9) / 3600
            print(f"  {indice:,}/{total:,}  {ritmo:,.1f} frag/s  "
                  f"quedan ~{restan:,.1f} h", flush=True)
    volcar()

    # Comprobación barata que evita un índice mudo: los vectores tienen que
    # estar normalizados (§5.2) y ninguno puede haber quedado en ceros.
    normas = np.linalg.norm(np.asarray(matriz[:min(total, 1000)]), axis=1)
    print(f"\nnormas de las primeras {len(normas):,} filas: "
          f"min {normas.min():.4f}  max {normas.max():.4f}  (deben ser 1.0000)")
    return indice


if __name__ == "__main__":
    # Banco de pruebas: mide en ESTA máquina lo que §4.3 llama «eficiencia
    # computacional», en vez de suponerlo. Codifica una muestra real de
    # fragmentos del corpus y extrapola al total.
    import torch

    from core.store import read_chunks

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    argumentos = sys.argv[1:]
    nombre = MODELO
    revision = REVISION
    if "--modelo" in argumentos:
        i = argumentos.index("--modelo")
        nombre = argumentos[i + 1]
        revision = None          # otros modelos no tienen sha fijado aquí
        argumentos = argumentos[:i] + argumentos[i + 2:]
    todo = "--todo" in argumentos
    if todo:
        argumentos.remove("--todo")
    n = int(argumentos[0]) if argumentos else 64

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"modelo      : {nombre}")
    print(f"revisión    : {revision or '(sin fijar)'}")
    print(f"dispositivo : {dispositivo}   hilos CPU: {torch.get_num_threads()}")

    if not CHUNKS.is_file():
        print(f"No existe {CHUNKS}. Corre antes `py -m chunking.chunker`.")
        raise SystemExit(1)

    if todo:
        # La corrida de verdad. Pensada para la máquina con GPU: se instala el
        # modelo (se descarga solo al cargarlo) y se ejecuta esto.
        print("\ncargando el modelo…")
        modelo = cargar_modelo(nombre, revision)
        inicio = time.perf_counter()
        hechos = codificar_corpus(modelo, nombre=nombre, revision=revision)
        print(f"\n{hechos:,} vectores en {EMBEDDINGS}")
        print(f"tiempo total: {(time.perf_counter() - inicio) / 3600:,.2f} h")
        raise SystemExit(0)

    # Muestra real del corpus, no texto inventado: la velocidad depende de la
    # longitud de los fragmentos y de su idioma.
    muestra: List = []
    for fragmento in read_chunks(CHUNKS):
        muestra.append(fragmento)
        if len(muestra) >= n:
            break

    print(f"\ncargando el modelo… (la primera vez se descarga de HuggingFace)")
    inicio = time.perf_counter()
    modelo = cargar_modelo(nombre, revision)
    print(f"cargado en {time.perf_counter() - inicio:,.1f} s")

    contar = contador_de_tokens(modelo)
    tokens = [contar(f.text) for f in muestra]
    tokens.sort()
    palabras = sorted(f.word_count for f in muestra)
    print(f"\nsobre {len(muestra)} fragmentos reales:")
    print(f"  palabras/fragmento : mediana {palabras[len(palabras)//2]}  max {palabras[-1]}")
    print(f"  tokens/fragmento   : mediana {tokens[len(tokens)//2]}  max {tokens[-1]}")
    print(f"  tokens por palabra : {sum(tokens)/max(sum(palabras),1):.2f}")
    print(f"  se pasan de {MAX_TOKENS}: {sum(1 for t in tokens if t > MAX_TOKENS)}")

    print("\ncodificando…")
    inicio = time.perf_counter()
    vectores = codificar(modelo, [f.text for f in muestra])
    segundos = time.perf_counter() - inicio

    ritmo = len(muestra) / segundos
    print(f"\nforma de la matriz : {vectores.shape}   (esperado: (n, {DIMENSION}))")
    print(f"norma del primero  : {float((vectores[0] ** 2).sum()) ** 0.5:.4f}  "
          f"(tiene que ser 1.0000 — §5.2)")
    print(f"tiempo             : {segundos:,.1f} s")
    print(f"ritmo              : {ritmo:,.2f} fragmentos/s")

    TOTAL_CORPUS = 91_021
    horas = TOTAL_CORPUS / ritmo / 3600
    print(f"\nEXTRAPOLACIÓN a {TOTAL_CORPUS:,} fragmentos: "
          f"{horas:,.1f} h ({horas*60:,.0f} min)")
