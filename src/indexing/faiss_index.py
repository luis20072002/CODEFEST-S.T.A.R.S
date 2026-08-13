"""Índice vectorial FAISS (§5): construye el `index.faiss` de la entrega.

    py -m indexing.faiss_index           # embeddings.npy → entrega/…/index.faiss

────────────────────────────────────────────────────────────────────────────────
POR QUÉ `IndexFlatIP` Y NO OTRO

§5.2 lo recomienda literalmente: «Para el volumen de documentos esperado en este
reto, un índice plano (IndexFlatIP con vectores normalizados, equivalente a
similitud coseno) es suficiente y garantiza resultados exactos. Los índices
aproximados son apropiados si el equipo trabaja con corpus de gran escala o si
el tiempo de respuesta es un factor crítico».

Ninguna de las dos excepciones aplica: son 143.962 vectores —una consulta son
~147 M multiplicaciones, milisegundos— y las consultas son **50**, de una sola
vez. Cambiar exactitud por velocidad que no necesitamos solo añadiría una fuente
de error en la métrica.

⚠️ **`IndexFlatIP` calcula producto interno, no coseno.** Son lo mismo *solo* si
los vectores tienen norma 1. Por eso `embedding/encoder.py` codifica con
`normalize_embeddings=True` y por eso este módulo lo **verifica** antes de
indexar en vez de darlo por hecho: con vectores sin normalizar, los fragmentos
largos ganarían por tener norma mayor, no por ser más relevantes, y el ranking
saldría mal sin ningún error visible.

────────────────────────────────────────────────────────────────────────────────
EL ORDEN, OTRA VEZ

`IndexFlatIP` asigna los identificadores internos **por orden de inserción**:
el primer vector añadido es el 0. Como se añaden en el orden de
`embeddings.npy`, que es el de `chunks.jsonl`, que es el de `metadata.jsonl`,
las tres numeraciones coinciden.

No se usa `IndexIDMap` a propósito: añadiría una capa de indirección para
resolver un problema que aquí no existe (no hay ids externos que preservar), y
la comprobación de `tools/verificar_indice.py` —recuperar un vector y ver que
FAISS devuelve su propio índice— cubre el riesgo de forma directa.
"""

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
DATOS = Path(__file__).resolve().parents[1] / "data"
EMBEDDINGS = DATOS / "embeddings.npy"

CARPETA_ENCODER = RAIZ / "entrega" / "base_vectorial" / "encoder_bge-m3"
INDICE = CARPETA_ENCODER / "index.faiss"

# Cuántos vectores se añaden por tanda. `IndexFlatIP.add()` copia los datos a su
# propio almacenamiento, así que trocear evita tener el `.npy` entero (562 MB)
# duplicado en memoria durante la copia.
TANDA = 8192

# Tolerancia al comprobar que las normas son 1. Los vectores vienen en float32 y
# de una GPU, así que exigir igualdad exacta fallaría por el último bit.
TOLERANCIA = 1e-3


class IndiceError(Exception):
    pass


def comprobar_vectores(matriz) -> None:
    """Falla ruidosamente si los vectores no sirven para `IndexFlatIP`.

    Las tres comprobaciones corresponden a fallos que NO lanzan ninguna
    excepción por su cuenta y que producirían un índice mudo:

    1. **Filas en cero** — una corrida de `--todo` interrumpida deja el final
       del `.npy` sin escribir, porque `open_memmap` lo reserva lleno de ceros.
       Un vector de ceros no recupera nada nunca. Se mira **la cola**, que es
       justo donde aparecen y donde no llega la comprobación del encoder.
    2. **Normas distintas de 1** — romperían la equivalencia
       producto interno = coseno en la que se apoya §5.2.
    3. **Tipo distinto de float32** — FAISS lo exige; con float64 falla al
       añadir, y con float16 lo convierte en silencio perdiendo precisión.
    """
    import numpy as np

    if matriz.dtype != np.float32:
        raise IndiceError(f"Los vectores deben ser float32; son {matriz.dtype}")
    if matriz.ndim != 2:
        raise IndiceError(f"Se esperaba una matriz 2D; tiene forma {matriz.shape}")

    total = matriz.shape[0]
    # Cabeza, medio y COLA. La cola es la que importa: es donde deja los ceros
    # una corrida truncada.
    muestras = [
        ("cabeza", np.asarray(matriz[:1000])),
        ("medio", np.asarray(matriz[total // 2: total // 2 + 1000])),
        ("cola", np.asarray(matriz[-1000:])),
    ]
    for nombre, bloque in muestras:
        if bloque.size == 0:
            continue
        normas = np.linalg.norm(bloque, axis=1)
        ceros = int((normas < TOLERANCIA).sum())
        if ceros:
            raise IndiceError(
                f"{ceros} vectores en cero en la {nombre} del archivo. "
                "Casi seguro la codificación no terminó: vuelve a correr "
                "`py -m embedding.encoder --todo`, que reanuda donde iba."
            )
        desvio = float(np.abs(normas - 1.0).max())
        if desvio > TOLERANCIA:
            raise IndiceError(
                f"Normas fuera de 1 en la {nombre} (desvío máximo {desvio:.4f}). "
                "§5.2 exige vectores normalizados para que el producto interno "
                "sea el coseno. Revisa `normalize_embeddings=True` en el encoder."
            )
        print(f"   {nombre:7} normas 1±{desvio:.2e}  ✔")


def build_index(ruta_embeddings: Path = EMBEDDINGS, salida: Path = INDICE):
    """Construye el `IndexFlatIP`, lo verifica y lo persiste. Devuelve el índice."""
    import faiss
    import numpy as np

    if not ruta_embeddings.is_file():
        raise IndiceError(
            f"No existe {ruta_embeddings}. Corre antes `py -m embedding.encoder --todo`."
        )

    # mmap_mode="r": no carga los 562 MB de golpe; se leen por tandas.
    matriz = np.load(ruta_embeddings, mmap_mode="r")
    total, dimension = matriz.shape
    print(f"vectores  : {total:,}   dimensión: {dimension}")

    print("comprobando los vectores:")
    comprobar_vectores(matriz)

    indice = faiss.IndexFlatIP(dimension)
    for inicio in range(0, total, TANDA):
        # np.ascontiguousarray: FAISS necesita memoria contigua en C; una
        # rebanada de un memmap no siempre lo es y fallaría de forma confusa.
        bloque = np.ascontiguousarray(matriz[inicio:inicio + TANDA], dtype=np.float32)
        indice.add(bloque)

    if indice.ntotal != total:
        raise IndiceError(
            f"El índice tiene {indice.ntotal:,} vectores y el archivo {total:,}. "
            "No coinciden, y el mapeo a metadata.jsonl dependería de eso."
        )

    salida.parent.mkdir(parents=True, exist_ok=True)
    # §1.4 exige `faiss.write_index()` por su nombre.
    faiss.write_index(indice, str(salida))
    return indice


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print(f"origen : {EMBEDDINGS}")
    print(f"salida : {INDICE}\n")

    indice = build_index()
    tam = INDICE.stat().st_size / 1024 / 1024

    print(f"\nvectores en el índice : {indice.ntotal:,}")
    print(f"tipo                  : IndexFlatIP (coseno exacto, §5.2)")
    print(f"tamaño en disco       : {tam:,.1f} MB")
    print(f"\nescrito en {INDICE}")
    print("\nSiguiente: `py -m tools.verificar_indice`")
