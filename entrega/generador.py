"""Generador de `resultados.jsonl` — entregable 4 de §1.4.

    python generador.py                 # la corrida real (necesita el encoder)
    python generador.py --simulado      # prueba de formato SIN el modelo
    python generador.py --consultas otro.pdf --salida otro.jsonl

────────────────────────────────────────────────────────────────────────────────
QUÉ ES ESTE ARCHIVO Y POR QUÉ IMPORTA MÁS QUE NINGÚN OTRO

§1.4 lo define como un «Script Python que utilice el índice, lea el archivo de
consultas y genere el archivo de resultados», y añade la frase que gobierna todo
el proyecto:

    **si `generador.py` no reproduce los resultados, la entrega se excluye de
    la evaluación.**

De ahí salen las tres reglas que este archivo respeta:

1. **No importa ningún loader ni el orquestador.** Solo el índice, el
   `metadata.jsonl`, el encoder y el archivo de consultas. Si importara un
   loader arrastraría `pdfplumber` y `pytesseract` a la máquina del jurado sin
   ninguna necesidad, y un fallo de esas dependencias tumbaría la entrega.
   *(Sí importa el chunker, indirectamente, a través de `retrieval.fragmentos`:
   solo usa `re` y la librería estándar, y evita duplicar el cortador de
   oraciones que §9.2.1 necesita. Ver `ESTADO.md` §8.)*

2. **La revisión del modelo está fijada.** `embedding/encoder.py` carga
   `BAAI/bge-m3` con el sha del commit. Sin eso, el jurado podría descargar otra
   versión, los vectores de la consulta dejarían de vivir en el espacio del
   índice y **el ranking cambiaría en silencio**. Es el riesgo real de §1.4.

3. **Todo empate se rompe de forma determinista**, por `chunk_id`. Con
   `IndexFlatIP` la búsqueda es exacta, así que los empates de puntuación son el
   único punto de variación entre corridas.

────────────────────────────────────────────────────────────────────────────────
QUÉ NECESITA PARA CORRER

    entrega/base_vectorial/encoder_bge-m3/index.faiss
    entrega/base_vectorial/encoder_bge-m3/metadata.jsonl
    src/data/Extracto_Preguntas_50_v2.pdf     (o el archivo que se le pase)

y `pip install sentence-transformers faiss-cpu pypdf`. **`faiss-cpu`, no
`faiss-gpu`**: esto se ejecuta en la máquina del jurado y puede no haber GPU.

La primera vez descarga el modelo (4,35 GB) de HuggingFace. Codificar 50
consultas en CPU son unos 8 minutos; en GPU, segundos.
"""

import argparse
import json
import sys
import time
from pathlib import Path

# El script vive en `entrega/`, fuera de `src/`, así que hay que decirle a
# Python dónde están los módulos del proyecto. Se hace con una ruta relativa a
# ESTE archivo, no al directorio desde el que se invoque: así funciona igual se
# ejecute desde donde se ejecute, que es lo que hará el jurado.
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from retrieval.consultas import (CONSULTAS, cargar_consultas,  # noqa: E402
                                 fenomeno_de_consulta)
from retrieval.search import (BONIFICACION_FENOMENO, FACTOR_IDIOMA_OTROS,  # noqa: E402
                              INDICE, MAX_POR_DOCUMENTO, METADATA,
                              N_DOCUMENTOS, N_FRAGMENTOS, Buscador)

SALIDA = RAIZ / "entrega" / "resultados.jsonl"


def generar(consultas, buscador, modelo=None, max_por_documento=MAX_POR_DOCUMENTO,
            bonificacion=BONIFICACION_FENOMENO,
            factor_idioma=FACTOR_IDIOMA_OTROS):
    """Produce un objeto de resultado por consulta, en orden.

    Con `modelo=None` funciona en **modo simulado**: en vez de codificar la
    consulta usa un vector del propio índice. No sirve para evaluar nada —los
    resultados no tienen relación con la consulta— pero permite comprobar que
    el formato de salida cumple §9.3.1 sin descargar 4,35 GB.

    `bonificacion > 1.0` empuja hacia el fenómeno de la consulta. El fenómeno
    sale de `fenomeno_de_consulta()`, que es una correspondencia **verificada a
    mano** sobre el archivo de ADL, no un campo del archivo.

    `factor_idioma < 1.0` penaliza los fragmentos fuera de {es, en} (§8.7). Va
    en 1.0 —desactivado— hasta que esté medido; ver `retrieval.search
    .aplicar_factor_idioma()` y `tools/barrer_factor_idioma.py`.
    """
    import numpy as np

    for posicion, (query_id, texto) in enumerate(consultas):
        fenomeno = fenomeno_de_consulta(query_id) if bonificacion != 1.0 else None
        comun = dict(query_id=query_id, max_por_documento=max_por_documento,
                     fenomeno=fenomeno, bonificacion=bonificacion,
                     factor_idioma=factor_idioma)
        if modelo is not None:
            resultado = buscador.buscar_texto(texto, modelo, **comun)
        else:
            # Vector del índice, repartido para que no salgan los mismos
            # siempre. Determinista: no se usa azar.
            indice = (posicion * 977) % buscador.indice.ntotal
            vector = np.asarray(buscador.indice.reconstruct(int(indice)))
            resultado = buscador.buscar(vector, consulta=texto, **comun)
        yield resultado


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Genera resultados.jsonl (§9.3)")
    parser.add_argument("--consultas", type=Path, default=CONSULTAS,
                        help="PDF o .txt con las consultas")
    parser.add_argument("--salida", type=Path, default=SALIDA)
    parser.add_argument("--indice", type=Path, default=INDICE)
    parser.add_argument("--metadata", type=Path, default=METADATA)
    parser.add_argument("--max-por-documento", type=int, default=MAX_POR_DOCUMENTO,
                        help="0 desactiva el tope de fragmentos por documento")
    parser.add_argument("--bonificacion", type=float, default=BONIFICACION_FENOMENO,
                        help="factor para los documentos del fenómeno de la "
                             "consulta (1.0 = desactivada; útil entre 1.02 y 1.05)")
    parser.add_argument("--factor-idioma", type=float, default=FACTOR_IDIOMA_OTROS,
                        help="factor <1.0 que penaliza los fragmentos fuera de "
                             "es/en (1.0 = desactivado). Elegirlo con "
                             "tools/barrer_factor_idioma.py, no a ojo")
    parser.add_argument("--simulado", action="store_true",
                        help="NO codifica: prueba de formato sin el modelo")
    args = parser.parse_args()

    if args.simulado and args.salida == SALIDA:
        # Que una prueba no pise el entregable de verdad.
        args.salida = SALIDA.with_name("resultados_simulado.jsonl")
        print("⚠️  MODO SIMULADO: los resultados NO responden a las consultas.")
        print(f"    Sirve solo para validar el formato. Salida: {args.salida.name}\n")

    consultas = cargar_consultas(args.consultas)
    print(f"consultas : {len(consultas)} de {args.consultas.name}")
    if args.bonificacion != 1.0:
        print(f"⚠️  bonificación por fenómeno ACTIVADA: ×{args.bonificacion}")
        print("    Depende del reparto q001–q016→F1, q017–q032→F2, q033–q050→F3,")
        print("    verificado a mano sobre el archivo de ADL. Ver ESTADO.md §5.")
    if args.factor_idioma != 1.0:
        print(f"⚠️  factor por idioma ACTIVADO: ×{args.factor_idioma} "
              f"fuera de {{es, en}}")
        print("    Si este valor no coincide con el que produjo el resultados.jsonl")
        print("    entregado, §1.4 excluye la entrega. Ver ESTADO.md §17.")
    if len(consultas) != 50:
        # §9.3 exige exactamente 50 líneas. Avisar, no fallar: si ADL entrega
        # otro conjunto, el script debe seguir sirviendo.
        print(f"⚠️  §9.3 espera 50 consultas y se han leído {len(consultas)}.")

    print("cargando índice y metadata…")
    buscador = Buscador(args.indice, args.metadata)
    print(f"índice    : {buscador.indice.ntotal:,} vectores")

    modelo = None
    if not args.simulado:
        from embedding.encoder import MODELO, REVISION, cargar_modelo

        print(f"modelo    : {MODELO}  revisión {REVISION}")
        print("cargando… (la primera vez descarga 4,35 GB)")
        modelo = cargar_modelo()

    inicio = time.perf_counter()
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    # Se escribe a un temporal y se renombra: una corrida interrumpida no debe
    # dejar un `resultados.jsonl` a medias, que es peor que no dejar ninguno.
    temporal = args.salida.with_suffix(args.salida.suffix + ".parcial")

    escritas = 0
    with open(temporal, "w", encoding="utf-8", newline="\n") as f:
        for resultado in generar(consultas, buscador, modelo,
                                 args.max_por_documento, args.bonificacion,
                                 args.factor_idioma):
            f.write(json.dumps(resultado.to_json(), ensure_ascii=False))
            f.write("\n")
            escritas += 1
            if escritas % 10 == 0:
                print(f"  {escritas}/{len(consultas)}", flush=True)

    temporal.replace(args.salida)
    buscador.cerrar()

    segundos = time.perf_counter() - inicio
    print(f"\nescritas {escritas} líneas en {args.salida}")
    print(f"tiempo   : {segundos:,.1f} s")
    print("\nSiguiente: `py -m tools.verificar_resultados` desde src/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
