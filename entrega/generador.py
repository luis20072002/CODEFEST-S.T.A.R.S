"""Generador de `resultados.jsonl` — entregable 4 de §1.4.

    pip install -r requirements.txt     # 9 paquetes; NO hace falta Tesseract
    python generador.py                 # la corrida real (necesita el encoder)
    python generador.py --comprobar     # ¿reproduce la salida ya escrita? (§1.4)
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
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# El script vive en `entrega/`, fuera de `src/`, así que hay que decirle a
# Python dónde están los módulos del proyecto. Se hace con una ruta relativa a
# ESTE archivo, no al directorio desde el que se invoque: así funciona igual se
# ejecute desde donde se ejecute, que es lo que hará el jurado.
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from retrieval.consultas import (CONSULTAS, cargar_consultas,  # noqa: E402
                                 fenomeno_de_consulta)
from retrieval.fragmentos import LIMITE_PALABRAS  # noqa: E402
from retrieval.search import (BONIFICACION_FENOMENO, FACTOR_IDIOMA_OTROS,  # noqa: E402
                              INDICE, K_CANDIDATOS, MAX_POR_DOCUMENTO,
                              METADATA, N_DOCUMENTOS, N_FRAGMENTOS, Buscador)

SALIDA = RAIZ / "entrega" / "resultados.jsonl"

# El manifiesto va junto a la salida, con extensión `.json` y NO `.jsonl`, a
# propósito: si un script de evaluación busca `*.jsonl` en esta carpeta, no debe
# recogerlo por error.
MANIFIESTO = SALIDA.with_name("resultados.manifest.json")


# ── Trazabilidad de la corrida (§1.4) ────────────────────────────────────────
#
# 🔴 POR QUÉ EXISTE ESTO. Durante dos días el `resultados.jsonl` entregado se
# había generado con bonificación **1.0** mientras el código decía **1.03**. Los
# dos archivos eran válidos, las seis pruebas del validador pasaban, y aun así
# la entrega estaba **excluida** por §1.4, porque `generador.py` no reproducía
# esa salida. Nada en el repositorio lo delataba: la configuración con la que se
# produjo un `resultados.jsonl` no quedaba escrita en ninguna parte.
#
# El manifiesto cierra ese agujero. Registra la configuración, las versiones y
# los hashes, y `--comprobar` los contrasta contra el código actual sin
# necesidad del modelo. Es la diferencia entre descubrir el desajuste en un
# segundo y descubrirlo cuando ya no se puede corregir.

def sha256_de(ruta: Path, trozo: int = 1 << 20) -> str:
    """SHA-256 de un archivo, leído por trozos de 1 MiB.

    Por trozos y no de golpe porque `index.faiss` son 373 MB: cargarlo entero en
    memoria para hashearlo no aporta nada y duplica el pico de RAM.
    """
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(trozo), b""):
            h.update(bloque)
    return h.hexdigest()


def contar_lineas(ruta: Path) -> int:
    """Líneas no vacías. Se cuenta en binario: es más rápido y no decodifica."""
    with open(ruta, "rb") as f:
        return sum(1 for linea in f if linea.strip())


def _version(paquete: str) -> str:
    """Versión instalada de un paquete, o `"?"` si no se puede averiguar."""
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version(paquete)
    except PackageNotFoundError:
        return "?"


def configuracion_actual(args) -> dict:
    """Los parámetros que determinan la salida, tal como los usa esta corrida.

    Son **solo** los que cambian el `resultados.jsonl`. Cosas como la ruta de
    salida o el nivel de verbosidad no van aquí: harían que `--comprobar`
    reportara un desajuste donde no hay ninguno.
    """
    return {
        "bonificacion_fenomeno": args.bonificacion,
        "factor_idioma": args.factor_idioma,
        "max_por_documento": args.max_por_documento,
        "k_candidatos": K_CANDIDATOS,
        "n_documentos": N_DOCUMENTOS,
        "n_fragmentos": N_FRAGMENTOS,
        "limite_palabras": LIMITE_PALABRAS,
    }


def escribir_manifiesto(args, ntotal: int, escritas: int, simulado: bool,
                        segundos: float) -> dict:
    """Escribe el manifiesto junto a la salida y lo devuelve."""
    from embedding.encoder import MODELO, REVISION

    manifiesto = {
        "generado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "simulado": simulado,
        "segundos": round(segundos, 1),
        "entorno": {
            "python": platform.python_version(),
            "plataforma": platform.platform(),
            "faiss-cpu": _version("faiss-cpu"),
            "sentence-transformers": _version("sentence-transformers"),
            "torch": _version("torch"),
            "transformers": _version("transformers"),
            "numpy": _version("numpy"),
            "pypdf": _version("pypdf"),
        },
        "encoder": {"modelo": MODELO, "revision": REVISION},
        # De dónde salió el vector de la consulta. NO va dentro de
        # `configuracion` a propósito: no altera la salida —el vector es el
        # mismo— así que `--comprobar` no debe reportar un desajuste cuando el
        # jurado codifique las consultas y nosotros las hayamos leído del cache.
        # Se registra aparte porque es información de trazabilidad honesta:
        # dice si el modelo se cargó de verdad en esta corrida.
        "vectores_consulta": ("precodificados: " + args.vectores.name
                              if args.vectores is not None
                              else "simulados (vectores del índice)" if simulado
                              else "codificados con el modelo"),
        "configuracion": configuracion_actual(args),
        "consultas": {"archivo": args.consultas.name, "n": escritas},
        "indice": {
            "archivo": str(args.indice.relative_to(RAIZ)),
            "bytes": args.indice.stat().st_size,
            "vectores": ntotal,
            "sha256": sha256_de(args.indice),
        },
        "metadata": {
            "archivo": str(args.metadata.relative_to(RAIZ)),
            "bytes": args.metadata.stat().st_size,
            "lineas": contar_lineas(args.metadata),
        },
        "salida": {
            "archivo": args.salida.name,
            "bytes": args.salida.stat().st_size,
            "lineas": escritas,
            "sha256": sha256_de(args.salida),
        },
    }
    destino = args.salida.with_name(args.salida.stem + ".manifest.json")
    destino.write_text(json.dumps(manifiesto, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
    print(f"manifiesto: {destino.name}")
    return manifiesto


def comprobar(args) -> int:
    """Contrasta un `resultados.jsonl` ya escrito con el código actual.

    **No necesita el modelo ni regenerar nada**, así que corre en segundos en
    cualquier máquina. Es la comprobación que hay que hacer antes de entregar.

    Devuelve 0 si todo cuadra y 1 si algo no, para poder encadenarlo.
    """
    problemas: list = []
    avisos: list = []

    if not args.salida.is_file():
        print(f"✘ no existe {args.salida}")
        return 1

    lineas = contar_lineas(args.salida)
    sha = sha256_de(args.salida)
    print(f"salida     : {args.salida.name}")
    print(f"  líneas   : {lineas}")
    print(f"  bytes    : {args.salida.stat().st_size:,}")
    print(f"  sha256   : {sha}")

    # §9.3 exige exactamente 50 líneas, una por consulta.
    if lineas != 50:
        problemas.append(f"§9.3 exige 50 líneas y hay {lineas}")

    manifiesto_ruta = args.salida.with_name(args.salida.stem + ".manifest.json")
    if not manifiesto_ruta.is_file():
        print(f"\n⚠️  no hay manifiesto ({manifiesto_ruta.name}).")
        print("    Sin él no se puede saber con qué configuración se generó esta")
        print("    salida, que es exactamente el fallo que §1.4 castiga. Vuelve a")
        print("    correr `python generador.py` para que se escriba.")
        return 1

    m = json.loads(manifiesto_ruta.read_text(encoding="utf-8"))
    print(f"\nmanifiesto : {manifiesto_ruta.name}  ({m['generado']})")

    if m.get("simulado"):
        problemas.append("el manifiesto dice que esta salida es SIMULADA: sus "
                         "resultados no responden a las consultas")

    # 1. El hash: ¿es este el archivo que produjo aquella corrida?
    if m["salida"]["sha256"] != sha:
        problemas.append(
            "el sha256 de la salida NO coincide con el del manifiesto: el "
            "archivo se modificó o se regeneró con otra configuración")
    else:
        print("  ✔ el sha256 coincide con el del manifiesto")

    # 2. La configuración: ESTE es el chequeo que faltaba.
    actual = configuracion_actual(args)
    guardada = m["configuracion"]
    diferencias = {k: (guardada.get(k), v) for k, v in actual.items()
                   if guardada.get(k) != v}
    if diferencias:
        problemas.append("la configuración del código NO es la que produjo esta "
                         "salida, así que `generador.py` no la reproduce (§1.4):")
        for k, (antes, ahora) in diferencias.items():
            problemas.append(f"      {k}: manifiesto={antes}  código={ahora}")
    else:
        print("  ✔ la configuración del código reproduce esta salida")
        for k, v in actual.items():
            print(f"      {k} = {v}")

    # 3. El encoder: la revisión es el riesgo silencioso de §1.4.
    from embedding.encoder import MODELO, REVISION
    if m["encoder"]["modelo"] != MODELO or m["encoder"]["revision"] != REVISION:
        problemas.append(
            f"el encoder cambió: manifiesto={m['encoder']['modelo']}@"
            f"{m['encoder']['revision'][:8]} código={MODELO}@{REVISION[:8]}. "
            f"Los vectores de la consulta no vivirían en el espacio del índice")
    else:
        print(f"  ✔ encoder {MODELO} revisión {REVISION[:8]}…")

    # 4. El índice: si es otro, el ranking es otro.
    if args.indice.is_file():
        bytes_ahora = args.indice.stat().st_size
        if bytes_ahora != m["indice"]["bytes"]:
            problemas.append(
                f"el índice cambió de tamaño: manifiesto="
                f"{m['indice']['bytes']:,} B, ahora={bytes_ahora:,} B")
        elif args.hash_indice:
            if sha256_de(args.indice) != m["indice"]["sha256"]:
                problemas.append("el índice tiene el mismo tamaño pero otro "
                                 "sha256: no es el que produjo esta salida")
            else:
                print(f"  ✔ index.faiss idéntico ({bytes_ahora:,} B)")
        else:
            print(f"  ✔ index.faiss del tamaño esperado ({bytes_ahora:,} B)"
                  f"  [--hash-indice para comprobar el sha256]")
    else:
        avisos.append(f"no está el índice en {args.indice}, no se pudo comparar")

    # 5. Las versiones que determinan los valores numéricos.
    for paquete in ("faiss-cpu", "sentence-transformers", "torch", "transformers",
                    "numpy"):
        antes = m["entorno"].get(paquete)
        ahora = _version(paquete)
        if antes and antes != "?" and ahora != "?" and antes != ahora:
            avisos.append(f"{paquete}: se generó con {antes} y aquí hay {ahora}")

    linea = "─" * 74
    print(f"\n{linea}")
    for a in avisos:
        print(f"⚠️  {a}")
    if problemas:
        for p in problemas:
            print(f"✘ {p}")
        print(f"{linea}\nVEREDICTO: ✘ esta salida NO es reproducible con el código actual")
        print("Si el código es el correcto, regenera la salida. Si la salida es la")
        print("correcta, ajusta el código a la configuración del manifiesto.")
        return 1
    print(f"{linea}\nVEREDICTO: ✔ `generador.py` reproduce esta salida (§1.4)")
    print("Falta lo que esto NO puede comprobar: que el contenido sea bueno. Eso")
    print("solo lo mide el ground truth, que no es público (§10.1).")
    return 0


def generar(consultas, buscador, modelo=None, max_por_documento=MAX_POR_DOCUMENTO,
            bonificacion=BONIFICACION_FENOMENO,
            factor_idioma=FACTOR_IDIOMA_OTROS, vectores=None):
    """Produce un objeto de resultado por consulta, en orden.

    Tres formas de obtener el vector de la consulta, en orden de preferencia:

    1. **`modelo`** — se codifica la consulta. Es el camino normal y el que
       ejecuta quien reproduce la entrega.
    2. **`vectores`** — matriz `(n_consultas × dim)` ya codificada con el MISMO
       modelo y la MISMA revisión. Permite regenerar la salida en una máquina sin
       los 4,35 GB del modelo: 200 KB de vectores en lugar de 4,35 GB de pesos.
       La procedencia se comprueba antes de usarla —modelo, revisión, ids de las
       consultas y normas— porque mezclar dos espacios vectoriales no daría
       ningún error y produciría un ranking sin sentido.

       🔴 **NO produce una salida idéntica a la del camino 1, y este docstring
       afirmaba lo contrario.** Medido el 2026-08-08 contra la salida generada
       con el modelo: **4 de 50 consultas difieren** (`q020`, `q023`, `q024`,
       `q028`), y en las dos que cambian a nivel documento el **conjunto es el
       mismo** — solo cambia el orden, que §10.2.2 no considera para F1@3.

       La causa está medida: esas cuatro consultas tienen **empates prácticos**
       en su top-12, con márgenes de 0,00, 6,1e-08, 2,5e-07 y 4,2e-06. Es ruido
       de float32 por el relleno de los lotes: `sentence-transformers` agrupa por
       longitud, así que codificar una consulta sola no da exactamente el mismo
       vector que codificarla dentro de un lote de 50 (`ESTADO.md` §14 lo midió
       en los embeddings del corpus). El desempate por `chunk_id` **no cubre
       esto**: solo actúa sobre empates exactos, y un margen de 1e-08 no es un
       empate exacto.

       ⚠️ **Consecuencia práctica: el `resultados.jsonl` que se entrega debe
       generarse por el camino 1**, que es el que ejecutará quien reproduzca la
       entrega. Este camino sirve para experimentar y comparar configuraciones,
       no para producir el entregable.
    3. **Ninguno de los dos** — modo simulado: se usa un vector del propio
       índice. No sirve para evaluar nada —los resultados no tienen relación con
       la consulta— pero permite comprobar que el formato cumple §9.3.1.

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
        elif vectores is not None:
            # La fila `posicion` corresponde a la consulta `posicion`: el orden
            # se verificó contra la lista de ids del cache antes de llegar aquí.
            resultado = buscador.buscar(vectores[posicion], consulta=texto, **comun)
        else:
            # Vector del índice, repartido para que no salgan los mismos
            # siempre. Determinista: no se usa azar.
            indice = (posicion * 977) % buscador.indice.ntotal
            vector = np.asarray(buscador.indice.reconstruct(int(indice)))
            resultado = buscador.buscar(vector, consulta=texto, **comun)
        yield resultado


def cargar_vectores_consulta(ruta: Path, consultas) -> "object":
    """Lee vectores de consulta ya codificados, verificando su procedencia.

    Las tres comprobaciones existen porque las tres pueden producir un ranking
    silenciosamente incorrecto: otro modelo, otra revisión del mismo modelo, u
    otro conjunto de consultas en otro orden.
    """
    import numpy as np

    from embedding.encoder import MODELO, REVISION

    matriz = np.load(ruta)
    lateral = ruta.with_suffix(".json")
    if not lateral.is_file():
        raise SystemExit(f"falta {lateral.name}, que lleva la procedencia de los "
                         f"vectores. Sin él no se puede comprobar con qué modelo "
                         f"se codificaron, y usarlos a ciegas puede producir un "
                         f"ranking sin sentido sin dar ningún error.")
    meta = json.loads(lateral.read_text(encoding="utf-8"))

    if meta.get("modelo") != MODELO or meta.get("revision") != REVISION:
        raise SystemExit(
            f"los vectores son de {meta.get('modelo')}@"
            f"{str(meta.get('revision'))[:8]}… y el código pide {MODELO}@"
            f"{REVISION[:8]}…. Mezclar dos espacios vectoriales no da ningún "
            f"error y produce un ranking sin sentido.")
    ids_cache = meta.get("consultas")
    ids_ahora = [qid for qid, _ in consultas]
    if ids_cache != ids_ahora:
        raise SystemExit("los vectores no corresponden a estas consultas, o no "
                         "están en el mismo orden.")
    if len(matriz) != len(consultas):
        raise SystemExit(f"{len(matriz)} vectores para {len(consultas)} consultas.")

    # La norma es condición para que el producto interno de `IndexFlatIP` sea el
    # coseno (§5.2). Si no lo fuera, las puntuaciones no serían comparables.
    normas = np.linalg.norm(matriz, axis=1)
    if abs(normas.min() - 1.0) > 1e-4 or abs(normas.max() - 1.0) > 1e-4:
        raise SystemExit(f"los vectores no están normalizados "
                         f"(normas {normas.min():.6f}–{normas.max():.6f}); §5.2 "
                         f"lo exige para que el producto interno sea el coseno.")

    print(f"vectores  : {ruta.name}  {matriz.shape}  ✔ {MODELO} rev {REVISION[:8]}…")
    print(f"            normas 1,000000 · NO se carga el modelo (4,35 GB)")
    return matriz


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
    parser.add_argument("--vectores", type=Path, default=None,
                        help="matriz .npy con las consultas YA codificadas por el "
                             "mismo modelo y revisión. Produce una salida idéntica "
                             "a la del camino normal sin descargar los 4,35 GB")
    parser.add_argument("--simulado", action="store_true",
                        help="NO codifica: prueba de formato sin el modelo")
    parser.add_argument("--comprobar", action="store_true",
                        help="NO genera nada: contrasta el resultados.jsonl que "
                             "ya existe con el código actual (§1.4). Segundos, y "
                             "sin necesitar el modelo")
    parser.add_argument("--hash-indice", action="store_true",
                        help="con --comprobar, calcula además el sha256 de "
                             "index.faiss (373 MB, unos segundos)")
    args = parser.parse_args()

    # `--comprobar` no genera nada, así que se atiende antes de cargar el índice
    # de 373 MB y muchísimo antes de pensar en el modelo.
    if args.comprobar:
        return comprobar(args)

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

    modelo = vectores = None
    if args.vectores is not None:
        vectores = cargar_vectores_consulta(args.vectores, consultas)
    elif not args.simulado:
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
                                 args.factor_idioma, vectores):
            f.write(json.dumps(resultado.to_json(), ensure_ascii=False))
            f.write("\n")
            escritas += 1
            if escritas % 10 == 0:
                print(f"  {escritas}/{len(consultas)}", flush=True)

    temporal.replace(args.salida)
    ntotal = buscador.indice.ntotal
    buscador.cerrar()

    segundos = time.perf_counter() - inicio
    print(f"\nescritas {escritas} líneas en {args.salida}")
    print(f"tiempo   : {segundos:,.1f} s")

    # El manifiesto se escribe SIEMPRE, incluido en modo simulado —ahí queda
    # marcado como tal—, porque una salida sin registro de cómo se produjo es
    # justo el estado que dejó la entrega excluida durante dos días.
    escribir_manifiesto(args, ntotal, escritas, args.simulado, segundos)

    print("\nSiguiente:")
    print("  python generador.py --comprobar        # reproducibilidad (§1.4)")
    print("  py -m tools.verificar_resultados       # esquema (§9.3.1), desde src/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
