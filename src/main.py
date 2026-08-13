"""Punto de entrada del proyecto: diagnóstico y mapa del pipeline.

    py -m main                 # diagnóstico: ¿está todo para reproducir la entrega?
    py -m main pipeline        # el orden de las etapas, con sus comandos y costes
    py -m main entregables     # el árbol de §1.4 y qué falta

No ejecuta ninguna etapa del pipeline ni duplica su lógica: cada módulo tiene su
propio `__main__` y su propio verificador. Esto responde a otra pregunta, la que
no tenía dueño: **¿esta máquina puede reproducir la entrega, y qué falta?**

────────────────────────────────────────────────────────────────────────────────
POR QUÉ HACE FALTA

Los fallos que han costado tiempo en este proyecto no fueron errores de lógica,
fueron **desajustes silenciosos** entre lo que el código esperaba y lo que la
máquina tenía:

  · `py` no era el Python del venv, así que `openpyxl` no estaba (`ESTADO.md` §6)
  · `py3langid` desapareció del venv y no estaba declarado en `requirements.txt`
  · `faiss-cpu` figuraba como no instalado cuando sí lo estaba, y al revés
  · el `resultados.jsonl` entregado se produjo con una configuración distinta a
    la del código, lo que por §1.4 **excluye la entrega**

Ninguno lanzaba una excepción donde se pudiera ver. Todos se detectan en
segundos si alguien pregunta por ellos, y eso es lo que hace este archivo.

⚠️ **Esto NO es un verificador de calidad.** No dice si la recuperación es
buena: eso solo lo mide el *ground truth*, que según §10.1 no es público.
Dice si la cañería está completa y coherente.
"""

import importlib.util
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DATOS = RAIZ / "src" / "data"
ENTREGA = RAIZ / "entrega"
ENCODER = ENTREGA / "base_vectorial" / "encoder_bge-m3"

# Lo que `entrega/generador.py` necesita de verdad, deducido de los imports de
# toda su cadena. Si falta uno de estos, §1.4 no se puede cumplir.
DEPS_JURADO = [
    ("faiss", "faiss-cpu"),
    ("sentence_transformers", "sentence-transformers"),
    ("pypdf", "pypdf"),
    ("numpy", "numpy"),
]

# Lo que hace falta ADEMÁS para reconstruir el índice desde `data_raw/`. Que
# falten no bloquea la entrega — §1.4 evalúa `generador.py`, no la
# reconstrucción (`ESTADO.md` §8)— pero sí impide re-extraer el corpus.
DEPS_CONSTRUCCION = [
    ("pdfplumber", "pdfplumber"),
    ("fitz", "pymupdf"),
    ("pytesseract", "pytesseract"),
    ("PIL", "pillow"),
    ("openpyxl", "openpyxl"),
    ("mapbox_vector_tile", "mapbox-vector-tile"),
    ("py3langid", "py3langid"),
]

LINEA = "─" * 78


def _instalado(modulo: str) -> bool:
    """¿Se puede importar, sin importarlo?

    `find_spec` no ejecuta el módulo, que es lo que se quiere aquí: importar
    `torch` de verdad tarda segundos y no hace falta para saber si está.
    """
    try:
        return importlib.util.find_spec(modulo) is not None
    except (ImportError, ValueError):
        return False


def _version(paquete: str) -> str:
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version(paquete)
    except PackageNotFoundError:
        return "?"


def _lineas(ruta: Path) -> int:
    with open(ruta, "rb") as f:
        return sum(1 for linea in f if linea.strip())


def _mb(ruta: Path) -> str:
    return f"{ruta.stat().st_size / 1e6:,.1f} MB"


def diagnostico() -> int:
    """Comprueba entorno y artefactos. Devuelve 1 si algo bloquea la entrega."""
    fallos, avisos = [], []

    print(LINEA)
    print("ENTORNO")
    print(LINEA)
    print(f"  python      : {sys.version.split()[0]}")
    print(f"  ejecutable  : {sys.executable}")
    # El aviso que más veces ha costado tiempo (`ESTADO.md` §6): `py` a secas no
    # es el intérprete del venv, y entonces faltan la mitad de los paquetes.
    if "venv" not in sys.executable.lower():
        avisos.append("este intérprete NO parece el del venv. Si algo sale como "
                      "no instalado, prueba `venv\\Scripts\\python.exe -m main`")

    print(f"\n  Para `generador.py` (§1.4) — si falta uno, la entrega no corre:")
    for modulo, paquete in DEPS_JURADO:
        ok = _instalado(modulo)
        print(f"    {'✔' if ok else '✘'} {paquete:24s} {_version(paquete)}")
        if not ok:
            fallos.append(f"falta {paquete}: `generador.py` no puede correr")

    print(f"\n  Para reconstruir el índice — no bloquea la entrega:")
    for modulo, paquete in DEPS_CONSTRUCCION:
        ok = _instalado(modulo)
        print(f"    {'✔' if ok else '·'} {paquete:24s} {_version(paquete)}")
        if not ok:
            avisos.append(f"sin {paquete} no se puede reconstruir el índice "
                          f"(pero sí reproducir la entrega)")

    # ── Artefactos ───────────────────────────────────────────────────────────
    print(f"\n{LINEA}")
    print("ARTEFACTOS")
    print(LINEA)

    vectores = metadata_lineas = None

    if ENCODER.joinpath("index.faiss").is_file():
        idx = ENCODER / "index.faiss"
        print(f"  ✔ index.faiss           {_mb(idx)}")
        if _instalado("faiss"):
            import faiss
            vectores = faiss.read_index(str(idx)).ntotal
            print(f"      {vectores:,} vectores")
    else:
        fallos.append("no está `index.faiss`. La base vectorial no viaja en git "
                      "(pesa 373 MB): se entrega por el canal de la competencia")
        print(f"  ✘ index.faiss           NO ESTÁ")

    if ENCODER.joinpath("metadata.jsonl").is_file():
        md = ENCODER / "metadata.jsonl"
        metadata_lineas = _lineas(md)
        print(f"  ✔ metadata.jsonl        {_mb(md)}  ·  {metadata_lineas:,} líneas")
    else:
        fallos.append("no está `metadata.jsonl`: sin él no hay `doc_id` → `fuente` "
                      "y §10.2.2 no puede emparejar ningún documento")
        print(f"  ✘ metadata.jsonl        NO ESTÁ")

    # El invariante de §5.3: una línea de metadata por vector del índice. Si se
    # desalinea, cada consulta devuelve la metadata de otro fragmento **sin
    # lanzar ninguna excepción**.
    if vectores is not None and metadata_lineas is not None:
        if vectores == metadata_lineas:
            print(f"  ✔ §5.3 alineados        {vectores:,} = {metadata_lineas:,}")
        else:
            fallos.append(f"§5.3 ROTO: {vectores:,} vectores contra "
                          f"{metadata_lineas:,} líneas de metadata. Cada consulta "
                          f"devolvería la metadata de otro fragmento, en silencio")

    consultas = DATOS / "Extracto_Preguntas_50_v2.pdf"
    print(f"  {'✔' if consultas.is_file() else '✘'} consultas             "
          f"{consultas.name}")
    if not consultas.is_file():
        fallos.append("no está el PDF de las 50 consultas")

    # ── Los entregables de §1.4 ──────────────────────────────────────────────
    print(f"\n{LINEA}")
    print("ENTREGABLES (§1.4)")
    print(LINEA)
    for nombre, ruta, obligatorio in [
        ("resultados.jsonl", ENTREGA / "resultados.jsonl", True),
        ("generador.py", ENTREGA / "generador.py", True),
        ("informe_tecnico.pdf", ENTREGA / "informe_tecnico.pdf", True),
        ("requirements.txt", ENTREGA / "requirements.txt", False),
    ]:
        if ruta.is_file():
            extra = ""
            if ruta.suffix == ".jsonl":
                n = _lineas(ruta)
                extra = f"  ·  {n} líneas"
                if n != 50:
                    fallos.append(f"§9.3 exige 50 líneas y `{nombre}` tiene {n}")
            print(f"  ✔ {nombre:22s} {extra}")
        else:
            print(f"  {'✘' if obligatorio else '·'} {nombre:22s} NO ESTÁ")
            if obligatorio:
                fallos.append(f"falta el entregable `{nombre}` de §1.4")

    # La comprobación que ninguna otra hace: ¿la salida entregada corresponde a
    # la configuración del código? Se delega en el manifiesto del generador.
    manifiesto = ENTREGA / "resultados.manifest.json"
    if manifiesto.is_file():
        m = json.loads(manifiesto.read_text(encoding="utf-8"))
        cfg = m.get("configuracion", {})
        print(f"\n  manifiesto de la salida ({m.get('generado', '?')}):")
        for k, v in cfg.items():
            print(f"      {k} = {v}")
        if m.get("simulado"):
            fallos.append("el `resultados.jsonl` entregado está marcado como "
                          "SIMULADO en su manifiesto: no responde a las consultas")
        print("\n  Contrástalo con el código:  python entrega/generador.py --comprobar")
    else:
        avisos.append("no hay `resultados.manifest.json`: no se puede saber con "
                      "qué configuración se generó la salida entregada. Regenérala "
                      "o corre `generador.py --comprobar` para el detalle")

    # ── Residuos que no deben viajar con la entrega ──────────────────────────
    basura = [p for p in ENTREGA.rglob("*")
              if p.is_file() and (p.suffix == ".parcial"
                                  or p.name.startswith("resultados_"))]
    if basura:
        print(f"\n  ⚠️ residuos en entrega/ ({len(basura)}):")
        for p in basura:
            print(f"      {p.relative_to(ENTREGA)}  ({_mb(p)})")
        avisos.append("hay archivos de trabajo dentro de `entrega/`. La base "
                      "vectorial se entrega a mano, así que viajarían con ella")

    # ── Veredicto ────────────────────────────────────────────────────────────
    print(f"\n{LINEA}")
    for a in avisos:
        print(f"⚠️  {a}")
    if fallos:
        for f in fallos:
            print(f"✘ {f}")
        print(f"{LINEA}")
        print(f"VEREDICTO: ✘ {len(fallos)} cosa(s) bloquean la entrega")
        return 1
    print(f"{LINEA}")
    print("VEREDICTO: ✔ entorno y artefactos coherentes")
    print("Esto NO dice que la recuperación sea buena: el ground truth no es "
          "público (§10.1).")
    return 0


def pipeline() -> int:
    """El orden de las etapas, con su coste real medido y su verificador."""
    etapas = [
        ("§2.1 Extracción", "py -m orchestrator",
         "~4 h", "data/documentos.jsonl", "1826 docs, 0 fallos"),
        ("§2.2 Preprocesado", "py -m preprocess.runner",
         "~4,5 min", "data/documentos_limpios.jsonl", "py -m tools.verificar_cleaner"),
        ("§3 Chunking", "py -m chunking.chunker",
         "~24 s", "data/chunks.jsonl", "py -m tools.verificar_chunker"),
        ("§4 Codificación", "py -m embedding.encoder --todo",
         "~2,8 h GPU", "data/embeddings.npy", "normas = 1,0"),
        ("§5 Metadata", "py -m indexing.metadata --tokens",
         "minutos", "metadata.jsonl", "Tabla 1 en español"),
        ("§5 Índice", "py -m indexing.faiss_index",
         "<1 min", "index.faiss", "py -m tools.verificar_indice"),
        ("§9 Resultados", "python entrega/generador.py",
         "~8 min CPU", "resultados.jsonl", "py -m tools.verificar_resultados"),
    ]
    print(LINEA)
    print("PIPELINE — cada etapa parte de la salida de la anterior")
    print(LINEA)
    for etapa, comando, coste, salida, verificacion in etapas:
        print(f"\n  {etapa}   ({coste})")
        print(f"    $ {comando}")
        print(f"    → {salida}")
        print(f"    ✓ {verificacion}")
    print(f"\n{LINEA}")
    print("🔴 NO hace falta correr las etapas 1–6: sus salidas ya existen y están")
    print("   verificadas. Re-extraer cuesta 4 h y re-codificar 2,8 h de GPU.")
    print("   Solo se re-corre una etapa si cambia lo que la anterior PRODUCE.")
    print("\n⚠️ `py` no es el Python del venv. Activa el venv o usa")
    print("   `venv\\Scripts\\python.exe -m …`  (ESTADO.md §6)")
    return 0


def entregables() -> int:
    print(LINEA)
    print("ÁRBOL DE §1.4")
    print(LINEA)
    print("""
    entrega/
      resultados.jsonl          50 líneas, una por consulta
      generador.py              reproduce resultados.jsonl desde el índice
      requirements.txt          9 paquetes; el jurado no necesita más
      informe_tecnico.pdf       máx. 8 páginas
      base_vectorial/
        encoder_bge-m3/
          index.faiss           IndexFlatIP, 91.021 × 1024
          metadata.jsonl        Tabla 1 en español + idioma + faiss_id
        grafo/                  bonus §7, opcional — VA DENTRO de base_vectorial/
          grafo.graphml
    """)
    print("El informe debe cubrir (§1.4, y §3.2 lo exige por nombre):")
    print("  · la estrategia de chunking y su justificación")
    print("  · el encoder y los criterios de §4.3 por los que se eligió")
    print("  · el tipo de índice FAISS")
    print("  · las dependencias de sistema (Tesseract) y el límite de tokens")
    print(f"\n{LINEA}")
    print("🔴 Si `generador.py` no reproduce `resultados.jsonl`, la entrega se")
    print("   EXCLUYE de la evaluación (§1.4, textual).")
    print("   Compruébalo con: python entrega/generador.py --comprobar")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    orden = sys.argv[1] if len(sys.argv) > 1 else "diagnostico"
    acciones = {"diagnostico": diagnostico, "pipeline": pipeline,
                "entregables": entregables}
    if orden in ("-h", "--help", "help"):
        print(__doc__)
        raise SystemExit(0)
    if orden not in acciones:
        print(f"orden desconocida: {orden}\nusa: {', '.join(acciones)}")
        raise SystemExit(2)
    raise SystemExit(acciones[orden]())
