"""Comprueba que `entrega/` corra en el Python del jurado (§1.4).

    py -m tools.verificar_python39 [<carpeta>]     # por defecto, `entrega/`

Devuelve 1 si algo fallaría. Encadenable con el resto de verificadores.

────────────────────────────────────────────────────────────────────────────────
POR QUÉ EXISTE

El comité respondió por escrito que «la evaluación se realizará con versiones de
Python >= 3.9.5, que son las recomendadas para FAISS». Y §1.4 es textual: si
`generador.py` no reproduce los resultados, la entrega **se excluye**. Un fallo
de importación en 3.9 no es un bug menor: es la entrega fuera.

🔴 **Ningún otro verificador puede ver esto**, y la razón es la misma lección que
destapó el grafo que pasaba 6/6 y no abría en ningún visor: todas las pruebas de
este proyecto se ejecutan en **el intérprete de esta máquina**, que es 3.13. Una
incompatibilidad con 3.9 es invisible por construcción para cualquier prueba que
consista en ejecutar el código aquí. Hay que mirarla estáticamente.

────────────────────────────────────────────────────────────────────────────────
QUÉ MIRA, Y POR QUÉ CADA COSA

  A. SINTAXIS — `ast.parse(feature_version=(3, 9))`. Atrapa `match`, los gestores
     de contexto entre paréntesis y demás gramática de 3.10+.

  B. ANOTACIONES PEP 604 — `int | None` es sintaxis **válida** en 3.9, así que el
     bloque A no la ve: es un `BinOp` corriente. Pero las anotaciones **se evalúan
     al definir la función**, y `type.__or__` no existe hasta 3.10, de modo que el
     import revienta con `TypeError`. La cura es `from __future__ import
     annotations`, que las deja como cadenas sin evaluar.
     ⚠️ Por eso este bloque no marca el operador, sino el operador **sin esa
     línea**: el mismo código con ella es correcto.

  C. APIs DE 3.10+ — funciones que existen aquí y no allí. Ninguna da error de
     sintaxis y todas fallan en tiempo de ejecución.

Lo que NO puede comprobar: que una dependencia declarada en `requirements.txt`
tenga rueda para 3.9. Eso solo se ve instalando.
"""

import ast
import sys
from pathlib import Path
from typing import List, Tuple

RAIZ = Path(__file__).resolve().parents[2]
ENTREGA = RAIZ / "entrega"

# Nombres que solo existen en 3.10 o posterior. Se buscan como atributo o como
# llamada; es una heurística deliberadamente estrecha, porque un falso positivo
# en un verificador cuesta más que un falso negativo: se deja de mirar.
APIS_NUEVAS = {
    "pairwise": "itertools.pairwise (3.10+)",
    "bit_count": "int.bit_count (3.10+)",
    "anext": "anext() (3.10+)",
    "aiter": "aiter() (3.10+)",
    "TypeAlias": "typing.TypeAlias (3.10+)",
    "ParamSpec": "typing.ParamSpec (3.10+)",
    "TypeGuard": "typing.TypeGuard (3.10+)",
}


def _archivos(carpeta: Path) -> List[Path]:
    return sorted(p for p in carpeta.rglob("*.py"))


def _mostrar(p: Path) -> str:
    """Ruta legible: relativa a la raíz si cuelga de ella, absoluta si no.

    `relative_to()` lanza `ValueError` cuando la ruta no está bajo la raíz, que
    es lo que pasa al apuntar el verificador a una carpeta con `../`.
    """
    try:
        return str(p.resolve().relative_to(RAIZ))
    except ValueError:
        return str(p)


def bloque_a(archivos: List[Path]) -> List[str]:
    """Sintaxis que 3.9 no sabe leer."""
    fallos = []
    for p in archivos:
        try:
            ast.parse(p.read_text(encoding="utf-8"), feature_version=(3, 9))
        except SyntaxError as e:
            fallos.append(f"{_mostrar(p)}:{e.lineno}  {e.msg}")
    return fallos


def _anotaciones(nodo: ast.AST) -> List[ast.expr]:
    if isinstance(nodo, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = nodo.args
        todos = args.posonlyargs + args.args + args.kwonlyargs
        salida = [a.annotation for a in todos if a.annotation]
        if nodo.returns:
            salida.append(nodo.returns)
        return salida
    if isinstance(nodo, ast.AnnAssign) and nodo.annotation:
        return [nodo.annotation]
    return []


def bloque_b(archivos: List[Path]) -> List[str]:
    """`X | Y` en anotaciones sin `from __future__ import annotations`."""
    fallos = []
    for p in archivos:
        arbol = ast.parse(p.read_text(encoding="utf-8"))
        protegido = any(
            isinstance(n, ast.ImportFrom) and n.module == "__future__"
            and any(a.name == "annotations" for a in n.names)
            for n in arbol.body
        )
        if protegido:
            continue
        vistos = set()
        for nodo in ast.walk(arbol):
            for anotacion in _anotaciones(nodo):
                for sub in ast.walk(anotacion):
                    if isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.BitOr):
                        vistos.add((anotacion.lineno, ast.unparse(anotacion)))
        for ln, txt in sorted(vistos):
            fallos.append(f"{_mostrar(p)}:{ln}  anotación `{txt}`")
    return fallos


def bloque_c(archivos: List[Path]) -> List[str]:
    """Funciones y nombres que no existen en 3.9."""
    fallos = []
    for p in archivos:
        arbol = ast.parse(p.read_text(encoding="utf-8"))
        for nodo in ast.walk(arbol):
            nombre = None
            if isinstance(nodo, ast.Attribute):
                nombre = nodo.attr
            elif isinstance(nodo, ast.Name):
                nombre = nodo.id
            if nombre in APIS_NUEVAS:
                fallos.append(
                    f"{_mostrar(p)}:{nodo.lineno}  {APIS_NUEVAS[nombre]}")
    return fallos


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    carpeta = Path(sys.argv[1]) if len(sys.argv) > 1 else ENTREGA
    if not carpeta.is_dir():
        print(f"No existe la carpeta {carpeta}.")
        return 1

    archivos = _archivos(carpeta)
    linea = "─" * 74
    print(linea)
    print(f"carpeta : {carpeta}")
    print(f"archivos: {len(archivos)} .py   ·   objetivo: Python 3.9.5 (el del jurado)")

    bloques: List[Tuple[str, List[str]]] = [
        ("A. SINTAXIS que 3.9 no sabe leer", bloque_a(archivos)),
        ("B. ANOTACIONES `X | Y` sin `from __future__ import annotations`",
         bloque_b(archivos)),
        ("C. APIs que no existen en 3.9", bloque_c(archivos)),
    ]

    total = 0
    for titulo, fallos in bloques:
        print(f"\n{linea}\n{titulo}")
        if not fallos:
            print("   ✔ PASA")
            continue
        total += len(fallos)
        for f in fallos:
            print(f"   ✖ {f}")

    print(f"\n{linea}")
    if total:
        print(f"VEREDICTO: ✖ {total} problema(s). En 3.9 `generador.py` fallaría")
        print("al importar, y §1.4 excluye la entrega que no reproduce.")
        print("El arreglo del bloque B es una línea: `from __future__ import")
        print("annotations` como primera sentencia tras el docstring.")
        print(linea)
        return 1
    print("VEREDICTO: ✔ la entrega debería importar y correr en Python 3.9.5")
    print("⚠️  Estático: no comprueba que las ruedas de pip existan para 3.9.")
    print(linea)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
