"""Contrato del fragmento (`Chunk`) — la unidad que se codifica e indexa.

Un `Document` es un archivo entero; un `Chunk` es uno de los pedazos en que el
Chunker (§3) lo parte. Es **el chunk, y no el documento, lo que se convierte en
un vector** dentro de FAISS y lo que se escribe como una línea del
`metadata.jsonl` de la entrega.

────────────────────────────────────────────────────────────────────────────
POR QUÉ ESTA CLASE TIENE EXACTAMENTE ESTOS CAMPOS

No son una elección de diseño nuestra: son **los ocho campos obligatorios de la
Tabla 1** del PDF (§3.4), más tres extras que §3.4 autoriza explícitamente
(«Los equipos pueden añadir campos adicionales (idioma, fecha de publicación,
título del documento, etc.) siempre que los campos obligatorios de la Tabla 1
estén presentes»).

La propiedad que hay que preservar, y que es la razón de ser de este archivo:

    **Los ocho campos de la Tabla 1 se construyen desde un `Chunk` sin volver
    a mirar el `Document` del que salió.**

Si faltara uno —digamos `source`— habría que reabrir `documentos_limpios.jsonl`
y cruzar por `doc_id` al momento de serializar. Eso no solo es lento: es la
clase de paso extra que un día se olvida y produce un `metadata.jsonl` con
`fuente` vacía. Y `fuente` es justo el campo por el que §10.2.2 empareja los
documentos con el *ground truth*: sin él, un documento acertado **no puntúa**.

────────────────────────────────────────────────────────────────────────────
NOMBRES EN INGLÉS AQUÍ, EN ESPAÑOL AL SERIALIZAR

Igual que `Document` y `CatalogEntry`, los identificadores van en inglés. La
Tabla 1 exige los nombres en español, y esa traducción se hace **una sola vez**,
en `to_metadata_record()` (más abajo), que es el único sitio del proyecto donde
aparecen las claves en español:

    source → fuente          format     → formato
    text   → texto           phenomenon → fenomeno

`position` y `num_tokens` no se traducen porque la Tabla 1 ya los nombra
`posicion` y `num_tokens`; el único ajuste es la tilde de `posicion`.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from core.document import Document

# Correspondencia inglés → Tabla 1. Vive aquí, junto a la clase que la aplica,
# para que no haya dos listas de campos que se puedan desincronizar.
# `indexing/metadata.py` debe usar `to_metadata_record()`, no rehacer el dict.
TABLA1_FIELDS = {
    "doc_id": "doc_id",
    "chunk_id": "chunk_id",
    "source": "fuente",
    "format": "formato",
    "phenomenon": "fenomeno",
    "position": "posicion",
    "num_tokens": "num_tokens",
    "text": "texto",
}


@dataclass
class Chunk:
    """Un fragmento de documento, listo para codificar e indexar."""

    # ── Los ocho de la Tabla 1 ───────────────────────────────────────────
    doc_id: str                        # documento de origen
    chunk_id: str                      # único DENTRO del documento (ver make_chunk_id)
    source: str                        # ruta/nombre del archivo original de ADL
    format: str                        # extensión real: "pdf", "json", "csv", …
    phenomenon: int                    # 1, 2 o 3
    position: int                      # ordinal dentro del documento, EMPIEZA EN 0
    num_tokens: int                    # tokens según el tokenizador del encoder
    text: str                          # texto del fragmento, sin modificaciones

    # ── Extras autorizados por §3.4 ──────────────────────────────────────
    # No van en la Tabla 1 pero sí sirven: `language` habilita el post-filtro
    # por idioma que §8.3 permite explícitamente, y `title` da contexto al
    # revisar resultados a mano.
    language: Optional[str] = None
    title: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Valida lo que, si sale mal, no se nota hasta que ya es tarde.

        Cada comprobación corresponde a un requisito del PDF que de otro modo
        fallaría en silencio: un `position` que empiece en 1, un `num_tokens`
        sin calcular o un fragmento vacío no lanzan ningún error por su cuenta
        — simplemente producen un `metadata.jsonl` que incumple la Tabla 1 o un
        vector que no recupera nada.
        """
        if not self.doc_id:
            raise ValueError("doc_id no puede estar vacío")
        if not self.chunk_id:
            raise ValueError("chunk_id no puede estar vacío")
        if not self.source:
            raise ValueError("source no puede estar vacío (§10.2.2 empareja por él)")
        if self.phenomenon not in (1, 2, 3):
            raise ValueError(f"phenomenon debe ser 1, 2 o 3 (recibido: {self.phenomenon!r})")
        if self.position < 0:
            raise ValueError(f"position empieza en 0 (§3.4); recibido: {self.position}")
        if self.num_tokens <= 0:
            raise ValueError(f"num_tokens debe ser positivo; recibido: {self.num_tokens}")
        if not self.text.strip():
            raise ValueError(f"{self.chunk_id}: un fragmento sin texto no recupera nada")

    # ── Construcción ─────────────────────────────────────────────────────

    @staticmethod
    def make_chunk_id(doc_id: str, position: int) -> str:
        """Construye el `chunk_id` a partir del documento y la posición.

        Formato: `F2-ESA-028#0007`.

        Tres decisiones metidas en una línea, y las tres importan:

        - **Se deriva de `doc_id` + `position`**, no de un contador global ni de
          un UUID. Así es **determinista**: dos corridas del chunker sobre el
          mismo corpus producen los mismos identificadores. §1.4 excluye la
          entrega si `generador.py` no reproduce los resultados, y un UUID
          rompería eso de la forma más silenciosa posible.
        - **Lleva el `doc_id` dentro**, así que es único en todo el corpus y no
          solo dentro del documento — que es lo que hace falta cuando los chunks
          de los 1826 documentos acaban mezclados en un único índice FAISS.
        - **La posición va rellenada a 4 dígitos** (`#0007`) para que el orden
          alfabético coincida con el numérico. Sin eso, `#10` va antes que `#2`
          y cualquier desempate por `chunk_id` —que es justo el que recomienda
          `ESTADO.md` §8 para que el ranking sea reproducible— barajaría los
          fragmentos de un mismo documento.

        El `:04d` desborda con gracia: un documento con 12.000 fragmentos da
        `#12000` (5 dígitos), que sigue siendo único y sigue ordenando bien
        entre sus iguales.
        """
        return f"{doc_id}#{position:04d}"

    @classmethod
    def from_document(
        cls,
        document: Document,
        *,
        text: str,
        position: int,
        num_tokens: int,
        **extra: Any,
    ) -> "Chunk":
        """Crea un `Chunk` heredando del `Document` todo lo que no cambia.

        Es la forma en que el Chunker debería crear los fragmentos, y no
        llamando al constructor a mano. El motivo es concreto: `source`,
        `format` y `phenomenon` son idénticos para todos los fragmentos de un
        documento, así que copiarlos a mano en el chunker es repetir código y
        abrir la puerta a que alguno se quede sin poner. Aquí se copian una vez
        y no hay forma de olvidarlos.

        Los argumentos van **obligatoriamente por nombre** (el `*` los fuerza)
        porque `text`, `position` y `num_tokens` son fáciles de confundir entre
        sí al leer una llamada — `Chunk.from_document(doc, t, 0, 51)` no dice
        nada, `position=0, num_tokens=51` sí.

        `**extra` va a `metadata`, para que el chunker pueda anotar lo que
        necesite (la estrategia con la que se cortó, si el chunk viene de una
        tabla, etc.) sin cambiar esta firma.
        """
        if document.phenomenon is None:
            raise ValueError(
                f"{document.doc_id}: phenomenon es None y la Tabla 1 lo exige. "
                "Debería venir del catálogo (core/catalog.py)."
            )

        return cls(
            doc_id=document.doc_id,
            chunk_id=cls.make_chunk_id(document.doc_id, position),
            source=document.source,
            format=document.format,
            phenomenon=document.phenomenon,
            position=position,
            num_tokens=num_tokens,
            text=text,
            language=document.language,
            title=document.title,
            metadata=dict(extra),      # copia: no se comparte con el llamante
        )

    # ── Serialización ────────────────────────────────────────────────────

    def to_metadata_record(self, *, extras: bool = True) -> Dict[str, Any]:
        """Devuelve el dict con las claves de la Tabla 1, en español.

        ⚠️ **Este es el único punto del proyecto donde se traduce al español.**
        Si `indexing/metadata.py` arma el dict por su cuenta, acabará habiendo
        dos listas de campos y un día se desincronizarán.

        `extras=False` deja **solo** los ocho obligatorios. Sirve para
        comprobar que el conjunto mínimo está completo sin que los campos
        opcionales estorben en la revisión.
        """
        record = {clave_es: getattr(self, campo_en)
                  for campo_en, clave_es in TABLA1_FIELDS.items()}
        if extras:
            # Los opcionales van después de los obligatorios y solo si tienen
            # valor: una clave `idioma: null` en 200.000 líneas es peso muerto.
            if self.language:
                record["idioma"] = self.language
            if self.title:
                record["titulo"] = self.title
            if self.metadata:
                record.update(self.metadata)
        return record

    @property
    def word_count(self) -> int:
        """Palabras del fragmento, contadas como las cuenta §9.2.1.

        No es lo mismo que `num_tokens`. El PDF usa **dos unidades distintas en
        dos sitios distintos** y confundirlas es un error caro:

        - `num_tokens` (Tabla 1) son tokens del **tokenizador del encoder**, y
          es lo que limita el chunking a 512 (§4.3).
        - **Palabras** es lo que limita los fragmentos de `resultados.jsonl` a
          250 (§9.2.1), y ahí un fragmento que se pase **se penaliza o se
          descarta** (§9.3.1).

        Un chunk de 512 tokens puede pasar de 250 palabras o no, según el
        idioma. Por eso la Fase 6 tiene que volver a medir en palabras y partir
        lo que haga falta, aunque el chunk fuera válido para el encoder.
        """
        return len(self.text.split())

    def __repr__(self) -> str:
        preview = (self.text[:57] + "…") if len(self.text) > 60 else self.text
        return (f"Chunk(chunk_id={self.chunk_id!r}, pos={self.position}, "
                f"num_tokens={self.num_tokens}, text={preview!r})")


if __name__ == "__main__":
    # Autoprueba del contrato: `py -m core.chunk` desde src/.
    # Comprueba lo único que de verdad importa aquí — que los ocho campos de la
    # Tabla 1 salgan de un Chunk sin tocar el Document.
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    documento = Document(
        doc_id="F2-ESA-028",
        source="F2_Seguridad_Espacial/ESA/ESA_space-environment-report.pdf",
        format="pdf",
        text="La órbita baja terrestre concentra la mayor parte de los objetos "
             "catalogados. Su densidad crece cada año.",
        phenomenon=2,
        title="ESA Space Environment Report",
        language="es",
    )

    fragmento = Chunk.from_document(
        documento,
        text=documento.text,
        position=0,
        num_tokens=27,
        estrategia="parrafo",
    )

    print(fragmento, "\n")

    registro = fragmento.to_metadata_record(extras=False)
    obligatorios = set(TABLA1_FIELDS.values())
    faltan = obligatorios - registro.keys()

    print("registro Tabla 1 (solo obligatorios):")
    for clave, valor in registro.items():
        texto = valor if not isinstance(valor, str) or len(valor) <= 46 else valor[:45] + "…"
        print(f"  {clave:12} = {texto!r}")

    print(f"\ncampos obligatorios presentes: {len(obligatorios) - len(faltan)}/8"
          + (f"  ✖ FALTAN: {faltan}" if faltan else "  ✔"))
    print(f"palabras (§9.2.1): {fragmento.word_count}   tokens (Tabla 1): {fragmento.num_tokens}")
    print("registro completo con extras:", fragmento.to_metadata_record().keys())

    # Las validaciones tienen que saltar. Se prueban las tres que más fácil se
    # cuelan: posición en 1 en vez de 0, num_tokens sin calcular y texto vacío.
    print("\nvalidaciones:")
    for descripcion, kwargs in [
        ("position negativa",   {"position": -1, "num_tokens": 5}),
        ("num_tokens en 0",     {"position": 0, "num_tokens": 0}),
        ("texto vacío",         {"position": 0, "num_tokens": 5, "text": "   "}),
    ]:
        try:
            Chunk.from_document(documento, **{"text": "algo", **kwargs})
        except ValueError as error:
            print(f"  ✔ {descripcion:20} → {error}")
        else:
            print(f"  ✖ {descripcion:20} → NO saltó, revisar __post_init__")
