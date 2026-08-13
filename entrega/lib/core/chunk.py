"""Contrato del fragmento (`Chunk`) — la unidad que se codifica y se indexa.

Un `Document` es un archivo completo; un `Chunk` es una de las porciones en que
la estrategia de fragmentación (§3) lo divide. Es el fragmento, y no el
documento, lo que se convierte en un vector dentro de FAISS y lo que se escribe
como una línea del `metadata.jsonl` entregado.

────────────────────────────────────────────────────────────────────────────
LOS CAMPOS SON LOS DE LA TABLA 1

Los ocho primeros campos son exactamente los que §3.4 declara obligatorios por
fragmento. Los tres restantes son extras que la propia §3.4 autoriza: «los
equipos pueden añadir campos adicionales (idioma, fecha de publicación, título
del documento, etc.) siempre que los campos obligatorios de la Tabla 1 estén
presentes».

La propiedad que esta clase garantiza, y que es su razón de ser:

    Los ocho campos de la Tabla 1 se construyen a partir de un `Chunk` sin
    volver a consultar el `Document` del que procede.

Si alguno faltara —`source`, por ejemplo— habría que reabrir el corpus
normalizado y cruzar por `doc_id` en el momento de serializar. Ese paso
adicional es el que produce, cuando se omite, un `metadata.jsonl` con el campo
`fuente` vacío.

────────────────────────────────────────────────────────────────────────────
NOMBRES EN INGLÉS AQUÍ, EN ESPAÑOL AL SERIALIZAR

La Tabla 1 exige los nombres de campo en español. Esa traducción se realiza una
sola vez, en `to_metadata_record()`, que es el único punto del sistema donde
aparecen las claves en español:

    source → fuente          format     → formato
    text   → texto           phenomenon → fenomeno

`position` y `num_tokens` no se traducen: la Tabla 1 ya los nombra `posicion` y
`num_tokens`, y el único ajuste es la tilde.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from core.document import Document

# Correspondencia inglés → Tabla 1. Se declara junto a la clase que la aplica
# para que no existan dos listas de campos susceptibles de desincronizarse.
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

    # ── Los ocho campos obligatorios de la Tabla 1 ───────────────────────
    doc_id: str                        # DOC_ID oficial del documento de origen
    chunk_id: str                      # identificador del fragmento (ver make_chunk_id)
    source: str                        # ruta relativa del archivo original de ADL
    format: str                        # extensión real: "pdf", "json", "csv", …
    phenomenon: int                    # 1, 2 o 3
    position: int                      # ordinal dentro del documento, empieza en 0
    num_tokens: int                    # tokens según el tokenizador del encoder
    text: str                          # texto del fragmento, sin modificaciones

    # ── Extras autorizados por §3.4 ──────────────────────────────────────
    # `language` habilita el post-filtro por idioma que §8.7 contempla de forma
    # explícita, y es el único extra que llega al `metadata.jsonl` entregado.
    language: Optional[str] = None
    title: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Valida los requisitos que, de incumplirse, fallarían en silencio.

        Cada comprobación corresponde a una exigencia del PDF que por sí sola no
        produciría ninguna excepción: una `position` que empiece en 1, un
        `num_tokens` sin calcular o un fragmento vacío generan un
        `metadata.jsonl` que incumple la Tabla 1, o un vector que no recupera
        nada, sin ningún aviso.
        """
        if not self.doc_id:
            raise ValueError("doc_id no puede estar vacío")
        if not self.chunk_id:
            raise ValueError("chunk_id no puede estar vacío")
        if not self.source:
            raise ValueError("source no puede estar vacío (Tabla 1)")
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

        Tres propiedades deliberadas:

        - **Se deriva de `doc_id` y `position`**, no de un contador global ni de
          un identificador aleatorio. Es por tanto determinista: dos ejecuciones
          de la fragmentación sobre el mismo corpus producen los mismos
          identificadores, condición necesaria para la reproducibilidad que
          exige §1.4.
        - **Incluye el `doc_id`**, de modo que es único en todo el corpus y no
          solo dentro de su documento, que es lo que se necesita cuando los
          fragmentos de los 1.826 documentos conviven en un único índice.
        - **La posición se rellena a cuatro dígitos**, para que el orden
          alfabético coincida con el numérico. Sin ese relleno `#10` precedería
          a `#2`, y el desempate determinista por `chunk_id` que usa el módulo
          de recuperación alteraría el orden de los fragmentos.

        El formato admite documentos con más de 9.999 fragmentos: la posición
        crece a cinco dígitos y el identificador sigue siendo único.
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
        """Crea un `Chunk` heredando del `Document` los campos que no varían.

        `source`, `format` y `phenomenon` son idénticos para todos los
        fragmentos de un documento. Copiarlos aquí una sola vez elimina la
        posibilidad de que alguno quede sin asignar.

        Los argumentos se pasan obligatoriamente por nombre —el `*` lo fuerza—
        porque `text`, `position` y `num_tokens` son fáciles de confundir entre
        sí en el punto de llamada.

        `**extra` se almacena en `metadata`, de modo que la fragmentación puede
        anotar información de diagnóstico sin modificar esta firma.
        """
        if document.phenomenon is None:
            raise ValueError(
                f"{document.doc_id}: phenomenon es None y la Tabla 1 lo exige."
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
        """Devuelve el registro con las claves de la Tabla 1, en español.

        Este es el único punto del sistema donde se traducen los nombres de
        campo. `extras=False` devuelve exclusivamente los ocho obligatorios, lo
        que permite comprobar que el conjunto mínimo está completo.

        QUÉ CAMPOS ADICIONALES SE ENTREGAN

        §3.4 autoriza campos extra, pero el `metadata.jsonl` tiene 91.021
        líneas: cada campo adicional se paga 91.021 veces. Se conserva
        únicamente `idioma`, porque §8.7 contempla filtrar los resultados por
        ese campo y el módulo de recuperación lo utiliza.

        Se descartaron el título del documento y los diagnósticos internos de la
        fragmentación: ninguna etapa del sistema los consume.
        """
        record = {clave_es: getattr(self, campo_en)
                  for campo_en, clave_es in TABLA1_FIELDS.items()}
        if extras and self.language:
            # Solo si tiene valor. Una clave `idioma: null` repetida 91.021
            # veces es peso muerto; 18 documentos del corpus no ofrecen señal
            # suficiente para determinar su idioma.
            record["idioma"] = self.language
        return record

    @property
    def word_count(self) -> int:
        """Palabras del fragmento, contadas como las cuenta §9.2.1.

        No equivale a `num_tokens`. El PDF emplea dos unidades distintas en dos
        lugares distintos:

        - `num_tokens` (Tabla 1) son tokens del tokenizador del encoder, y es la
          magnitud que limita el tamaño de los fragmentos del índice (§4.3).
        - Las **palabras** son la unidad que limita a 250 los fragmentos de
          `resultados.jsonl` (§9.2.1); un fragmento que se pase «será penalizado
          o descartado» (§9.3.1).

        Un fragmento válido para el encoder puede superar las 250 palabras, y de
        ahí que el módulo de recuperación vuelva a medir en palabras y divida lo
        que haga falta al construir la salida.
        """
        return len(self.text.split())

    def __repr__(self) -> str:
        preview = (self.text[:57] + "…") if len(self.text) > 60 else self.text
        return (f"Chunk(chunk_id={self.chunk_id!r}, pos={self.position}, "
                f"num_tokens={self.num_tokens}, text={preview!r})")
