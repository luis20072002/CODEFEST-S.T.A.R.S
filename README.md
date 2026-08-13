# CODEFEST AD ASTRA 2026 — Etapa 1 · Equipo S.T.A.R.S

Base de conocimiento vectorial construida sobre el corpus de fuentes abiertas provisto por ADL,
y módulo de recuperación que, ante consultas en lenguaje natural, devuelve los documentos y
fragmentos más relevantes.

**La entrega, en cinco líneas:**

- **1.826 documentos** de los **tres fenómenos** — los ocho formatos del corpus, 0 fallos de
  extracción — fragmentados en **91.021 fragmentos** sin cortar una sola oración (§3.3).
- Codificados con **`BAAI/bge-m3`** (licencia MIT, revisión fijada por sha de commit) e
  indexados en **`IndexFlatIP`** con vectores normalizados = coseno exacto (§5.2).
- **50 consultas → `resultados.jsonl`**, 3 documentos y 10 fragmentos por consulta, verificado.
- **Bonus §7 entregado**: grafo de conocimiento de 3.375 nodos, **e integrado en la
  recuperación** según §8.5.
- **Ningún modelo generativo en ninguna etapa**, conforme a §4.2 y §8.3.

**Equipo S.T.A.R.S**

| Integrante | Rol |
|---|---|
| **Luis Mendoza** | líder |
| Deiner González | |
| Valeria Berrio | |
| Mark Pastrana | |

---

## 1. Dónde está cada entregable

> ### 📦 [**ABRIR LA CARPETA COMPARTIDA CON LA ENTREGA COMPLETA**](https://tecnoutb-my.sharepoint.com/:f:/g/personal/luiangulo_utb_edu_co/IgAUdiG1bpAUSonnyEtmQDoeAezPlJtK0FQ4N6p0CaB-y6A?e=kuBXne)
>
> Contiene la carpeta `entrega/` completa, con el árbol de §1.4 íntegro, incluida la base
> vectorial. **Es la ubicación de referencia para evaluar.**

<sub>URL directa: `https://tecnoutb-my.sharepoint.com/:f:/g/personal/luiangulo_utb_edu_co/IgAUdiG1bpAUSonnyEtmQDoeAezPlJtK0FQ4N6p0CaB-y6A?e=kuBXne`</sub>

La entrega se reparte en dos ubicaciones **por una única razón técnica**: `index.faiss`
(372,8 MB) y `metadata.jsonl` (240,1 MB) superan el límite de 100 MB por archivo que impone
GitHub, de modo que no pueden alojarse en el repositorio. **Todo lo demás está en los dos
sitios**, y es idéntico byte a byte.

| | carpeta compartida | este repositorio |
|---|:---:|:---:|
| `resultados.jsonl` · `resultados.manifest.json` | ✅ | ✅ |
| `generador.py` · `requirements.txt` · `lib/` | ✅ | ✅ |
| `informe_tecnico.pdf` (8 páginas) | ✅ | ✅ |
| `base_vectorial/grafo/` — bonus §7 y el mapa de §8.5 | ✅ | ✅ |
| `base_vectorial/encoder_bge-m3/` — **613 MB en dos archivos** | ✅ | ❌ límite de GitHub |
| `src/` — pipeline de construcción, **no es entregable** | ❌ | ✅ |

> **Para ejecutar `generador.py` desde este repositorio** solo falta descargar
> `base_vectorial/encoder_bge-m3/` de la carpeta compartida y colocarla dentro de
> `entrega/base_vectorial/`. El apartado 3 lo explica paso a paso.

---

## 2. Conformidad con §1.4 — dónde está cada requisito

| §1.4 exige | Archivo | Estado |
|---|---|---|
| **1. Base vectorial**: índice FAISS y almacén de metadata, en subcarpetas por encoder | `entrega/base_vectorial/encoder_bge-m3/` | ✅ solo en la carpeta compartida |
| ├ `index.faiss` serializado con `faiss.write_index()` | `index.faiss` · 372.822.061 B | ✅ |
| ├ `metadata.jsonl`, un objeto JSON por línea, con los campos de la Tabla 1, en el orden de los identificadores internos de FAISS | `metadata.jsonl` · 91.021 líneas | ✅ |
| └ Grafo de conocimiento en `grafo/` como `grafo.graphml` | `grafo/grafo.graphml` · 3.375 nodos · **integrado en la recuperación (§8.5)** | ✅ bonus §7 · en ambas ubicaciones |
| **2. Archivo de resultados**, JSON Lines, `resultados.jsonl`, exactamente 50 líneas `q001`–`q050` | `entrega/resultados.jsonl` · 50 líneas | ✅ |
| **3. Documento técnico** en PDF, máximo 8 páginas | `entrega/informe_tecnico.pdf` · **8 páginas** | ✅ |
| **4. Script Python** `generador.py` que use el índice, lea las consultas y genere los resultados | `entrega/generador.py` | ✅ |

**Árbol de la entrega:**

```
entrega/
  resultados.jsonl              50 lineas, una por consulta (§9.3)
  resultados.manifest.json      configuracion, versiones y hashes de la corrida
  generador.py                  reproduce resultados.jsonl desde el indice
  requirements.txt              9 paquetes
  informe_tecnico.pdf           documento tecnico, 8 paginas (§1.4)
  informe_tecnico.md            fuente en Markdown del anterior
  lib/                          modulos que consume generador.py
    retrieval/ chunking/ core/ embedding/ graph/
    data/Extracto_Preguntas_50_v2.pdf     archivo de consultas
  base_vectorial/
    encoder_bge-m3/             ← SOLO en la carpeta compartida (613 MB)
      index.faiss               IndexFlatIP, 91.021 x 1.024
      metadata.jsonl            Tabla 1 + idioma + faiss_id, 91.021 lineas
    grafo/                      ← en ambas ubicaciones
      grafo.graphml             bonus §7 · 3.375 nodos, 3.460 aristas
      chunk_index.json          mapa chunk_id→faiss_id que usa §8.5
```

### Cuatro aclaraciones para evitar malentendidos

1. **`lib/` es la carpeta extra autorizada.** El comité confirmó por escrito que «pueden
   agregar una carpeta extra `lib` que consuma `generador.py`». Contiene únicamente los
   módulos que el generador necesita, de modo que `entrega/` es **autocontenida**: se puede
   copiar sola a cualquier máquina y ejecutar.
2. **`src/` NO es un entregable.** Es el pipeline que construyó el índice, incluido por
   transparencia. `generador.py` no depende de él.
3. **`informe_tecnico.md` NO es el entregable.** El entregable es el `.pdf`; el `.md` es su
   fuente y se incluye para que el documento sea editable y auditable.
4. **`chunk_index.json` no lo pide §1.4, pero hace falta para reproducir.** Es el mapa
   `chunk_id → faiss_id` (3.218 entradas, 90 KB) que consume la integración del grafo de §8.5.
   Sin él el grafo no puede aportar ningún fragmento: `generador.py` lo detecta, lo avisa por
   pantalla y desactiva §8.5 en lugar de fingir que se aplica — pero la salida ya **no
   coincidiría** con la entregada. Va junto a `grafo.graphml`, y está en las dos ubicaciones.

---

## 3. Reproducir `resultados.jsonl` — paso a paso

### Paso 1 · Obtener la base vectorial

**Si trabajas desde la carpeta compartida, sáltate este paso: ya está todo.**

Desde este repositorio faltan los dos archivos grandes. Descarga
`base_vectorial/encoder_bge-m3/` de la carpeta compartida y colócala dentro de
`entrega/base_vectorial/`, de modo que el árbol quede así:

```
entrega/base_vectorial/encoder_bge-m3/index.faiss      ← se descarga
entrega/base_vectorial/encoder_bge-m3/metadata.jsonl   ← se descarga
entrega/base_vectorial/grafo/grafo.graphml             ← ya viene en el repositorio
entrega/base_vectorial/grafo/chunk_index.json          ← ya viene en el repositorio
```

**Comprobación de integridad tras la descarga** (recomendable: son archivos grandes y una
transferencia truncada no siempre avisa):

| archivo | bytes | SHA-256 |
|---|---|---|
| `index.faiss` | 372.822.061 | `3b8ec63e3e218e0532e87d8f069045184580737560023f0c5c3119d5456def0a` |
| `metadata.jsonl` | 240.054.195 | 91.021 líneas |
| `grafo.graphml` | 2.300.287 | `dc93631ffa40…` |
| `chunk_index.json` | 90.841 | `b470e2c1dc34…` |
| `resultados.jsonl` | 801.865 | `3617145844f10f81b3b3bb3214d1e240691c3d954b71fc415ba1298dd3fb44ef` |

No hace falta calcularlos a mano: el **paso 3** contrasta automáticamente contra
`resultados.manifest.json` el SHA-256 del índice, el número de líneas del `metadata.jsonl` y el
SHA-256 de la salida.

### Paso 2 · Instalar las dependencias

```bash
pip install -r entrega/requirements.txt
```

Son **nueve paquetes**. No hace falta Tesseract, ni bibliotecas de lectura de PDF, ni GPU:
todo eso pertenece a la construcción del índice, que ya está hecha.

### Paso 3 · Verificar la reproducibilidad **sin descargar el modelo** (segundos)

```bash
python entrega/generador.py --comprobar --hash-indice
```

Contrasta el `resultados.jsonl` entregado contra el código y el manifiesto: SHA-256 del
archivo de resultados, SHA-256 del índice, configuración de recuperación y modelo con su
revisión. Debe terminar en:

```
VEREDICTO: ✔ `generador.py` reproduce esta salida (§1.4)
```

> Puede emitir avisos del tipo `faiss-cpu: se generó con 1.15.0 y aquí hay 1.14.3`. **Son
> informativos, no fallos.** Indican que el entorno actual difiere del que produjo la salida;
> lo que decide es el contraste de SHA-256 y de configuración, que el propio comando hace.
> La configuración **previa a la integración del grafo** se verificó idéntica byte a byte en dos
> sesiones independientes, con descarga fresca del modelo y versiones distintas de biblioteca.
> El detalle, y el alcance exacto de esa evidencia, están en la sección 8 del informe técnico.

### Paso 4 · Regenerar el archivo desde cero

```bash
python entrega/generador.py
```

Descarga el encoder `BAAI/bge-m3` (4,35 GB) desde HuggingFace **con la revisión fijada por sha
de commit**, codifica las 50 consultas y escribe `resultados.jsonl`. Unos 8 minutos en CPU,
segundos en GPU. El resultado debe coincidir byte a byte con el entregado:

```bash
python entrega/generador.py --comprobar --hash-indice
```

`generador.py` resuelve todas sus rutas a partir de su propia ubicación, de modo que funciona
desde cualquier directorio de trabajo.

---

## 4. Verificación

Todas las etapas tienen verificador propio y **devuelven código de salida distinto de cero si
algo falla**, de modo que pueden encadenarse. Estado en la última ejecución:

| verificador | qué cubre | resultado |
|---|---|---|
| `tools.verificar_resultados` | §9.3.1 y §10.2.2 sobre el entregable | ✅ **6/6** |
| `tools.verificar_grafo` | §7, del GraphML a su interoperabilidad | ✅ **7/7** |
| `generador.py --comprobar --hash-indice` | §1.4, reproducibilidad por hash | ✅ **en verde** |
| `tools.verificar_cleaner` | §2.2, normalización | ✅ 5/5 |
| `evaluation.metrics` | §10, implementación de NDCG@10 y F1@3 | ✅ 12/12 |
| `tools.verificar_chunker` | §3, fragmentación | ✅ 5/5 · requiere `chunks.jsonl` |
| `tools.verificar_indice` | §5, invariantes del índice | ✅ 5/5 · requiere `embeddings.npy` |

**Los tres primeros se ejecutan sobre los entregables**, así que cualquiera que descargue la
entrega puede reproducirlos tal cual. Los cuatro restantes parten de archivos de trabajo del
pipeline —`documentos_limpios.jsonl`, `chunks.jsonl`, `embeddings.npy` y los juicios de
evaluación— que no forman parte de la entrega por tamaño, de modo que su resultado procede de
las sesiones de construcción. No bloquean nada: §1.4 evalúa `generador.py` sobre el índice ya
construido, y eso son exactamente los tres primeros.

Un criterio gobierna estas pruebas: **una verificación que toma su entrada de la misma fuente
que valida no verifica nada.** Por eso la comprobación del invariante de §5.3 no le pregunta al
índice por su propio contenido, sino que parte de la matriz de embeddings que produjo el
encoder; y por eso la prueba de interoperabilidad del GraphML lee el XML en crudo, no el grafo
ya cargado — se añadió justo después de descubrir que un archivo que pasaba las otras seis no
abría en ningún visor.

⚠️ **Lo que estas pruebas NO miden es la calidad de la recuperación.** Eso solo lo determina el
ground truth de §10.1, que no es público. En particular, el efecto del grafo sobre NDCG@10 y
F1@3 **no está medido**; lo que sí se midió es su estabilidad, y está documentado en el informe.

### El que valida el entregable (§9.3.1 y §10.2.2)

```bash
cd src
python -m tools.verificar_resultados
```

Seis comprobaciones sobre `resultados.jsonl`: 50 líneas y JSON válido, arrays de exactamente
3 documentos y 10 fragmentos, todos los campos de la Tabla 2 presentes y no vacíos, `rank`
consecutivos desde 1, ningún fragmento por encima de 250 palabras, y **trazabilidad**: que
todo `doc_id` y todo `chunk_id` de la salida exista en `metadata.jsonl`.

### El del grafo (§7)

```bash
python -m tools.verificar_grafo
```

Siete comprobaciones: estructura dirigida y tipada, trazabilidad de cada arista hasta su
`chunk_id`, canonicalización, integridad al releerse, cobertura por fenómeno, ausencia de
dependencias generativas e interoperabilidad del GraphML.

### Los demás

```bash
python -m tools.verificar_cleaner    # §2.2, 5 comprobaciones
python -m tools.verificar_chunker    # §3,   5
python -m tools.verificar_indice     # §5,   5   (requiere archivos de trabajo internos)
python -m evaluation.metrics         # §10,  12  (NDCG@10 y F1@3)
python -m tools.informe_a_pdf        # regenera el informe y cuenta las paginas reales
python -m tools.construir_chunk_index # regenera chunk_index.json desde metadata.jsonl
```

> El último no hace falta para evaluar —`chunk_index.json` ya va en la entrega— pero demuestra
> que ese archivo **se deriva del `metadata.jsonl` entregado** y no es un dato externo: sale de
> leer su campo `faiss_id`, sin modelo y sin GPU.

---

## 5. Resumen técnico

| | |
|---|---|
| **Corpus** | 1.826 documentos · F1 459 · F2 479 · F3 888 · 28,9 M palabras · 0 fallos de extracción |
| **Fragmentación** (§3) | cascada híbrida de 4 niveles · 91.021 fragmentos · 0 oraciones cortadas sobre 42.000 cortes de prosa |
| **Reto lingüístico** | las 50 consultas están **todas en español** y el corpus es **55% inglés**: la recuperación es cross-lingual (es→en), y ese fue el criterio que decidió el encoder |
| **Encoder** (§4) | `BAAI/bge-m3` · revisión `5617a9f61b028005a4858fdac845db406aefb181` · licencia MIT · 1.024 dimensiones |
| **Índice** (§5) | `IndexFlatIP` con vectores normalizados, equivalente a coseno exacto |
| **Recuperación** (§8) | 200 candidatos · agregación por max pooling (§8.6) · bonificación por fenómeno y factor por idioma (§8.7) |
| **Grafo** (§7, bonus) | 3.375 nodos · 3.460 aristas · NER con `urchade/gliner_multi-v2.1`, Apache 2.0 |
| **Grafo en la recuperación** (§8.5) | bonificación tope 1,03 · peso de entidad inverso al log del grado · enlazado literal, sin modelo |

**Ninguna etapa emplea modelos generativos.** La extracción y la normalización son
procesamiento determinista de texto; la representación vectorial procede de un encoder de la
familia BERT; y la recuperación opera exclusivamente sobre vectores, puntuaciones de similitud
y metadata, conforme a §4.2 y §8.3.

Las decisiones de diseño, con la medición que sostiene cada una, están en
**`entrega/informe_tecnico.pdf`**.

### Decisiones alineadas con las aclaraciones oficiales del comité

- **`formato` registra la extensión real del archivo, en minúsculas.** Conforme a la
  aclaración de que la enumeración de la Tabla 1 es ilustrativa y no exhaustiva.
- **El emparejamiento a nivel de documento se hace por `doc_id`**, y el `doc_id` empleado es
  el **DOC_ID oficial de `Indice_Datos_Codefest.xlsx`**, no un identificador propio. Se
  verificó que los 1.818 identificadores del almacén de metadata y los 195 que aparecen en el
  archivo de resultados existen todos en ese inventario. El campo `fuente` se conserva como
  metadata, con la ruta relativa completa.
- **Los 10 fragmentos y los 3 documentos se entregan siempre**, en las 50 consultas, sin
  aplicar ningún umbral de corte.

---

## 6. Cómo se construyó el índice

No es necesario para reproducir los resultados —§1.4 evalúa `generador.py` sobre el índice ya
construido— pero el pipeline completo está en `src/` y es ejecutable:

```bash
pip install -r requirements.txt      # incluye extraccion, OCR y tabulares
cd src
python -m orchestrator               # extraccion (§2.1), ~4 h
python -m preprocess.runner          # normalizacion (§2.2), ~4,5 min
python -m chunking.chunker --tokens  # fragmentacion (§3)
python -m embedding.encoder --todo   # codificacion (§4), ~2,8 h en GPU T4
python -m indexing.metadata --tokens # almacen de metadata (§5.3)
python -m indexing.faiss_index       # indice FAISS (§5)
```

Requiere además **Tesseract OCR 5.4.0** con los idiomas `spa` y `eng`, dependencia de sistema
que `pip` no instala; las instrucciones para Windows y Debian/Ubuntu están en la cabecera de
`requirements.txt`. **No hace falta para reproducir los resultados**: el OCR se ejecutó aguas
arriba y su producto está contenido en el índice.

### Estructura del repositorio

```
entrega/       los entregables de §1.4
src/           pipeline de construccion (no es entregable)
  core/        contratos de Document y Chunk, catalogo, persistencia
  loaders/     un extractor por familia de formato (§2.1)
  preprocess/  normalizacion y deteccion de idioma (§2.2)
  chunking/    fragmentacion (§3)
  embedding/   encoder (§4)
  indexing/    indice FAISS y almacen de metadata (§5)
  retrieval/   recuperacion y construccion de la salida (§8, §9)
  graph/       grafo de conocimiento (§7, bonus)
  evaluation/  NDCG@10 y F1@3 (§10)
  tools/       verificadores y herramientas de diagnostico
```

---

## 7. Contacto

Ante cualquier duda sobre la entrega, escribir a **Luis Mendoza**, líder del equipo S.T.A.R.S.
