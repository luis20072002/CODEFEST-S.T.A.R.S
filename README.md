# CODEFEST AD ASTRA 2026 — Etapa 1 · Equipo S.T.A.R.S

Base de conocimiento vectorial construida sobre el corpus de fuentes abiertas provisto por ADL,
y módulo de recuperación que, ante consultas en lenguaje natural, devuelve los documentos y
fragmentos más relevantes.

**Equipo S.T.A.R.S**

| Integrante | Rol |
|---|---|
| **Luis Mendoza** | líder |
| Deiner González | |
| Valeria Berrio | |
| Mark Pastrana | |

---

## 1. Dónde está cada entregable

La entrega se reparte en dos ubicaciones **por una única razón técnica**: `index.faiss`
(372,8 MB) y `metadata.jsonl` (240,1 MB) superan el límite de 100 MB por archivo que impone
GitHub, de modo que no pueden alojarse en el repositorio.

###  Carpeta compartida — la entrega completa, con el árbol de §1.4 íntegro

###  [**Abrir la carpeta compartida con la entrega completa**](https://tecnoutb-my.sharepoint.com/:f:/g/personal/luiangulo_utb_edu_co/IgCeTmlebbmlR4nVOBW2QI9tAUSLjTZHMSBbiFMkykK7zoQ?e=5KsHor)

Contiene la carpeta `entrega/` completa, tal como la especifica §1.4, incluida la base
vectorial. **Es la ubicación de referencia para evaluar.**

<sub>URL directa: `https://tecnoutb-my.sharepoint.com/:f:/g/personal/luiangulo_utb_edu_co/IgCeTmlebbmlR4nVOBW2QI9tAUSLjTZHMSBbiFMkykK7zoQ?e=5KsHor`</sub>

###  Este repositorio

Contiene lo mismo **excepto `base_vectorial/`**, más el código de construcción del índice
(`src/`), que no forma parte de los entregables pero documenta cómo se produjo todo.

> **Para ejecutar `generador.py` desde este repositorio** hay que descargar la carpeta
> `base_vectorial/` de la carpeta compartida y colocarla dentro de `entrega/`. El apartado 3
> lo explica paso a paso.

---

## 2. Conformidad con §1.4 — dónde está cada requisito

| §1.4 exige | Archivo | Estado |
|---|---|---|
| **1. Base vectorial**: índice FAISS y almacén de metadata, en subcarpetas por encoder | `entrega/base_vectorial/encoder_bge-m3/` | ✅ carpeta compartida |
| ├ `index.faiss` serializado con `faiss.write_index()` | `index.faiss` · 372.822.061 B | ✅ |
| ├ `metadata.jsonl`, un objeto JSON por línea, con los campos de la Tabla 1, en el orden de los identificadores internos de FAISS | `metadata.jsonl` · 91.021 líneas | ✅ |
| └ Grafo de conocimiento en `grafo/` como `grafo.graphml` | `grafo/grafo.graphml` · 3.375 nodos · **integrado en la recuperación (§8.5)** | ✅ bonus §7 |
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
  base_vectorial/               ← solo en la carpeta compartida
    encoder_bge-m3/
      index.faiss               IndexFlatIP, 91.021 x 1.024
      metadata.jsonl            Tabla 1 + idioma + faiss_id, 91.021 lineas
    grafo/
      grafo.graphml             bonus §7 · 3.375 nodos, 3.460 aristas
      chunk_index.json          mapa chunk_id→faiss_id que usa §8.5
```

### Tres aclaraciones para evitar malentendidos

1. **`lib/` es la carpeta extra autorizada.** El comité confirmó por escrito que «pueden
   agregar una carpeta extra `lib` que consuma `generador.py`». Contiene únicamente los
   módulos que el generador necesita, de modo que `entrega/` es **autocontenida**: se puede
   copiar sola a cualquier máquina y ejecutar.
2. **`src/` NO es un entregable.** Es el pipeline que construyó el índice, incluido por
   transparencia. `generador.py` no depende de él.
3. **`informe_tecnico.md` NO es el entregable.** El entregable es el `.pdf`; el `.md` es su
   fuente y se incluye para que el documento sea editable y auditable.

---

## 3. Reproducir `resultados.jsonl` — paso a paso

### Paso 1 · Obtener la base vectorial

Descarga `base_vectorial/` de la carpeta compartida y colócala dentro de `entrega/`, de modo
que quede así:

```
entrega/base_vectorial/encoder_bge-m3/index.faiss
entrega/base_vectorial/encoder_bge-m3/metadata.jsonl
entrega/base_vectorial/grafo/grafo.graphml
```

**Comprobación de integridad tras la descarga** (opcional pero recomendable: son archivos
grandes y una transferencia truncada no siempre avisa):

| archivo | bytes | SHA-256 |
|---|---|---|
| `index.faiss` | 372.822.061 | `3b8ec63e3e218e0532e87d8f069045184580737560023f0c5c3119d5456def0a` |
| `metadata.jsonl` | 240.054.195 | 91.021 líneas |
| `resultados.jsonl` | 801.865 | `3617145844f10f81b3b3bb3214d1e240691c3d954b71fc415ba1298dd3fb44ef` |

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

> Puede emitir avisos del tipo `numpy: se generó con 2.0.2 y aquí hay 2.5.1`. **Son
> informativos, no fallos.** Indican que el entorno actual difiere del que produjo la salida.
> La reproducibilidad se verificó como idéntica byte a byte en dos sesiones independientes con
> versiones distintas de biblioteca; el detalle está en la sección 8 del informe técnico.

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
algo falla**, de modo que pueden encadenarse.

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
```

---

## 5. Resumen técnico

| | |
|---|---|
| **Corpus** | 1.826 documentos · F1 459 · F2 479 · F3 888 · 28,9 M palabras · 0 fallos de extracción |
| **Fragmentación** (§3) | cascada híbrida de 4 niveles · 91.021 fragmentos · 0 oraciones cortadas sobre 42.000 cortes de prosa |
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
