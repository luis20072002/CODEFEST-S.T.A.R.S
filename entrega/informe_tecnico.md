# Informe técnico

**CODEFEST AD ASTRA 2026 — Etapa 1: Base de Conocimiento Vectorial**
Equipo S.T.A.R.S · 8 de agosto de 2026

---

## 1. Objeto y alcance

Este documento describe la base de conocimiento vectorial construida sobre el corpus de fuentes
abiertas provisto por ADL y el módulo de recuperación que, ante consultas en lenguaje natural,
devuelve los documentos y fragmentos más relevantes. Cubre los cuatro puntos que exige §1.4 —
estrategia de fragmentación y su justificación, encoder y criterios de elección, tipo de índice
FAISS, y las dependencias necesarias para reproducir los resultados — y documenta el
procedimiento de verificación de cada etapa.

El sistema indexa **la totalidad del corpus de los tres fenómenos**, conforme a §1.3: 1.826
archivos distribuidos en F1 (IA e innovación en entornos militares, 459), F2 (seguridad espacial
y órbita baja, 479) y F3 (dinámicas territoriales en América Latina, 888).

Ninguna etapa emplea modelos generativos. La extracción y la normalización son procesamiento de
texto determinista; la representación vectorial se obtiene de un encoder de la familia BERT; y
la recuperación opera exclusivamente sobre vectores, puntuaciones de similitud y metadata,
conforme a la prohibición de §4.2 y §8.3.

---

## 2. Arquitectura del pipeline

El procesamiento se organiza en seis etapas encadenadas, cada una con una salida persistente en
disco. La persistencia intermedia es una decisión de diseño, no un detalle de implementación:
permite reejecutar cualquier etapa sin repetir las anteriores, cuyo coste no es simétrico.

```
  data_raw/  1.826 archivos
      │
      │ (1) Extracción §2.1 — 6 loaders según formato        ~4 h
      ▼
  documentos.jsonl  1.826 documentos, texto crudo
      │
      │ (2) Normalización §2.2 — 8 pasos + detección de idioma   ~4,5 min
      ▼
  documentos_limpios.jsonl  1.826 documentos, texto limpio
      │
      │ (3) Fragmentación §3 — cascada híbrida de 4 niveles     ~24 s
      ▼
  chunks.jsonl  91.021 fragmentos
      │
      │ (4) Codificación §4 — BAAI/bge-m3, 1024 dimensiones     ~2,8 h GPU
      ▼
  embeddings.npy  matriz (91.021 × 1024) float32
      │
      │ (5) Indexación §5 — IndexFlatIP + almacén de metadata   <1 min
      ▼
  index.faiss + metadata.jsonl
      │
      │ (6) Recuperación §8 y salida §9                          ~8 min CPU
      ▼
  resultados.jsonl  50 líneas
```

**Invariante de orden.** La fila *i* de `embeddings.npy` corresponde a la línea *i* de
`chunks.jsonl` y a la línea *i* de `metadata.jsonl`, que a su vez es el identificador interno
*i* del índice FAISS. Este invariante materializa el requisito de §5.3 y es comprobable: cada
registro de metadata incluye un campo `faiss_id` con su número de línea. Un desalineamiento en
esta cadena no produciría ninguna excepción y devolvería la metadata de un fragmento distinto
al recuperado, por lo que se verifica explícitamente (§9 de este informe).

---

## 3. Extracción del corpus (§2.1)

### 3.1 Composición y enrutamiento

El corpus es heterogéneo: 954 JSON, 759 PDF, 73 PBF (mosaicos vectoriales), 26 CSV, 8 JPG, 4
XLSX, 1 AVIF y 1 TXT. Se implementó un loader por familia de formato, todos con la misma
interfaz (`load(path, entry) -> list[Document]`), y un orquestador que enruta cada archivo según
su extensión real.

El inventario en Excel provisto por ADL es la autoridad sobre qué constituye un documento y
sobre su `doc_id`, que se toma tal cual en lugar de generarse, por trazabilidad. El campo
`formato` se deriva de la extensión del archivo y no de la columna `Tipo` del inventario, que
agrupa `.pbf` y `.avif` bajo «Otro».

**Única excepción a esa autoridad: la verificación por número mágico.** Dos archivos con
extensión `.pdf` comienzan con `<!DOCTYPE html>` y no son PDF. El orquestador comprueba la firma
`%PDF-` y, cuando no está presente, corrige `formato` a `"html"` y enruta el archivo al loader
correspondiente. No se trata de una heurística: un archivo que no lleva su propia firma se
desmiente a sí mismo. El corpus no se modifica en disco en ningún caso, porque §10.2.2 empareja
los documentos por el campo `fuente`, definido en la Tabla 1 como el nombre del archivo original
provisto por ADL.

Para los formatos JSON se emplea **selección explícita de campos** en lugar de un recorrido
genérico del árbol. El corpus contiene seis esquemas distintos, y un recorrido genérico
incorporaría al texto indexado campos como `url`, `authors` o `pdf_links`, que no son contenido
del documento. Se verificó que `body_text` y `body_paragraphs` son idénticos en los 229 archivos
donde ambos aparecen, por lo que nunca se concatenan.

### 3.2 Reconocimiento óptico de caracteres

§2.1 recomienda OCR para imágenes con texto relevante. El OCR clásico no es un modelo
generativo, por lo que su uso no incurre en la prohibición de §4.2. Se aplicaron dos criterios
distintos, ambos decididos con medición:

**OCR de página completa: activado.** Cincuenta y un PDF del corpus son documentos escaneados
sin capa de texto y producían cero caracteres. Cuando un documento completo rinde menos de 200
caracteres extraíbles, cada página se rasteriza y se somete a OCR. Los 51 pasaron de cero a una
mediana de 5.310 palabras, ninguno por debajo de 313.

La resolución se fijó en **150 DPI**, no en 300, midiendo la confianza que reporta el propio
motor sobre once páginas de cuatro documentos en español e inglés:

| Resolución | Confianza media | Palabras | Caracteres acentuados | Tiempo |
|---|---|---|---|---|
| **150 DPI** | **91,95** | 3.343 | **378** | **52,8 s** |
| 200 DPI | 92,35 | 3.368 | 380 | 60,7 s |
| 300 DPI | 91,34 | 3.331 | 343 | 81,6 s |

300 DPI resulta simultáneamente peor y un 55 % más lento: menor confianza y un 9 % menos de
caracteres acentuados. La confianza del motor es mejor criterio que el recuento de palabras,
porque más palabras puede significar más ruido.

**OCR de figuras: desactivado.** Se evaluó sobre una muestra de 178 figuras extraídas de 60 PDF
repartidos por todo el corpus. Solo el 19 % aportaba texto aprovechable, y ese porcentaje estaba
sistemáticamente degradado: la sigla «AI» no aparecía correctamente escrita ni una vez, y cuatro
archivos la contenían como «Al» con ele minúscula — el error clásico de confusión I/l, aplicado
justamente al término central del fenómeno 1. Puesto que la Tabla 1 define `texto` como «texto
original del fragmento, sin modificaciones», indexar transcripciones con esa tasa de error
introduciría en el índice afirmaciones que el documento no contiene. La funcionalidad se
conserva tras un interruptor, no eliminada, para permitir su reevaluación.

Los idiomas de OCR se limitan a `spa+eng`. Se comprobó que los 51 documentos rescatados son 47
en español y 4 en inglés, ninguno en portugués: las fuentes lusófonas del corpus se entregan en
JSON y PBF, no como PDF escaneados.

### 3.3 Resultado

```
documentos cargados : 1.826        fallos : 0
palabras totales    : 28.881.496
por fenómeno        : F1 459 · F2 479 · F3 888
rescatados por OCR  : 51
documentos sin texto: 8
```

La correspondencia es **1:1 entre archivos y documentos**, propiedad relevante para la
evaluación: cada `doc_id` designa un archivo real y cada `fuente` empareja directamente con el
ground truth de §10.2.2. Los 8 documentos sin texto son cinco imágenes fotográficas sin texto
que extraer y tres archivos JSON que son manifiestos del proceso de descarga —contienen campos
como `total_publicaciones` o `scraped_at`— y no contenido documental.

---

## 4. Normalización (§2.2)

Se implementaron ocho pasos deterministas. Cuatro corresponden a lo que §2.2 requiere
—normalización a UTF-8 y forma NFC, eliminación de caracteres de control y espaciado
redundante, supresión de encabezados y pies repetidos, y detección de idioma— y cuatro
adicionales que el corpus hizo necesarios.

**Principio rector: se elimina ruido, nunca información.** Cada regla que suprime contenido va
acompañada de la medición que demuestra que lo suprimido no era contenido del documento. Este
criterio descartó, entre otras, la idea de indexar únicamente las columnas textuales de los
archivos tabulares, que habría eliminado identificadores capaces de discriminar un registro.

| Paso adicional | Justificación medida |
|---|---|
| `decode_cid()` | 18 documentos contienen marcadores `(cid:N)` en lugar de caracteres, por fuentes empotradas sin tabla ToUnicode; dos de ellos en más del 99 % del texto. Se busca el desplazamiento constante entre índice de glifo y código de carácter y se acepta únicamente si produce palabras funcionales reales. Un documento pasó de 473 palabras ilegibles a 1.556 legibles, y otro de 22.896 a 52.269. |
| `strip_control_chars()` | 21 PDF emiten `\x07` en la posición del espacio y 6 lo emiten dentro de las palabras. Una regla fija falla en uno de los dos grupos, por lo que se evalúan ambas variantes por documento y se conserva la que produce más palabras plausibles. La función es pura, de modo que el comportamiento sigue siendo determinista. |
| `dehyphenate()` | Reunifica palabras partidas por guion de fin de línea. |
| `strip_anonymous_columns()` | Cuatro archivos CSV incluyen una columna sin encabezado con 231.830 líneas. Se verificó que forman secuencias `+1` con reinicios a cero cada 10.000 registros, es decir el índice del exporte y no un dato. Se elimina únicamente cuando el valor es un número sin texto. |

El resultado sobre los 1.826 documentos es una reducción de 8.245.459 caracteres (3,85 %), sin
que ningún documento con texto quedara vacío.

**Detección de idioma.** Se emplea un clasificador estadístico basado en n-gramas de bytes, no
un modelo generativo, con licencia BSD y salida determinista — condición que descarta
alternativas que muestrean aleatoriamente y varían entre ejecuciones. No se restringe la lista
de idiomas candidatos, porque el corpus contiene documentos legítimos en árabe, ruso, chino,
coreano, japonés y francés; restringirla haría que el campo mintiera precisamente donde más se
notaría. Distribución resultante:

| Idioma | Documentos | Porcentaje |
|---|---|---|
| Inglés | 1.004 | 55,0 % |
| Español | 641 | 35,1 % |
| Portugués | 108 | 5,9 % |
| Otros (fr, zh, ar, ru, ja, ko, de) | 55 | 3,0 % |
| Sin señal suficiente | 18 | 1,0 % |

El orden de las etapas es significativo: la limpieza precede a la detección. En sentido inverso,
la marca de agua de un documento —que ocupa el 69,3 % de su texto extraído— provocaba que un
documento en inglés se clasificara como ruso.

---

## 5. Fragmentación (§3)

### 5.1 Estrategia: cascada híbrida de cuatro niveles

§3.2 admite estrategias híbridas siempre que se justifiquen explícitamente. La implementada
recorre cuatro niveles en orden, descendiendo solo cuando el anterior no basta:

1. **Agrupación de bloques.** Los saltos `\n\n` del texto normalizado delimitan párrafos en
   documentos de prosa y filas en documentos tabulares. Se agrupan bloques completos hasta
   agotar el presupuesto.
2. **Retroceso a límite de oración.** Cuando el bloque excede el presupuesto, el corte retrocede
   al final de la última oración completa que cabe. Es la formulación literal de §3.3.
3. **Separadores secundarios.** Punto y coma, dos puntos, puntos guía y saltos de línea simples,
   para unidades que no son oraciones.
4. **Corte por longitud**, como último recurso.

**Justificación de la cascada frente a una estrategia simple.** Se midió, sobre los 1.826
documentos normalizados, cuántos bloques `\n\n` exceden por sí solos el presupuesto:

| | Bloques `\n\n` | No caben en el presupuesto |
|---|---|---|
| Prosa (1.715 documentos) | 57.613 | **17.218 (29,89 %)** |
| Tabular (103 documentos) | 276.925 | 83 (0,03 %) |

Una estrategia «por párrafo» habría producido 17.218 fragmentos por encima del límite del
encoder, que este truncaría sin emitir aviso alguno. El nivel 2 es por tanto necesario, no una
mejora opcional. En la ejecución entregada, los niveles 3 y 4 —los únicos donde la completitud
lingüística y el tope de longitud pueden entrar en conflicto— intervienen en el **1,24 %** de
los fragmentos, y lo hacen sobre celdas de tabla, no sobre prosa.

**Un `\n\n` no es siempre frontera de oración.** Los extractores de PDF emiten un salto de
bloque también en los cambios de columna y de página, de modo que agrupar bloques completos
partía oraciones. Se resuelve uniendo un bloque con el siguiente únicamente cuando el primero
no cierra oración **y** el segundo comienza en minúscula. Las dos condiciones conjuntas
distinguen una oración realmente partida de un encabezado o un pie de figura, que legítimamente
carecen de punto final.

Se evaluó y descartó el uso de un divisor recursivo de biblioteca. Su lista de separadores por
omisión no incluye el punto, por lo que no mantiene las oraciones íntegras e incumple §3.3, que
es requisito obligatorio; sus últimos recursos parten palabras; y cuenta caracteres en lugar de
tokens.

### 5.2 Tratamiento de los documentos tabulares

Los 103 documentos tabulares (CSV, XLSX y PBF) concentran el 52,1 % de las palabras del corpus
desde el 5,6 % de los archivos. §2.1 indica que cada fila «**puede**» tratarse como unidad de
fragmentación independiente. Esa lectura literal produciría 276.925 fragmentos tabulares, el
87,7 % del índice.

La granularidad elegida agrupa filas completas hasta agotar el presupuesto, lo que **reduce los
fragmentos tabulares en un 83,6 % sin perder una sola palabra**. En el índice entregado el
reparto resultante es:

| | Fragmentos | Porcentaje |
|---|---|---|
| Prosa | 45.245 | 49,7 % |
| Tabular | 45.776 | 50,3 % |

No se consideró excluir estos documentos. §1.3 establece que el conjunto provisto para los tres
fenómenos constituye el corpus, y un documento excluido del índice no puede recuperarse en
ninguna circunstancia.

### 5.3 Verificación y resultado

```
fragmentos             : 91.021
documentos con fragmento: 1.818   (los 1.826 menos los 8 sin texto)
documentos con texto y sin fragmentos : 0
```

Cinco pruebas automáticas, todas superadas: **§3.3** — ningún corte parte una oración, sobre más
de 42.000 cortes de prosa examinados; **cobertura** — cada documento se reconstruye carácter a
carácter concatenando sus fragmentos; **Tabla 1** — los ocho campos presentes, `posicion`
consecutiva desde 0 y 91.021 `chunk_id` únicos; **presupuesto**; y **determinismo** — dos
ejecuciones producen idénticos identificadores y textos.

La prueba de §3.3 se define sobre el **corte** y no sobre el fragmento aislado. Preguntar si un
fragmento termina en signo de puntuación produce más de doce mil falsos positivos, porque los
encabezados y pies de figura carecen legítimamente de punto final. El síntoma inequívoco es la
pareja: un fragmento que no cierra seguido de otro que comienza en minúscula.

---

## 6. Codificación semántica (§4)

### 6.1 Modelo seleccionado

**`BAAI/bge-m3`**, revisión `5617a9f61b028005a4858fdac845db406aefb181`, 1.024 dimensiones.

| Criterio de §4.3 | Verificación |
|---|---|
| Familia encoder (§4.2) | Arquitectura XLM-RoBERTa. No es un decoder. |
| Soporte multilingüe | Más de 100 idiomas; español, inglés y portugués nativos. |
| **Alineamiento translingüe** | Entrenado con objetivo explícito translingüe (MIRACL, MKQA). |
| Recuperación densa | Diseñado para recuperación, no para clasificación ni similitud de pares — criterio que descarta modelos orientados a minería de textos paralelos. |
| Licencia | **MIT**, una de las tres que §4.3 prefiere. |
| Eficiencia computacional | Ver §6.2. |

**El criterio determinante es el alineamiento translingüe, no el soporte multilingüe.** El
conjunto de evaluación entregado presenta las cincuenta consultas **en español**, mientras que
el corpus es 55,0 % inglés. La recuperación es por tanto translingüe español→inglés, propiedad
más exigente que comprender varios idiomas por separado. Se prefirió este modelo frente a
alternativas de dimensión equivalente y licencia igualmente permisiva por una razón operativa
adicional: no requiere prefijos de instrucción en la consulta y en el pasaje, cuya omisión en
uno de los dos lados degrada la calidad sin producir ningún error.

Se emplea únicamente la modalidad densa. El modelo genera también representaciones dispersas y
multi-vector, que no son representables en un índice de un vector por fragmento.

### 6.2 Coste y verificación

§4.3 enumera la eficiencia computacional entre los criterios de selección. Medida sobre
fragmentos reales del corpus en CPU sin GPU, la codificación completa con este modelo requeriría
del orden de 363 horas, frente a las 12 horas de un modelo de 384 dimensiones de la misma
familia. Se conservó el modelo de mayor calidad translingüe y la codificación se ejecutó en una
máquina con GPU (2,8 horas, ≈14 fragmentos/s).

Esta decisión no afecta a la reproducibilidad que evalúa §1.4, que consiste en que
`generador.py` reproduzca los resultados **a partir del índice ya construido**. La construcción
del índice no forma parte de lo evaluado.

La matriz resultante se verificó sobre la totalidad de las filas:

```
forma        : (91.021 × 1.024) float32     filas nulas : 0     NaN o Inf : 0
normas       : 1,000000 exacto, desviación máxima 1,19e-07
```

La normalización es condición necesaria para que el producto interno del índice sea el coseno
(§5.2). Como prueba de discriminación del espacio vectorial, el coseno entre 200 vectores
repartidos por todo el índice arroja media 0,485, mínimo 0,232 y máximo 1,000: el espacio
discrimina, y no ha colapsado.

### 6.3 Declaración del límite de tokens

§4.3 indica que los modelos encoder tienen un límite de tokens de entrada, «comúnmente 512», y
que los fragmentos deben diseñarse para no superar **dicho límite**. El límite del modelo
empleado es de **8.192 tokens**, valor comprobado en tiempo de ejecución sobre la longitud
máxima de secuencia efectiva que el modelo declara.

La distribución real de `num_tokens` en el índice entregado, medida sobre los 91.021 fragmentos
con el tokenizador del propio modelo, es:

| Mediana | Percentil 95 | Máximo | Por encima de 512 | Por encima de 8.192 |
|---|---|---|---|---|
| 692 | 930 | 6.867 | 68,24 % | **0** |

**Ningún fragmento se trunca.** El máximo real se sitúa un 16 % por debajo del límite efectivo.
Se declara explícitamente esta cifra porque el valor de 512 tokens que §4.3 cita como habitual
no es el aplicable a este encoder, y porque la relación entre palabras y tokens en este corpus
—2,145 tokens por palabra— es sensiblemente superior a la de un corpus de prosa homogénea: los
documentos tabulares, mitad del índice, contienen identificadores y códigos que el tokenizador
fragmenta con densidad muy superior a la del texto corrido.

Se contempló el uso de un segundo encoder con fusión de rankings por CombSUM o CombMNZ, previsto
en §4.4 y §8.4. Se descartó: al fusionarse por fragmento, el mismo `chunk_id` debe existir en
ambos índices y los fragmentos deben respetar el límite del encoder más restrictivo. Con el
68,24 % de los fragmentos por encima de 512 tokens, un segundo encoder con ese tope truncaría
dos tercios del índice sin emitir aviso.

---

## 7. Índice vectorial (§5)

### 7.1 Tipo de índice

**`IndexFlatIP` con vectores normalizados**, serializado con `faiss.write_index()`.

| Decisión | Justificación |
|---|---|
| `IndexFlatIP` | §5.2 lo recomienda para el volumen esperado en el reto. Con 91.021 vectores una consulta se resuelve en milisegundos y el conjunto de evaluación son 50 consultas: sustituir búsqueda exacta por búsqueda aproximada introduciría error en la métrica a cambio de una velocidad innecesaria. |
| Vectores normalizados | Convierte el producto interno en similitud coseno exacta. |
| Sin `IndexIDMap` | Los identificadores internos, asignados por orden de inserción, ya coinciden con los números de línea del almacén de metadata. La indirección resolvería un problema inexistente. |

Dimensiones finales: **91.021 vectores × 1.024 dimensiones**, 372.822.061 bytes.

### 7.2 Almacén de metadata

`metadata.jsonl` contiene una línea JSON por fragmento, con los ocho campos obligatorios de la
Tabla 1 nombrados en español (`doc_id`, `chunk_id`, `fuente`, `formato`, `fenomeno`, `posicion`,
`num_tokens`, `texto`) y dos campos adicionales de los que §3.4 autoriza:

- **`idioma`**, porque §8.7 contempla el filtrado por este campo y el módulo de recuperación lo
  utiliza (§8.3 de este informe).
- **`faiss_id`**, el número de línea, que convierte el invariante de §5.3 en una propiedad
  comprobable en lugar de una suposición.

Se descartaron otros campos candidatos. Cada campo adicional se paga 91.021 veces, y ninguna
etapa del sistema consumía el título del documento ni los diagnósticos internos del fragmentador.

El campo `formato` registra la extensión real del archivo. La Tabla 1 enumera «pdf, html o md»,
pero el 58 % del corpus no pertenece a ninguno de esos tres formatos, y §1.3 y §2.1 describen
explícitamente el tratamiento de CSV, XLSX e imágenes. El campo `fuente` registra la ruta
relativa completa del archivo original, y no solo su nombre, porque el corpus contiene 59
nombres de archivo repetidos que afectan a 186 documentos; §10.2.2 empareja los documentos por
este campo, de modo que un nombre ambiguo introduciría un error de emparejamiento.

`metadata.jsonl` ocupa 240.054.195 bytes en 91.021 líneas.

---

## 8. Recuperación (§8) y construcción de la salida (§9)

El módulo de recuperación transforma una consulta en tres documentos y diez fragmentos mediante
cinco pasos: codificación de la consulta, búsqueda en el índice, agregación a nivel documento,
filtrado, y división al límite de palabras de la salida. Toda la operación se realiza sobre
vectores, puntuaciones y metadata.

### 8.1 Búsqueda y agregación

**Se recuperan 200 candidatos** y no diez. Entre la búsqueda y la respuesta operan tres filtros
que descartan candidatos, y con un índice plano el coste de solicitar más es marginal, porque el
trabajo consiste en recorrer los 91.021 vectores en cualquier caso.

**La relevancia de un documento se agrega por máximo** de las puntuaciones de sus fragmentos.
§8.6 admite máximo, suma o media. La suma es inadecuada en este corpus: un único archivo tabular
aporta 33.396 fragmentos frente a una mediana de 5 por documento, de modo que sumar premiaría el
tamaño y los tres puestos de la respuesta los ocuparían invariablemente los mismos archivos.

Cuando los candidatos recuperados no contienen al menos tres documentos distintos, la búsqueda
amplía progresivamente el número de candidatos hasta alcanzarlos. §9.3.1 penaliza o descarta los
objetos cuyos arrays no midan exactamente 3 y 10 elementos, por lo que cumplir esa condición
prevalece sobre cualquier preferencia de diversidad.

**El almacén de metadata se consulta por desplazamiento de bytes**, manteniendo en memoria un
índice de 91.021 posiciones (~700 KB) en lugar de los 229 MB del archivo completo. Quien
reproduzca los resultados no necesita medio gigabyte de memoria para leer diez fragmentos.

### 8.2 Filtros aplicados

**Tope de dos fragmentos por documento.** Sin él, una consulta puede consumir siete de sus diez
posiciones con fragmentos de un mismo documento. El tope es una preferencia: si tras aplicarlo
no se alcanzan diez fragmentos, se realiza una segunda pasada sin restricción, porque cumplir
§9.3.1 es obligatorio.

**Deduplicación por contenido.** El índice contiene 91.021 fragmentos pero 87.424 textos únicos:
3.597 fragmentos (4,0 %) repiten contenido, en dos patrones distintos —filas tabulares idénticas
repartidas entre documentos, y secciones repetidas dentro de un mismo informe—. Puesto que
§10.2.1 evalúa la relevancia sobre el campo `text`, dos fragmentos de contenido idéntico
ocuparían dos posiciones con la misma información. La clave de comparación normaliza únicamente
el espaciado, verificado: colapsa exactamente los tres pares que el tokenizador ya trataba como
equivalentes, y ninguno más. No se normalizan mayúsculas ni puntuación, porque dos textos que
difieren en ellas difieren realmente.

**Bonificación por fenómeno: factor 1,03.** La proporción de documentos devueltos que pertenecen
al fenómeno de la consulta era del 80,7 %, con el fenómeno 1 en el 60,4 %. La causa está
identificada: las etiquetas de fenómeno describen el observatorio de origen y no el contenido, y
existen fuentes catalogadas en un fenómeno que publican abundantemente sobre el tema de otro. Se
descartó el filtrado duro que §8.7 permitiría, porque su fallo es irreversible: un documento
relevante de otro fenómeno quedaría inalcanzable. La bonificación multiplicativa desplaza sin
cerrar la puerta. El factor se determinó comparando 1,00, 1,03, 1,05 y 1,10 mediante *pooling*
—metodología TREC— sobre juicios de relevancia elaborados manualmente, puntuados con las
métricas de §10. El factor 1,10 no deja sobrevivir ningún documento de otro fenómeno, es decir
equivale a un filtro. Resultado aplicado: la correspondencia asciende al 91,3 %.

**Factor por idioma: 0,97 para idiomas distintos de español e inglés.** Varias organizaciones del
corpus publican el mismo informe en varios idiomas, que constituyen documentos distintos con
`doc_id` distintos. Una consulta podía dedicar posiciones a versiones traducidas del mismo
contenido, y en un caso ocupar dos de sus tres puestos de documento con dos traducciones del
mismo resumen ejecutivo. El factor se determinó por barrido:

| Factor | Fragmentos fuera de {es, en} | Conjuntos de documentos alterados |
|---|---|---|
| 1,00 | 16 de 500, en 8 consultas | — |
| 0,99 | 11 de 500, en 8 consultas | 2 de 50 |
| **0,97** | **5 de 500, en 4 consultas** | **5 de 50** |
| 0,95 y menores | **0** | 5 de 50 |

Los factores de 0,95 y menores no dejan sobrevivir ningún fragmento en otro idioma y constituyen
por tanto un filtro. El factor 0,99 altera el conjunto de documentos en solo dos consultas,
porque en el resto únicamente reordena los mismos tres — y §10.2.2 establece que F1@3 es una
métrica de conjunto que no considera el orden. El valor 0,97 corresponde a una banda de
tolerancia del 3 % en similitud coseno: solo se prefiere la versión en español o inglés cuando
ambas son semánticamente casi equivalentes. Los documentos sin idioma detectado **no se
penalizan**, porque la ausencia de etiqueta indica falta de señal del clasificador y no un
idioma distinto.

### 8.3 Límite de 250 palabras (§9.2.1)

El 90,6 % de los fragmentos del índice excede las 250 palabras, de modo que la división es el
caso normal y no la excepción. **El límite se aplica al construir la salida, no al índice**: los
fragmentos extensos producen vectores con más contexto, y se dividen únicamente al reportarse.

La división respeta la misma completitud lingüística de §3.3, reutilizando el divisor de
oraciones del fragmentador para evitar dos implementaciones divergentes de la misma regla.
Verificado sobre el índice completo: 175.025 subfragmentos, mediana de 169 palabras, máximo 250,
ninguno por encima del límite y ningún fragmento que pierda texto.

Se reporta **un subfragmento por fragmento**, no todos. §9.2.1 prohíbe situar dos en una misma
posición del ranking, pero no obliga a reportarlos todos; reportándolos, las diez posiciones
procederían de unos cinco fragmentos distintos y las mitades sin contenido relevante ocuparían
posiciones altas, que es precisamente lo que penaliza NDCG@10. La selección del subfragmento se
realiza por **solapamiento léxico** con la consulta y no por similitud vectorial: codificar los
subfragmentos en tiempo de consulta trasladaría un coste de horas a quien reproduzca los
resultados, mientras que el solapamiento es determinista, cuesta microsegundos y no constituye un
modelo.

### 8.4 Formato de salida

`resultados.jsonl` contiene 50 líneas en el orden `q001`–`q050`, con exactamente tres documentos
y diez fragmentos por consulta, cada uno con los campos de la Tabla 2. Seis pruebas automáticas
verifican estructura, tamaños de los arrays, presencia y no vacuidad de los campos, consecutividad
de las posiciones, límite de 250 palabras, y **trazabilidad**: que todo `doc_id` y todo `chunk_id`
de la salida exista en `metadata.jsonl`. Esta última condición es necesaria porque §10.2.2
empareja los documentos por el campo `fuente`, y la única vía desde `doc_id` hasta `fuente` es el
almacén de metadata.

---

## 9. Verificación y reproducibilidad (§1.4)

### 9.1 Estrategia de verificación

Cada etapa dispone de un verificador independiente que devuelve código de salida distinto de cero
ante cualquier fallo, de modo que puede encadenarse. En total, 33 comprobaciones automáticas:

| Etapa | Comprobaciones | Estado |
|---|---|---|
| Normalización (§2.2) | 5 | Superadas |
| Fragmentación (§3) | 5 | Superadas |
| Índice (§5) | 5 | Superadas |
| Salida (§9.3.1, §10.2.2) | 6 | Superadas |
| Métricas (§10) | 12 | Superadas |

Un criterio metodológico gobierna estas pruebas: **una verificación que toma su entrada de la
misma fuente que valida no verifica nada**. La comprobación del invariante de §5.3 no interroga
al índice sobre su propio contenido, sino que parte de la matriz de embeddings que produjo el
encoder, y confirma que la fila *i* de ese archivo se corresponde con la posición *i* del índice.

Las métricas de §10 se implementaron para comparar decisiones de diseño entre sí. Dos detalles de
las fórmulas se respetan explícitamente: la ganancia del DCG de la fórmula (8) es **lineal**,
`rᵢ / log₂(i+1)`, y no exponencial —diferencia que altera el resultado en cuanto la relevancia es
graduada—; y el denominador de R@3 en la fórmula (12) se limita a `mín(|D*|, 3)`.

### 9.2 Reproducibilidad

`generador.py` cumple tres condiciones:

1. **No importa ningún loader ni el orquestador.** Arrastrar las dependencias de extracción y OCR
   a la máquina donde se reproduce la entrega añadiría formas de fallo sin ninguna necesidad,
   dado que el resultado de esas etapas está contenido en el índice.
2. **La revisión del modelo está fijada por su identificador de commit**, y no solo su nombre. Sin
   ello podría descargarse una versión distinta del modelo, los vectores de la consulta dejarían
   de residir en el mismo espacio que los del índice y el ranking cambiaría sin producir error.
3. **Los empates de puntuación se resuelven de forma determinista** por `chunk_id`, sin depender
   del orden que devuelva la biblioteca de búsqueda. Con búsqueda exacta, los empates son el único
   punto de variación posible entre ejecuciones.

Cada ejecución registra un manifiesto (`resultados.manifest.json`) con la configuración aplicada,
las versiones de las bibliotecas, el modelo con su revisión, y las sumas SHA-256 del índice y de
la salida. El modo `--comprobar` contrasta ese registro con el código en ejecución y señala
cualquier divergencia, sin necesidad de descargar el modelo ni regenerar la salida.

La prueba de reproducibilidad se ejecutó eliminando el archivo de resultados y regenerándolo desde
cero, obteniéndose un archivo idéntico, confirmado por suma SHA-256 en dos máquinas distintas.

### 9.3 Dependencias

Reproducir los resultados requiere **nueve paquetes de Python**, declarados en
`entrega/requirements.txt`: cuatro importados directamente (`faiss-cpu`, `sentence-transformers`,
`pypdf`, `numpy`) y cinco transitivas fijadas por determinar los valores numéricos de los
embeddings. El conjunto es deliberadamente reducido: cada dependencia adicional es una vía más
para que la instalación falle, y una instalación fallida impide ejecutar el generador.

**Se declara `faiss-cpu` y no `faiss-gpu`**, dado que el entorno de evaluación puede carecer de
GPU y un índice plano no requiere entrenamiento.

⚠️ **Dependencia de sistema, no capturada por los gestores de paquetes de Python: Tesseract OCR
5.4.0** con los idiomas `spa` y `eng`. Es necesaria **únicamente para reconstruir el índice desde
el corpus**, no para reproducir los resultados: el reconocimiento óptico se ejecutó aguas arriba
y su producto está contenido en el índice. La verificación de este hecho es directa: la ejecución
que generó el archivo de resultados entregado se realizó en un entorno sin Tesseract y sin las
bibliotecas de extracción de PDF instaladas. El procedimiento de instalación para Windows y para
Debian/Ubuntu se documenta en la cabecera de `requirements.txt`.

---

## 10. Limitaciones conocidas

Se declaran por transparencia metodológica.

1. **Las métricas de §10 no se han evaluado contra el ground truth**, que no es público durante el
   reto (§10.1). Las comparaciones realizadas para elegir los parámetros de recuperación emplean
   *pooling* sobre juicios de relevancia elaborados manualmente: son válidas para comparar
   configuraciones entre sí, pero sus valores absolutos no son comparables con los de la
   evaluación oficial.
2. **La bonificación por fenómeno presupone** que el ground truth se construyó dentro del fenómeno
   de cada consulta. Si se construyó por tema, el ajuste sería contraproducente. No es verificable
   con la información disponible.
3. **Los documentos tabulares constituyen la mitad del índice** y son, en su mayoría, volcados
   bibliográficos. Su inclusión es obligada por §1.3, y la granularidad elegida reduce su peso en
   el índice sin excluirlos, pero compiten en el ranking con documentos de prosa.
4. **La detección de idioma presenta un error estimado del 1 %**, concentrado en documentos cuya
   extracción es defectuosa. Por este motivo el idioma se emplea como factor de puntuación y nunca
   como filtro.
5. **El componente de grafo de conocimiento (§7) no se implementó.** Es un componente bonus
   opcional.

---

## 11. Estructura de la entrega

```
entrega/
  resultados.jsonl                        50 líneas, una por consulta
  generador.py                            reproduce resultados.jsonl desde el índice
  requirements.txt                        9 paquetes
  informe_tecnico.pdf                     este documento
  base_vectorial/
    encoder_bge-m3/
      index.faiss                         IndexFlatIP, 91.021 × 1.024
      metadata.jsonl                      Tabla 1 + idioma + faiss_id, 91.021 líneas
```

Ejecución:

```
pip install -r requirements.txt
python generador.py
python generador.py --comprobar
```
