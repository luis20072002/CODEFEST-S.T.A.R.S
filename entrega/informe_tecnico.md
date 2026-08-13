# Informe técnico

**CODEFEST AD ASTRA 2026 — Etapa 1: Base de Conocimiento Vectorial**
Equipo S.T.A.R.S · 9 de agosto de 2026

---

## 1. Objeto y alcance

Este documento describe la base de conocimiento vectorial construida sobre el corpus de fuentes
abiertas provisto por ADL y el módulo de recuperación que, ante consultas en lenguaje natural,
devuelve los documentos y fragmentos más relevantes. Cubre los puntos exigidos —estrategia de
fragmentación y su justificación, encoder y criterios de elección, tipo de índice FAISS, el
grafo de conocimiento, y las dependencias necesarias para reproducir los resultados— y documenta
el procedimiento de verificación de cada etapa.

El sistema indexa **la totalidad del corpus de los tres fenómenos**, conforme a lo establecido en
las bases: 1.826 archivos distribuidos en F1 (IA e innovación en entornos militares, 459), F2
(seguridad espacial y órbita baja, 479) y F3 (dinámicas territoriales en América Latina, 888).

Ninguna etapa emplea modelos generativos. La extracción y la normalización son procesamiento de
texto determinista; la representación vectorial se obtiene de un encoder de la familia BERT; y
la recuperación opera exclusivamente sobre vectores, puntuaciones de similitud y metadata,
conforme a la prohibición de decoders en la construcción del índice y en la recuperación.

---

## 2. Arquitectura del pipeline

El procesamiento se organiza en seis etapas encadenadas, cada una con una salida persistente en
disco. La persistencia intermedia es una decisión de diseño, no un detalle de implementación:
permite reejecutar cualquier etapa sin repetir las anteriores, cuyo coste no es simétrico.

| # | Etapa | Salida persistente | Coste |
|---|---|---|---|
| 1 | Extracción — 6 loaders según formato | `documentos.jsonl` · 1.826 documentos | ~4 h |
| 2 | Normalización — 8 pasos + idioma | `documentos_limpios.jsonl` | ~4,5 min |
| 3 | Fragmentación — cascada de 4 niveles | `chunks.jsonl` · 91.021 fragmentos | ~24 s |
| 4 | Codificación — `bge-m3`, 1.024 dim. | `embeddings.npy` (91.021 × 1.024) | ~2,8 h GPU |
| 5 | Indexación — `IndexFlatIP` + metadata | `index.faiss` + `metadata.jsonl` | <1 min |
| 6 | Recuperación y construcción de la salida | `resultados.jsonl` · 50 líneas | ~8 min CPU |

**Invariante de orden.** La fila *i* de `embeddings.npy` corresponde a la línea *i* de
`chunks.jsonl` y a la línea *i* de `metadata.jsonl`, que a su vez es el identificador interno
*i* del índice FAISS. Este invariante materializa el requisito de correspondencia entre el
identificador interno del índice y el fragmento, y es comprobable: cada registro de metadata
incluye un campo `faiss_id` con su número de línea. Un desalineamiento en esta cadena no
produciría ninguna excepción y devolvería la metadata de un fragmento distinto al recuperado,
por lo que se verifica explícitamente (sección 10).

---

## 3. Extracción del corpus

### 3.1 Composición y enrutamiento

El corpus es heterogéneo: 954 JSON, 759 PDF, 73 PBF (mosaicos vectoriales), 26 CSV, 8 JPG, 4
XLSX, 1 AVIF y 1 TXT. Se implementó un loader por familia de formato, todos con la misma
interfaz (`load(path, entry) -> list[Document]`), y un orquestador que enruta cada archivo según
su extensión real.

El inventario en Excel provisto por ADL es la autoridad sobre qué constituye un documento y sobre
su `doc_id`, que se toma tal cual por trazabilidad. El campo `formato` se deriva de la extensión
real y no de la columna `Tipo` del inventario, que agrupa `.pbf` y `.avif` bajo «Otro».

**Única excepción a esa autoridad: la verificación por número mágico.** Dos archivos con extensión
`.pdf` comienzan con `<!DOCTYPE html>`. El orquestador comprueba la firma `%PDF-` y, si falta,
corrige `formato` a `"html"` y reencamina el archivo. No es una heurística: un archivo que no lleva
su propia firma se desmiente a sí mismo. El corpus no se modifica en disco en ningún caso, porque
la evaluación empareja los documentos por el campo `fuente`, definido en la Tabla 1 como el nombre
del archivo original.

En JSON se emplea **selección explícita de campos** y no un recorrido genérico del árbol: el corpus
contiene seis esquemas distintos y un recorrido genérico incorporaría al texto indexado campos como
`url`, `authors` o `pdf_links`. Se verificó que `body_text` y `body_paragraphs` son idénticos en los
229 archivos donde ambos aparecen, por lo que nunca se concatenan.

### 3.2 Reconocimiento óptico de caracteres

Las bases recomiendan OCR para imágenes con texto relevante. El OCR clásico no es un modelo
generativo, por lo que su uso no incurre en la prohibición de decoders. Se aplicaron dos criterios
distintos, ambos decididos con medición:

**OCR de página completa: activado.** Cincuenta y un PDF son documentos escaneados sin capa de
texto y producían cero caracteres. Cuando un documento rinde menos de 200 caracteres extraíbles,
cada página se rasteriza y se somete a OCR: los 51 pasaron de cero a una mediana de 5.310
palabras, ninguno por debajo de 313.

La resolución se fijó en **150 DPI**, no en 300, midiendo la confianza que reporta el propio motor
sobre once páginas en español e inglés: 150 DPI da confianza 91,95 y 378 caracteres acentuados en
52,8 s, frente a 91,34 y 343 en 81,6 s a 300 DPI. La resolución alta es simultáneamente peor y un
55 % más lenta. Se toma la confianza del motor como criterio, y no el recuento de palabras, porque
más palabras puede significar más ruido.

**OCR de figuras: desactivado.** Sobre una muestra de 178 figuras de 60 PDF, solo el 19 % aportaba
texto aprovechable, y degradado de forma sistemática: la sigla «AI» no aparecía bien escrita ni una
vez, y cuatro archivos la contenían como «Al» con ele minúscula —confusión I/l aplicada al término
central del fenómeno 1—. Puesto que la Tabla 1 define `texto` como «texto original del fragmento,
sin modificaciones», indexar transcripciones con esa tasa introduciría afirmaciones que el
documento no contiene. La funcionalidad se conserva tras un interruptor, para permitir reevaluarla.

Los idiomas de OCR se limitan a `spa+eng`: los 51 documentos rescatados son 47 en español y 4 en
inglés, ninguno en portugués, porque las fuentes lusófonas se entregan en JSON y PBF.

### 3.3 Resultado

**1.826 documentos, 0 fallos, 28.881.496 palabras** (F1 459 · F2 479 · F3 888), con 51 documentos
rescatados por OCR y 8 sin texto. La correspondencia es **1:1 entre archivos y documentos**,
propiedad relevante para la evaluación: cada `doc_id` designa un archivo real y cada `fuente`
empareja directamente con el ground truth. Los 8 sin texto son cinco imágenes fotográficas y tres
JSON que son manifiestos del proceso de descarga, no contenido documental.

---

## 4. Normalización

Se implementaron ocho pasos deterministas. Cuatro corresponden a lo requerido —normalización a
UTF-8 y forma NFC, eliminación de caracteres de control y espaciado redundante, supresión de
encabezados y pies repetidos, y detección de idioma— y cuatro adicionales que el corpus hizo
necesarios.

**Principio rector: se elimina ruido, nunca información.** Cada regla que suprime contenido va
acompañada de la medición que demuestra que lo suprimido no era contenido del documento. Este
criterio descartó, entre otras, la idea de indexar únicamente las columnas textuales de los
archivos tabulares, que habría eliminado identificadores capaces de discriminar un registro.

| Paso adicional | Justificación medida |
|---|---|
| `decode_cid()` | 18 documentos traen marcadores `(cid:N)` en lugar de caracteres —fuentes empotradas sin tabla ToUnicode—, dos de ellos en más del 99 % del texto. Se busca el desplazamiento constante entre glifo y código, y se acepta solo si produce palabras funcionales reales. Un documento pasó de 473 palabras ilegibles a 1.556 legibles. |
| `strip_control_chars()` | 21 PDF emiten `\x07` en la posición del espacio y 6 lo emiten dentro de las palabras: una regla fija falla en uno de los dos grupos. Se evalúan ambas variantes por documento y gana la que produce más palabras plausibles. La función es pura, luego determinista. |
| `dehyphenate()` | Reunifica palabras partidas por guion de fin de línea. |
| `strip_anonymous_columns()` | Cuatro CSV incluyen una columna sin encabezado con 231.830 líneas. Se verificó que son secuencias `+1` con reinicios cada 10.000 registros: el índice del exporte, no un dato. Se elimina solo cuando el valor es un número sin texto. |

El resultado sobre los 1.826 documentos es una reducción de 8.245.459 caracteres (3,85 %), sin
que ningún documento con texto quedara vacío.

**Detección de idioma.** Clasificador estadístico de n-gramas de bytes, no un modelo generativo,
con licencia BSD y salida determinista —condición que descarta alternativas que muestrean al azar
y varían entre ejecuciones—. No se restringe la lista de idiomas candidatos, porque el corpus
contiene documentos legítimos en árabe, ruso, chino, coreano, japonés y francés y restringirla
haría que el campo mintiera precisamente donde más se notaría. Distribución: **inglés 55,0 %,
español 35,1 %, portugués 5,9 %**, otros 3,0 % y 1,0 % sin señal suficiente.

El orden de las etapas es significativo: la limpieza precede a la detección. En sentido inverso,
la marca de agua de un documento —el 69,3 % de su texto extraído— hacía que un documento en
inglés se clasificara como ruso.

---

## 5. Fragmentación

### 5.1 Estrategia: cascada híbrida de cuatro niveles

Las bases admiten estrategias híbridas siempre que se justifiquen explícitamente. La implementada
recorre cuatro niveles en orden, descendiendo solo cuando el anterior no basta:

1. **Agrupación de bloques.** Los saltos `\n\n` del texto normalizado delimitan párrafos en
   documentos de prosa y filas en documentos tabulares. Se agrupan bloques completos hasta
   agotar el presupuesto.
2. **Retroceso a límite de oración.** Cuando el bloque excede el presupuesto, el corte retrocede
   al final de la última oración completa que cabe. Es la formulación literal del requisito
   obligatorio de completitud lingüística.
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
omisión no incluye el punto, por lo que no mantiene las oraciones íntegras e incumple el requisito
obligatorio de completitud lingüística; sus últimos recursos parten palabras; y cuenta caracteres
en lugar de tokens.

### 5.2 Tratamiento de los documentos tabulares

Los 103 documentos tabulares (CSV, XLSX y PBF) concentran el 52,1 % de las palabras del corpus
desde el 5,6 % de los archivos. Las bases indican que cada fila «**puede**» tratarse como unidad
de fragmentación independiente. Esa lectura literal produciría 276.925 fragmentos tabulares, el
87,7 % del índice.

La granularidad elegida agrupa filas completas hasta agotar el presupuesto, lo que **reduce los
fragmentos tabulares en un 83,6 % sin perder una sola palabra**. En el índice entregado el
reparto resultante es:

| | Fragmentos | Porcentaje |
|---|---|---|
| Prosa | 45.245 | 49,7 % |
| Tabular | 45.776 | 50,3 % |

No se consideró excluir estos documentos. Las bases establecen que el conjunto provisto para los
tres fenómenos constituye el corpus, y un documento excluido del índice no puede recuperarse en
ninguna circunstancia.

### 5.3 Verificación y resultado

```
fragmentos             : 91.021
documentos con fragmento: 1.818   (los 1.826 menos los 8 sin texto)
documentos con texto y sin fragmentos : 0
```

Cinco pruebas automáticas, todas superadas: **completitud lingüística** — ningún corte parte una
oración, sobre más de 42.000 cortes de prosa examinados; **cobertura** — cada documento se
reconstruye carácter a carácter concatenando sus fragmentos; **Tabla 1** — los ocho campos
presentes, `posicion` consecutiva desde 0 y 91.021 `chunk_id` únicos; **presupuesto**; y
**determinismo** — dos ejecuciones producen idénticos identificadores y textos.

La prueba de completitud se define sobre el **corte** y no sobre el fragmento aislado. Preguntar
si un fragmento termina en signo de puntuación produce más de doce mil falsos positivos, porque
los encabezados y pies de figura carecen legítimamente de punto final. El síntoma inequívoco es la
pareja: un fragmento que no cierra seguido de otro que comienza en minúscula.

---

## 6. Codificación semántica

### 6.1 Modelo seleccionado

**`BAAI/bge-m3`**, revisión `5617a9f61b028005a4858fdac845db406aefb181`, 1.024 dimensiones.

| Criterio | Verificación |
|---|---|
| Familia encoder | Arquitectura XLM-RoBERTa. No es un decoder. |
| Soporte multilingüe | Más de 100 idiomas; español, inglés y portugués nativos. |
| **Alineamiento translingüe** | Entrenado con objetivo explícito translingüe (MIRACL, MKQA). |
| Recuperación densa | Diseñado para recuperación, no para clasificación ni similitud de pares — criterio que descarta modelos orientados a minería de textos paralelos. |
| Licencia | **MIT**, una de las tres licencias preferidas. |
| Eficiencia computacional | Ver sección 6.2. |

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

La eficiencia computacional figura entre los criterios de selección exigidos. Medida sobre
fragmentos reales del corpus en CPU sin GPU, la codificación completa con este modelo requeriría
del orden de 363 horas, frente a las 12 horas de un modelo de 384 dimensiones de la misma
familia. Se conservó el modelo de mayor calidad translingüe y la codificación se ejecutó en una
máquina con GPU (2,8 horas, ≈14 fragmentos/s).

Esta decisión no afecta a la reproducibilidad evaluada, que consiste en que `generador.py`
reproduzca los resultados **a partir del índice ya construido**. La construcción del índice no
forma parte de lo evaluado.

La matriz resultante se verificó sobre la totalidad de las filas:

```
forma        : (91.021 × 1.024) float32     filas nulas : 0     NaN o Inf : 0
normas       : 1,000000 exacto, desviación máxima 1,19e-07
```

La normalización es condición necesaria para que el producto interno del índice sea el coseno.
Como prueba de discriminación del espacio vectorial, el coseno entre 200 vectores repartidos por
todo el índice arroja media 0,485, mínimo 0,232 y máximo 1,000: el espacio discrimina, y no ha
colapsado.

### 6.3 Declaración del límite de tokens

La especificación indica que los modelos encoder tienen un límite de tokens de entrada,
«comúnmente 512», y que los fragmentos deben diseñarse para no superar **dicho límite**. El
límite del modelo empleado es de **8.192 tokens**, valor comprobado en tiempo de ejecución sobre
la longitud máxima de secuencia efectiva que el modelo declara.

La distribución real de `num_tokens` en el índice entregado, medida sobre los 91.021 fragmentos
con el tokenizador del propio modelo, es:

| Mediana | Percentil 95 | Máximo | Por encima de 512 | Por encima de 8.192 |
|---|---|---|---|---|
| 692 | 930 | 6.867 | 68,24 % | **0** |

**Ningún fragmento se trunca.** El máximo real se sitúa un 16 % por debajo del límite efectivo.
Se declara explícitamente esta cifra porque el valor de 512 tokens citado como habitual no es el
aplicable a este encoder, y porque la relación entre palabras y tokens en este corpus —2,145
tokens por palabra— es sensiblemente superior a la de un corpus de prosa homogénea: los
documentos tabulares, mitad del índice, contienen identificadores y códigos que el tokenizador
fragmenta con densidad muy superior a la del texto corrido.

Se contempló el uso de un segundo encoder con fusión de rankings por CombSUM o CombMNZ, previsto
en las bases. Se descartó: al fusionarse por fragmento, el mismo `chunk_id` debe existir en ambos
índices y los fragmentos deben respetar el límite del encoder más restrictivo. Con el 68,24 % de
los fragmentos por encima de 512 tokens, un segundo encoder con ese tope truncaría dos tercios
del índice sin emitir aviso.

---

## 7. Índice vectorial

### 7.1 Tipo de índice

**`IndexFlatIP` con vectores normalizados**, serializado con `faiss.write_index()`.

| Decisión | Justificación |
|---|---|
| `IndexFlatIP` | Es el índice recomendado para el volumen esperado en el reto. Con 91.021 vectores una consulta se resuelve en milisegundos y el conjunto de evaluación son 50 consultas: sustituir búsqueda exacta por búsqueda aproximada introduciría error en la métrica a cambio de una velocidad innecesaria. |
| Vectores normalizados | Convierte el producto interno en similitud coseno exacta. |
| Sin `IndexIDMap` | Los identificadores internos, asignados por orden de inserción, ya coinciden con los números de línea del almacén de metadata. La indirección resolvería un problema inexistente. |

Dimensiones finales: **91.021 vectores × 1.024 dimensiones**, 372.822.061 bytes.

### 7.2 Almacén de metadata

`metadata.jsonl` contiene una línea JSON por fragmento, con los ocho campos obligatorios de la
Tabla 1 nombrados en español (`doc_id`, `chunk_id`, `fuente`, `formato`, `fenomeno`, `posicion`,
`num_tokens`, `texto`) y dos campos adicionales, de los que la especificación autoriza:

- **`idioma`**, porque el filtrado por metadata está contemplado y el módulo de recuperación lo
  utiliza (sección 8.2).
- **`faiss_id`**, el número de línea, que convierte el invariante de orden en una propiedad
  comprobable en lugar de una suposición.

Se descartaron otros campos candidatos. Cada campo adicional se paga 91.021 veces, y ninguna
etapa del sistema consumía el título del documento ni los diagnósticos internos del fragmentador.

El campo `formato` registra la extensión real del archivo. La Tabla 1 enumera «pdf, html o md»,
pero el 58 % del corpus no pertenece a ninguno de esos tres formatos, y las bases describen
explícitamente el tratamiento de CSV, XLSX e imágenes. El campo `fuente` registra la ruta
relativa completa del archivo original, y no solo su nombre, porque el corpus contiene 59
nombres de archivo repetidos que afectan a 186 documentos; la evaluación empareja los documentos
por este campo, de modo que un nombre ambiguo introduciría un error de emparejamiento.

`metadata.jsonl` ocupa 240.054.195 bytes en 91.021 líneas.

---

## 8. Recuperación y construcción de la salida

El módulo de recuperación transforma una consulta en tres documentos y diez fragmentos mediante
cinco pasos: codificación de la consulta, búsqueda en el índice, agregación a nivel documento,
filtrado, y división al límite de palabras de la salida. Toda la operación se realiza sobre
vectores, puntuaciones y metadata.

### 8.1 Búsqueda y agregación

**Se recuperan 200 candidatos** y no diez, porque entre la búsqueda y la respuesta operan tres
filtros que descartan candidatos y, con un índice plano, pedir más es marginal: el trabajo consiste
en recorrer los 91.021 vectores en cualquier caso. Si los candidatos no contienen tres documentos
distintos, la búsqueda los amplía progresivamente hasta alcanzarlos: la evaluación automática
penaliza los objetos cuyos arrays no midan exactamente 3 y 10, y cumplirlo prevalece sobre
cualquier otra preferencia.

**La relevancia de un documento se agrega por máximo** de las puntuaciones de sus fragmentos. Las
bases admiten máximo, suma o media; la suma es inadecuada aquí porque un solo archivo tabular
aporta 33.396 fragmentos frente a una mediana de 5 por documento, de modo que premiaría el tamaño
y los tres puestos los ocuparían invariablemente los mismos archivos.

**El almacén de metadata se consulta por desplazamiento de bytes**, manteniendo en memoria un
índice de 91.021 posiciones (~700 KB) en vez de los 229 MB del archivo: quien reproduzca los
resultados no necesita medio gigabyte de memoria para leer diez fragmentos.

### 8.2 Filtros aplicados

**Tope de dos fragmentos por documento.** Sin él, una consulta puede consumir siete de sus diez
posiciones con fragmentos de un mismo documento. El tope es una preferencia: si tras aplicarlo
no se alcanzan diez fragmentos, se realiza una segunda pasada sin restricción, porque cumplir el
formato de salida es obligatorio.

**Deduplicación por contenido.** El índice contiene 91.021 fragmentos pero 87.424 textos únicos:
3.597 fragmentos (4,0 %) repiten contenido, en dos patrones distintos —filas tabulares idénticas
repartidas entre documentos, y secciones repetidas dentro de un mismo informe—. Puesto que la
evaluación juzga la relevancia sobre el campo `text`, dos fragmentos de contenido idéntico
ocuparían dos posiciones con la misma información. La clave de comparación normaliza únicamente
el espaciado, verificado: colapsa exactamente los tres pares que el tokenizador ya trataba como
equivalentes, y ninguno más. No se normalizan mayúsculas ni puntuación, porque dos textos que
difieren en ellas difieren realmente.

**Bonificación por fenómeno: factor 1,03.** La proporción de documentos devueltos pertenecientes al
fenómeno de la consulta era del 80,7 %. La causa está identificada: las etiquetas de fenómeno
describen el observatorio de origen y no el contenido, y hay fuentes catalogadas en un fenómeno que
publican abundantemente sobre el tema de otro. Se descartó el filtrado duro por metadata, que
estaría permitido, porque su fallo es irreversible: un documento relevante de otro fenómeno
quedaría inalcanzable, mientras que una bonificación multiplicativa desplaza sin cerrar la puerta.
El factor se determinó comparando 1,00, 1,03, 1,05 y 1,10 mediante *pooling* —metodología TREC—
sobre juicios de relevancia elaborados manualmente y puntuados con las métricas oficiales; 1,10 no
deja sobrevivir ningún documento de otro fenómeno, es decir equivale a un filtro. Aplicado, la
correspondencia asciende al 91,3 %.

**Factor por idioma: 0,97 para idiomas distintos de español e inglés.** Varias organizaciones
publican el mismo informe en varios idiomas, que son documentos distintos con `doc_id` distintos;
una consulta llegaba a ocupar dos de sus tres puestos con dos traducciones del mismo resumen
ejecutivo. Determinado por barrido: sin factor sobreviven 16 fragmentos de 500 fuera de {es, en}
en 8 consultas; con 0,99 quedan 11, pero solo altera el conjunto de documentos en 2 consultas —en
el resto reordena los mismos tres, y F1@3 no considera el orden—; con 0,97 quedan **5 en 4
consultas** y cambian 5 conjuntos; con 0,95 y menores no sobrevive **ninguno**, lo que constituye
un filtro encubierto. El valor elegido corresponde a una banda de tolerancia del 3 % en similitud
coseno: se prefiere la versión en español o inglés solo cuando ambas son semánticamente casi
equivalentes. Los documentos sin idioma detectado **no se penalizan**, porque la ausencia de
etiqueta indica falta de señal del clasificador y no un idioma distinto.

### 8.3 Límite de 250 palabras

El 90,6 % de los fragmentos excede las 250 palabras, de modo que dividir es el caso normal y no la
excepción. **El límite se aplica al construir la salida, no al índice**: los fragmentos extensos
producen vectores con más contexto y se dividen solo al reportarse. La división respeta la misma
completitud lingüística del fragmentado, reutilizando el divisor de oraciones del fragmentador para
evitar dos implementaciones divergentes de la misma regla. Verificado sobre el índice completo:
175.025 subfragmentos, mediana 169 palabras, máximo 250, ninguno por encima y ninguno que pierda
texto.

Se reporta **un subfragmento por fragmento**, no todos. El formato prohíbe situar dos en una misma
posición, pero no obliga a reportarlos todos; haciéndolo, las diez posiciones procederían de unos
cinco fragmentos y las mitades sin contenido relevante ocuparían posiciones altas, que es lo que
penaliza NDCG@10. El subfragmento se elige por **solapamiento léxico** con la consulta y no por
similitud vectorial: codificarlos en tiempo de consulta trasladaría un coste de horas a quien
reproduzca los resultados, mientras que el solapamiento es determinista y no constituye un modelo.

### 8.4 Formato de salida

`resultados.jsonl` contiene 50 líneas en el orden `q001`–`q050`, con exactamente tres documentos
y diez fragmentos por consulta, cada uno con los campos de la Tabla 2. Seis pruebas automáticas
verifican estructura, tamaños de los arrays, presencia y no vacuidad de los campos, consecutividad
de las posiciones, límite de 250 palabras, y **trazabilidad**: que todo `doc_id` y todo `chunk_id`
de la salida exista en `metadata.jsonl`. Esta última condición es necesaria porque la evaluación
empareja los documentos por el campo `fuente`, y la única vía desde `doc_id` hasta `fuente` es el
almacén de metadata.

---

## 9. Grafo de conocimiento

Componente opcional, entregado como `base_vectorial/grafo/grafo.graphml`. La especificación pide
`G = (E, R, T)` con `T ⊆ E × R × E`: un grafo dirigido con relaciones tipadas. Se implementa como
multigrafo dirigido, porque dos entidades pueden estar unidas por varias relaciones distintas —un
Estado puede a la vez desarrollar y operar una tecnología— y un grafo simple conservaría solo una.

**Alcance: el grafo se construye pero no se integra en la recuperación.** Las bases contemplan
combinarlo con los resultados vectoriales, pero lo enuncian como posibilidad, y la puntuación
adicional se concede por implementar el componente. Integrarlo alteraría `resultados.jsonl` y
obligaría a repetir la validación de reproducibilidad, mientras que su efecto sobre NDCG@10 y F1@3
no es medible sin el ground truth. Es el mismo criterio que descartó el filtrado duro por fenómeno:
no asumir un riesgo irreversible a cambio de una mejora no verificable.

**Extracción de entidades: `urchade/gliner_multi-v2.1`**, revisión
`443d26d654e0324125a96bebd8e796c14ff2efe6`, licencia Apache 2.0. Es un **encoder bidireccional**
—arquitectura mDeBERTa-v3—, de modo que no incurre en la prohibición de decoders. Se eligió por ser
*zero-shot*: recibe las etiquetas de tipo en tiempo de inferencia, lo que permite solicitar
directamente categorías del dominio. Un reconocedor clásico de entidades, limitado a persona,
organización y lugar, nunca devolvería «sistema de armas autónomo», que es el ejemplo literal de la
especificación. Se emplean nueve etiquetas con umbral de confianza 0,5, elegido examinando las
entidades marginales y no el recuento: un umbral de 0,7 pierde nombres de país y uno de 0,3 admite
plurales genéricos, y **un nodo genérico es peor que una entidad ausente**, porque atrae relaciones
de todo el corpus sin designar nada.

**Extracción de relaciones sin modelo generativo.** Las bases autorizan explícitamente heurísticas
basadas en patrones lingüísticos, y esa es la vía empleada: dentro de una misma unidad de texto,
cuando entre dos entidades aparece un verbo de un inventario curado de **trece relaciones**, se
emite la tripleta correspondiente, con tratamiento de voz pasiva, negación y coordinación. Se
descartó por principio el uso de un modelo generativo para extraer tripletas, que es la vía obvia
y la que está prohibida en la construcción del índice.

**Unidad de entrada.** El modelo tiene una ventana de 384 unidades, mientras que los fragmentos
del índice tienen mediana de 692 tokens: pasarlos completos dejaría sin explorar la segunda mitad
de la mayoría, sin ningún aviso. Se reutiliza el fragmentador de la sección 5 con un presupuesto de
320, aprovechando que recibe la función de conteo como parámetro. El conteo es `máx(palabras,
subtokens)`, porque el modelo mantiene **dos contabilidades independientes** —su troceador separa
cada signo de puntuación como unidad propia— y acotar solo una permite que la otra se desborde;
son además los mismos textos, los de puntuación densa, los que desbordan por ambos lados. Con ese
criterio se recorrieron **127.070 unidades sin un solo truncamiento**, en 1 h 57 min sobre GPU.

Se excluyeron **trece documentos**, no un formato completo: volcados bibliográficos identificados
por marcadores inequívocos (`PMID:`, `NCT Number:`), que concentran el 93,8 % de los fragmentos
tabulares. La exclusión se midió antes de aplicarla: rinden once entidades por unidad, y son
afiliaciones hospitalarias reales pero ajenas a los tres fenómenos. Se conservan en cambio los
catálogos y los mosaicos vectoriales, que sí aportan entidades del dominio.

**Resultado.**

| | |
|---|---|
| Nodos · aristas | **3.375 · 3.460**, en un archivo GraphML de 2,30 MB |
| Tipos de entidad | ORG 1.332 · TEC 756 · LOC 545 · PER 404 · NORM 189 · GRP 149 |
| Relaciones más frecuentes | `opera` 709 · `desarrolla` 633 · `lanza` 601 · `coopera_con` 357 |
| Cobertura | 521 de 1.818 documentos (28,7 %), con los tres fenómenos representados |
| Entidades más conectadas | Estados Unidos (249), China (195), Rusia (128), NASA (107) |

Las trece relaciones se emplean todas y los seis tipos de entidad están representados. Que los
nodos más conectados sean actores geopolíticos concretos, y no términos genéricos, es la señal de
que la canonicalización y el umbral funcionan: la identidad de un nodo es el par **(tipo, clave
canónica)** y no la clave sola, porque «Amazon» designa una región en el fenómeno 3 y una empresa
en el 1, y unificarlas afirmaría que una selva desarrolla servicios en la nube.

**Trazabilidad.** Cada arista conserva el `doc_id` y el `chunk_id` del fragmento donde se observó
la relación, más hasta diez identificadores adicionales de evidencia. Se verifica que todos existen
en el `metadata.jsonl` entregado: un identificador que no estuviera en el índice no permitiría
rastrear la evidencia textual exigida.

---

## 10. Verificación y reproducibilidad

### 10.1 Estrategia de verificación

Cada etapa dispone de un verificador independiente que devuelve código de salida distinto de cero
ante cualquier fallo, de modo que puede encadenarse. En total, 40 comprobaciones automáticas:

| Etapa | Comprobaciones | Estado |
|---|---|---|
| Normalización | 5 | Superadas |
| Fragmentación | 5 | Superadas |
| Índice | 5 | Superadas |
| Grafo | 7 | Superadas |
| Salida | 6 | Superadas |
| Métricas | 12 | Superadas |

Un criterio metodológico gobierna estas pruebas: **una verificación que toma su entrada de la
misma fuente que valida no verifica nada**. La comprobación del invariante de orden no interroga
al índice sobre su propio contenido, sino que parte de la matriz de embeddings que produjo el
encoder, y confirma que la fila *i* de ese archivo se corresponde con la posición *i* del índice.

El criterio tiene una segunda cara, que el grafo hizo explícita: **una comprobación tampoco puede
leer el artefacto con la misma biblioteca que lo escribió**. Las pruebas del grafo se apoyaban en
la biblioteca que lo genera, que interpreta el identificador de arista como clave local a cada par
de nodos, mientras que GraphML lo exige globalmente único: el archivo superaba todas las pruebas y
ningún visor podía abrirlo. La comprobación se rehízo sobre el XML en bruto, que es como lo lee
cualquier consumidor externo.

Las métricas oficiales se implementaron para comparar decisiones de diseño entre sí. Dos detalles
de las fórmulas se respetan explícitamente: la ganancia del DCG es **lineal**, `rᵢ / log₂(i+1)`, y
no exponencial —diferencia que altera el resultado en cuanto la relevancia es graduada—; y el
denominador de R@3 se limita a `mín(|D*|, 3)`.

### 10.2 Reproducibilidad

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

### 10.3 Dependencias

Reproducir los resultados requiere **nueve paquetes de Python**, declarados en
`entrega/requirements.txt`: cuatro importados directamente (`faiss-cpu`, `sentence-transformers`,
`pypdf`, `numpy`) y cinco transitivas fijadas por determinar los valores numéricos de los
embeddings. El conjunto es deliberadamente reducido: cada dependencia adicional es una vía más para
que la instalación falle, y una instalación fallida impide ejecutar el generador. **Se declara
`faiss-cpu` y no `faiss-gpu`**, dado que el entorno de evaluación puede carecer de GPU y un índice
plano no requiere entrenamiento.

**Dependencia de sistema, no capturada por los gestores de paquetes: Tesseract OCR 5.4.0** con los
idiomas `spa` y `eng`. Es necesaria **únicamente para reconstruir el índice desde el corpus**, no
para reproducir los resultados, ya que el reconocimiento óptico se ejecutó aguas arriba y su
producto está contenido en el índice. La comprobación es directa: la ejecución que generó el
archivo entregado se realizó en un entorno sin Tesseract y sin las bibliotecas de extracción de PDF
instaladas. El procedimiento de instalación para Windows y Debian/Ubuntu se documenta en la
cabecera de `requirements.txt`.

---

## 11. Limitaciones conocidas

Se declaran por transparencia metodológica.

1. **Las métricas oficiales no se han evaluado contra el ground truth**, que no es público durante
   el reto. Las comparaciones empleadas para elegir los parámetros de recuperación usan *pooling*
   sobre juicios de relevancia elaborados manualmente: son válidas para comparar configuraciones
   entre sí, pero sus valores absolutos no son comparables con los de la evaluación oficial.
2. **La bonificación por fenómeno presupone** que el ground truth se construyó dentro del fenómeno
   de cada consulta; si se construyó por tema, el ajuste sería contraproducente. No es verificable
   con la información disponible.
3. **Los documentos tabulares son la mitad del índice** y en su mayoría volcados bibliográficos. Su
   inclusión es obligada por las bases y la granularidad elegida reduce su peso, pero compiten en
   el ranking con documentos de prosa.
4. **La detección de idioma presenta un error estimado del 1 %**, concentrado en documentos de
   extracción defectuosa. Por eso el idioma se emplea como factor de puntuación y nunca como filtro.
5. **La precisión del grafo está acotada por el método.** Revisadas manualmente las aristas de mayor
   peso, algo más de la mitad son defendibles. Los errores se agrupan en clases identificadas: nodos
   genéricos que el umbral no descarta, construcciones contrastivas, títulos de cita en notas al pie
   y enumeraciones de elementos del mismo tipo. Las tres últimas exigirían análisis sintáctico —vía
   autorizada y no implementada— y constituyen el techo de un extractor por patrones. El grafo es
   además disperso: el 80,9 % de sus aristas se apoya en una sola observación del corpus.

---

## 12. Estructura de la entrega

```
entrega/  resultados.jsonl (50 líneas) · generador.py · requirements.txt (9 paquetes)
          informe_tecnico.pdf
  base_vectorial/encoder_bge-m3/  index.faiss (IndexFlatIP, 91.021 × 1.024)
                                  metadata.jsonl (Tabla 1 + idioma + faiss_id)
                 grafo/           grafo.graphml (3.375 nodos · 3.460 aristas)
```

Ejecución: `pip install -r requirements.txt`, después `python generador.py` y, para contrastar la
salida con el manifiesto, `python generador.py --comprobar`.
