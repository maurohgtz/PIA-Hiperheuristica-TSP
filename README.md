# Hiperheurística de selección para el TSP

Proyecto final de Algoritmia y Optimización · Mauro Herrera.

El proyecto selecciona entre búsqueda local 2-opt, recocido simulado con 2-opt y MST con refinamiento 2-opt. El conjunto del curso contiene 500 instancias euclidianas. Cada etiqueta se calculó al ejecutar las tres heurísticas sobre la misma instancia. Los dos selectores pueden aplicarse a coordenadas nuevas.

[Reporte técnico en PDF](PIA_Hiperheuristica_TSP.pdf)

## Resultados

En las 500 instancias, 2-opt recibió 365 etiquetas (73.0 %), SA recibió 41 (8.2 %) y MST recibió 94 (18.8 %). Hubo 242 empates en el costo mínimo.

| Método | Aciertos en prueba | Accuracy | Regret medio | Elecciones de costo mínimo |
|---|---:|---:|---:|---:|
| Baseline de entrenamiento, siempre 2-opt | 109/149 | 73.15 % | 0.02840 | 109/149 |
| Clasificador CART | 109/149 | 73.15 % | 0.02840 | 109/149 |
| Selector complementario por distancia | 58/149 | 38.93 % | 0.02346 | 115/149 |

El clasificador principal alcanzó la misma accuracy que el baseline. El selector por distancia redujo el regret medio un 17.4 % en esta prueba. La columna final incluye las elecciones que empataban en el costo mínimo y recibieron otra etiqueta por la regla de desempate.

## Ejecutar

Desde la carpeta que contiene `main.py`, en PowerShell:

```powershell
py -m pip install -r requirements.txt
py main.py
```

En macOS o Linux se puede usar `python3` en lugar de `py`. La ejecución completa regenera los CSV, los dos selectores, las figuras y el PDF. Los resultados y el PDF están incluidos. El proceso se comprobó con Python 3.12.14, NumPy 2.3.5, pandas 2.2.3, scikit-learn 1.8.0, Matplotlib 3.10.8 y ReportLab 4.4.9.

Para ejecutar una etapa por separado:

| Comando | Resultado |
|---|---|
| `py main.py etiquetar` | Ejecuta las tres heurísticas y guarda los costos y las rutas. |
| `py main.py entrenar` | Selecciona los modelos en entrenamiento, evalúa la prueba y exporta los selectores. |
| `py main.py verificar` | Recalcula los 1,500 costos desde los tours y comprueba etiquetas, predicciones y métricas. |
| `py main.py reporte` | Genera las figuras y el PDF con las tablas de resultados. |

El comando `reporte` admite el argumento opcional `--repo` seguido de la URL de GitHub o GitLab para incorporarla al PDF. La dirección se conserva en `repositorio.txt` para las siguientes ejecuciones.

## Predecir

Ejemplo reproducible del dataset incluido:

```powershell
py main.py predecir --instancia inst_0235
py main.py predecir --instancia inst_0235 --criterio distancia
```

Para una instancia nueva, el CSV debe tener los encabezados `x,y` y una ciudad por fila, con coordenadas en [0,1]. El programa calcula sus características y devuelve el nombre de la heurística elegida. Las distribuciones admitidas son `uniform`, `clustered` y `border`.

```powershell
py main.py predecir --csv ciudades.csv --distribucion uniform
```

El criterio predeterminado, `acierto`, usa el clasificador. El criterio `distancia` usa el árbol que estima pérdidas de distancia.

## Diseño

- Se mantiene la partición original de 351 instancias de entrenamiento y 149 de prueba. Cada partición contiene las 27 combinaciones de tamaño y distribución.
- Las funciones de costo, 2-opt de mejor mejora, vecino más cercano, recocido y MST en preorden provienen de la Tarea 2. El recocido usa 4,500 iteraciones con enfriamiento exponencial. En coordenadas unitarias usa temperaturas 0.40 y 0.001, equivalentes a 40 y 0.1 en la escala original. El sufijo de la instancia fija las semillas de las rutas iniciales y del recocido.
- El costo es la longitud del circuito cerrado, registrada a 12 decimales. En caso de igualdad, la prioridad de la etiqueta es 2-opt, SA y MST. La columna `n_best` indica cuántos métodos alcanzaron el mínimo.
- El ajuste principal compara 22 configuraciones de CART y KNN mediante cinco pliegues de entrenamiento. El identificador solo sirve para localizar una instancia. Los costos y el ganador se usan como objetivos o medidas de evaluación y se excluyen de las variables de entrada.
- El análisis complementario ajusta un árbol de regresión con tres salidas, una por algoritmo. Elige la menor pérdida esperada. El PDF analiza este criterio y el efecto del desempate sobre la interpretación de la accuracy.

## Archivos

| Ubicación | Contenido |
|---|---|
| `datos/train.csv`, `datos/test.csv` | Particiones originales sin etiquetas. |
| `datos/instancias.zip` | Los 500 CSV de coordenadas del curso. Se leen directamente del ZIP. |
| `src/algoritmos.py`, `src/etiquetado.py` | Heurísticas y generación de las etiquetas. |
| `src/datos.py`, `src/modelo.py`, `src/distancia.py` | Características, validación, selección y evaluación. |
| `src/verificacion.py`, `src/figuras.py`, `src/informe.py` | Comprobaciones y reproducción de figuras e informe. |
| `resultados/train_etiquetado.csv`, `resultados/test_etiquetado.csv` | Rasgos, tres costos, etiqueta y número de ganadores. |
| `resultados/predicciones_prueba.csv` | Una fila por instancia de prueba con predicción, baseline, costos y regret de cada estrategia. |
| `resultados/rutas.jsonl` | Tres rutas por instancia. Los índices empiezan en 0 y el cierre a la primera ciudad es implícito. |
| `resultados/selector.json`, `resultados/selector_distancia.json` | Modelos ajustados para predecir instancias nuevas. |
| Otros CSV y JSON de `resultados/` | Balance, validación cruzada, matrices, importancia, patrones por grupo, métricas y controles. |
| `figuras/` | Matrices de confusión, ganadores por grupo y recorridos de ejemplo. |

El regret se expresa en unidades de distancia. Para cada fila se resta el mínimo de los tres costos observados al costo de la heurística elegida. En las instancias de cinco ciudades, la verificación calcula además el óptimo global por enumeración.
