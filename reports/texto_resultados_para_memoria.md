## Evaluación del modelo

Una vez generado el conjunto de datos y extraídas las características acústicas por ventanas temporales, se entrenó un modelo de clasificación basado en Random Forest para detectar indicios de manipulación por audio splicing. El sistema no clasifica únicamente ventanas aisladas, sino que posteriormente agrupa las predicciones a nivel de archivo para decidir si un audio completo debe considerarse limpio o sospechoso.

Durante las primeras pruebas se observó que la clasificación directa de ventanas presentaba un fuerte desbalance entre clases, ya que el número de ventanas limpias era muy superior al número de ventanas asociadas a regiones manipuladas. Por este motivo, se reformuló parcialmente el enfoque, incorporando características de discontinuidad entre ventanas consecutivas y evaluando el comportamiento del modelo mediante distintos umbrales de decisión.

La evaluación final se realizó sobre 63 archivos de audio generados: 21 audios limpios y 42 audios manipulados. Para la decisión final a nivel de archivo se seleccionó un umbral de 0,50, al ofrecer un equilibrio razonable entre sensibilidad y reducción de falsos positivos.

Con este umbral, el modelo obtuvo los siguientes resultados:

| Métrica | Valor |
|---|---:|
| Accuracy | 0,7778 |
| Precision | 0,8182 |
| Recall | 0,8571 |
| F1-score | 0,8372 |

La matriz de confusión obtenida fue la siguiente:

| | Predicho limpio | Predicho manipulado |
|---|---:|---:|
| Real limpio | 13 | 8 |
| Real manipulado | 6 | 36 |

Estos resultados indican que el sistema detectó correctamente 36 de los 42 audios manipulados, mientras que 6 audios manipulados no fueron detectados. En el caso de los audios limpios, 13 fueron clasificados correctamente y 8 fueron marcados como sospechosos. Aunque el número de falsos positivos todavía es relevante, el resultado es coherente con un prototipo de análisis forense orientado a señalar posibles regiones sospechosas para una revisión posterior.

Además de la clasificación global del archivo, se evaluó la capacidad del sistema para localizar temporalmente la región manipulada. Con el umbral seleccionado, el sistema consiguió localizar correctamente la zona alterada en 30 de los 42 audios manipulados, lo que supone una tasa de localización de 0,7143.

La comparación de métricas por umbral muestra que los umbrales bajos aumentan el recall, pero generan más falsos positivos, mientras que los umbrales altos reducen los falsos positivos a costa de dejar más manipulaciones sin detectar. Por ello, el umbral 0,50 se considera una solución intermedia adecuada para este prototipo.

En conjunto, los resultados obtenidos permiten concluir que la metodología propuesta es viable como prueba de concepto. El sistema es capaz de detectar patrones compatibles con audio splicing y de aproximar la localización temporal de una parte relevante de las manipulaciones. No obstante, los resultados deben interpretarse con cautela, ya que el conjunto de datos empleado es reducido y las manipulaciones han sido generadas de forma controlada. Como trabajo futuro, sería recomendable ampliar el dataset, incorporar audios de mayor variedad acústica y comparar el rendimiento con otros modelos de aprendizaje automático o arquitecturas profundas.
