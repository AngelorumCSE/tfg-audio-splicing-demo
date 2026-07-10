# Detector de manipulación por splicing en audio

Aplicación de demostración desarrollada para el TFG: **Verificación forense de grabaciones de audio mediante IA: detección y localización de manipulaciones**.

> Proyecto desarrollado como parte del **Trabajo de Fin de Estudios del Grado en Ingeniería Informática de UNIR** (2026). Autor: Ángel Carlos Soler Encinas. Código publicado bajo **licencia MIT** (véase `LICENSE`).

La aplicación permite analizar audios de dos formas:

1. **Ejemplos precargados**: audios limpios y manipulados incluidos con el proyecto, con ground truth conocido.
2. **Audio subido por el usuario**: cualquier archivo de audio compatible puede procesarse dinámicamente para obtener una predicción y una curva temporal de sospecha.

La salida incluye la predicción global, el *tamper score*, los intervalos temporales sospechosos, la curva de scores por ventana, el espectrograma y una tabla descargable con los resultados por ventana.

## Demo online

La aplicación puede probarse desde el siguiente enlace:

https://tfg-audio-splicing-demo.streamlit.app/

## Contenido de la carpeta

- `app_tfg.py`: aplicación principal desarrollada con Streamlit. Incluye comentarios explicativos del pipeline de inferencia.
- `requirements.txt`: librerías necesarias para ejecutar la aplicación.
- `src/`: scripts empleados durante el desarrollo experimental: generación del dataset, extracción de características, entrenamiento y evaluación.
- `data/demo/`: ejemplos precargados de la aplicación (12 audios cross-source de LibriSpeech con su manifiesto y características). Se regeneran con `preparar_ejemplos_demo.py`; la app los usa automáticamente si existen.
- `data/generated/`: muestra de audios del dataset intra-fuente original, incluida únicamente en la entrega académica (no se publica en el repositorio por contener voz de personas identificables). El conjunto completo se regenera con `src/generar_dataset_splicing.py` a partir de las grabaciones originales.
- `data/processed/window_features_borde.csv`: características por ventana del experimento intra-fuente completo (5 194 ventanas; evidencia de las secciones 5.3 a 5.5 de la memoria).
- `data/libri/`: conjunto cross-source derivado de LibriSpeech (120 audios, manifiesto y características por ventana; sección 6.8 de la memoria).
- `data/manifests/splicing_manifest.csv`: manifiesto del dataset intra-fuente con el ground truth.
- `models/`: modelos entrenados (`modelo_libri.joblib` es el que usa la aplicación; `random_forest_borde.joblib`, el del experimento intra-fuente).
- `reports/`: métricas y gráficos finales del experimento.

### Novedades de esta versión (mejoras de ingeniería y rigor)

- `analisis_avanzado/08_desglose_y_ablacion_cross.py`: desglose de detección/localización por tipo de empalme en el umbral operativo y ablación/comparativa de clasificadores sobre el conjunto cross-source (salidas en `reports/avanzado/desglose_cross_por_tipo.*` y `ablacion_baselines_cross.*`). También `make desglose-cross`.
- `analisis_avanzado/09_comprobacion_espanol.py`: comprobación exploratoria de validez en español con el modelo ya entrenado (Anexo B de la memoria). También `make espanol`.
- `analisis_avanzado/10_iou_localizacion_cross.py`: localización con criterio estricto (IoU) sobre el conjunto cross-source, out-of-fold (sección 6.8 de la memoria). También `make iou-cross`.
- `analisis_avanzado/11_transferencia_partialspoof.py`: transferencia exploratoria del detector, sin reentrenar, al benchmark público PartialSpoof (sección 6.9 de la memoria; salidas en `reports/avanzado/transferencia_partialspoof.*`). También `make transferencia`.

- **Coherencia artefacto–memoria**: los bundles joblib persisten ahora el umbral operativo (`best_threshold = 0.50`, el usado en la memoria y la app) y conservan `best_threshold_f1` como referencia; el slider de la app se inicializa con el umbral persistido.
- Constantes centralizadas: la app y los scripts toman SR/ventana/salto/umbral de `src/config.py` (con valores por defecto si no está disponible).
- `src/evaluar_por_archivo_borde.py` usa el umbral operativo de `config.py` (antes tenía 0.30 hardcodeado de una fase intermedia).
- Pruebas ampliadas (9): cobertura de `balancear_train`, fusión por `max_gap` e IoU sin solape.

- `src/config.py`: configuración central (rutas, frecuencia de muestreo, geometría de ventana, semilla, umbrales y parámetros del modelo). Elimina los números «mágicos» repetidos entre scripts.
- `src/posproceso.py`: lógica pura de posprocesado (agrupación de intervalos, solape, IoU y balanceo), aislada de scikit-learn para poder probarse de forma unitaria.
- `src/evaluar_por_archivo_cv.py`: **evaluación por archivo sin fuga de datos** mediante validación cruzada agrupada (GroupKFold por `archivo_base`). Es la verificación rigurosa recomendada (ver más abajo).
- `tests/test_posproceso.py`: pruebas unitarias de la lógica de posprocesado.
- `Makefile`: orquestador para reproducir el pipeline de extremo a extremo (`make help`).
- `src/informe_forense.py`: genera un **informe forense en PDF** por audio con hash SHA-256 (cadena de custodia), usado por la app.
- `src/features_inferencia.py`: pipeline de características por ventana (ventaneo + base + delta) reutilizable en inferencia.
- `analisis_avanzado/`: análisis de excelencia (validación cruzada y ROC/PR/AUC, ablación, comparativa de modelos, robustez, explicabilidad, validación externa y variabilidad entre semillas (`07_multisemilla_libri.py`)). Ver `analisis_avanzado/README.md`.

### Novedades en la aplicación (app_tfg.py)

- Descarga de **informe forense en PDF** (con hash SHA-256) para cada audio analizado.
- **Procesamiento por lotes**: varios audios con tabla-resumen descargable.
- Visualización de los intervalos sospechosos sobre la **forma de onda**, además del espectrograma.

## Validación sin fuga de datos (recomendada)

La evaluación por archivo reportada en la memoria (`evaluar_umbrales_por_archivo_borde.py`) puntúa los 63 archivos con un modelo entrenado sobre 15 de los 21 audios base; por tanto, ~45 de esos archivos pertenecen a audios vistos en entrenamiento y sus métricas son una estimación **optimista** (in-sample).

Para una estimación honesta de la generalización, ejecuta:

```bash
python3 src/evaluar_por_archivo_cv.py        # GroupKFold por archivo_base (out-of-fold)
```

Cada archivo se puntúa con un modelo que **no** vio su audio base. Comparar `reports/evaluacion_por_archivo_cv.csv` con la evaluación in-sample permite cuantificar el optimismo de esta última. Este es el procedimiento citado en el apartado de limitaciones de la memoria.

## Pruebas

```bash
python3 tests/test_posproceso.py      # runner mínimo, sin dependencias extra
# o, si tienes pytest:  python3 -m pytest tests/ -q
```

## Reproducibilidad (Makefile)

```bash
make help     # lista los objetivos disponibles
make all      # dataset -> features -> labels -> train -> eval -> summary
make cv       # evaluación sin fuga de datos
make test     # pruebas unitarias
```

## Instalación local

Se recomienda usar Python 3.10 o superior.

Instalar dependencias:

```bash
python3 -m pip install -r requirements.txt
```

## Ejecución local

Desde esta carpeta, ejecutar:

```bash
python3 -m streamlit run app_tfg.py
```

Después se abrirá la aplicación en el navegador.

## Uso

1. Seleccionar el modo de análisis en el panel lateral:
   - `Ejemplos precargados`.
   - `Subir audio propio`.
2. Seleccionar un ejemplo o subir un archivo de audio.
3. Ajustar el umbral de decisión si se desea.
4. Revisar la predicción global: audio limpio o sospechoso/manipulado.
5. Consultar el *tamper score* y los intervalos sospechosos.
6. Revisar la curva temporal y el espectrograma.
7. Descargar los resultados por ventana en CSV si se desea.

## Formatos de audio

Formatos recomendados: WAV, FLAC u OGG. También se permite subir MP3/M4A, aunque su correcta carga puede depender del backend de audio disponible en el entorno donde se ejecute la aplicación.

## Criterio de decisión

El umbral operativo del sistema es **0.50**, seleccionado en la memoria (análisis de umbrales, evitando el punto degenerado de umbrales bajos y atendiendo al equilibrio entre verdaderos y falsos positivos). Este umbral operativo se persiste como `best_threshold` dentro de los bundles joblib, de modo que la aplicación y cualquier script de inferencia usan por defecto el mismo criterio que la memoria; como referencia se conserva también `best_threshold_f1` (el mejor umbral por F1 en validación, más sensible pero con muchos más falsos positivos).

## Limitaciones

Esta aplicación es una prueba de concepto. El dataset empleado es reducido y las manipulaciones se han generado de forma controlada. Para audios subidos por el usuario no existe ground truth, por lo que la salida debe interpretarse como una ayuda de cribado y no como una herramienta forense definitiva.

## Robustez en audios subidos

La demo web admite WAV, MP3, M4A, OGG y FLAC. Para evitar errores de códec en Streamlit Community Cloud, los audios subidos se convierten temporalmente a WAV mono 16 kHz mediante FFmpeg cuando es necesario. Además, la versión web limita la subida a 20 MB y analiza como máximo los primeros 120 segundos para evitar agotar la memoria del servidor.

## Licencia

Este proyecto se distribuye bajo licencia MIT (archivo `LICENSE`). Desarrollado como parte del Trabajo de Fin de Estudios del Grado en Ingeniería Informática de UNIR.
