# Detector de manipulación por splicing en audio

Aplicación de demostración desarrollada para el TFG: **Verificación forense de grabaciones de audio mediante IA: detección y localización de manipulaciones**.

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
- `data/generated/`: audios de ejemplo incluidos con la entrega reducida.
- `data/processed/window_features_borde.csv`: características por ventana de los ejemplos precargados.
- `data/manifests/splicing_manifest.csv`: manifest con la información de los ejemplos y el ground truth.
- `models/random_forest_borde.joblib`: modelo final usado por la aplicación.
- `reports/`: métricas y gráficos finales del experimento.

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

El umbral principal usado en la evaluación final es 0.50. Este umbral se seleccionó porque ofrece un equilibrio razonable entre precisión y recall en el conjunto de prueba generado.

## Limitaciones

Esta aplicación es una prueba de concepto. El dataset empleado es reducido y las manipulaciones se han generado de forma controlada. Para audios subidos por el usuario no existe ground truth, por lo que la salida debe interpretarse como una ayuda de cribado y no como una herramienta forense definitiva.
