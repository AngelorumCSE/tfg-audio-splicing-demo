# Detector de manipulacion por splicing en audio

Aplicacion de demostracion desarrollada para el TFG: Verificacion forense de grabaciones de audio mediante IA: deteccion y localizacion de manipulaciones.

La aplicacion permite seleccionar un audio del dataset generado, reproducirlo, visualizar su curva temporal de sospecha, mostrar el espectrograma y consultar los intervalos detectados como sospechosos por el modelo.

## Contenido de la carpeta

- app_tfg.py: aplicacion principal desarrollada con Streamlit.
- requirements.txt: librerias necesarias para ejecutar la aplicacion.
- models/random_forest_borde.joblib: modelo entrenado de deteccion de bordes/manipulacion.
- data/generated/: audios generados para la prueba.
- data/processed/window_features_borde.csv: caracteristicas por ventana utilizadas por la app.
- data/manifests/splicing_manifest.csv: manifest con la informacion de los audios y el ground truth.

## Instalacion

Se recomienda usar Python 3.10 o superior.

Para instalar las dependencias:

python3 -m pip install -r requirements.txt

## Ejecucion

Desde esta misma carpeta, ejecutar:

python3 -m streamlit run app_tfg.py

Despues se abrira la aplicacion en el navegador.

## Uso

1. Seleccionar un audio en el panel lateral.
2. Ajustar el umbral de decision si se desea.
3. Revisar la prediccion global: audio limpio o sospechoso/manipulado.
4. Consultar el tamper score.
5. Revisar la curva temporal y el espectrograma.
6. Comprobar si los intervalos predichos coinciden con el ground truth cuando el audio es manipulado.

## Criterio de decision

El umbral principal usado en la evaluacion final es 0.50. Este umbral se selecciono porque ofrece un equilibrio razonable entre precision y recall.

## Limitaciones

Esta aplicacion es una prueba de concepto. El dataset empleado es reducido y las manipulaciones se han generado de forma controlada. Por tanto, los resultados no deben interpretarse como una herramienta forense definitiva, sino como una demostracion funcional del metodo desarrollado.
