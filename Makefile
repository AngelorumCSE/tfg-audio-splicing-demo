# Orquestador del pipeline del TFG (reproducibilidad de extremo a extremo).
# Uso:  make all   |   make dataset features train eval cv test   |   make app
PY ?= python3

.PHONY: all dataset features labels train eval cv summary app test multiseed desglose-cross clean help

all: dataset features labels train eval summary ## Reproduce el pipeline completo

dataset: ## 1) Genera el dataset de splicing (WAV + manifiesto)
	$(PY) src/generar_dataset_splicing.py

features: ## 2) Extrae características por ventana y deltas
	$(PY) src/extraer_caracteristicas_ventanas.py
	$(PY) src/generar_features_delta.py

labels: ## 3) Genera la etiqueta de borde
	$(PY) src/generar_etiquetas_borde.py

train: ## 4) Entrena el modelo final (Random Forest de bordes)
	$(PY) src/entrenar_modelo_borde.py

eval: ## 5) Evaluación por archivo IN-SAMPLE (la reportada en la memoria)
	$(PY) src/evaluar_umbrales_por_archivo_borde.py

cv: ## 5b) Evaluación por archivo SIN fuga de datos (out-of-fold, recomendada)
	$(PY) src/evaluar_por_archivo_cv.py

summary: ## 6) Gráficos y resumen final
	$(PY) src/generar_resumen_final_resultados.py

app: ## Lanza la aplicación de demostración (Streamlit)
	$(PY) -m streamlit run app_tfg.py

test: ## Pruebas unitarias de la lógica de posprocesado (no requieren sklearn)
	$(PY) tests/test_posproceso.py

multiseed: ## Variabilidad entre semillas del detector cross-source (requiere data/libri)
	$(PY) analisis_avanzado/07_multisemilla_libri.py

desglose-cross: ## Desglose por tipo y ablación/comparativa cross-source (requiere data/libri)
	$(PY) analisis_avanzado/08_desglose_y_ablacion_cross.py

clean: ## Borra artefactos regenerables (no toca raw_wav ni código)
	rm -f reports/evaluacion_por_archivo_cv.csv reports/resumen_por_archivo_cv.txt

help: ## Muestra esta ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
