# Análisis avanzado (rigor científico y robustez)

Estos scripts amplían el TFG con los análisis que más refuerzan una evaluación
sólida: validación sin fuga de datos, curvas ROC/PR, ablación, comparativa de
modelos, robustez y explicabilidad, además de validación con datos externos.

Todos se ejecutan **desde la carpeta `Codigo_y_Resultados`** y reutilizan
`src/config.py` y `src/posproceso.py`. Requieren las dependencias de
`requirements.txt` (scikit-learn, librosa, matplotlib…); `shap` es opcional.

Las salidas se guardan en `reports/avanzado/`.

## Scripts y orden sugerido

| Script | Qué hace | Salidas principales |
|---|---|---|
| `01_validacion_cv.py` | Validación cruzada agrupada (out-of-fold) + ROC/PR y AUC a nivel de ventana y de archivo, con IC95 % por bootstrap. | `validacion_cv_resumen.txt`, `roc_pr_ventana.png`, `roc_pr_archivo.png`, `metricas_por_umbral_cv.csv` |
| `02_ablacion_baselines.py` | Ablación con/sin características delta y comparativa RF vs LogReg, SVM, HistGradientBoosting (mismo protocolo honesto). | `ablacion.csv`, `comparativa_modelos.csv`, `resumen_ablacion_baselines.txt` |
| `03_robustez.py` | Degradación frente a recompresión MP3, ruido (SNR) y remuestreo. | `robustez.csv`, `robustez.png` |
| `04_explicabilidad.py` | Importancia por permutación (sobre prueba) agregada por familia y tipo; SHAP opcional. | `importancia_permutacion.csv`, `importancia_por_familia.png`, `explicabilidad_resumen.txt` |
| `05_validacion_externa.py` | Generalización con audios independientes (hablantes no vistos). Requiere un corpus externo. | `validacion_externa_detalle.csv`, `validacion_externa_resumen.txt` |
| `06_diagnostico_por_tipo.py` | Desglose por tipo de empalme del experimento intra-fuente (`mismo_audio` vs `mismo_hablante`), out-of-fold. | `diagnostico_por_tipo.csv`, `diagnostico_por_tipo.txt` |
| `07_multisemilla_libri.py` | Variabilidad del detector cross-source entre 5 semillas (media ± desviación típica de ROC-AUC y PR-AUC). | `multisemilla_libri.csv`, `multisemilla_libri.txt` |
| `08_desglose_y_ablacion_cross.py` | Sobre el conjunto cross-source (LibriSpeech): desglose de detección/localización por tipo en los umbrales operativos y ablación/comparativa de clasificadores, por hablante. | `desglose_cross_por_tipo.csv/.txt`, `ablacion_baselines_cross.csv/.txt` |

```bash
cd Codigo_y_Resultados
python3 analisis_avanzado/01_validacion_cv.py
python3 analisis_avanzado/02_ablacion_baselines.py
python3 analisis_avanzado/03_robustez.py
python3 analisis_avanzado/04_explicabilidad.py
# requiere preparar un corpus externo (ver cabecera del script):
python3 analisis_avanzado/05_validacion_externa.py --dir data/externo
python3 analisis_avanzado/06_diagnostico_por_tipo.py
python3 analisis_avanzado/07_multisemilla_libri.py   # varianza entre semillas (cross-source)
python3 analisis_avanzado/08_desglose_y_ablacion_cross.py   # desglose y ablación cross-source
```

Los scripts `03` y `04` admiten `--sufijo _libri` para la reejecución sobre el
detector cross-source; los artefactos entregados en `reports/avanzado/` con sufijo
`_libri` (robustez, importancias, SHAP) proceden de esa reejecución.

## Correspondencia con la memoria

| Análisis | Sección de la memoria | Artefactos |
|---|---|---|
| Validación sin fuga de datos (`01`) | §6.7 y Tabla 8 | `validacion_cv_resumen.txt`, `roc_pr_*.png` |
| Ablación y líneas base intra-fuente (`02`) | §6.9, Tabla 9 | `ablacion.csv`, `comparativa_modelos.csv` |
| Robustez (`03`, con `--sufijo _libri`) | §6.10, Figura 15 | `robustez_libri.csv/.png` |
| Explicabilidad (`04`, con `--sufijo _libri`) | §6.10, Figuras 16–17 | `importancia_*_libri.*`, `shap_summary_libri.png` |
| Diagnóstico por tipo intra-fuente (`06`) | §6.7–6.8 (motivación de la reformulación) | `diagnostico_por_tipo.*` |
| Variabilidad multisemilla (`07`) | §6.8 (0,71 ± 0,02 / 0,89 ± 0,01) | `multisemilla_libri.*` |
| Desglose y ablación cross-source (`08`) | §6.8 (localización por tipo, recall solo-cross) y §6.9 (párrafo final) | `desglose_cross_por_tipo.*`, `ablacion_baselines_cross.*` |

> Nota: estos scripts no modifican el modelo entregado ni las cifras de la memoria;
> producen la evidencia adicional de `reports/avanzado/`.
