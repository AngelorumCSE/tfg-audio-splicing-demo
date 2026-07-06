# Reconstrucción del experimento: splicing cross-source (LibriSpeech)

Esta carpeta reconstruye el experimento de forma que el detector tenga **señal real
que aprender**: empalmes en los que el fragmento insertado procede de una **fuente
distinta** (otro hablante y, opcionalmente, otro entorno), que es el caso forense
detectable. La evaluación es **por hablante** (se prueba con hablantes nunca vistos),
sin fuga de datos. Corresponde al apartado 6.8 de la memoria.

## Paso 1 · Descargar LibriSpeech dev-clean (~337 MB, ~40 hablantes, 16 kHz)

```bash
cd Codigo_y_Resultados
curl -L -O https://www.openslr.org/resources/12/dev-clean.tar.gz
tar xzf dev-clean.tar.gz        # crea ./LibriSpeech/dev-clean/<hablante>/<capítulo>/*.flac
```

> Este paso solo es necesario para **regenerar** el conjunto desde cero. La entrega ya
> incluye los 120 audios generados en `data/libri/audios/`, junto con su manifiesto y
> las características por ventana, por lo que el Paso 3 puede ejecutarse directamente.

## Paso 2 · Generar el dataset cross-source

```bash
python3 reconstruccion/01_generar_dataset_libri.py \
        --libri LibriSpeech/dev-clean --salida data/libri --n-host 30 --n-insert 10
```

Crea `data/libri/audios/*.wav` y `data/libri/manifest_libri.csv` con, por cada
hablante host: una versión limpia y tres empalmes (otro hablante, otro hablante con
ruido, y mismo hablante como caso difícil), con ground truth temporal.

> Si avisa de que faltan hablantes, reduce `--n-host`/`--n-insert` (deben sumar ≤ nº de
> hablantes del corpus; dev-clean tiene ~40).

## Paso 3 · Entrenar y evaluar (validación por hablante)

```bash
python3 reconstruccion/02_entrenar_evaluar_libri.py --datos data/libri --folds 5
```

Genera en `reconstruccion/reports/`:
- `resumen_libri.txt` — ROC-AUC por archivo, **desglose por tipo de empalme** y métricas por umbral.
- `roc_pr_libri.png` — curvas ROC y Precision-Recall.
- `por_tipo_libri.csv`, `metricas_por_umbral_libri.csv`.
- y el modelo `reconstruccion/modelo_libri.joblib`.

Estos artefactos son la fuente de las cifras del apartado 6.8 de la memoria (ROC-AUC
por archivo 0,722; PR-AUC 0,896; desglose por tipo 0,999/0,634/0,534; matriz y
localización por umbral). Los análisis complementarios sobre este mismo conjunto
(multisemilla, desglose en el umbral operativo y ablación/comparativa) están en
`analisis_avanzado/` (scripts 07 y 08).

## Papel de esta carpeta en el trabajo

El experimento original (empalmes intra-fuente) sirvió para descubrir, mediante
validación rigurosa, que la tarea así planteada no era detectable y que la validación
ingenua inflaba los resultados. A partir de ahí se reformuló el problema hacia el
empalme cross-source —el caso forense relevante— y se evaluó por hablante. Esta
iteración (hipótesis → validación crítica → reformulación → resultado honesto) es el
eje metodológico del trabajo, descrito en los apartados 6.7 y 6.8 de la memoria.
