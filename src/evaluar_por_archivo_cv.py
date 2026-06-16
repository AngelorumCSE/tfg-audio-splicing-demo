"""
Evaluación por archivo SIN fuga de datos (validación cruzada agrupada).

Motivación
----------
La evaluación por archivo original (evaluar_umbrales_por_archivo_borde.py) puntúa
los 63 archivos con un modelo entrenado sobre 15 de los 21 audios base. Como
consecuencia, ~45 de los 63 archivos evaluados pertenecen a audios base vistos en
entrenamiento, por lo que sus métricas son una estimación OPTIMISTA (in-sample)
de la generalización.

Este script corrige esa limitación: genera, para CADA archivo, una predicción
"fuera de muestra" (out-of-fold) mediante validación cruzada agrupada por
archivo_base (GroupKFold). En cada pliegue, el modelo se entrena con los audios
base de los pliegues restantes y se puntúan únicamente los archivos del pliegue
retenido, de modo que ningún archivo es puntuado por un modelo que haya visto su
audio base ni ninguna de sus variantes (limpia / splice).

El resto del pipeline (balanceo 1:3 solo en entrenamiento, agregación por archivo
con tamper score = máximo, agrupación de intervalos y criterio de localización por
solape) replica exactamente el del experimento original, de forma que la única
diferencia con la evaluación reportada en la memoria es la ausencia de fuga de
datos. Así, este script permite confirmar de forma honesta el comportamiento del
prototipo.

Uso
---
    python3 src/evaluar_por_archivo_cv.py            # 5 pliegues por defecto
    python3 src/evaluar_por_archivo_cv.py --folds 6  # nº de pliegues configurable

Salidas
-------
    reports/evaluacion_por_archivo_cv.csv   (detalle por archivo y umbral)
    reports/resumen_por_archivo_cv.txt      (resumen de métricas por umbral)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)

try:
    import config as C
    from posproceso import agrupar_intervalos, hay_solape, balancear_train
except ImportError:  # permite ejecutar el script como "python3 src/evaluar_por_archivo_cv.py"
    import sys
    sys.path.append(str(Path(__file__).parent))
    import config as C
    from posproceso import agrupar_intervalos, hay_solape, balancear_train

OUT_CSV = C.REPORTS_DIR / "evaluacion_por_archivo_cv.csv"
OUT_TXT = C.REPORTS_DIR / "resumen_por_archivo_cv.txt"


# --------------------------------------------------------------------------- #
# Núcleo: scores out-of-fold y evaluación por archivo.
# --------------------------------------------------------------------------- #
def scores_out_of_fold(df: pd.DataFrame, feature_cols, n_folds: int) -> pd.Series:
    """Devuelve, para cada ventana, una probabilidad de la clase positiva
    obtenida por un modelo que NO vio su archivo_base (out-of-fold)."""
    scores = pd.Series(np.nan, index=df.index)
    gkf = GroupKFold(n_splits=n_folds)
    groups = df["archivo_base"]

    for k, (tr_idx, te_idx) in enumerate(gkf.split(df[feature_cols], df["etiqueta_borde"], groups), start=1):
        df_tr = df.iloc[tr_idx]
        df_tr_bal = balancear_train(df_tr)
        model = RandomForestClassifier(
            n_estimators=C.N_ESTIMATORS,
            min_samples_leaf=C.MIN_SAMPLES_LEAF,
            random_state=C.RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(df_tr_bal[feature_cols], df_tr_bal["etiqueta_borde"])
        proba = model.predict_proba(df.iloc[te_idx][feature_cols])[:, 1]
        scores.iloc[te_idx] = proba
        n_base_test = df.iloc[te_idx]["archivo_base"].nunique()
        print(f"  Pliegue {k}/{n_folds}: {len(te_idx)} ventanas de prueba, {n_base_test} audios base.")
    return scores


def leer_manifest():
    with open(C.MANIFEST_CSV, newline="", encoding="utf-8-sig") as f:
        return {r["archivo_generado"]: r for r in csv.DictReader(f, delimiter=";")}


def evaluar_por_umbral(df: pd.DataFrame, manifest: dict, threshold: float):
    filas = []
    for archivo, df_audio in df.groupby("archivo_generado"):
        df_audio = df_audio.sort_values("indice_ventana")
        meta = manifest.get(archivo, {})
        manipulado = int(meta.get("manipulado", 0))
        intervalos = agrupar_intervalos(df_audio, threshold)
        detectado = 1 if intervalos else 0
        loc = ""
        if manipulado == 1:
            gi, gf = float(meta["inicio_insercion_s"]), float(meta["fin_insercion_s"])
            loc = 1 if any(hay_solape(iv, gi, gf) for iv in intervalos) else 0
        filas.append({
            "threshold": threshold, "archivo_generado": archivo,
            "tipo_splicing": meta.get("tipo_splicing", "desconocido"),
            "manipulado_real": manipulado, "detectado_archivo": detectado,
            "tamper_score": float(df_audio["score_sospecha"].max()),
            "num_intervalos_predichos": len(intervalos),
            "acierto_localizacion": loc,
        })
    return pd.DataFrame(filas)


def main():
    parser = argparse.ArgumentParser(description="Evaluación por archivo sin fuga de datos (GroupKFold).")
    parser.add_argument("--folds", type=int, default=5, help="Número de pliegues (por defecto 5).")
    args = parser.parse_args()

    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(C.FEATURES_BORDE_CSV, sep=";")
    feature_cols = [c for c in df.columns if c not in C.META_COLS]

    n_base = df["archivo_base"].nunique()
    n_folds = min(args.folds, n_base)
    print(f"Validación cruzada agrupada por archivo_base: {n_folds} pliegues sobre {n_base} audios base.")

    df = df.copy()
    df["score_sospecha"] = scores_out_of_fold(df, feature_cols, n_folds)

    manifest = leer_manifest()
    detalles, resumen = [], []
    for thr in C.THRESHOLDS:
        res = evaluar_por_umbral(df, manifest, thr)
        detalles.append(res)
        y_true = res["manipulado_real"].astype(int)
        y_pred = res["detectado_archivo"].astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        man = res[res["manipulado_real"] == 1]
        loc = man["acierto_localizacion"].replace("", 0).astype(int).sum()
        resumen.append({
            "threshold": thr,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "localizados": int(loc), "total_manipulados": int(len(man)),
            "tasa_localizacion": (loc / len(man)) if len(man) else 0.0,
        })

    pd.concat(detalles, ignore_index=True).to_csv(OUT_CSV, index=False, sep=";")
    resumen_df = pd.DataFrame(resumen)

    texto = [
        "Evaluación por archivo SIN fuga de datos (out-of-fold, GroupKFold por archivo_base)",
        "=" * 78,
        "",
        "Cada archivo se puntúa con un modelo que no vio su audio base. Comparar estas",
        "cifras con las del experimento in-sample (evaluacion_umbrales_por_archivo_borde.csv)",
        "permite cuantificar el optimismo de la evaluación in-sample.",
        "",
        resumen_df.to_string(index=False),
    ]
    OUT_TXT.write_text("\n".join(texto), encoding="utf-8")

    print()
    print(resumen_df.to_string(index=False))
    print()
    print(f"Detalle -> {OUT_CSV}")
    print(f"Resumen -> {OUT_TXT}")


if __name__ == "__main__":
    main()
