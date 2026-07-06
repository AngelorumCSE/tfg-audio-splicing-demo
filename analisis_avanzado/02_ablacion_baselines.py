# Proyecto desarrollado como parte del Trabajo de Fin de Estudios del Grado en
# Ingeniería Informática de UNIR (2026). Autor: Ángel Carlos Soler Encinas. Licencia MIT.
"""
Estudio de ablación (con/sin características delta) y comparativa de clasificadores.

Todo se evalúa con el MISMO protocolo honesto: GroupKFold por archivo_base,
puntuación out-of-fold, agregación por archivo (tamper score = máximo) y métricas
a nivel de archivo. Así se mide:

  1) ABLACIÓN: cuánto aporta la familia de características delta (discontinuidad),
     comparando "solo base" frente a "base + delta". Justifica empíricamente la
     decisión de diseño central del TFG.
  2) BASELINES: Random Forest frente a Regresión Logística, SVM (RBF) e
     Histogram Gradient Boosting, sobre el conjunto completo de características.

Uso:
    cd Codigo_y_Resultados
    python3 analisis_avanzado/02_ablacion_baselines.py --folds 5

Salidas (reports/avanzado/):
    ablacion.csv, comparativa_modelos.csv, resumen_ablacion_baselines.txt
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
)

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
import config as C
from posproceso import balancear_train

OUT_DIR = C.REPORTS_DIR / "avanzado"


def estimadores():
    """Factorías de modelos (cada llamada crea uno nuevo, no entrenado)."""
    return {
        "RandomForest": lambda: RandomForestClassifier(
            n_estimators=C.N_ESTIMATORS, min_samples_leaf=C.MIN_SAMPLES_LEAF,
            random_state=C.RANDOM_STATE, n_jobs=-1),
        "RegresiónLogística": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=C.RANDOM_STATE)),
        "SVM (RBF)": lambda: make_pipeline(
            StandardScaler(), SVC(kernel="rbf", probability=True, random_state=C.RANDOM_STATE)),
        "HistGradientBoosting": lambda: HistGradientBoostingClassifier(
            random_state=C.RANDOM_STATE),
    }


def oof_por_archivo(df, feature_cols, factory, n_folds):
    """Puntuación out-of-fold por ventana y agregación por archivo (máximo)."""
    score = pd.Series(np.nan, index=df.index)
    gkf = GroupKFold(n_splits=n_folds)
    for tr, te in gkf.split(df[feature_cols], df["etiqueta_borde"], groups=df["archivo_base"]):
        bal = balancear_train(df.iloc[tr])
        m = factory()
        m.fit(bal[feature_cols], bal["etiqueta_borde"])
        score.iloc[te] = m.predict_proba(df.iloc[te][feature_cols])[:, 1]
    df = df.assign(_score=score.values)
    tamper = df.groupby("archivo_generado")["_score"].max()
    return tamper


def metricas_archivo(tamper, y_file, thr):
    yp = (tamper.loc[y_file.index].values >= thr).astype(int)
    yt = y_file.values
    return {
        "accuracy": accuracy_score(yt, yp),
        "precision": precision_score(yt, yp, zero_division=0),
        "recall": recall_score(yt, yp, zero_division=0),
        "f1": f1_score(yt, yp, zero_division=0),
        "roc_auc": roc_auc_score(yt, tamper.loc[y_file.index].values),
        "pr_auc": average_precision_score(yt, tamper.loc[y_file.index].values),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(C.FEATURES_BORDE_CSV, sep=";").copy()
    all_feats = [c for c in df.columns if c not in C.META_COLS]
    base_feats = [c for c in all_feats if not c.startswith(("delta_prev_", "delta_next_"))]
    n_folds = min(args.folds, df["archivo_base"].nunique())

    with open(C.MANIFEST_CSV, newline="", encoding="utf-8-sig") as f:
        manifest = {r["archivo_generado"]: r for r in csv.DictReader(f, delimiter=";")}
    archivos = df["archivo_generado"].unique()
    y_file = pd.Series({a: int(manifest.get(a, {}).get("manipulado", 0)) for a in archivos})

    rf = estimadores()["RandomForest"]

    # 1) Ablación
    print("Ablación (RandomForest, GroupKFold)...")
    abl = []
    for nombre, feats in [("Solo base (36)", base_feats), ("Base + delta (108)", all_feats)]:
        tamper = oof_por_archivo(df, feats, rf, n_folds)
        m = metricas_archivo(tamper, y_file, C.DEFAULT_THRESHOLD)
        abl.append({"conjunto": nombre, "n_features": len(feats), **{k: round(v, 4) for k, v in m.items()}})
    df_abl = pd.DataFrame(abl); df_abl.to_csv(OUT_DIR / "ablacion.csv", index=False, sep=";")

    # 2) Baselines (conjunto completo)
    print("Comparativa de modelos (GroupKFold)...")
    comp = []
    for nombre, factory in estimadores().items():
        try:
            tamper = oof_por_archivo(df, all_feats, factory, n_folds)
            m = metricas_archivo(tamper, y_file, C.DEFAULT_THRESHOLD)
            comp.append({"modelo": nombre, **{k: round(v, 4) for k, v in m.items()}})
            print(f"  {nombre}: F1={m['f1']:.3f}  ROC-AUC={m['roc_auc']:.3f}")
        except Exception as exc:
            comp.append({"modelo": nombre, "error": str(exc)})
            print(f"  {nombre}: ERROR {exc}")
    df_comp = pd.DataFrame(comp); df_comp.to_csv(OUT_DIR / "comparativa_modelos.csv", index=False, sep=";")

    txt = [
        "ABLACIÓN Y COMPARATIVA DE MODELOS (out-of-fold, GroupKFold por archivo_base)",
        "=" * 78, "",
        f"Umbral de decisión: {C.DEFAULT_THRESHOLD} | Pliegues: {n_folds}", "",
        "1) Ablación de las características de discontinuidad (delta):",
        df_abl.to_string(index=False), "",
        "   La diferencia de F1/AUC entre 'solo base' y 'base + delta' cuantifica la",
        "   aportación de las deltas, núcleo de la hipótesis del trabajo.", "",
        "2) Comparativa de clasificadores (conjunto completo de 108 características):",
        df_comp.to_string(index=False), "",
        "   Todos los modelos se evalúan con el mismo protocolo honesto (out-of-fold).",
    ]
    (OUT_DIR / "resumen_ablacion_baselines.txt").write_text("\n".join(txt), encoding="utf-8")
    print("\n".join(txt))


if __name__ == "__main__":
    main()
