# Proyecto desarrollado como parte del Trabajo de Fin de Estudios del Grado en
# Ingeniería Informática de UNIR (2026). Autor: Ángel Carlos Soler Encinas. Licencia MIT.
"""
Validación cruzada agrupada SIN fuga de datos + curvas ROC/PR y AUC.

Genera, mediante GroupKFold por archivo_base, una puntuación fuera de muestra
(out-of-fold) para cada ventana, de modo que ninguna ventana es puntuada por un
modelo que haya visto su audio base. A partir de esas puntuaciones calcula:

  - Métricas a nivel de VENTANA: ROC-AUC y PR-AUC (independientes del umbral).
  - Métricas a nivel de ARCHIVO: tamper score = máximo de las ventanas del audio;
    ROC-AUC y PR-AUC por archivo, métricas en el umbral 0,50 e intervalo de
    confianza al 95 % por bootstrap sobre los archivos.

Es la evaluación rigurosa que se recomienda para confirmar, de forma honesta, las
cifras in-sample reportadas en la memoria.

Uso:
    cd Codigo_y_Resultados
    python3 analisis_avanzado/01_validacion_cv.py --folds 5 --bootstrap 2000

Salidas (en reports/avanzado/):
    validacion_cv_resumen.txt, roc_pr_ventana.png, roc_pr_archivo.png,
    metricas_por_umbral_cv.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
import config as C
from posproceso import agrupar_intervalos, hay_solape, balancear_train

OUT_DIR = C.REPORTS_DIR / "avanzado"


def oof_scores(df, feature_cols, n_folds):
    s = pd.Series(np.nan, index=df.index)
    gkf = GroupKFold(n_splits=n_folds)
    for tr, te in gkf.split(df[feature_cols], df["etiqueta_borde"], groups=df["archivo_base"]):
        bal = balancear_train(df.iloc[tr])
        m = RandomForestClassifier(n_estimators=C.N_ESTIMATORS, min_samples_leaf=C.MIN_SAMPLES_LEAF,
                                   random_state=C.RANDOM_STATE, n_jobs=-1)
        m.fit(bal[feature_cols], bal["etiqueta_borde"])
        s.iloc[te] = m.predict_proba(df.iloc[te][feature_cols])[:, 1]
    return s.values


def curvas(y, score, titulo, path):
    fpr, tpr, _ = roc_curve(y, score)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y, score)
    ap = average_precision_score(y, score)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(fpr, tpr, color="#1F6FB2", label=f"ROC (AUC = {roc_auc:.3f})")
    ax[0].plot([0, 1], [0, 1], "--", color="#aaaaaa")
    ax[0].set_xlabel("Tasa de falsos positivos"); ax[0].set_ylabel("Tasa de verdaderos positivos")
    ax[0].set_title(f"ROC – {titulo}"); ax[0].legend(loc="lower right"); ax[0].grid(alpha=0.3)
    ax[1].plot(rec, prec, color="#B00020", label=f"PR (AP = {ap:.3f})")
    ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision")
    ax[1].set_title(f"Precision-Recall – {titulo}"); ax[1].legend(loc="lower left"); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)
    return roc_auc, ap


def bootstrap_ci(y_true, y_pred, metric, B, rng):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    n = len(y_true); vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        try:
            vals.append(metric(y_true[idx], y_pred[idx]))
        except Exception:
            pass
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(C.FEATURES_BORDE_CSV, sep=";").copy()
    feature_cols = [c for c in df.columns if c not in C.META_COLS]
    n_folds = min(args.folds, df["archivo_base"].nunique())
    print(f"GroupKFold por archivo_base con {n_folds} pliegues...")
    df["score_sospecha"] = oof_scores(df, feature_cols, n_folds)

    # --- nivel ventana ---
    roc_w, ap_w = curvas(df["etiqueta_borde"].values, df["score_sospecha"].values,
                         "ventana (out-of-fold)", OUT_DIR / "roc_pr_ventana.png")

    # --- nivel archivo ---
    with open(C.MANIFEST_CSV, newline="", encoding="utf-8-sig") as f:
        manifest = {r["archivo_generado"]: r for r in csv.DictReader(f, delimiter=";")}
    files = df.groupby("archivo_generado")
    tamper = files["score_sospecha"].max()
    y_file = pd.Series({a: int(manifest.get(a, {}).get("manipulado", 0)) for a in tamper.index})
    roc_f, ap_f = curvas(y_file.values, tamper.loc[y_file.index].values,
                         "archivo (tamper score OOF)", OUT_DIR / "roc_pr_archivo.png")

    # --- métricas por umbral (detección + localización), out-of-fold ---
    rng = np.random.default_rng(C.RANDOM_STATE)
    filas, resumen_050 = [], {}
    for thr in C.THRESHOLDS:
        recs = []
        for archivo, da in files:
            meta = manifest.get(archivo, {})
            manip = int(meta.get("manipulado", 0))
            ivs = agrupar_intervalos(da.sort_values("indice_ventana"), thr)
            det = 1 if ivs else 0
            loc = ""
            if manip == 1:
                gi, gf = float(meta["inicio_insercion_s"]), float(meta["fin_insercion_s"])
                loc = 1 if any(hay_solape(iv, gi, gf) for iv in ivs) else 0
            recs.append({"manip": manip, "det": det, "loc": loc})
        rr = pd.DataFrame(recs)
        yt, yp = rr["manip"].astype(int), rr["det"].astype(int)
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        man = rr[rr["manip"] == 1]; locd = man["loc"].replace("", 0).astype(int).sum()
        fila = {"umbral": thr, "accuracy": accuracy_score(yt, yp),
                "precision": precision_score(yt, yp, zero_division=0),
                "recall": recall_score(yt, yp, zero_division=0),
                "f1": f1_score(yt, yp, zero_division=0),
                "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                "localizacion": f"{locd}/{len(man)}"}
        filas.append(fila)
        if abs(thr - C.DEFAULT_THRESHOLD) < 1e-9:
            ci_f1 = bootstrap_ci(yt.values, yp.values, lambda a, b: f1_score(a, b, zero_division=0), args.bootstrap, rng)
            ci_acc = bootstrap_ci(yt.values, yp.values, accuracy_score, args.bootstrap, rng)
            resumen_050 = {**fila, "f1_ci95": ci_f1, "acc_ci95": ci_acc}
    df_umbral = pd.DataFrame(filas)
    df_umbral.to_csv(OUT_DIR / "metricas_por_umbral_cv.csv", index=False, sep=";")

    txt = [
        "VALIDACIÓN CRUZADA AGRUPADA (out-of-fold) — sin fuga de datos",
        "=" * 70, "",
        f"Pliegues: {n_folds} (GroupKFold por archivo_base)",
        "",
        "AUC (independiente del umbral):",
        f"  Ventana  -> ROC-AUC = {roc_w:.3f} | PR-AUC = {ap_w:.3f}",
        f"  Archivo  -> ROC-AUC = {roc_f:.3f} | PR-AUC = {ap_f:.3f}",
        "",
        f"Métricas por archivo en el umbral {C.DEFAULT_THRESHOLD} (out-of-fold):",
        f"  accuracy={resumen_050.get('accuracy', float('nan')):.4f}  "
        f"precision={resumen_050.get('precision', float('nan')):.4f}  "
        f"recall={resumen_050.get('recall', float('nan')):.4f}  "
        f"f1={resumen_050.get('f1', float('nan')):.4f}",
        f"  IC95% bootstrap -> F1 {resumen_050.get('f1_ci95')}  |  accuracy {resumen_050.get('acc_ci95')}",
        f"  matriz TN/FP/FN/TP = {resumen_050.get('tn')}/{resumen_050.get('fp')}/"
        f"{resumen_050.get('fn')}/{resumen_050.get('tp')}  |  localización {resumen_050.get('localizacion')}",
        "",
        "Tabla completa por umbral en metricas_por_umbral_cv.csv",
        "",
        "NOTA: compara estas cifras con la evaluación in-sample de la memoria. La",
        "diferencia (normalmente a la baja) cuantifica el optimismo del enfoque in-sample.",
    ]
    (OUT_DIR / "validacion_cv_resumen.txt").write_text("\n".join(txt), encoding="utf-8")
    print("\n".join(txt))
    print(f"\nFiguras y resumen en {OUT_DIR}")


if __name__ == "__main__":
    main()
