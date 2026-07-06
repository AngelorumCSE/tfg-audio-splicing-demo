# Proyecto desarrollado como parte del Trabajo de Fin de Estudios del Grado en
# Ingeniería Informática de UNIR (2026). Autor: Ángel Carlos Soler Encinas. Licencia MIT.
"""
Variabilidad entre semillas del detector cross-source (LibriSpeech).

Repite la evaluación out-of-fold (GroupKFold de 5 pliegues POR HABLANTE, la misma
de reconstruccion/02) con varias semillas que controlan el submuestreo de
negativos y el Random Forest, y reporta media y desviación típica del ROC-AUC y
PR-AUC por archivo. GroupKFold es determinista, por lo que la variabilidad
observada procede íntegramente del modelo y del balanceo.

Uso:
    cd Codigo_y_Resultados
    python3 analisis_avanzado/07_multisemilla_libri.py
Salida:
    reports/avanzado/multisemilla_libri.csv / .txt
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
import config as C  # noqa: E402

FEATURES = Path("data/libri/window_features_libri.csv")
OUT_DIR = Path("reports/avanzado")
SEEDS = [42, 7, 13, 101, 2026]
N_FOLDS = 5

META = {"archivo_generado", "id_hablante_base", "tipo_splicing", "manipulado",
        "indice_ventana", "inicio_ventana_s", "fin_ventana_s", "centro_ventana_s",
        "etiqueta_borde", "inicio_insercion_s", "fin_insercion_s"}


def balancear(df_train: pd.DataFrame, seed: int) -> pd.DataFrame:
    pos = df_train[df_train["etiqueta_borde"] == 1]
    neg = df_train[df_train["etiqueta_borde"] == 0]
    n_neg = min(len(neg), len(pos) * C.RATIO_NEGATIVOS)
    neg_s = neg.sample(n=n_neg, random_state=seed)
    return pd.concat([pos, neg_s]).sample(frac=1, random_state=seed)


def evaluar_semilla(df: pd.DataFrame, feats: list, seed: int) -> tuple:
    scores = pd.Series(np.nan, index=df.index)
    gkf = GroupKFold(n_splits=N_FOLDS)
    for tr, te in gkf.split(df[feats], df["etiqueta_borde"], groups=df["id_hablante_base"]):
        bal = balancear(df.iloc[tr], seed)
        m = RandomForestClassifier(n_estimators=C.N_ESTIMATORS,
                                   min_samples_leaf=C.MIN_SAMPLES_LEAF,
                                   random_state=seed, n_jobs=-1)
        m.fit(bal[feats], bal["etiqueta_borde"])
        scores.iloc[te] = m.predict_proba(df.iloc[te][feats])[:, 1]
    df = df.assign(score_sospecha=scores.values)
    g = df.groupby("archivo_generado")
    files = pd.DataFrame({"tamper": g["score_sospecha"].max(),
                          "manipulado": g["manipulado"].first()})
    roc = roc_auc_score(files["manipulado"], files["tamper"])
    ap = average_precision_score(files["manipulado"], files["tamper"])
    return roc, ap


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FEATURES, sep=";")
    feats = [c for c in df.columns if c not in META and pd.api.types.is_numeric_dtype(df[c])]
    print(f"ventanas={len(df)} caracteristicas={len(feats)} archivos={df['archivo_generado'].nunique()}")
    filas = []
    for seed in SEEDS:
        roc, ap = evaluar_semilla(df, feats, seed)
        filas.append({"semilla": seed, "roc_auc_archivo": round(roc, 4), "pr_auc_archivo": round(ap, 4)})
        print(f"semilla {seed}: ROC-AUC={roc:.4f}  PR-AUC={ap:.4f}")
    res = pd.DataFrame(filas)
    res.to_csv(OUT_DIR / "multisemilla_libri.csv", index=False, sep=";")
    resumen = (
        "VARIABILIDAD ENTRE SEMILLAS — detector cross-source (GroupKFold por hablante)\n"
        + "=" * 70 + "\n\n"
        + res.to_string(index=False) + "\n\n"
        + f"ROC-AUC por archivo: media={res['roc_auc_archivo'].mean():.4f}  "
          f"desv. tipica={res['roc_auc_archivo'].std(ddof=1):.4f}\n"
        + f"PR-AUC por archivo:  media={res['pr_auc_archivo'].mean():.4f}  "
          f"desv. tipica={res['pr_auc_archivo'].std(ddof=1):.4f}\n\n"
        + "La particion (GroupKFold por hablante) es determinista; la variabilidad\n"
          "procede del submuestreo de negativos y del Random Forest.\n")
    (OUT_DIR / "multisemilla_libri.txt").write_text(resumen, encoding="utf-8")
    print("\n" + resumen)


if __name__ == "__main__":
    main()
