"""
Barrido de umbrales de decisión del modelo base por ventana (accuracy, precision, recall y F1)
para estudiar el compromiso entre detección y falsos positivos.
"""
from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

FEATURES_CSV = Path("data/processed/window_features.csv")
MODEL_PATH = Path("models/random_forest_base.joblib")
OUT_CSV = Path("reports/evaluacion_umbrales.csv")

payload = joblib.load(MODEL_PATH)
model = payload["model"]
feature_cols = payload["feature_cols"]

df = pd.read_csv(FEATURES_CSV, sep=";")

X = df[feature_cols]
y = df["etiqueta"]
groups = df["archivo_base"]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=42
)

_, test_idx = next(splitter.split(X, y, groups=groups))

X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]

proba = model.predict_proba(X_test)[:, 1]

umbrales = [0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05]

rows = []

for umbral in umbrales:
    y_pred = (proba >= umbral).astype(int)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()

    rows.append({
        "umbral": umbral,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "positivos_predichos": int(y_pred.sum())
    })

out = pd.DataFrame(rows)
out.to_csv(OUT_CSV, index=False, sep=";")

print(out)
print()
print(f"Guardado en: {OUT_CSV}")
print()
print("Resumen probabilidades clase sospechosa:")
print(pd.Series(proba).describe())
print()
print("Probabilidades máximas:")
print(sorted(proba, reverse=True)[:20])
