# Proyecto desarrollado como parte del Trabajo de Fin de Estudios del Grado en
# Ingeniería Informática de UNIR (2026). Autor: Ángel Carlos Soler Encinas. Licencia MIT.
"""
Barrido fino de umbrales a nivel de archivo, reutilizando el modelo de ventanas para localizar
el mejor punto de operación.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

sys.path.append("src")

from entrenar_modelo_archivo import (
    cargar_modelo_ventanas,
    construir_dataset_por_archivo,
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

DATA_IN = Path("data/processed/window_features_borde.csv")
WINDOW_MODEL_IN = Path("models/random_forest_borde.joblib")

OUT_CSV = Path("reports/evaluacion_umbrales_finos_modelo_archivo.csv")
OUT_TXT = Path("reports/resumen_umbrales_finos_modelo_archivo.txt")

RANDOM_STATE = 42

df = pd.read_csv(DATA_IN, sep=";")

window_model, window_feature_cols = cargar_modelo_ventanas(WINDOW_MODEL_IN)

X_window = df[window_feature_cols].fillna(0)
df["score_borde"] = window_model.predict_proba(X_window)[:, 1]

df_file = construir_dataset_por_archivo(df)

meta_cols_file = [
    "archivo_generado",
    "archivo_base",
    "id_hablante_base",
    "genero_base",
    "tipo_splicing",
    "manipulado_real",
]

feature_cols = [c for c in df_file.columns if c not in meta_cols_file]

X = df_file[feature_cols].fillna(0)
y = df_file["manipulado_real"].astype(int)
groups = df_file["archivo_base"]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.30,
    random_state=RANDOM_STATE
)

train_idx, test_idx = next(splitter.split(X, y, groups))

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

model = RandomForestClassifier(
    n_estimators=500,
    random_state=RANDOM_STATE,
    class_weight="balanced",
    min_samples_leaf=1,
    max_features="sqrt",
)

model.fit(X_train, y_train)

proba = model.predict_proba(X_test)[:, 1]

rows = []

for thr in np.arange(0.01, 0.76, 0.01):
    y_pred = (proba >= thr).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    rows.append({
        "threshold": round(float(thr), 2),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    })

out = pd.DataFrame(rows)
out.to_csv(OUT_CSV, index=False, sep=";")

best_f1 = out.sort_values(["f1", "accuracy", "precision"], ascending=False).head(10)
sin_fp = out[out["fp"] == 0].sort_values(["recall", "f1", "accuracy"], ascending=False).head(10)

texto = []
texto.append("Evaluación fina de umbrales del modelo por archivo")
texto.append("=" * 70)
texto.append("")
texto.append("Top 10 umbrales por F1-score:")
texto.append(best_f1.to_string(index=False))
texto.append("")
texto.append("Mejores umbrales sin falsos positivos:")
texto.append(sin_fp.to_string(index=False))
texto.append("")
texto.append("Resumen probabilidades del conjunto de prueba:")
texto.append(str(pd.Series(proba).describe()))
texto.append("")
texto.append("Probabilidades ordenadas de mayor a menor:")
texto.append(str(sorted([round(float(x), 4) for x in proba], reverse=True)))

OUT_TXT.write_text("\n".join(texto), encoding="utf-8")

print("\n".join(texto))
print()
print(f"CSV guardado en: {OUT_CSV}")
print(f"Resumen guardado en: {OUT_TXT}")
