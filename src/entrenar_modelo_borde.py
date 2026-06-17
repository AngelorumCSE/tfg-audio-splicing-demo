"""
Entrena el Random Forest sobre la etiqueta de borde (características base + delta). Es el modelo
principal del prototipo intra-fuente, orientado a marcar las ventanas próximas a un empalme.
"""
from pathlib import Path
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

FEATURES_CSV = Path("data/processed/window_features_borde.csv")
MODEL_OUT = Path("models/random_forest_borde.joblib")
REPORT_OUT = Path("reports/evaluacion_modelo_borde.txt")
IMPORTANCES_OUT = Path("reports/importancia_caracteristicas_borde.csv")
THRESHOLDS_OUT = Path("reports/evaluacion_umbrales_borde.csv")

RANDOM_STATE = 42
RATIO_NEGATIVOS = 3

META_COLS = [
    "id_registro",
    "archivo_generado",
    "archivo_base",
    "id_hablante_base",
    "genero_base",
    "tipo_splicing",
    "manipulado",
    "indice_ventana",
    "inicio_ventana_s",
    "fin_ventana_s",
    "centro_ventana_s",
    "etiqueta",
    "etiqueta_borde",
    "solape_manipulacion_s",
]

MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)

def balancear_train(df_train):
    positivos = df_train[df_train["etiqueta_borde"] == 1]
    negativos = df_train[df_train["etiqueta_borde"] == 0]

    n_neg = min(len(negativos), len(positivos) * RATIO_NEGATIVOS)

    negativos_sample = negativos.sample(
        n=n_neg,
        random_state=RANDOM_STATE
    )

    df_bal = pd.concat([positivos, negativos_sample], axis=0)
    df_bal = df_bal.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    return df_bal

def evaluar_umbrales(y_true, proba):
    umbrales = [0.70, 0.60, 0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05]
    filas = []

    for umbral in umbrales:
        y_pred = (proba >= umbral).astype(int)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        filas.append({
            "umbral": umbral,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "positivos_predichos": int(y_pred.sum())
        })

    return pd.DataFrame(filas)

def main():
    df = pd.read_csv(FEATURES_CSV, sep=";")

    feature_cols = [c for c in df.columns if c not in META_COLS]

    X = df[feature_cols]
    y = df["etiqueta_borde"]
    groups = df["archivo_base"]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=RANDOM_STATE
    )

    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    df_train = df.iloc[train_idx].copy()
    df_test = df.iloc[test_idx].copy()

    df_train_bal = balancear_train(df_train)

    X_train = df_train_bal[feature_cols]
    y_train = df_train_bal["etiqueta_borde"]

    X_test = df_test[feature_cols]
    y_test = df_test["etiqueta_borde"]

    model = RandomForestClassifier(
        n_estimators=500,
        random_state=RANDOM_STATE,
        min_samples_leaf=2,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]

    df_umbrales = evaluar_umbrales(y_test, proba)
    df_umbrales.to_csv(THRESHOLDS_OUT, index=False, sep=";")

    best = df_umbrales.sort_values("f1", ascending=False).iloc[0]
    best_umbral = float(best["umbral"])

    y_pred = (proba >= best_umbral).astype(int)

    report = []
    report.append("Evaluación del modelo de detección de bordes de splicing")
    report.append("=" * 70)
    report.append("")
    report.append(f"Filas totales: {len(df)}")
    report.append(f"Ventanas entrenamiento originales: {len(df_train)}")
    report.append(f"Ventanas entrenamiento balanceadas: {len(df_train_bal)}")
    report.append(f"Ventanas prueba: {len(df_test)}")
    report.append("")
    report.append("Distribución entrenamiento original:")
    report.append(str(df_train["etiqueta_borde"].value_counts().sort_index()))
    report.append("")
    report.append("Distribución entrenamiento balanceado:")
    report.append(str(df_train_bal["etiqueta_borde"].value_counts().sort_index()))
    report.append("")
    report.append("Distribución prueba:")
    report.append(str(df_test["etiqueta_borde"].value_counts().sort_index()))
    report.append("")
    report.append("Evaluación por umbral:")
    report.append(df_umbrales.to_string(index=False))
    report.append("")
    report.append(f"Mejor umbral preliminar por F1: {best_umbral}")
    report.append("")
    report.append("Classification report:")
    report.append(classification_report(y_test, y_pred, zero_division=0))
    report.append("")
    report.append("Matriz de confusión:")
    report.append("[[TN FP]")
    report.append(" [FN TP]]")
    report.append(str(confusion_matrix(y_test, y_pred)))

    REPORT_OUT.write_text("\n".join(report), encoding="utf-8")

    joblib.dump(
        {
            "model": model,
            "feature_cols": feature_cols,
            "meta_cols": META_COLS,
            "best_threshold": best_umbral,
            "target": "etiqueta_borde",
        },
        MODEL_OUT
    )

    importances = pd.DataFrame({
        "caracteristica": feature_cols,
        "importancia": model.feature_importances_
    }).sort_values("importancia", ascending=False)

    importances.to_csv(IMPORTANCES_OUT, index=False, sep=";")

    print("Modelo de bordes entrenado correctamente.")
    print(f"Modelo guardado en: {MODEL_OUT}")
    print(f"Informe guardado en: {REPORT_OUT}")
    print(f"Umbrales guardados en: {THRESHOLDS_OUT}")
    print(f"Importancias guardadas en: {IMPORTANCES_OUT}")
    print()
    print("Evaluación por umbral:")
    print(df_umbrales)
    print()
    print(f"Mejor umbral por F1: {best_umbral}")
    print()
    print("Matriz de confusión con mejor umbral:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()
