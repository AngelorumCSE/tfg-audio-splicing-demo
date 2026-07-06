# Proyecto desarrollado como parte del Trabajo de Fin de Estudios del Grado en
# Ingeniería Informática de UNIR (2026). Autor: Ángel Carlos Soler Encinas. Licencia MIT.
"""
Entrena el Random Forest base por ventana usando solo las características de la ventana actual
(sin diferencias delta). Sirve de línea base del prototipo.
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

FEATURES_CSV = Path("data/processed/window_features.csv")
MODEL_OUT = Path("models/random_forest_base.joblib")
REPORT_OUT = Path("reports/evaluacion_modelo_base.txt")
IMPORTANCES_OUT = Path("reports/importancia_caracteristicas.csv")

MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)

# Columnas que no son características numéricas del modelo
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
    "etiqueta",
    "solape_manipulacion_s",
]

def main():
    df = pd.read_csv(FEATURES_CSV, sep=";")

    feature_cols = [c for c in df.columns if c not in META_COLS]

    X = df[feature_cols]
    y = df["etiqueta"]

    # Para evitar fuga de información, se separa por archivo_base,
    # no por ventanas individuales.
    groups = df["archivo_base"]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=42
    )

    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    df_train = df.iloc[train_idx]
    df_test = df.iloc[test_idx]

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        max_depth=None,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    report = []
    report.append("Evaluación del modelo base Random Forest")
    report.append("=" * 50)
    report.append("")
    report.append(f"Filas totales: {len(df)}")
    report.append(f"Ventanas entrenamiento: {len(X_train)}")
    report.append(f"Ventanas prueba: {len(X_test)}")
    report.append("")
    report.append("Archivos base en entrenamiento:")
    for a in sorted(df_train["archivo_base"].unique()):
        report.append(f"- {a}")
    report.append("")
    report.append("Archivos base en prueba:")
    for a in sorted(df_test["archivo_base"].unique()):
        report.append(f"- {a}")

    report.append("")
    report.append("Distribución de etiquetas en entrenamiento:")
    report.append(str(y_train.value_counts().sort_index()))
    report.append("")
    report.append("Distribución de etiquetas en prueba:")
    report.append(str(y_test.value_counts().sort_index()))
    report.append("")

    report.append("Métricas:")
    report.append(f"Accuracy:  {acc:.4f}")
    report.append(f"Precision: {prec:.4f}")
    report.append(f"Recall:    {rec:.4f}")
    report.append(f"F1-score:  {f1:.4f}")
    report.append("")

    report.append("Matriz de confusión:")
    report.append("[[TN FP]")
    report.append(" [FN TP]]")
    report.append(str(cm))
    report.append("")

    report.append("Classification report:")
    report.append(classification_report(y_test, y_pred, zero_division=0))

    REPORT_OUT.write_text("\n".join(report), encoding="utf-8")

    joblib.dump(
        {
            "model": model,
            "feature_cols": feature_cols,
            "meta_cols": META_COLS,
        },
        MODEL_OUT
    )

    importances = pd.DataFrame({
        "caracteristica": feature_cols,
        "importancia": model.feature_importances_
    }).sort_values("importancia", ascending=False)

    importances.to_csv(IMPORTANCES_OUT, index=False, sep=";")

    print("Modelo entrenado correctamente.")
    print(f"Modelo guardado en: {MODEL_OUT}")
    print(f"Informe guardado en: {REPORT_OUT}")
    print(f"Importancia de características guardada en: {IMPORTANCES_OUT}")
    print()
    print("Métricas principales:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print()
    print("Matriz de confusión:")
    print(cm)

if __name__ == "__main__":
    main()
