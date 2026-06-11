from pathlib import Path
import numpy as np
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
    classification_report,
)

DATA_IN = Path("data/processed/window_features_borde.csv")
WINDOW_MODEL_IN = Path("models/random_forest_borde.joblib")

OUT_FEATURES = Path("data/processed/file_level_features.csv")
OUT_MODEL = Path("models/random_forest_archivo.joblib")
OUT_REPORT = Path("reports/evaluacion_modelo_archivo.txt")
OUT_THRESHOLDS = Path("reports/evaluacion_umbrales_modelo_archivo.csv")
OUT_IMPORTANCES = Path("reports/importancia_caracteristicas_archivo.csv")

FEATURE_THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]
EVAL_THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]

RANDOM_STATE = 42


def cargar_modelo_ventanas(path):
    obj = joblib.load(path)

    if isinstance(obj, dict):
        print("Modelo de ventanas cargado como diccionario.")
        print(f"Claves disponibles: {list(obj.keys())}")
        model = obj["model"]
        feature_cols = obj["feature_cols"]
    else:
        print("Modelo de ventanas cargado directamente.")
        model = obj
        feature_cols = None

    return model, feature_cols


def longitudes_tramos(mask):
    lengths = []
    actual = 0

    for value in mask:
        if value:
            actual += 1
        else:
            if actual > 0:
                lengths.append(actual)
            actual = 0

    if actual > 0:
        lengths.append(actual)

    return lengths


def media_top(scores, k):
    if len(scores) == 0:
        return 0.0

    k = min(k, len(scores))
    return float(np.mean(np.sort(scores)[-k:]))


def construir_dataset_por_archivo(df):
    rows = []

    for archivo, g in df.groupby("archivo_generado", sort=False):
        g = g.sort_values("inicio_ventana_s").copy()
        scores = g["score_borde"].to_numpy(dtype=float)

        if len(scores) > 1:
            diffs = np.abs(np.diff(scores))
        else:
            diffs = np.array([0.0])

        first = g.iloc[0]

        row = {
            "archivo_generado": archivo,
            "archivo_base": first.get("archivo_base", ""),
            "id_hablante_base": first.get("id_hablante_base", ""),
            "genero_base": first.get("genero_base", ""),
            "tipo_splicing": first.get("tipo_splicing", ""),
            "manipulado_real": int(g["manipulado"].max()),
            "n_ventanas": int(len(scores)),
            "duracion_s": float(g["fin_ventana_s"].max()),
            "score_max": float(np.max(scores)),
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
            "score_min": float(np.min(scores)),
            "score_median": float(np.median(scores)),
            "score_p75": float(np.percentile(scores, 75)),
            "score_p90": float(np.percentile(scores, 90)),
            "score_p95": float(np.percentile(scores, 95)),
            "score_p99": float(np.percentile(scores, 99)),
            "score_top3_mean": media_top(scores, 3),
            "score_top5_mean": media_top(scores, 5),
            "score_top10_mean": media_top(scores, 10),
            "delta_score_mean": float(np.mean(diffs)),
            "delta_score_std": float(np.std(diffs)),
            "delta_score_max": float(np.max(diffs)),
            "delta_score_p90": float(np.percentile(diffs, 90)),
            "delta_score_p95": float(np.percentile(diffs, 95)),
        }

        for thr in FEATURE_THRESHOLDS:
            key = int(thr * 100)
            mask = scores >= thr
            lengths = longitudes_tramos(mask)

            row[f"n_ventanas_ge_{key}"] = int(mask.sum())
            row[f"prop_ventanas_ge_{key}"] = float(mask.mean())
            row[f"n_tramos_ge_{key}"] = int(len(lengths))
            row[f"max_tramo_ventanas_ge_{key}"] = int(max(lengths) if lengths else 0)

        rows.append(row)

    return pd.DataFrame(rows)


def evaluar_por_umbral(y_true, proba):
    rows = []

    for thr in EVAL_THRESHOLDS:
        y_pred = (proba >= thr).astype(int)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        rows.append({
            "threshold": thr,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        })

    return pd.DataFrame(rows)


def main():
    OUT_FEATURES.parent.mkdir(parents=True, exist_ok=True)
    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)

    print("Cargando datos de ventanas...")
    df = pd.read_csv(DATA_IN, sep=";")

    print("Cargando modelo de ventanas/bordes...")
    window_model, window_feature_cols = cargar_modelo_ventanas(WINDOW_MODEL_IN)

    if window_feature_cols is None:
        meta_cols = [
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
            "centro_ventana_s",
            "etiqueta_borde",
        ]
        window_feature_cols = [c for c in df.columns if c not in meta_cols]

    missing = [c for c in window_feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas para el modelo de ventanas: {missing}")

    print(f"Columnas usadas por el modelo de ventanas: {len(window_feature_cols)}")

    X_window = df[window_feature_cols].fillna(0)
    df["score_borde"] = window_model.predict_proba(X_window)[:, 1]

    print("Construyendo dataset por archivo...")
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

    print(f"Archivos totales: {len(df_file)}")
    print(f"Archivos limpios: {(y == 0).sum()}")
    print(f"Archivos manipulados: {(y == 1).sum()}")
    print(f"Características por archivo: {len(feature_cols)}")

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

    train_files = df_file.iloc[train_idx]["archivo_base"].unique()
    test_files = df_file.iloc[test_idx]["archivo_base"].unique()

    model_eval = RandomForestClassifier(
        n_estimators=500,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        min_samples_leaf=1,
        max_features="sqrt",
    )

    print("Entrenando modelo por archivo para evaluación...")
    model_eval.fit(X_train, y_train)

    proba_test = model_eval.predict_proba(X_test)[:, 1]

    thresholds_df = evaluar_por_umbral(y_test, proba_test)
    thresholds_df.to_csv(OUT_THRESHOLDS, index=False, sep=";")

    best_row = thresholds_df.sort_values(
        ["f1", "accuracy", "precision", "recall"],
        ascending=False
    ).iloc[0]

    best_threshold = float(best_row["threshold"])
    y_pred_best = (proba_test >= best_threshold).astype(int)

    acc = accuracy_score(y_test, y_pred_best)
    prec = precision_score(y_test, y_pred_best, zero_division=0)
    rec = recall_score(y_test, y_pred_best, zero_division=0)
    f1 = f1_score(y_test, y_pred_best, zero_division=0)
    cm = confusion_matrix(y_test, y_pred_best, labels=[0, 1])

    print("Entrenando modelo final por archivo con todos los datos...")
    final_model = RandomForestClassifier(
        n_estimators=500,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        min_samples_leaf=1,
        max_features="sqrt",
    )

    final_model.fit(X, y)

    df_file["score_modelo_archivo"] = final_model.predict_proba(X)[:, 1]
    df_file["prediccion_modelo_archivo"] = (
        df_file["score_modelo_archivo"] >= best_threshold
    ).astype(int)

    df_file.to_csv(OUT_FEATURES, index=False, sep=";")

    bundle = {
        "model": final_model,
        "feature_cols": feature_cols,
        "threshold": best_threshold,
        "source_window_model": str(WINDOW_MODEL_IN),
        "window_feature_cols": window_feature_cols,
        "target": "manipulado_real",
    }

    joblib.dump(bundle, OUT_MODEL)

    importances = pd.DataFrame({
        "caracteristica": feature_cols,
        "importancia": final_model.feature_importances_,
    }).sort_values("importancia", ascending=False)

    importances.to_csv(OUT_IMPORTANCES, index=False, sep=";")

    report = []
    report.append("Evaluación del modelo por archivo")
    report.append("=" * 60)
    report.append("")
    report.append("Objetivo:")
    report.append("Clasificar el audio completo como limpio o manipulado a partir de")
    report.append("estadísticas agregadas de los scores temporales del modelo de bordes.")
    report.append("")
    report.append(f"Archivos totales: {len(df_file)}")
    report.append(f"Archivos limpios: {(y == 0).sum()}")
    report.append(f"Archivos manipulados: {(y == 1).sum()}")
    report.append(f"Características por archivo: {len(feature_cols)}")
    report.append("")
    report.append("División de evaluación:")
    report.append(f"Archivos base en entrenamiento: {len(train_files)}")
    for item in sorted(train_files):
        report.append(f"- {item}")
    report.append("")
    report.append(f"Archivos base en prueba: {len(test_files)}")
    for item in sorted(test_files):
        report.append(f"- {item}")
    report.append("")
    report.append("Evaluación por umbral:")
    report.append(thresholds_df.to_string(index=False))
    report.append("")
    report.append(f"Mejor umbral según F1-score: {best_threshold}")
    report.append("")
    report.append("Métricas con el mejor umbral:")
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
    report.append(classification_report(y_test, y_pred_best, zero_division=0))
    report.append("")
    report.append("Archivos generados:")
    report.append(f"- {OUT_FEATURES}")
    report.append(f"- {OUT_MODEL}")
    report.append(f"- {OUT_THRESHOLDS}")
    report.append(f"- {OUT_IMPORTANCES}")

    OUT_REPORT.write_text("\n".join(report), encoding="utf-8")

    print()
    print("Modelo por archivo entrenado correctamente.")
    print(f"Modelo guardado en: {OUT_MODEL}")
    print(f"Dataset por archivo guardado en: {OUT_FEATURES}")
    print(f"Informe guardado en: {OUT_REPORT}")
    print(f"Umbrales guardados en: {OUT_THRESHOLDS}")
    print(f"Importancias guardadas en: {OUT_IMPORTANCES}")
    print()
    print("Métricas principales:")
    print(f"Mejor umbral: {best_threshold}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print()
    print("Matriz de confusión:")
    print(cm)


if __name__ == "__main__":
    main()
