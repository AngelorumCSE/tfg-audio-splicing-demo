"""
Evalúa el modelo de borde a nivel de archivo: agrega los scores por ventana (tamper score =
máximo) y decide con un umbral fijo, reportando detección y localización.
"""
from pathlib import Path
import csv
import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

FEATURES_CSV = Path("data/processed/window_features_borde.csv")
MANIFEST_CSV = Path("data/manifests/splicing_manifest.csv")
MODEL_PATH = Path("models/random_forest_borde.joblib")

OUT_CSV = Path("reports/evaluacion_por_archivo_borde.csv")
OUT_TXT = Path("reports/resumen_evaluacion_por_archivo_borde.txt")

THRESHOLD = 0.30
MAX_GAP_S = 0.75
MIN_DURATION_S = 0.50


def leer_manifest():
    with open(MANIFEST_CSV, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        return {r["archivo_generado"]: r for r in reader}


def cargar_modelo_y_columnas():
    objeto = joblib.load(MODEL_PATH)

    if isinstance(objeto, dict):
        print("Modelo cargado como diccionario.")
        print("Claves disponibles:", list(objeto.keys()))

        modelo = (
            objeto.get("model")
            or objeto.get("modelo")
            or objeto.get("clf")
            or objeto.get("classifier")
        )

        feature_cols = (
            objeto.get("feature_cols")
            or objeto.get("features")
            or objeto.get("columnas")
            or objeto.get("columnas_caracteristicas")
        )

        if modelo is None:
            raise ValueError(
                "El archivo joblib es un diccionario, pero no se ha encontrado ninguna clave de modelo reconocida."
            )

        return modelo, feature_cols

    print("Modelo cargado directamente.")
    return objeto, None


def obtener_columnas_caracteristicas(df, modelo, feature_cols_guardadas):
    if feature_cols_guardadas is not None:
        return list(feature_cols_guardadas)

    if hasattr(modelo, "feature_names_in_"):
        return list(modelo.feature_names_in_)

    columnas_excluir = {
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
    }

    return [
        c for c in df.columns
        if c not in columnas_excluir and pd.api.types.is_numeric_dtype(df[c])
    ]


def agrupar_intervalos(df_audio):
    positivos = df_audio[df_audio["pred_ventana"] == 1].copy()

    if positivos.empty:
        return []

    intervalos = []

    inicio_actual = float(positivos.iloc[0]["inicio_ventana_s"])
    fin_actual = float(positivos.iloc[0]["fin_ventana_s"])
    score_max = float(positivos.iloc[0]["score_sospecha"])

    for _, row in positivos.iloc[1:].iterrows():
        ini = float(row["inicio_ventana_s"])
        fin = float(row["fin_ventana_s"])
        score = float(row["score_sospecha"])

        if ini <= fin_actual + MAX_GAP_S:
            fin_actual = max(fin_actual, fin)
            score_max = max(score_max, score)
        else:
            if fin_actual - inicio_actual >= MIN_DURATION_S:
                intervalos.append((inicio_actual, fin_actual, score_max))

            inicio_actual = ini
            fin_actual = fin
            score_max = score

    if fin_actual - inicio_actual >= MIN_DURATION_S:
        intervalos.append((inicio_actual, fin_actual, score_max))

    return intervalos


def hay_solape(intervalo_pred, inicio_gt, fin_gt):
    ini_pred, fin_pred, _ = intervalo_pred
    return max(ini_pred, inicio_gt) <= min(fin_pred, fin_gt)


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(FEATURES_CSV, sep=";")
    manifest = leer_manifest()
    modelo, feature_cols_guardadas = cargar_modelo_y_columnas()

    feature_cols = obtener_columnas_caracteristicas(df, modelo, feature_cols_guardadas)

    print(f"Columnas usadas por el modelo: {len(feature_cols)}")

    X = df[feature_cols].copy()
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    df["score_sospecha"] = modelo.predict_proba(X)[:, 1]
    df["pred_ventana"] = (df["score_sospecha"] >= THRESHOLD).astype(int)

    filas = []

    for archivo, df_audio in df.groupby("archivo_generado"):
        df_audio = df_audio.sort_values("indice_ventana").copy()
        meta = manifest.get(archivo, {})

        manipulado_real = int(meta.get("manipulado", 0))
        tipo_splicing = meta.get("tipo_splicing", "desconocido")
        tamper_score = float(df_audio["score_sospecha"].max())

        intervalos = agrupar_intervalos(df_audio)
        detectado_archivo = 1 if intervalos else 0

        inicio_gt = ""
        fin_gt = ""
        acierto_localizacion = ""

        if manipulado_real == 1:
            inicio_gt = float(meta["inicio_insercion_s"])
            fin_gt = float(meta["fin_insercion_s"])
            acierto_localizacion = 1 if any(
                hay_solape(intervalo, inicio_gt, fin_gt)
                for intervalo in intervalos
            ) else 0

        texto_intervalos = " | ".join(
            [f"{ini:.3f}-{fin:.3f} ({score:.3f})" for ini, fin, score in intervalos]
        )

        filas.append({
            "archivo_generado": archivo,
            "manipulado_real": manipulado_real,
            "tipo_splicing": tipo_splicing,
            "inicio_gt_s": inicio_gt,
            "fin_gt_s": fin_gt,
            "tamper_score": round(tamper_score, 4),
            "detectado_archivo": detectado_archivo,
            "num_intervalos_predichos": len(intervalos),
            "acierto_localizacion": acierto_localizacion,
            "intervalos_predichos": texto_intervalos
        })

    resultados = pd.DataFrame(filas)
    resultados.to_csv(OUT_CSV, index=False, sep=";")

    y_true = resultados["manipulado_real"].astype(int)
    y_pred = resultados["detectado_archivo"].astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    manipulados = resultados[resultados["manipulado_real"] == 1].copy()
    localizados = manipulados["acierto_localizacion"].astype(int).sum()
    total_manipulados = len(manipulados)

    resumen = []
    resumen.append("Evaluación por archivo del modelo de bordes")
    resumen.append("=" * 60)
    resumen.append("")
    resumen.append(f"Archivos evaluados: {len(resultados)}")
    resumen.append(f"Archivos limpios: {(resultados['manipulado_real'] == 0).sum()}")
    resumen.append(f"Archivos manipulados: {(resultados['manipulado_real'] == 1).sum()}")
    resumen.append(f"Umbral utilizado: {THRESHOLD}")
    resumen.append("")
    resumen.append("Métricas de detección por archivo:")
    resumen.append(f"Accuracy:  {acc:.4f}")
    resumen.append(f"Precision: {prec:.4f}")
    resumen.append(f"Recall:    {rec:.4f}")
    resumen.append(f"F1-score:  {f1:.4f}")
    resumen.append("")
    resumen.append("Matriz de confusión:")
    resumen.append("[[TN FP]")
    resumen.append(" [FN TP]]")
    resumen.append(str(cm))
    resumen.append("")
    resumen.append("Localización temporal:")
    resumen.append(f"Manipulados con intervalo localizado: {localizados}/{total_manipulados}")

    if total_manipulados > 0:
        resumen.append(f"Tasa de localización: {localizados / total_manipulados:.4f}")

    resumen.append("")
    resumen.append("Resultados por tipo de splicing:")
    resumen.append(str(resultados.groupby(["tipo_splicing", "manipulado_real", "detectado_archivo"]).size()))

    OUT_TXT.write_text("\n".join(resumen), encoding="utf-8")

    print()
    print("\n".join(resumen))
    print()
    print(f"CSV guardado en: {OUT_CSV}")
    print(f"Resumen guardado en: {OUT_TXT}")


if __name__ == "__main__":
    main()
