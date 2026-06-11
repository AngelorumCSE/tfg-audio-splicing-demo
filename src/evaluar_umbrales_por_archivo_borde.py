from pathlib import Path
import csv
import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

FEATURES_CSV = Path("data/processed/window_features_borde.csv")
MANIFEST_CSV = Path("data/manifests/splicing_manifest.csv")
MODEL_PATH = Path("models/random_forest_borde.joblib")

OUT_CSV = Path("reports/evaluacion_umbrales_por_archivo_borde.csv")
OUT_TXT = Path("reports/resumen_umbrales_por_archivo_borde.txt")

THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]


def leer_manifest():
    with open(MANIFEST_CSV, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        return {r["archivo_generado"]: r for r in reader}


def cargar_modelo_y_columnas():
    objeto = joblib.load(MODEL_PATH)

    if isinstance(objeto, dict):
        modelo = objeto.get("model")
        feature_cols = objeto.get("feature_cols")
        if modelo is None:
            raise ValueError("No se ha encontrado la clave 'model' dentro del joblib.")
        return modelo, feature_cols

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


def agrupar_intervalos(df_audio, threshold, max_gap_s=0.75, min_duration_s=0.50):
    positivos = df_audio[df_audio["score_sospecha"] >= threshold].copy()

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

        if ini <= fin_actual + max_gap_s:
            fin_actual = max(fin_actual, fin)
            score_max = max(score_max, score)
        else:
            if fin_actual - inicio_actual >= min_duration_s:
                intervalos.append((inicio_actual, fin_actual, score_max))

            inicio_actual = ini
            fin_actual = fin
            score_max = score

    if fin_actual - inicio_actual >= min_duration_s:
        intervalos.append((inicio_actual, fin_actual, score_max))

    return intervalos


def hay_solape(intervalo_pred, inicio_gt, fin_gt):
    ini_pred, fin_pred, _ = intervalo_pred
    return max(ini_pred, inicio_gt) <= min(fin_pred, fin_gt)


def evaluar_threshold(df, manifest, threshold):
    filas = []

    for archivo, df_audio in df.groupby("archivo_generado"):
        df_audio = df_audio.sort_values("indice_ventana").copy()
        meta = manifest.get(archivo, {})

        manipulado_real = int(meta.get("manipulado", 0))
        tipo_splicing = meta.get("tipo_splicing", "desconocido")

        tamper_score = float(df_audio["score_sospecha"].max())
        intervalos = agrupar_intervalos(df_audio, threshold)

        detectado_archivo = 1 if intervalos else 0

        acierto_localizacion = ""

        if manipulado_real == 1:
            inicio_gt = float(meta["inicio_insercion_s"])
            fin_gt = float(meta["fin_insercion_s"])

            acierto_localizacion = 1 if any(
                hay_solape(intervalo, inicio_gt, fin_gt)
                for intervalo in intervalos
            ) else 0

        filas.append({
            "threshold": threshold,
            "archivo_generado": archivo,
            "tipo_splicing": tipo_splicing,
            "manipulado_real": manipulado_real,
            "detectado_archivo": detectado_archivo,
            "tamper_score": tamper_score,
            "num_intervalos_predichos": len(intervalos),
            "acierto_localizacion": acierto_localizacion,
        })

    return pd.DataFrame(filas)


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(FEATURES_CSV, sep=";")
    manifest = leer_manifest()
    modelo, feature_cols_guardadas = cargar_modelo_y_columnas()
    feature_cols = obtener_columnas_caracteristicas(df, modelo, feature_cols_guardadas)

    X = df[feature_cols].copy()
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    df["score_sospecha"] = modelo.predict_proba(X)[:, 1]

    all_results = []
    resumen_rows = []

    for threshold in THRESHOLDS:
        resultados = evaluar_threshold(df, manifest, threshold)
        all_results.append(resultados)

        y_true = resultados["manipulado_real"].astype(int)
        y_pred = resultados["detectado_archivo"].astype(int)

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        manipulados = resultados[resultados["manipulado_real"] == 1].copy()
        localizados = manipulados["acierto_localizacion"].replace("", 0).astype(int).sum()
        total_manipulados = len(manipulados)

        resumen_rows.append({
            "threshold": threshold,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "localizados": localizados,
            "total_manipulados": total_manipulados,
            "tasa_localizacion": localizados / total_manipulados if total_manipulados else 0,
        })

    detalle = pd.concat(all_results, ignore_index=True)
    resumen = pd.DataFrame(resumen_rows)

    detalle.to_csv(OUT_CSV, index=False, sep=";")

    texto = []
    texto.append("Evaluación de umbrales por archivo")
    texto.append("=" * 60)
    texto.append("")
    texto.append(resumen.to_string(index=False))

    OUT_TXT.write_text("\n".join(texto), encoding="utf-8")

    print(resumen.to_string(index=False))
    print()
    print(f"Detalle guardado en: {OUT_CSV}")
    print(f"Resumen guardado en: {OUT_TXT}")


if __name__ == "__main__":
    main()
