"""
Genera un informe visual por audio (curva temporal de sospecha, intervalos detectados y
espectrograma) usando el modelo de borde.
"""
from pathlib import Path
import argparse
import csv
import joblib
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

FEATURES_CSV = Path("data/processed/window_features_borde.csv")
MANIFEST_CSV = Path("data/manifests/splicing_manifest.csv")
AUDIO_DIR = Path("data/generated")
MODEL_PATH = Path("models/random_forest_borde.joblib")
OUT_BASE = Path("reports/predicciones")

def leer_manifest():
    with open(MANIFEST_CSV, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        return {r["archivo_generado"]: r for r in reader}

def agrupar_intervalos(df_audio, score_col, threshold, max_gap_s=0.75, min_duration_s=0.5):
    positivos = df_audio[df_audio[score_col] >= threshold].copy()

    if positivos.empty:
        return []

    intervalos = []

    inicio_actual = float(positivos.iloc[0]["inicio_ventana_s"])
    fin_actual = float(positivos.iloc[0]["fin_ventana_s"])
    score_max = float(positivos.iloc[0][score_col])

    for _, row in positivos.iloc[1:].iterrows():
        ini = float(row["inicio_ventana_s"])
        fin = float(row["fin_ventana_s"])
        score = float(row[score_col])

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Nombre del archivo WAV dentro de data/generated/")
    parser.add_argument("--threshold", type=float, default=None, help="Umbral de sospecha. Si no se indica, usa el mejor umbral del modelo.")
    args = parser.parse_args()

    audio_file = args.audio
    audio_path = AUDIO_DIR / audio_file

    if not audio_path.exists():
        raise FileNotFoundError(f"No existe el audio: {audio_path}")

    payload = joblib.load(MODEL_PATH)
    model = payload["model"]
    feature_cols = payload["feature_cols"]
    threshold = args.threshold if args.threshold is not None else payload.get("best_threshold", 0.3)

    df = pd.read_csv(FEATURES_CSV, sep=";")
    df_audio = df[df["archivo_generado"] == audio_file].copy()

    if df_audio.empty:
        raise ValueError(f"No hay features para el archivo: {audio_file}")

    df_audio = df_audio.sort_values("indice_ventana").reset_index(drop=True)

    X = df_audio[feature_cols]
    proba = model.predict_proba(X)[:, 1]

    df_audio["score_sospecha"] = proba
    df_audio["score_suavizado"] = (
        pd.Series(proba)
        .rolling(window=3, center=True, min_periods=1)
        .mean()
        .values
    )

    tamper_score = float(df_audio["score_suavizado"].max())

    manifest = leer_manifest()
    meta = manifest.get(audio_file, {})

    intervalos_pred = agrupar_intervalos(
        df_audio,
        score_col="score_suavizado",
        threshold=threshold
    )

    out_dir = OUT_BASE / Path(audio_file).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Guardar scores por ventana
    df_audio[[
        "archivo_generado",
        "indice_ventana",
        "inicio_ventana_s",
        "fin_ventana_s",
        "etiqueta",
        "etiqueta_borde",
        "score_sospecha",
        "score_suavizado"
    ]].to_csv(out_dir / "window_scores.csv", index=False, sep=";")

    # Figura 1: curva de sospecha
    plt.figure(figsize=(12, 4))
    plt.plot(df_audio["centro_ventana_s"], df_audio["score_suavizado"], label="Score suavizado")
    plt.axhline(threshold, linestyle="--", label=f"Umbral = {threshold}")

    if meta.get("manipulado") == "1":
        ini_gt = float(meta["inicio_insercion_s"])
        fin_gt = float(meta["fin_insercion_s"])
        plt.axvspan(ini_gt, fin_gt, alpha=0.2, label="Ground truth")

    for ini, fin, _ in intervalos_pred:
        plt.axvspan(ini, fin, alpha=0.15)

    plt.xlabel("Tiempo (s)")
    plt.ylabel("Score de sospecha")
    plt.title(f"Curva de sospecha - {audio_file}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "score_timeline.png", dpi=150)
    plt.close()

    # Figura 2: espectrograma con timeline
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    plt.figure(figsize=(12, 5))
    librosa.display.specshow(S, sr=sr, x_axis="time", y_axis="hz")

    if meta.get("manipulado") == "1":
        ini_gt = float(meta["inicio_insercion_s"])
        fin_gt = float(meta["fin_insercion_s"])
        plt.axvspan(ini_gt, fin_gt, alpha=0.2, label="Ground truth")

    for ini, fin, _ in intervalos_pred:
        plt.axvspan(ini, fin, alpha=0.15)

    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Espectrograma y regiones sospechosas - {audio_file}")
    plt.tight_layout()
    plt.savefig(out_dir / "spectrograma_timeline.png", dpi=150)
    plt.close()

    # Resumen
    resumen = []
    resumen.append(f"Audio analizado: {audio_file}")
    resumen.append(f"Tamper score: {tamper_score:.4f}")
    resumen.append(f"Umbral utilizado: {threshold}")
    resumen.append("")

    if meta.get("manipulado") == "1":
        resumen.append("Ground truth:")
        resumen.append(f"- Tipo: {meta.get('tipo_splicing')}")
        resumen.append(f"- Inicio inserción: {meta.get('inicio_insercion_s')} s")
        resumen.append(f"- Fin inserción: {meta.get('fin_insercion_s')} s")
    else:
        resumen.append("Ground truth: audio limpio")

    resumen.append("")
    resumen.append("Intervalos sospechosos predichos:")

    if intervalos_pred:
        for ini, fin, score in intervalos_pred:
            resumen.append(f"- {ini:.3f}s - {fin:.3f}s | score máximo = {score:.4f}")
    else:
        resumen.append("- No se han detectado intervalos por encima del umbral.")

    (out_dir / "resumen.txt").write_text("\n".join(resumen), encoding="utf-8")

    print(f"Informe generado en: {out_dir}")
    print(f"Tamper score: {tamper_score:.4f}")
    print(f"Intervalos predichos: {len(intervalos_pred)}")

if __name__ == "__main__":
    main()
