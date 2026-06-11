from pathlib import Path
import csv
import numpy as np
import pandas as pd
import librosa

AUDIO_DIR = Path("data/generated")
MANIFEST = Path("data/manifests/splicing_manifest.csv")
OUT_CSV = Path("data/processed/window_features.csv")

SR_ESPERADO = 16000
VENTANA_S = 1.0
SALTO_S = 0.5
SOLAPE_MINIMO_S = 0.25

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)


def leer_manifest(path: Path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        return list(reader)


def solape_intervalos(a_ini, a_fin, b_ini, b_fin):
    inicio = max(a_ini, b_ini)
    fin = min(a_fin, b_fin)
    return max(0.0, fin - inicio)


def etiqueta_ventana(inicio_v, fin_v, row):
    if row["manipulado"] != "1":
        return 0, 0.0

    inicio_m = float(row["inicio_insercion_s"])
    fin_m = float(row["fin_insercion_s"])

    solape = solape_intervalos(inicio_v, fin_v, inicio_m, fin_m)

    if solape >= SOLAPE_MINIMO_S:
        return 1, solape

    return 0, solape


def extraer_features(y, sr):
    # Aseguramos que no haya valores raros
    if len(y) == 0:
        raise ValueError("Ventana vacía")

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    features = {}

    for i in range(13):
        features[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
        features[f"mfcc_{i+1}_std"] = float(np.std(mfcc[i]))

    # Energía RMS
    rms = librosa.feature.rms(y=y)
    features["rms_mean"] = float(np.mean(rms))
    features["rms_std"] = float(np.std(rms))

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features["zcr_mean"] = float(np.mean(zcr))
    features["zcr_std"] = float(np.std(zcr))

    # Centroide espectral
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features["spectral_centroid_mean"] = float(np.mean(centroid))
    features["spectral_centroid_std"] = float(np.std(centroid))

    # Ancho de banda espectral
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features["spectral_bandwidth_mean"] = float(np.mean(bandwidth))
    features["spectral_bandwidth_std"] = float(np.std(bandwidth))

    # Rolloff espectral
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features["spectral_rolloff_mean"] = float(np.mean(rolloff))
    features["spectral_rolloff_std"] = float(np.std(rolloff))

    return features


def procesar_audio(row):
    audio_path = AUDIO_DIR / row["archivo_generado"]

    if not audio_path.exists():
        raise FileNotFoundError(f"No existe el archivo: {audio_path}")

    y, sr = librosa.load(audio_path, sr=SR_ESPERADO, mono=True)

    ventana_muestras = int(VENTANA_S * sr)
    salto_muestras = int(SALTO_S * sr)

    registros = []
    total_muestras = len(y)

    idx = 0
    for inicio in range(0, total_muestras - ventana_muestras + 1, salto_muestras):
        fin = inicio + ventana_muestras

        inicio_s = inicio / sr
        fin_s = fin / sr

        ventana = y[inicio:fin]

        etiqueta, solape_s = etiqueta_ventana(inicio_s, fin_s, row)
        features = extraer_features(ventana, sr)

        registro = {
            "id_registro": row["id_registro"],
            "archivo_generado": row["archivo_generado"],
            "archivo_base": row["archivo_base"],
            "id_hablante_base": row["id_hablante_base"],
            "genero_base": row["genero_base"],
            "tipo_splicing": row["tipo_splicing"],
            "manipulado": row["manipulado"],
            "indice_ventana": idx,
            "inicio_ventana_s": round(inicio_s, 3),
            "fin_ventana_s": round(fin_s, 3),
            "etiqueta": etiqueta,
            "solape_manipulacion_s": round(solape_s, 3),
        }

        registro.update(features)
        registros.append(registro)

        idx += 1

    return registros


def main():
    rows = leer_manifest(MANIFEST)

    all_records = []

    for i, row in enumerate(rows, start=1):
        print(f"[{i}/{len(rows)}] Procesando: {row['archivo_generado']}")
        registros = procesar_audio(row)
        all_records.extend(registros)

    df = pd.DataFrame(all_records)
    df.to_csv(OUT_CSV, index=False, sep=";")

    print()
    print(f"CSV generado: {OUT_CSV}")
    print(f"Ventanas totales: {len(df)}")
    print("Distribución de etiquetas:")
    print(df["etiqueta"].value_counts().sort_index())

    print()
    print("Distribución por tipo de splicing:")
    print(df.groupby(["tipo_splicing", "etiqueta"]).size())


if __name__ == "__main__":
    main()
