# Proyecto desarrollado como parte del Trabajo de Fin de Estudios del Grado en
# Ingeniería Informática de UNIR (2026). Autor: Ángel Carlos Soler Encinas. Licencia MIT.
"""
Extracción de características por ventana del prototipo intra-fuente.

Recorre cada audio del manifiesto en ventanas solapadas (1 s, salto 0,5 s) y calcula,
para cada ventana, 36 descriptores espectro-temporales:
  - 13 MFCC, cada uno con media y desviación típica  -> 26
  - RMS, ZCR, centroide, ancho de banda y rolloff, cada uno con media y desv. típica -> 10
Cada ventana se etiqueta como sospechosa (1) si solapa al menos 0,25 s con el intervalo
de manipulación anotado en el manifiesto. El resultado se vuelca a un CSV que alimenta
el cálculo de las diferencias delta y el entrenamiento.

Uso:
    cd Codigo_y_Resultados
    python3 src/extraer_caracteristicas_ventanas.py
"""
from pathlib import Path
import csv
import numpy as np
import pandas as pd
import librosa

AUDIO_DIR = Path("data/generated")
MANIFEST = Path("data/manifests/splicing_manifest.csv")
OUT_CSV = Path("data/processed/window_features.csv")

try:
    import sys as _sys
    _sys.path.append(str(Path(__file__).resolve().parent))
    from config import SR as SR_ESPERADO, VENTANA_S, SALTO_S
except ImportError:
    SR_ESPERADO = 16000      # frecuencia de muestreo de trabajo (Hz)
    VENTANA_S = 1.0          # longitud de la ventana de análisis (s)
    SALTO_S = 0.5            # desplazamiento entre ventanas consecutivas (s)
SOLAPE_MINIMO_S = 0.25   # solape mínimo con la zona manipulada para etiquetar como sospechosa

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)


def leer_manifest(path: Path):
    """Lee el manifiesto (CSV con ';') y devuelve una lista de filas como diccionarios."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        return list(reader)


def solape_intervalos(a_ini, a_fin, b_ini, b_fin):
    """Segundos de solape entre dos intervalos [a_ini, a_fin] y [b_ini, b_fin] (0 si no solapan)."""
    inicio = max(a_ini, b_ini)
    fin = min(a_fin, b_fin)
    return max(0.0, fin - inicio)


def etiqueta_ventana(inicio_v, fin_v, row):
    """Etiqueta una ventana: 1 si es manipulada y solapa >= SOLAPE_MINIMO_S con la inserción.

    Devuelve (etiqueta, segundos_de_solape). Los audios limpios devuelven siempre (0, 0.0).
    """
    if row["manipulado"] != "1":
        return 0, 0.0

    inicio_m = float(row["inicio_insercion_s"])
    fin_m = float(row["fin_insercion_s"])

    solape = solape_intervalos(inicio_v, fin_v, inicio_m, fin_m)

    if solape >= SOLAPE_MINIMO_S:
        return 1, solape

    return 0, solape


def extraer_features(y, sr):
    """Calcula los 36 descriptores base (media y desv. típica) de una ventana de señal `y`."""
    if len(y) == 0:
        raise ValueError("Ventana vacía")

    features = {}

    # 13 MFCC -> media y desviación típica de cada coeficiente (26 descriptores)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
        features[f"mfcc_{i+1}_std"] = float(np.std(mfcc[i]))

    # Descriptores energéticos y espectrales -> media y desviación típica (10 descriptores)
    rms = librosa.feature.rms(y=y)                                  # energía (Root Mean Square)
    features["rms_mean"] = float(np.mean(rms))
    features["rms_std"] = float(np.std(rms))

    zcr = librosa.feature.zero_crossing_rate(y)                     # tasa de cruces por cero
    features["zcr_mean"] = float(np.mean(zcr))
    features["zcr_std"] = float(np.std(zcr))

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)        # centro de masa del espectro
    features["spectral_centroid_mean"] = float(np.mean(centroid))
    features["spectral_centroid_std"] = float(np.std(centroid))

    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)      # dispersión en torno al centroide
    features["spectral_bandwidth_mean"] = float(np.mean(bandwidth))
    features["spectral_bandwidth_std"] = float(np.std(bandwidth))

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)          # frecuencia que acumula el 85% de energía
    features["spectral_rolloff_mean"] = float(np.mean(rolloff))
    features["spectral_rolloff_std"] = float(np.std(rolloff))

    return features


def procesar_audio(row):
    """Trocea un audio en ventanas solapadas y devuelve un registro (metadatos + features) por ventana."""
    audio_path = AUDIO_DIR / row["archivo_generado"]

    if not audio_path.exists():
        raise FileNotFoundError(f"No existe el archivo: {audio_path}")

    y, sr = librosa.load(audio_path, sr=SR_ESPERADO, mono=True)

    ventana_muestras = int(VENTANA_S * sr)
    salto_muestras = int(SALTO_S * sr)

    registros = []
    total_muestras = len(y)

    idx = 0
    # Ventana deslizante: avanza de salto_muestras en salto_muestras hasta agotar la señal.
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
