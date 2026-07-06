# Proyecto desarrollado como parte del Trabajo de Fin de Estudios del Grado en
# Ingeniería Informática de UNIR (2026). Autor: Ángel Carlos Soler Encinas. Licencia MIT.
"""
Pruebas de robustez del detector frente a degradaciones realistas.

Aplica a los audios manipulados varias transformaciones que aparecen en escenarios
reales —recompresión MP3 a distintos bitrates, ruido aditivo a distintas SNR y
remuestreo agresivo— y mide cómo cambian la tasa de detección, la tasa de
localización y el tamper score medio respecto a la condición original (limpia).

Mostrar dónde se degrada el sistema es un valor forense: delimita sus límites de
aplicabilidad. Reutiliza la MISMA función de extracción de características del
pipeline (extraer_caracteristicas_ventanas.extraer_features) para garantizar
coherencia con el entrenamiento.

Uso:
    cd Codigo_y_Resultados
    python3 analisis_avanzado/03_robustez.py --n 42        # nº de audios manipulados a usar

Salidas (reports/avanzado/):
    robustez.csv, robustez.png
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import librosa
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
import config as C
from posproceso import agrupar_intervalos, hay_solape
from features_inferencia import construir_df_caracteristicas  # pipeline de características compartido

OUT_DIR = C.REPORTS_DIR / "avanzado"
GENERATED_DIR = C.GENERATED_DIR   # se puede sobrescribir por --generated

try:
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG = "ffmpeg"


# ---------------- perturbaciones (devuelven y a 16 kHz mono) ----------------
def pert_identidad(y, sr):
    return y

def pert_ruido(snr_db):
    def f(y, sr):
        pot = np.mean(y ** 2) + 1e-12
        ruido = np.random.default_rng(C.RANDOM_STATE).normal(0, np.sqrt(pot / (10 ** (snr_db / 10))), len(y))
        return (y + ruido).astype(np.float32)
    return f

def pert_resample(sr_inter):
    def f(y, sr):
        bajo = librosa.resample(y, orig_sr=sr, target_sr=sr_inter)
        return librosa.resample(bajo, orig_sr=sr_inter, target_sr=sr).astype(np.float32)
    return f

def pert_mp3(bitrate):
    def f(y, sr):
        with tempfile.TemporaryDirectory() as tmp:
            wav, mp3 = Path(tmp) / "a.wav", Path(tmp) / "a.mp3"
            sf.write(wav, y, sr)
            subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(wav),
                            "-b:a", bitrate, str(mp3)], check=True)
            y2, _ = librosa.load(mp3, sr=sr, mono=True)
        return y2.astype(np.float32)
    return f


def evaluar(model, feature_cols, audios, manifest, perturbacion):
    det, loc, scores = 0, 0, []
    for archivo in audios:
        y, _ = librosa.load(GENERATED_DIR / archivo, sr=C.SR, mono=True)
        y = perturbacion(y, C.SR)
        df = construir_df_caracteristicas(y, C.SR)
        for c in feature_cols:
            if c not in df.columns:
                df[c] = 0.0
        df["score_sospecha"] = model.predict_proba(df[feature_cols])[:, 1]
        ts = float(df["score_sospecha"].max()); scores.append(ts)
        ivs = agrupar_intervalos(df, C.DEFAULT_THRESHOLD)
        if ivs:
            det += 1
            meta = manifest[archivo]
            gi, gf = float(meta["inicio_insercion_s"]), float(meta["fin_insercion_s"])
            if any(hay_solape(iv, gi, gf) for iv in ivs):
                loc += 1
    n = len(audios)
    return {"deteccion": det / n, "localizacion": loc / n, "tamper_medio": float(np.mean(scores))}


def main():
    global GENERATED_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=42, help="nº de audios manipulados a evaluar")
    ap.add_argument("--model", default=str(C.MODEL_BORDE), help="ruta al modelo .joblib")
    ap.add_argument("--manifest", default=str(C.MANIFEST_CSV), help="manifiesto del dataset")
    ap.add_argument("--generated", default=str(C.GENERATED_DIR), help="carpeta de audios")
    ap.add_argument("--tipos", default="", help="tipos de empalme a incluir, separados por comas "
                    "(p.ej. cross_speaker,cross_speaker_ruido); vacío = todos los manipulados")
    ap.add_argument("--sufijo", default="", help="sufijo de los archivos de salida (p.ej. _libri)")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_DIR = Path(args.generated)

    bundle = joblib.load(args.model)
    model, feature_cols = bundle["model"], list(bundle["feature_cols"])
    with open(args.manifest, newline="", encoding="utf-8-sig") as f:
        manifest = {r["archivo_generado"]: r for r in csv.DictReader(f, delimiter=";")}
    tipos = {t.strip() for t in args.tipos.split(",") if t.strip()}
    manip = [a for a, r in manifest.items()
             if str(r["manipulado"]) == "1" and (not tipos or r.get("tipo_splicing") in tipos)][: args.n]

    pruebas = {
        "Original": pert_identidad,
        "MP3 128k": pert_mp3("128k"), "MP3 96k": pert_mp3("96k"), "MP3 64k": pert_mp3("64k"),
        "Ruido 30 dB": pert_ruido(30), "Ruido 20 dB": pert_ruido(20), "Ruido 10 dB": pert_ruido(10),
        "Resample 8 kHz": pert_resample(8000),
    }
    filas = []
    for nombre, p in pruebas.items():
        print(f"Evaluando: {nombre} ...")
        try:
            r = evaluar(model, feature_cols, manip, manifest, p)
            filas.append({"perturbacion": nombre, **{k: round(v, 4) for k, v in r.items()}})
        except Exception as exc:
            filas.append({"perturbacion": nombre, "error": str(exc)})
            print("  ERROR:", exc)
    df = pd.DataFrame(filas)
    suf = args.sufijo
    df.to_csv(OUT_DIR / f"robustez{suf}.csv", index=False, sep=";")

    if "deteccion" in df.columns:
        fig, ax = plt.subplots(figsize=(11, 4.5))
        x = range(len(df))
        ax.plot(x, df["deteccion"], marker="o", label="Tasa de detección")
        ax.plot(x, df["localizacion"], marker="s", label="Tasa de localización")
        ax.plot(x, df["tamper_medio"], marker="^", label="Tamper score medio")
        ax.set_xticks(list(x)); ax.set_xticklabels(df["perturbacion"], rotation=30, ha="right")
        ax.set_ylim(0, 1.05); ax.grid(alpha=0.3); ax.legend()
        ax.set_title("Robustez del detector frente a degradaciones")
        fig.tight_layout(); fig.savefig(OUT_DIR / f"robustez{suf}.png", dpi=180); plt.close(fig)

    print(df.to_string(index=False))
    print(f"\nResultados en {OUT_DIR}")


if __name__ == "__main__":
    main()
