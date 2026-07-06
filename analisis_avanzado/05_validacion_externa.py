# Proyecto desarrollado como parte del Trabajo de Fin de Estudios del Grado en
# Ingeniería Informática de UNIR (2026). Autor: Ángel Carlos Soler Encinas. Licencia MIT.
"""
Validación EXTERNA con audios independientes (hablantes/dispositivos no vistos).

Es la prueba más exigente y la que más refuerza el TFG: mide si el detector
generaliza a voces que NO participaron en el entrenamiento. Dado un directorio con
grabaciones limpias externas (p. ej. de Common Voice en español o LibriSpeech),
el script:

  1) Normaliza cada audio a 16 kHz mono.
  2) Crea una versión manipulada por splicing del mismo audio, registrando el
     intervalo real insertado (ground truth).
  3) Extrae las 108 características y puntúa con el modelo ya entrenado
     (random_forest_borde.joblib), que NUNCA vio estas voces.
  4) Evalúa detección (los limpios deben dar negativo; los manipulados, positivo)
     y localización temporal.

Cómo preparar el corpus (ejemplo con Common Voice en español):
  - Descarga un subconjunto de https://commonvoice.mozilla.org/es/datasets
  - Convierte algunos clips a WAV mono y colócalos en una carpeta, p. ej.:
        ffmpeg -i clip.mp3 -ac 1 -ar 16000 data/externo/clip01.wav
  - Usa 15-30 audios de >= 6 s, de hablantes variados, distintos de los del TFG.

Uso:
    cd Codigo_y_Resultados
    python3 analisis_avanzado/05_validacion_externa.py --dir data/externo

Salidas (reports/avanzado/):
    validacion_externa_detalle.csv, validacion_externa_resumen.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import librosa
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
import config as C
from posproceso import agrupar_intervalos, hay_solape
from features_inferencia import construir_df_caracteristicas

OUT_DIR = C.REPORTS_DIR / "avanzado"
MIN_DUR = 6.0   # s mínimos para poder generar un splice razonable


def generar_splice(y, sr, rng):
    """Inserta un fragmento del propio audio y devuelve (y_manipulado, inicio_gt, fin_gt)."""
    dur = len(y) / sr
    frag_s = rng.uniform(1.0, min(2.5, dur / 4))
    frag_n = int(frag_s * sr)
    src_ini = int(rng.uniform(1.0, max(1.1, dur - frag_s - 1.0)) * sr)
    fragmento = y[src_ini:src_ini + frag_n]
    pos = int(rng.uniform(1.0, max(1.1, dur - 1.0)) * sr)
    y_man = np.concatenate([y[:pos], fragmento, y[pos:]])
    return y_man.astype(np.float32), pos / sr, (pos + frag_n) / sr


def puntuar(model, feature_cols, y, sr):
    df = construir_df_caracteristicas(y, sr)
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    df["score_sospecha"] = model.predict_proba(df[feature_cols])[:, 1]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, required=True, help="carpeta con WAV limpios externos")
    ap.add_argument("--threshold", type=float, default=C.DEFAULT_THRESHOLD)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    carpeta = Path(args.dir)
    wavs = sorted([*carpeta.glob("*.wav"), *carpeta.glob("*.flac"), *carpeta.glob("*.ogg")])
    if not wavs:
        print(f"No se encontraron audios en {carpeta}. Revisa las instrucciones de la cabecera.")
        sys.exit(1)

    bundle = joblib.load(C.MODEL_BORDE)
    model, feature_cols = bundle["model"], list(bundle["feature_cols"])
    rng = np.random.default_rng(C.RANDOM_STATE)

    filas = []
    for wav in wavs:
        try:
            y, _ = librosa.load(wav, sr=C.SR, mono=True)
        except Exception as exc:
            print(f"  [omitido] {wav.name}: {exc}"); continue
        if len(y) / C.SR < MIN_DUR:
            print(f"  [omitido] {wav.name}: dura < {MIN_DUR}s"); continue

        # versión limpia (negativo)
        df_c = puntuar(model, feature_cols, y, C.SR)
        ts_c = float(df_c["score_sospecha"].max())
        det_c = 1 if agrupar_intervalos(df_c, args.threshold) else 0
        filas.append({"archivo": wav.name, "clase": "limpio", "manipulado_real": 0,
                      "detectado": det_c, "tamper_score": round(ts_c, 4), "localizado": ""})

        # versión manipulada (positivo)
        y_man, gi, gf = generar_splice(y, C.SR, rng)
        df_m = puntuar(model, feature_cols, y_man, C.SR)
        ts_m = float(df_m["score_sospecha"].max())
        ivs = agrupar_intervalos(df_m, args.threshold)
        det_m = 1 if ivs else 0
        loc_m = 1 if any(hay_solape(iv, gi, gf) for iv in ivs) else 0
        filas.append({"archivo": wav.name, "clase": "manipulado", "manipulado_real": 1,
                      "detectado": det_m, "tamper_score": round(ts_m, 4), "localizado": loc_m})
        print(f"  {wav.name}: limpio ts={ts_c:.3f} ({'det' if det_c else 'ok'}) | "
              f"manip ts={ts_m:.3f} ({'det' if det_m else 'no-det'}, loc={'sí' if loc_m else 'no'})")

    det = pd.DataFrame(filas)
    det.to_csv(OUT_DIR / "validacion_externa_detalle.csv", index=False, sep=";")

    yt = det["manipulado_real"].astype(int); yp = det["detectado"].astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    man = det[det["manipulado_real"] == 1]
    locd = man["localizado"].replace("", 0).astype(int).sum()
    n_audios = det["archivo"].nunique()

    txt = [
        "VALIDACIÓN EXTERNA (audios independientes, hablantes no vistos en entrenamiento)",
        "=" * 78, "",
        f"Audios externos usados: {n_audios}  ->  {len(det)} muestras ({(yt==0).sum()} limpias, {(yt==1).sum()} manipuladas)",
        f"Umbral: {args.threshold}", "",
        f"Accuracy:  {accuracy_score(yt, yp):.4f}",
        f"Precision: {precision_score(yt, yp, zero_division=0):.4f}",
        f"Recall:    {recall_score(yt, yp, zero_division=0):.4f}",
        f"F1-score:  {f1_score(yt, yp, zero_division=0):.4f}",
        f"Matriz TN/FP/FN/TP = {tn}/{fp}/{fn}/{tp}",
        f"Localización: {locd}/{len(man)}"
        + (f"  (tasa {locd/len(man):.4f})" if len(man) else ""),
        "",
        "Esta es la estimación de generalización más fiable del prototipo: si se mantiene",
        "razonable aquí, el sistema funciona más allá del conjunto controlado del TFG.",
    ]
    (OUT_DIR / "validacion_externa_resumen.txt").write_text("\n".join(txt), encoding="utf-8")
    print("\n" + "\n".join(txt))


if __name__ == "__main__":
    main()
