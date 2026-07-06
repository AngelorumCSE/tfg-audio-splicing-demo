# Proyecto desarrollado como parte del Trabajo de Fin de Estudios del Grado en
# Ingeniería Informática de UNIR (2026). Autor: Ángel Carlos Soler Encinas. Licencia MIT.
"""
Evaluación EXPLORATORIA de transferencia sobre PartialSpoof (dev).

Pregunta: ¿qué ocurre cuando el detector cross-source de este TFG, entrenado para
empalmes de VOZ GENUINA (LibriSpeech), se aplica TAL CUAL (sin reentrenar ni
recalibrar) a un benchmark público de audio parcialmente falso, donde los segmentos
insertados son VOZ SINTÉTICA (TTS/VC)?

Los materiales difieren por diseño (sección 6.8 de la memoria): allí el indicio es la
artificialidad del fragmento; aquí, la discontinuidad de la unión. Este experimento
CUANTIFICA esa distancia (transferencia), en lugar de dejarla como argumento.

Protocolo:
  - PartialSpoof v1.2 (Zhang et al., 2021; Zenodo 5766198, CC BY 4.0), partición dev.
  - Muestra aleatoria reproducible (semilla 42): N bonafide + N parcialmente falsos.
  - Puntuación: mismas 108 características y modelo_libri.joblib; tamper = máximo por
    ventana (idéntico a la app y a los scripts 08-10).
  - Métricas: ROC-AUC y PR-AUC por archivo (bonafide=0, parcial=1), tasas al umbral
    operativo (0,50) y de cribado (0,30) y, como análisis secundario, el solape de los
    intervalos predichos con los tramos sintéticos reales (etiquetas de segmento a
    resolución 0,16 s).

Uso:
    cd Codigo_y_Resultados
    python3 analisis_avanzado/11_transferencia_partialspoof.py \
        --dir ~/Desktop/PartialSpoof_data/database --n 250

Salidas: reports/avanzado/transferencia_partialspoof.{txt,csv}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
import config as C
from features_inferencia import construir_df_caracteristicas
from posproceso import agrupar_intervalos, hay_solape

OUT = Path("reports/avanzado")
RES_SEG = 0.16  # resolución de las etiquetas de segmento utilizadas (s)


def cargar_protocolo(base: Path) -> pd.DataFrame:
    f = base / "protocols" / "PartialSpoof_LA_cm_protocols" / "PartialSpoof.LA.cm.dev.trl.txt"
    df = pd.read_csv(f, sep=r"\s+", header=None,
                     names=["speaker", "utt", "c3", "c4", "label"])
    return df[["utt", "label"]]


def tramos_sintenticos(seglab: np.ndarray, valor_spoof) -> list[tuple[float, float]]:
    """Convierte el vector de etiquetas por segmento en intervalos temporales (s)."""
    tramos, ini = [], None
    for i, v in enumerate(seglab):
        if v == valor_spoof and ini is None:
            ini = i * RES_SEG
        elif v != valor_spoof and ini is not None:
            tramos.append((ini, i * RES_SEG)); ini = None
    if ini is not None:
        tramos.append((ini, len(seglab) * RES_SEG))
    return tramos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(Path.home() / "Desktop/PartialSpoof_data/database"))
    ap.add_argument("--n", type=int, default=250, help="archivos por clase")
    args = ap.parse_args()
    base = Path(args.dir).expanduser()
    wav_dir = base / "dev" / "con_wav"

    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(C.RANDOM_STATE)

    proto = cargar_protocolo(base)
    bona = proto[proto.label == "bonafide"].utt.tolist()
    spoof = proto[proto.label == "spoof"].utt.tolist()
    sel = ([(u, 0) for u in rng.choice(bona, args.n, replace=False)]
           + [(u, 1) for u in rng.choice(spoof, args.n, replace=False)])

    # Etiquetas de segmento (convención inferida de los bonafide: su valor único = genuino)
    seglabs = np.load(base / "segment_labels" / f"dev_seglab_{RES_SEG:.2f}.npy",
                      allow_pickle=True).item()
    vals_bona = {v for u in bona[:200] if u in seglabs for v in np.unique(seglabs[u])}
    valor_spoof = None
    if len(vals_bona) == 1:
        unico = vals_bona.pop()
        valor_spoof = 1 - int(unico) if str(unico).isdigit() or isinstance(unico, (int, np.integer)) else None

    bundle = joblib.load("models/modelo_libri.joblib")
    model, cols = bundle["model"], list(bundle["feature_cols"])

    filas, saltados = [], 0
    for utt, y in sel:
        f = wav_dir / f"{utt}.wav"
        if not f.exists():
            for ext in (".flac", ".WAV"):
                if f.with_suffix(ext).exists(): f = f.with_suffix(ext); break
        try:
            audio, sr = sf.read(f)
        except Exception:
            saltados += 1; continue
        if audio.ndim > 1: audio = audio.mean(axis=1)
        if len(audio) < int(1.2 * sr):
            saltados += 1; continue
        df = construir_df_caracteristicas(audio, sr)
        for c in cols:
            if c not in df.columns: df[c] = 0.0
        df["score_sospecha"] = model.predict_proba(df[cols])[:, 1]
        if "indice_ventana" not in df.columns: df["indice_ventana"] = range(len(df))
        tamper = float(df["score_sospecha"].max())
        fila = {"utt": utt, "clase": y, "tamper": round(tamper, 4), "n_ventanas": len(df)}
        if y == 1 and valor_spoof is not None and utt in seglabs:
            gt = tramos_sintenticos(np.asarray(seglabs[utt]).ravel(), valor_spoof)
            ivs = agrupar_intervalos(df, 0.30)
            fila["algun_tramo_gt"] = len(gt)
            fila["solape_con_sintetico@0.30"] = int(any(
                hay_solape(iv, a, b) for iv in ivs for a, b in gt)) if gt else ""
        filas.append(fila)

    t = pd.DataFrame(filas)
    t.to_csv(OUT / "transferencia_partialspoof.csv", sep=";", index=False)

    yv, sv = t["clase"].values, t["tamper"].values
    roc = roc_auc_score(yv, sv)
    ap_ = average_precision_score(yv, sv)
    lineas = [
        "TRANSFERENCIA EXPLORATORIA A PARTIALSPOOF (dev) — detector cross-source SIN reentrenar",
        "=" * 84, "",
        f"Muestra: {int((t.clase==0).sum())} bonafide + {int((t.clase==1).sum())} parcialmente falsos "
        f"(semilla {C.RANDOM_STATE}; {saltados} descartados por duración/lectura)",
        f"Detector: modelo_libri.joblib (108 características, entrenado con empalmes de voz genuina)", "",
        f"ROC-AUC por archivo: {roc:.3f}   |   PR-AUC: {ap_:.3f}   (0,5 = azar)",
        f"Tamper mediano bonafide: {np.median(sv[yv==0]):.3f}  |  parcial: {np.median(sv[yv==1]):.3f}",
        f"Marcados como sospechosos @0,50: bonafide {int((sv[yv==0]>=0.5).sum())}/{int((yv==0).sum())} "
        f"| parcial {int((sv[yv==1]>=0.5).sum())}/{int((yv==1).sum())}",
        f"Marcados como sospechosos @0,30: bonafide {int((sv[yv==0]>=0.3).sum())}/{int((yv==0).sum())} "
        f"| parcial {int((sv[yv==1]>=0.3).sum())}/{int((yv==1).sum())}",
    ]
    if "solape_con_sintetico@0.30" in t.columns and valor_spoof is not None:
        det = t[(t.clase == 1) & (t["solape_con_sintetico@0.30"] != "")]
        if len(det):
            lineas.append(f"Solape de intervalos predichos (@0,30) con tramos sintéticos reales: "
                          f"{int(pd.to_numeric(det['solape_con_sintetico@0.30']).sum())}/{len(det)}")
    lineas += ["",
        "Lectura: PartialSpoof inserta VOZ SINTÉTICA; este detector modela discontinuidades",
        "de VOZ GENUINA y no fue entrenado ni calibrado para este material. El valor del",
        "experimento es cuantificar esa transferencia (o su ausencia), en línea con la",
        "brecha out-of-domain que reporta la literatura (He et al., 2025), y complementar",
        "el argumento de no-comparabilidad de la sección 6.8 con una medición directa."]
    (OUT / "transferencia_partialspoof.txt").write_text("\n".join(lineas), encoding="utf-8")
    print("\n".join(lineas))


if __name__ == "__main__":
    main()
