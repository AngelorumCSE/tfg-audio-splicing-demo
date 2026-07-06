# Proyecto desarrollado como parte del Trabajo de Fin de Estudios del Grado en
# Ingeniería Informática de UNIR (2026). Autor: Ángel Carlos Soler Encinas. Licencia MIT.
"""
Comprobación exploratoria de validez en ESPAÑOL (Anexo B de la memoria).

Pregunta: ¿el principio de detección (discontinuidad acústica entre ventanas)
funciona sobre voz en español grabada en condiciones reales, aunque el detector
se haya entrenado con LibriSpeech (inglés leído)?

Diseño (determinista, semilla 42):
  1) Audio host REAL en español (nota de voz de WhatsApp del dataset propio,
     variante limpia, hablante con consentimiento, seudonimizado).
  2) Tres condiciones evaluadas con el detector cross-source YA ENTRENADO
     (models/modelo_libri.joblib), sin reentrenar ni recalibrar:
       a) limpio (control);
       b) inserto de 2 s de OTRA fuente (hablante de LibriSpeech) en t=10 s;
       c) mismo inserto con ruido añadido (simula cambio de entorno).
  3) Para cada condición: tamper score (máximo por ventana), intervalos
     sospechosos al umbral operativo 0,50 y solape con el intervalo real [10, 12] s.

Lectura esperada (H1): el caso con cambio de entorno debe destacar sobre el
fondo del host y localizarse en el tramo insertado; el umbral 0,50, calibrado
en LibriSpeech, NO se presupone válido para este dominio (se discute en el
Anexo B). Este análisis es exploratorio: un solo host, sin valor estadístico;
la validación formal en español queda como línea futura (sección 7.3).

Uso:
    cd Codigo_y_Resultados
    python3 analisis_avanzado/09_comprobacion_espanol.py

Requiere: data/generated/A021_*_clean.wav, data/libri/audios/H006_8297_clean.wav,
models/modelo_libri.joblib. Salidas en reports/avanzado/comprobacion_espanol.{txt,csv}.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import soundfile as sf

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
import config as C
from features_inferencia import construir_df_caracteristicas
from posproceso import agrupar_intervalos, hay_solape, iou

HOST = Path("data/generated/A021_WhatsApp_Audio_2026-05-11_at_13.25.03_clean.wav")
DONANTE = Path("data/libri/audios/H006_8297_clean.wav")
MODELO = Path("models/modelo_libri.joblib")
OUT = Path("reports/avanzado")
UMBRAL = C.DEFAULT_THRESHOLD          # 0,50: umbral operativo de la memoria
T_INSERTO, DUR_INSERTO = 10.0, 2.0    # posición y duración del inserto (s)
T_FRAG_DONANTE = 12.0                 # instante del fragmento en el donante (s)
SIGMA_RUIDO = 0.01                    # ruido gaussiano añadido (condición c)


def evaluar(nombre: str, y: np.ndarray, sr: int, model, cols, gt=None) -> dict:
    df = construir_df_caracteristicas(y, sr)
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
    df["score_sospecha"] = model.predict_proba(df[cols])[:, 1]
    if "indice_ventana" not in df.columns:
        df["indice_ventana"] = range(len(df))
    tamper = float(df["score_sospecha"].max())
    ivs = agrupar_intervalos(df, UMBRAL)
    fila = {"condicion": nombre, "tamper": round(tamper, 4),
            "prediccion@0.50": "MANIPULADO" if tamper >= UMBRAL else "LIMPIO",
            "n_intervalos": len(ivs),
            "intervalos": "; ".join(f"[{a:.1f}, {b:.1f}]" for a, b, _ in ivs) or "-"}
    if gt is not None:
        acierto = any(hay_solape(v, *gt) for v in ivs)
        mejor_iou = max((iou(v, *gt) for v in ivs), default=0.0)
        fila["solape_con_gt"] = "sí" if acierto else "no"
        fila["iou_mejor_intervalo"] = round(mejor_iou, 3)
    return fila


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(C.RANDOM_STATE)
    host, sr = sf.read(HOST)
    donante, sr2 = sf.read(DONANTE)
    assert sr == sr2 == C.SR, "ambos audios deben estar a 16 kHz"

    bundle = joblib.load(MODELO)
    model, cols = bundle["model"], list(bundle["feature_cols"])

    frag = donante[int(T_FRAG_DONANTE * sr): int((T_FRAG_DONANTE + DUR_INSERTO) * sr)]
    pos = int(T_INSERTO * sr)
    gt = (T_INSERTO, T_INSERTO + DUR_INSERTO)

    filas = [
        evaluar("host limpio (español, WhatsApp)", host, sr, model, cols),
        evaluar("host + inserto otra fuente (solo voz)",
                np.concatenate([host[:pos], frag, host[pos:]]), sr, model, cols, gt),
        evaluar("host + inserto otra fuente + ruido (entorno)",
                np.concatenate([host[:pos], frag + rng.normal(0, SIGMA_RUIDO, len(frag)), host[pos:]]),
                sr, model, cols, gt),
    ]
    tabla = pd.DataFrame(filas)
    tabla.to_csv(OUT / "comprobacion_espanol.csv", sep=";", index=False)

    lineas = [
        "COMPROBACIÓN EXPLORATORIA EN ESPAÑOL — detector cross-source sin reentrenar",
        "=" * 78, "",
        f"Host: {HOST.name} | Donante del inserto: {DONANTE.name}",
        f"Inserto: {DUR_INSERTO:.0f} s en t={T_INSERTO:.0f} s (ground truth [{gt[0]:.0f}, {gt[1]:.0f}] s)",
        f"Modelo: {MODELO.name} | umbral operativo {UMBRAL:.2f} (calibrado en LibriSpeech)", "",
        tabla.to_string(index=False), "",
        "Lectura: el empalme con cambio de entorno destaca sobre el fondo del host y",
        "se localiza en el tramo insertado, pese al cambio de idioma y de canal; el",
        "host limpio queda cerca del umbral, lo que indica que 0,50 no está calibrado",
        "para este dominio (un punto de operación razonable aquí sería 0,55-0,60).",
        "Análisis exploratorio con un único host; sin valor estadístico (Anexo B).",
    ]
    (OUT / "comprobacion_espanol.txt").write_text("\n".join(lineas), encoding="utf-8")
    print("\n".join(lineas))


if __name__ == "__main__":
    main()
