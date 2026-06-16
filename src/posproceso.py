"""
Posprocesado de scores por ventana (funciones puras, sin dependencias de modelo).

Contiene la lógica de:
  - agrupación de ventanas sospechosas consecutivas en intervalos temporales,
  - comprobación de solape entre un intervalo predicho y el real (localización),
  - solape relativo (IoU) como métrica de localización más estricta,
  - balanceo por submuestreo de la clase negativa.

Al estar aislada de scikit-learn, esta lógica puede probarse de forma unitaria
(ver tests/test_posproceso.py).
"""
from __future__ import annotations
from typing import List, Tuple

import pandas as pd

try:
    import config as C
    MAX_GAP_S, MIN_DURATION_S, RATIO_NEGATIVOS, RANDOM_STATE = (
        C.MAX_GAP_S, C.MIN_DURATION_S, C.RATIO_NEGATIVOS, C.RANDOM_STATE)
except Exception:  # valores por defecto si config no está disponible
    MAX_GAP_S, MIN_DURATION_S, RATIO_NEGATIVOS, RANDOM_STATE = 0.75, 0.50, 3, 42

Intervalo = Tuple[float, float, float]


def agrupar_intervalos(df_audio: pd.DataFrame, threshold: float,
                       max_gap_s: float = MAX_GAP_S,
                       min_duration_s: float = MIN_DURATION_S) -> List[Intervalo]:
    """Agrupa ventanas con score_sospecha >= threshold en intervalos (inicio, fin, score_max).

    Dos ventanas se funden si la siguiente empieza antes de `fin_actual + max_gap_s`.
    Un intervalo se conserva solo si su duración alcanza `min_duration_s`.
    """
    positivos = df_audio[df_audio["score_sospecha"] >= threshold].copy()
    if positivos.empty:
        return []
    positivos = positivos.sort_values("inicio_ventana_s")

    intervalos: List[Intervalo] = []
    inicio = float(positivos.iloc[0]["inicio_ventana_s"])
    fin = float(positivos.iloc[0]["fin_ventana_s"])
    score_max = float(positivos.iloc[0]["score_sospecha"])

    for _, row in positivos.iloc[1:].iterrows():
        ini = float(row["inicio_ventana_s"])
        f = float(row["fin_ventana_s"])
        s = float(row["score_sospecha"])
        if ini <= fin + max_gap_s:
            fin = max(fin, f)
            score_max = max(score_max, s)
        else:
            if fin - inicio >= min_duration_s:
                intervalos.append((inicio, fin, score_max))
            inicio, fin, score_max = ini, f, s

    if fin - inicio >= min_duration_s:
        intervalos.append((inicio, fin, score_max))
    return intervalos


def hay_solape(intervalo_pred: Intervalo, inicio_gt: float, fin_gt: float) -> bool:
    """True si el intervalo predicho y el real [inicio_gt, fin_gt] se solapan."""
    ini_pred, fin_pred, _ = intervalo_pred
    return max(ini_pred, inicio_gt) <= min(fin_pred, fin_gt)


def iou(intervalo_pred: Intervalo, inicio_gt: float, fin_gt: float) -> float:
    """Solape relativo (Intersection over Union) entre intervalo predicho y real."""
    ini_pred, fin_pred, _ = intervalo_pred
    inter = max(0.0, min(fin_pred, fin_gt) - max(ini_pred, inicio_gt))
    union = (fin_pred - ini_pred) + (fin_gt - inicio_gt) - inter
    return inter / union if union > 0 else 0.0


def balancear_train(df_train: pd.DataFrame, etiqueta: str = "etiqueta_borde",
                    ratio: int = RATIO_NEGATIVOS, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Submuestreo de la clase negativa con proporción 1:ratio (solo en entrenamiento)."""
    pos = df_train[df_train[etiqueta] == 1]
    neg = df_train[df_train[etiqueta] == 0]
    n_neg = min(len(neg), len(pos) * ratio)
    neg_s = neg.sample(n=n_neg, random_state=random_state)
    return pd.concat([pos, neg_s]).sample(frac=1, random_state=random_state).reset_index(drop=True)
