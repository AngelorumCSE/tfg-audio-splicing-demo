"""
Construcción del vector de características por ventana para inferencia.

Replica exactamente el pipeline de entrenamiento (ventaneo de 1 s con salto de
0,5 s -> características base por ventana -> diferencias delta con la ventana
anterior y posterior), reutilizando la función canónica
extraer_caracteristicas_ventanas.extraer_features para garantizar coherencia.

Lo usan los scripts de robustez y de validación externa.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import config as C
from extraer_caracteristicas_ventanas import extraer_features

_META = ("indice_ventana", "inicio_ventana_s", "fin_ventana_s")


def construir_df_caracteristicas(y: np.ndarray, sr: int) -> pd.DataFrame:
    """Devuelve un DataFrame con las 108 columnas (base + delta) por ventana."""
    L = int(C.VENTANA_S * sr)
    H = int(C.SALTO_S * sr)
    if len(y) < L:
        y = np.pad(y, (0, L - len(y)))

    registros = []
    for idx, ini in enumerate(range(0, len(y) - L + 1, H)):
        seg = y[ini:ini + L]
        reg = {"indice_ventana": idx,
               "inicio_ventana_s": round(ini / sr, 3),
               "fin_ventana_s": round((ini + L) / sr, 3)}
        reg.update(extraer_features(seg, sr))
        registros.append(reg)

    df = pd.DataFrame(registros).sort_values("indice_ventana").reset_index(drop=True)
    base = [c for c in df.columns if c not in _META]
    partes = [df]
    for col in base:
        partes.append(pd.DataFrame({
            f"delta_prev_{col}": (df[col] - df[col].shift(1)).abs(),
            f"delta_next_{col}": (df[col] - df[col].shift(-1)).abs(),
        }))
    return pd.concat(partes, axis=1).fillna(0)
