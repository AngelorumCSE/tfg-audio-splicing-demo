"""
Calcula las diferencias delta de cada característica respecto a la ventana anterior y posterior,
que capturan los cambios bruscos (discontinuidades) en los que se apoya la detección.
"""
from pathlib import Path
import pandas as pd
import numpy as np

IN_CSV = Path("data/processed/window_features.csv")
OUT_CSV = Path("data/processed/window_features_delta.csv")

META_COLS = [
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
]

df = pd.read_csv(IN_CSV, sep=";")

feature_cols = [c for c in df.columns if c not in META_COLS]

df = df.sort_values(["archivo_generado", "indice_ventana"]).reset_index(drop=True)

new_parts = [df]

for col in feature_cols:
    prev = df.groupby("archivo_generado")[col].shift(1)
    next_ = df.groupby("archivo_generado")[col].shift(-1)

    new_parts.append(pd.DataFrame({
        f"delta_prev_{col}": (df[col] - prev).abs(),
        f"delta_next_{col}": (df[col] - next_).abs(),
    }))

df_out = pd.concat(new_parts, axis=1)

# Las primeras/últimas ventanas no tienen anterior/siguiente.
# Se sustituyen por 0 para que el modelo pueda usarlas.
df_out = df_out.fillna(0)

df_out.to_csv(OUT_CSV, index=False, sep=";")

print(f"CSV generado: {OUT_CSV}")
print(f"Filas: {len(df_out)}")
print(f"Columnas originales: {len(df.columns)}")
print(f"Columnas nuevas: {len(df_out.columns)}")
print()
print("Distribución de etiquetas:")
print(df_out["etiqueta"].value_counts().sort_index())
