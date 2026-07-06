# Proyecto desarrollado como parte del Trabajo de Fin de Estudios del Grado en
# Ingeniería Informática de UNIR (2026). Autor: Ángel Carlos Soler Encinas. Licencia MIT.
"""
Añade la etiqueta de borde (1 si la ventana está a ±0,75 s de una inserción) al CSV de
características delta, generando el conjunto de entrenamiento del modelo de borde.
"""
from pathlib import Path
import pandas as pd
import csv

FEATURES_CSV = Path("data/processed/window_features_delta.csv")
MANIFEST_CSV = Path("data/manifests/splicing_manifest.csv")
OUT_CSV = Path("data/processed/window_features_borde.csv")

MARGEN_BORDE_S = 0.75

df = pd.read_csv(FEATURES_CSV, sep=";")

with open(MANIFEST_CSV, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f, delimiter=";")
    manifest_rows = list(reader)

intervalos = {}

for r in manifest_rows:
    if r["manipulado"] == "1":
        intervalos[r["archivo_generado"]] = {
            "inicio": float(r["inicio_insercion_s"]),
            "fin": float(r["fin_insercion_s"]),
        }

def calcular_etiqueta_borde(row):
    archivo = row["archivo_generado"]

    if archivo not in intervalos:
        return 0

    centro = (float(row["inicio_ventana_s"]) + float(row["fin_ventana_s"])) / 2

    inicio = intervalos[archivo]["inicio"]
    fin = intervalos[archivo]["fin"]

    cerca_inicio = abs(centro - inicio) <= MARGEN_BORDE_S
    cerca_fin = abs(centro - fin) <= MARGEN_BORDE_S

    return 1 if (cerca_inicio or cerca_fin) else 0

df["centro_ventana_s"] = (df["inicio_ventana_s"] + df["fin_ventana_s"]) / 2
df["etiqueta_borde"] = df.apply(calcular_etiqueta_borde, axis=1)

df.to_csv(OUT_CSV, index=False, sep=";")

print(f"CSV generado: {OUT_CSV}")
print(f"Filas: {len(df)}")
print()
print("Distribución etiqueta original:")
print(df["etiqueta"].value_counts().sort_index())
print()
print("Distribución etiqueta_borde:")
print(df["etiqueta_borde"].value_counts().sort_index())
print()
print("Distribución etiqueta_borde por tipo_splicing:")
print(df.groupby(["tipo_splicing", "etiqueta_borde"]).size())
