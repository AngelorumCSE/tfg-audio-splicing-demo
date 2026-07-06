# Proyecto desarrollado como parte del Trabajo de Fin de Estudios del Grado en
# Ingeniería Informática de UNIR (2026). Autor: Ángel Carlos Soler Encinas. Licencia MIT.
"""
Explicabilidad del modelo: importancia por permutación y agregación por familia.

La importancia por permutación se calcula sobre la PARTICIÓN DE PRUEBA (no sobre
entrenamiento), por lo que mide la pérdida real de rendimiento al "romper" cada
variable. Se agrega además por familia de características (MFCC, RMS, ZCR, centroide,
ancho de banda, rolloff) y por tipo (ventana actual frente a diferencias delta),
lo que permite verificar empíricamente la hipótesis del TFG: las características de
discontinuidad (delta) concentran la mayor parte de la importancia.

Si la librería 'shap' está instalada, genera además un resumen SHAP (opcional).

Uso:
    cd Codigo_y_Resultados
    python3 analisis_avanzado/04_explicabilidad.py --repeats 10

Salidas (reports/avanzado/):
    importancia_permutacion.csv, importancia_por_familia.csv, importancia_familias.png
    (y shap_summary.png si shap está disponible)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.inspection import permutation_importance

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
import config as C
from posproceso import balancear_train

OUT_DIR = C.REPORTS_DIR / "avanzado"


def familia(col: str) -> str:
    core = col.replace("delta_prev_", "").replace("delta_next_", "")
    if "mfcc" in core: return "MFCC"
    if "rms" in core: return "RMS (energía)"
    if "zcr" in core: return "ZCR"
    if "centroid" in core: return "Centroide espectral"
    if "bandwidth" in core: return "Ancho de banda"
    if "rolloff" in core: return "Rolloff"
    return "Otra"


def tipo(col: str) -> str:
    if col.startswith("delta_prev_"): return "delta (anterior)"
    if col.startswith("delta_next_"): return "delta (posterior)"
    return "ventana actual"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--features", default=str(C.FEATURES_BORDE_CSV),
                    help="CSV de características (por defecto el intra-fuente; usa "
                         "data/libri/window_features_libri.csv para el modelo cross-source)")
    ap.add_argument("--sufijo", default="", help="sufijo para los archivos de salida (p.ej. _libri)")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features, sep=";")
    feats = [c for c in df.columns if c not in C.META_COLS]

    # mismo esquema de partición que el entrenamiento del modelo final
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=C.RANDOM_STATE)
    tr, te = next(gss.split(df[feats], df["etiqueta_borde"], groups=df["archivo_base"]))
    bal = balancear_train(df.iloc[tr])
    model = RandomForestClassifier(n_estimators=C.N_ESTIMATORS, min_samples_leaf=C.MIN_SAMPLES_LEAF,
                                   random_state=C.RANDOM_STATE, n_jobs=-1)
    model.fit(bal[feats], bal["etiqueta_borde"])

    print("Calculando importancia por permutación sobre la partición de prueba...")
    pi = permutation_importance(model, df.iloc[te][feats], df.iloc[te]["etiqueta_borde"],
                                n_repeats=args.repeats, random_state=C.RANDOM_STATE, n_jobs=-1)
    imp = pd.DataFrame({"caracteristica": feats, "importancia": pi.importances_mean,
                        "desv": pi.importances_std})
    imp["familia"] = imp["caracteristica"].map(familia)
    imp["tipo"] = imp["caracteristica"].map(tipo)
    imp = imp.sort_values("importancia", ascending=False)
    suf = args.sufijo
    imp.to_csv(OUT_DIR / f"importancia_permutacion{suf}.csv", index=False, sep=";")

    # agregaciones
    por_familia = imp.groupby("familia")["importancia"].sum().sort_values(ascending=False)
    por_tipo = imp.groupby("tipo")["importancia"].sum().sort_values(ascending=False)
    total = imp["importancia"].clip(lower=0).sum() + 1e-12
    frac_delta = imp[imp["tipo"].str.startswith("delta")]["importancia"].clip(lower=0).sum() / total
    por_familia.to_csv(OUT_DIR / f"importancia_por_familia{suf}.csv", sep=";")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    por_familia.plot.bar(ax=ax[0], color="#1F6FB2"); ax[0].set_title("Importancia por familia")
    ax[0].set_ylabel("Suma de importancia (permutación)"); ax[0].tick_params(axis="x", rotation=30)
    por_tipo.plot.bar(ax=ax[1], color="#B00020"); ax[1].set_title("Importancia: ventana actual vs delta")
    ax[1].tick_params(axis="x", rotation=15)
    fig.tight_layout(); fig.savefig(OUT_DIR / f"importancia_familias{suf}.png", dpi=180); plt.close(fig)

    txt = [
        "EXPLICABILIDAD — importancia por permutación (partición de prueba)",
        "=" * 70, "",
        f"Fracción de importancia en características delta (discontinuidad): {frac_delta:.1%}",
        "", "Top 15 características:",
        imp.head(15)[["caracteristica", "importancia", "desv"]].to_string(index=False),
        "", "Importancia agregada por familia:", por_familia.to_string(),
        "", "Importancia por tipo:", por_tipo.to_string(),
    ]
    (OUT_DIR / f"explicabilidad_resumen{suf}.txt").write_text("\n".join(txt), encoding="utf-8")
    print("\n".join(txt))

    # SHAP opcional
    try:
        import shap
        print("\nshap disponible: generando resumen SHAP (muestra)...")
        sample = df.iloc[te][feats].sample(min(300, len(te)), random_state=C.RANDOM_STATE)
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(sample)
        # Normalizar a la matriz (n_muestras, n_caracteristicas) de la clase "manipulado" (1).
        # shap antiguo devuelve una lista [clase0, clase1]; shap nuevo, un array 3D (n, p, n_clases).
        if isinstance(sv, list):
            sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, 1] if sv.shape[-1] == 2 else sv[1]
        shap.summary_plot(sv, sample, show=False, max_display=15)
        plt.tight_layout(); plt.savefig(OUT_DIR / f"shap_summary{suf}.png", dpi=160, bbox_inches="tight"); plt.close()
        print(f"SHAP guardado en shap_summary{suf}.png")
    except ImportError:
        print("\n(shap no instalado: se omite el resumen SHAP. Para activarlo: pip install shap)")
    except Exception as exc:
        print(f"\n(SHAP omitido por error: {exc})")


if __name__ == "__main__":
    main()
