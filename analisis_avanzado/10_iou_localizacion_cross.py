# Proyecto desarrollado como parte del Trabajo de Fin de Estudios del Grado en
# Ingeniería Informática de UNIR (2026). Autor: Ángel Carlos Soler Encinas. Licencia MIT.
"""
Localización con criterio ESTRICTO (IoU) en el conjunto cross-source.

La memoria mide la localización con un criterio permisivo (solape no vacío,
sección 6.6) y deja el IoU como refinamiento. Este script lo calcula, con el
mismo protocolo out-of-fold del detector cross-source (GroupKFold por hablante,
balanceo 1:3 solo en entrenamiento, semilla 42):

  - Para cada audio manipulado, IoU del mejor intervalo predicho frente al
    intervalo real de inserción, en los umbrales 0,50 (operativo) y 0,30 (cribado).
  - Agregados: IoU medio sobre todos los manipulados, IoU medio sobre los
    aciertos (solape > 0) y fracción de archivos con IoU >= 0,3 y >= 0,5,
    con desglose por tipo de empalme.

Uso:
    cd Codigo_y_Resultados
    python3 analisis_avanzado/10_iou_localizacion_cross.py

Requiere data/libri/window_features_libri.csv y manifest_libri.csv.
Salidas: reports/avanzado/iou_localizacion_cross.{txt,csv}.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
import config as C
from posproceso import agrupar_intervalos, iou, balancear_train

DATOS = Path("data/libri")
OUT = Path("reports/avanzado")
META = {"archivo_generado", "archivo_base", "id_hablante_base", "tipo_splicing",
        "manipulado", "indice_ventana", "inicio_ventana_s", "fin_ventana_s",
        "centro_ventana_s", "etiqueta_borde", "score_sospecha"}
UMBRALES = (0.50, 0.30)


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATOS / "window_features_libri.csv", sep=";")
    if "centro_ventana_s" not in df.columns:
        df["centro_ventana_s"] = (df["inicio_ventana_s"] + df["fin_ventana_s"]) / 2
    feats = [c for c in df.columns if c not in META]
    with open(DATOS / "manifest_libri.csv", newline="", encoding="utf-8-sig") as f:
        manifest = {r["archivo_generado"]: r for r in csv.DictReader(f, delimiter=";")}

    # Puntuaciones out-of-fold (protocolo de reconstruccion/02 y script 08).
    scores = pd.Series(np.nan, index=df.index)
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(df[feats], df["etiqueta_borde"], groups=df["id_hablante_base"]):
        bal = balancear_train(df.iloc[tr])
        m = RandomForestClassifier(n_estimators=C.N_ESTIMATORS, min_samples_leaf=C.MIN_SAMPLES_LEAF,
                                   random_state=C.RANDOM_STATE, n_jobs=-1)
        m.fit(bal[feats], bal["etiqueta_borde"])
        scores.iloc[te] = m.predict_proba(df.iloc[te][feats])[:, 1]
    df = df.assign(score_sospecha=scores.values)

    filas, lineas = [], ["LOCALIZACIÓN CON IoU — detector cross-source (OOF, GroupKFold por hablante)",
                         "=" * 78, ""]
    for thr in UMBRALES:
        por_tipo: dict[str, list[float]] = {}
        for arch, meta in manifest.items():
            if meta["manipulado"] != "1":
                continue
            gi, gf = fnum(meta["inicio_insercion_s"]), fnum(meta["fin_insercion_s"])
            da = df[df["archivo_generado"] == arch].sort_values("indice_ventana")
            ivs = agrupar_intervalos(da, thr)
            mejor = max((iou(v, gi, gf) for v in ivs), default=0.0)
            por_tipo.setdefault(meta["tipo_splicing"], []).append(mejor)
            filas.append({"umbral": thr, "archivo": arch, "tipo": meta["tipo_splicing"],
                          "iou_mejor": round(mejor, 4)})
        todos = [v for vs in por_tipo.values() for v in vs]
        aciertos = [v for v in todos if v > 0]
        lineas += [f"Umbral {thr:.2f}:",
                   f"  IoU medio (todos los manipulados, n={len(todos)}):    {np.mean(todos):.3f}",
                   f"  IoU medio (solo aciertos con solape, n={len(aciertos)}): "
                   f"{np.mean(aciertos):.3f}" if aciertos else "  sin aciertos",
                   f"  Archivos con IoU >= 0,3: {sum(v >= 0.3 for v in todos)}/{len(todos)}"
                   f"  |  IoU >= 0,5: {sum(v >= 0.5 for v in todos)}/{len(todos)}"]
        for t in sorted(por_tipo):
            vs = por_tipo[t]
            lineas.append(f"    {t:<22} IoU medio {np.mean(vs):.3f} | >=0,3: "
                          f"{sum(v >= 0.3 for v in vs)}/{len(vs)}")
        lineas.append("")
    lineas += ["Lectura: el IoU penaliza los intervalos predichos demasiado anchos o",
               "descentrados; complementa la tasa de solape de la memoria (secciones 6.6",
               "y 6.8) con un criterio estricto, como se proponía en la línea futura."]
    pd.DataFrame(filas).to_csv(OUT / "iou_localizacion_cross.csv", sep=";", index=False)
    (OUT / "iou_localizacion_cross.txt").write_text("\n".join(lineas), encoding="utf-8")
    print("\n".join(lineas))


if __name__ == "__main__":
    main()
