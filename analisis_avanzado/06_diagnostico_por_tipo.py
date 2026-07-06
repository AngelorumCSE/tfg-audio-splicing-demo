# Proyecto desarrollado como parte del Trabajo de Fin de Estudios del Grado en
# Ingeniería Informática de UNIR (2026). Autor: Ángel Carlos Soler Encinas. Licencia MIT.
"""
Diagnóstico honesto por TIPO de empalme (out-of-fold, sin fuga de datos).

Pregunta clave: ¿el detector distingue mejor los empalmes 'mismo_hablante'
(fragmento de OTRA grabación, con discontinuidad acústica real) que los
'mismo_audio' (fragmento de la MISMA grabación, casi sin discontinuidad)?

Si así fuera, existe un resultado limitado pero real y defendible: el sistema
detecta empalmes entre grabaciones distintas, no ediciones dentro de la misma toma.

Calcula, con puntuaciones out-of-fold (GroupKFold por archivo_base) y agregación
por archivo (tamper = máximo):
  - ROC-AUC separando limpios vs cada tipo de empalme.
  - Tasa de detección y de localización por tipo, a varios umbrales.
  - Tabla del compromiso detección de manipulados vs falsos positivos en limpios.

Uso:
    cd Codigo_y_Resultados
    python3 analisis_avanzado/06_diagnostico_por_tipo.py

Salidas (reports/avanzado/):
    diagnostico_por_tipo.csv, diagnostico_por_tipo.txt
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
import config as C
from posproceso import agrupar_intervalos, hay_solape, balancear_train

OUT_DIR = C.REPORTS_DIR / "avanzado"
N_FOLDS = 5


def oof(df, feats):
    s = pd.Series(np.nan, index=df.index)
    gkf = GroupKFold(n_splits=min(N_FOLDS, df["archivo_base"].nunique()))
    for tr, te in gkf.split(df[feats], df["etiqueta_borde"], groups=df["archivo_base"]):
        bal = balancear_train(df.iloc[tr])
        m = RandomForestClassifier(n_estimators=C.N_ESTIMATORS, min_samples_leaf=C.MIN_SAMPLES_LEAF,
                                   random_state=C.RANDOM_STATE, n_jobs=-1)
        m.fit(bal[feats], bal["etiqueta_borde"])
        s.iloc[te] = m.predict_proba(df.iloc[te][feats])[:, 1]
    return s.values


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(C.FEATURES_BORDE_CSV, sep=";").copy()
    feats = [c for c in df.columns if c not in C.META_COLS]
    df["score"] = oof(df, feats)

    with open(C.MANIFEST_CSV, newline="", encoding="utf-8-sig") as f:
        manifest = {r["archivo_generado"]: r for r in csv.DictReader(f, delimiter=";")}

    # tamper score y tipo por archivo
    info = []
    for archivo, da in df.groupby("archivo_generado"):
        meta = manifest.get(archivo, {})
        da = da.sort_values("indice_ventana").rename(columns={"score": "score_sospecha"})
        info.append({
            "archivo": archivo,
            "tipo": meta.get("tipo_splicing", "none"),
            "manipulado": int(meta.get("manipulado", 0)),
            "tamper": float(da["score_sospecha"].max()),
            "df": da, "meta": meta,
        })
    info = pd.DataFrame(info)

    limpios = info[info["tipo"] == "none"]
    m_audio = info[info["tipo"] == "mismo_audio"]
    m_habla = info[info["tipo"] == "mismo_hablante"]

    def auc_contra_limpios(sub):
        y = np.r_[np.zeros(len(limpios)), np.ones(len(sub))]
        s = np.r_[limpios["tamper"].values, sub["tamper"].values]
        return roc_auc_score(y, s) if len(set(y)) == 2 else float("nan")

    auc_audio = auc_contra_limpios(m_audio)
    auc_habla = auc_contra_limpios(m_habla)

    # detección y localización por tipo a varios umbrales
    filas = []
    for thr in [0.30, 0.35, 0.40, 0.45, 0.50]:
        def det_loc(sub, con_loc):
            d = l = 0
            for _, row in sub.iterrows():
                ivs = agrupar_intervalos(row["df"], thr)
                if ivs:
                    d += 1
                    if con_loc:
                        gi, gf = float(row["meta"]["inicio_insercion_s"]), float(row["meta"]["fin_insercion_s"])
                        if any(hay_solape(iv, gi, gf) for iv in ivs):
                            l += 1
            return d, l
        # FP en limpios
        fp, _ = det_loc(limpios, False)
        da_d, da_l = det_loc(m_audio, True)
        ha_d, ha_l = det_loc(m_habla, True)
        filas.append({
            "umbral": thr,
            "FP_limpios": f"{fp}/{len(limpios)}",
            "det_mismo_audio": f"{da_d}/{len(m_audio)}",
            "loc_mismo_audio": f"{da_l}/{len(m_audio)}",
            "det_mismo_hablante": f"{ha_d}/{len(m_habla)}",
            "loc_mismo_hablante": f"{ha_l}/{len(m_habla)}",
        })
    tabla = pd.DataFrame(filas)
    tabla.to_csv(OUT_DIR / "diagnostico_por_tipo.csv", index=False, sep=";")

    txt = [
        "DIAGNÓSTICO POR TIPO DE EMPALME (out-of-fold, sin fuga de datos)",
        "=" * 70, "",
        "Capacidad de separar LIMPIOS vs cada tipo (ROC-AUC, 0,5 = azar):",
        f"  Limpios vs 'mismo_audio'    (misma grabación):   ROC-AUC = {auc_audio:.3f}",
        f"  Limpios vs 'mismo_hablante' (otra grabación):    ROC-AUC = {auc_habla:.3f}",
        "",
        "Tasa de falsos positivos en limpios, y detección/localización por tipo:",
        tabla.to_string(index=False),
        "",
        "Lectura: si 'mismo_hablante' tiene ROC-AUC y detección claramente mayores que",
        "'mismo_audio', el sistema detecta empalmes con discontinuidad acústica real",
        "(entre grabaciones distintas) aunque no las ediciones dentro de la misma toma.",
    ]
    (OUT_DIR / "diagnostico_por_tipo.txt").write_text("\n".join(txt), encoding="utf-8")
    print("\n".join(txt))


if __name__ == "__main__":
    main()
