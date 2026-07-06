# Proyecto desarrollado como parte del Trabajo de Fin de Estudios del Grado en
# Ingeniería Informática de UNIR (2026). Autor: Ángel Carlos Soler Encinas. Licencia MIT.
"""
Desglose por tipo en el umbral operativo y ablación/comparativa en CROSS-SOURCE.

Complementa a reconstruccion/02_entrenar_evaluar_libri.py y a 02_ablacion_baselines.py:
  1) Reproduce las puntuaciones out-of-fold del detector cross-source (GroupKFold
     por hablante, mismo protocolo y semilla) y VERIFICA que coinciden con
     reconstruccion/reports/resumen_libri.txt (ROC-AUC 0,722; matriz 22/8/37/53 a 0,50).
  2) Calcula, en los umbrales 0,50 y 0,30, la detección y la localización POR TIPO
     de empalme (cross_speaker, cross_speaker_ruido, same_speaker): el "recall
     solo-cross" citado en la memoria (§6.8).
  3) Repite la ablación (36 características base vs 108 con deltas) y la comparativa
     de clasificadores (RF, regresión logística, SVM RBF, HistGradientBoosting) con
     el MISMO protocolo out-of-fold, pero sobre el dataset cross-source (§6.9).

Proyecto desarrollado como parte del Trabajo de Fin de Estudios del
Grado en Ingeniería Informática de UNIR.

Uso:
    cd Codigo_y_Resultados
    python3 analisis_avanzado/08_desglose_y_ablacion_cross.py

Requiere data/libri/window_features_libri.csv (generado por reconstruccion/02) y
data/libri/manifest_libri.csv. Salidas en reports/avanzado/:
    desglose_cross_por_tipo.txt / .csv  y  ablacion_baselines_cross.txt / .csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
import config as C
from posproceso import balancear_train, agrupar_intervalos, hay_solape

DATOS = Path("data/libri")
OUT = Path("reports/avanzado")
META = {"archivo_generado", "archivo_base", "id_hablante_base", "tipo_splicing",
        "manipulado", "indice_ventana", "inicio_ventana_s", "fin_ventana_s",
        "centro_ventana_s", "etiqueta_borde", "score_sospecha"}


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def cargar_manifest():
    with open(DATOS / "manifest_libri.csv", newline="", encoding="utf-8-sig") as f:
        return {r["archivo_generado"]: r for r in csv.DictReader(f, delimiter=";")}


def oof_scores(df, feats, modelo_factory, folds=5):
    """Puntuaciones out-of-fold con GroupKFold por hablante (protocolo de reconstruccion/02)."""
    s = pd.Series(np.nan, index=df.index)
    gkf = GroupKFold(n_splits=folds)
    for tr, te in gkf.split(df[feats], df["etiqueta_borde"], groups=df["id_hablante_base"]):
        bal = balancear_train(df.iloc[tr])
        m = modelo_factory()
        m.fit(bal[feats], bal["etiqueta_borde"])
        s.iloc[te] = m.predict_proba(df.iloc[te][feats])[:, 1]
    return s.values


def por_archivo(df, scores):
    d = df.copy()
    d["score_sospecha"] = scores
    g = d.groupby("archivo_generado")
    return d, pd.DataFrame({"tamper": g["score_sospecha"].max(),
                            "tipo": g["tipo_splicing"].first(),
                            "manipulado": g["manipulado"].first()})


def metricas_archivo(files, thr):
    yt = files["manipulado"].astype(int).values
    yp = (files["tamper"] >= thr).astype(int).values
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    return tn, fp, fn, tp


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATOS / "window_features_libri.csv", sep=";")
    if "centro_ventana_s" not in df.columns:
        df["centro_ventana_s"] = (df["inicio_ventana_s"] + df["fin_ventana_s"]) / 2
    feats = [c for c in df.columns if c not in META]
    base = [c for c in feats if not c.startswith(("delta_prev_", "delta_next_"))]
    manifest = cargar_manifest()
    print(f"Ventanas: {len(df)} | feats: {len(feats)} (base: {len(base)}) | "
          f"hablantes: {df['id_hablante_base'].nunique()} | archivos: {df['archivo_generado'].nunique()}")

    rf = lambda: RandomForestClassifier(n_estimators=C.N_ESTIMATORS, min_samples_leaf=C.MIN_SAMPLES_LEAF,
                                        random_state=C.RANDOM_STATE, n_jobs=-1)

    # ---------- 1) Reproducción y verificación ----------
    scores = oof_scores(df, feats, rf)
    dfx, files = por_archivo(df, scores)
    roc = roc_auc_score(files["manipulado"], files["tamper"])
    ap = average_precision_score(files["manipulado"], files["tamper"])
    tn, fp, fn, tp = metricas_archivo(files, 0.50)
    print(f"[verificación] ROC-AUC={roc:.3f} PR-AUC={ap:.3f} | matriz@0.50 = {tn}/{fp}/{fn}/{tp} "
          f"(esperado 0.722/0.896 y 22/8/37/53)")

    # ---------- 2) Desglose por tipo en 0,50 y 0,30 ----------
    lineas = ["DESGLOSE POR TIPO EN EL UMBRAL OPERATIVO — detector cross-source (OOF, por hablante)",
              "=" * 78, "",
              f"Verificación de reproducción: ROC-AUC={roc:.3f} | PR-AUC={ap:.3f} | "
              f"matriz por archivo @0,50 (TN/FP/FN/TP) = {tn}/{fp}/{fn}/{tp}", ""]
    filas_csv = []
    for thr in (0.50, 0.30):
        lineas.append(f"Umbral {thr:.2f}:")
        det_por_tipo = {}
        for t in ("cross_speaker", "cross_speaker_ruido", "same_speaker"):
            sub = files[files["tipo"] == t]
            det = int((sub["tamper"] >= thr).sum())
            # localización con intervalos (mismo posproceso del pipeline)
            loc = 0
            for arch in sub.index:
                meta = manifest[arch]
                da = dfx[dfx["archivo_generado"] == arch].sort_values("indice_ventana")
                ivs = agrupar_intervalos(da, thr)
                gi, gf = fnum(meta["inicio_insercion_s"]), fnum(meta["fin_insercion_s"])
                if gi is not None and any(hay_solape(iv, gi, gf) for iv in ivs):
                    loc += 1
            det_por_tipo[t] = (det, loc, len(sub))
            lineas.append(f"  {t:<20} detección {det:>2}/{len(sub)}  localización {loc:>2}/{len(sub)}")
            filas_csv.append({"umbral": thr, "tipo": t, "n": len(sub), "detectados": det, "localizados": loc})
        cs, cr = det_por_tipo["cross_speaker"], det_por_tipo["cross_speaker_ruido"]
        rec_cross = (cs[0] + cr[0]) / (cs[2] + cr[2])
        loc_cross = (cs[1] + cr[1]) / (cs[2] + cr[2])
        lineas.append(f"  -> SOLO empalmes cross-source (n={cs[2]+cr[2]}): "
                      f"recall {rec_cross:.3f} | localización {loc_cross:.3f}")
        lineas.append("")
    pd.DataFrame(filas_csv).to_csv(OUT / "desglose_cross_por_tipo.csv", index=False, sep=";")
    (OUT / "desglose_cross_por_tipo.txt").write_text("\n".join(lineas), encoding="utf-8")
    print("\n".join(lineas))

    # ---------- 3) Ablación y comparativa en cross-source ----------
    configs = [
        ("Solo base (36) — RF", base, rf),
        ("Base + delta (108) — RF", feats, rf),
        ("RegresiónLogística (108)", feats,
         lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=C.RANDOM_STATE))),
        ("SVM RBF (108)", feats,
         lambda: make_pipeline(StandardScaler(), SVC(probability=True, random_state=C.RANDOM_STATE))),
        ("HistGradientBoosting (108)", feats,
         lambda: HistGradientBoostingClassifier(random_state=C.RANDOM_STATE)),
    ]
    filas = []
    for nombre, cols, factory in configs:
        s = oof_scores(df, cols, factory)
        _, fl = por_archivo(df, s)
        r = roc_auc_score(fl["manipulado"], fl["tamper"])
        a = average_precision_score(fl["manipulado"], fl["tamper"])
        tn2, fp2, fn2, tp2 = metricas_archivo(fl, 0.50)
        prec = tp2 / (tp2 + fp2) if tp2 + fp2 else 0.0
        rec = tp2 / (tp2 + fn2) if tp2 + fn2 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        filas.append({"configuracion": nombre, "roc_auc": round(r, 4), "pr_auc": round(a, 4),
                      "precision@0.50": round(prec, 4), "recall@0.50": round(rec, 4), "f1@0.50": round(f1, 4)})
        print(f"[{nombre}] ROC-AUC={r:.4f} PR-AUC={a:.4f} P={prec:.3f} R={rec:.3f} F1={f1:.3f}")
    tabla = pd.DataFrame(filas)
    tabla.to_csv(OUT / "ablacion_baselines_cross.csv", index=False, sep=";")
    txt = ["ABLACIÓN Y COMPARATIVA DE MODELOS EN CROSS-SOURCE (OOF, GroupKFold por hablante)",
           "=" * 78, "", tabla.to_string(index=False), "",
           "Línea base trivial ('todo manipulado'): precision 0,75 (90/120), recall 1,0, F1 0,857.",
           "Protocolo idéntico a reconstruccion/02: balanceo 1:3 solo en train, semilla 42."]
    (OUT / "ablacion_baselines_cross.txt").write_text("\n".join(txt), encoding="utf-8")
    print("Salidas escritas en reports/avanzado/")


if __name__ == "__main__":
    main()
