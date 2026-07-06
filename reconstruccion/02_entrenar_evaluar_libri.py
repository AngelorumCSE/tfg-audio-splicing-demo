# Proyecto desarrollado como parte del Trabajo de Fin de Estudios del Grado en
# Ingeniería Informática de UNIR (2026). Autor: Ángel Carlos Soler Encinas. Licencia MIT.
"""
Entrena y evalúa el detector de splicing CROSS-SOURCE con validación POR HABLANTE.

Pasos:
  1) Extrae las 108 características por ventana de cada audio del dataset LibriSpeech
     (reutiliza features_inferencia) y etiqueta los bordes de inserción (±0,75 s).
  2) Evalúa SIN fuga de datos con GroupKFold agrupando POR HABLANTE (id_hablante_base):
     se entrena con unos hablantes y se prueba con hablantes nunca vistos.
  3) Reporta ROC/PR y AUC a nivel de archivo, métricas por umbral, y un DESGLOSE POR
     TIPO de empalme (cross_speaker, cross_speaker_ruido, same_speaker) para mostrar
     dónde funciona el sistema y dónde no.
  4) Entrena el modelo final con todos los datos y lo guarda para la app.

Uso:
    cd Codigo_y_Resultados
    python3 reconstruccion/02_entrenar_evaluar_libri.py --datos data/libri --folds 5

Salidas (reconstruccion/reports/):
    resumen_libri.txt, roc_pr_libri.png, metricas_por_umbral_libri.csv,
    por_tipo_libri.csv  y  modelo  reconstruccion/modelo_libri.joblib
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
import config as C
from features_inferencia import construir_df_caracteristicas
from posproceso import balancear_train, agrupar_intervalos, hay_solape

META = {"archivo_generado", "archivo_base", "id_hablante_base", "tipo_splicing",
        "manipulado", "indice_ventana", "inicio_ventana_s", "fin_ventana_s",
        "centro_ventana_s", "etiqueta_borde", "score_sospecha"}
OUT_DIR = Path(__file__).resolve().parent / "reports"


def cargar_manifest(datos):
    with open(Path(datos) / "manifest_libri.csv", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f, delimiter=";"))


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def construir_dataset(datos):
    """Construye el DataFrame de ventanas con características y etiqueta_borde."""
    rows = cargar_manifest(datos)
    audio_dir = Path(datos) / "audios"
    partes = []
    for i, r in enumerate(rows, start=1):
        y, _ = librosa.load(audio_dir / r["archivo_generado"], sr=C.SR, mono=True)
        df = construir_df_caracteristicas(y, C.SR)
        df["archivo_generado"] = r["archivo_generado"]
        df["archivo_base"] = r["archivo_base"]
        df["id_hablante_base"] = r["id_hablante_base"]
        df["tipo_splicing"] = r["tipo_splicing"]
        df["manipulado"] = int(r["manipulado"])
        df["centro_ventana_s"] = (df["inicio_ventana_s"] + df["fin_ventana_s"]) / 2
        a, b = fnum(r["inicio_insercion_s"]), fnum(r["fin_insercion_s"])
        if a is None or b is None:
            df["etiqueta_borde"] = 0
        else:
            cerca = (df["centro_ventana_s"].sub(a).abs() <= C.MARGEN_BORDE_S) | \
                    (df["centro_ventana_s"].sub(b).abs() <= C.MARGEN_BORDE_S)
            df["etiqueta_borde"] = cerca.astype(int)
        partes.append(df)
        if i % 20 == 0:
            print(f"  procesados {i}/{len(rows)} audios...")
    return pd.concat(partes, ignore_index=True)


def oof(df, feats, n_folds):
    s = pd.Series(np.nan, index=df.index)
    gkf = GroupKFold(n_splits=n_folds)
    for k, (tr, te) in enumerate(gkf.split(df[feats], df["etiqueta_borde"],
                                           groups=df["id_hablante_base"]), start=1):
        bal = balancear_train(df.iloc[tr])
        m = RandomForestClassifier(n_estimators=C.N_ESTIMATORS, min_samples_leaf=C.MIN_SAMPLES_LEAF,
                                   random_state=C.RANDOM_STATE, n_jobs=-1)
        m.fit(bal[feats], bal["etiqueta_borde"])
        s.iloc[te] = m.predict_proba(df.iloc[te][feats])[:, 1]
        print(f"  pliegue {k}/{n_folds}: {df.iloc[te]['id_hablante_base'].nunique()} hablantes de prueba")
    return s.values


def curvas(y, score, path):
    fpr, tpr, _ = roc_curve(y, score); roc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y, score); apv = average_precision_score(y, score)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(fpr, tpr, color="#1F6FB2", label=f"ROC (AUC={roc:.3f})"); ax[0].plot([0, 1], [0, 1], "--", color="#aaa")
    ax[0].set_title("ROC por archivo"); ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(rec, prec, color="#B00020", label=f"PR (AP={apv:.3f})")
    ax[1].set_title("Precision-Recall por archivo"); ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)
    return roc, apv


def matriz_confusion_fig(df_umbral, umbral, path):
    """Dibuja la matriz de confusión por archivo en el umbral de operación (out-of-fold)."""
    fila = df_umbral.iloc[(df_umbral["umbral"] - umbral).abs().argmin()]
    cm = np.array([[int(fila["tn"]), int(fila["fp"])], [int(fila["fn"]), int(fila["tp"])]])
    nombres = ["VN", "FP", "FN", "VP"]
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Limpio", "Manipulado"])
    ax.set_yticks([0, 1], ["Limpio", "Manipulado"])
    ax.set_xlabel("Predicción del sistema"); ax.set_ylabel("Realidad (ground truth)")
    ax.set_title(f"Matriz de confusión por archivo — umbral {umbral:.2f}")
    vmax = cm.max()
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{nombres[i * 2 + j]}\n{cm[i, j]}", ha="center", va="center",
                    fontsize=13, color="white" if cm[i, j] > 0.6 * vmax else "#222")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", default="data/libri")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Extrayendo características...")
    df = construir_dataset(args.datos)
    feats = [c for c in df.columns if c not in META]
    print(f"Ventanas: {len(df)} | características: {len(feats)} | "
          f"hablantes: {df['id_hablante_base'].nunique()} | archivos: {df['archivo_generado'].nunique()}")
    # Guardar las características (para explicabilidad/robustez del modelo cross-source)
    df.to_csv(Path(args.datos) / "window_features_libri.csv", index=False, sep=";")

    n_folds = min(args.folds, df["id_hablante_base"].nunique())
    print(f"Validación cruzada por hablante: {n_folds} pliegues")
    df["score_sospecha"] = oof(df, feats, n_folds)

    # agregación por archivo
    g = df.groupby("archivo_generado")
    tamper = g["score_sospecha"].max()
    tipo = g["tipo_splicing"].first()
    manip = g["manipulado"].first()
    files = pd.DataFrame({"tamper": tamper, "tipo": tipo, "manipulado": manip})

    roc_f, ap_f = curvas(files["manipulado"].values, files["tamper"].values, OUT_DIR / "roc_pr_libri.png")

    # AUC por tipo (limpios como negativos)
    limpios = files[files["manipulado"] == 0]
    por_tipo = []
    for t in ["cross_speaker", "cross_speaker_ruido", "same_speaker"]:
        sub = files[files["tipo"] == t]
        if len(sub) == 0:
            continue
        y = np.r_[np.zeros(len(limpios)), np.ones(len(sub))]
        s = np.r_[limpios["tamper"].values, sub["tamper"].values]
        a = roc_auc_score(y, s) if len(set(y)) == 2 else float("nan")
        por_tipo.append({"tipo": t, "n": len(sub), "roc_auc_vs_limpios": round(a, 3),
                         "tamper_medio": round(float(sub["tamper"].mean()), 3)})
    pd.DataFrame(por_tipo).to_csv(OUT_DIR / "por_tipo_libri.csv", index=False, sep=";")

    # métricas por umbral (detección + localización con intervalos)
    manifest = {r["archivo_generado"]: r for r in cargar_manifest(args.datos)}
    filas = []
    for thr in C.THRESHOLDS:
        recs = []
        for arch, da in df.groupby("archivo_generado"):
            meta = manifest[arch]; mr = int(meta["manipulado"])
            ivs = agrupar_intervalos(da.sort_values("indice_ventana"), thr)
            det = 1 if ivs else 0; loc = ""
            if mr == 1:
                gi, gf = fnum(meta["inicio_insercion_s"]), fnum(meta["fin_insercion_s"])
                loc = 1 if any(hay_solape(iv, gi, gf) for iv in ivs) else 0
            recs.append({"manip": mr, "det": det, "loc": loc})
        rr = pd.DataFrame(recs); yt, yp = rr["manip"].astype(int), rr["det"].astype(int)
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        man = rr[rr["manip"] == 1]; locd = man["loc"].replace("", 0).astype(int).sum()
        filas.append({"umbral": thr, "accuracy": round(accuracy_score(yt, yp), 4),
                      "precision": round(precision_score(yt, yp, zero_division=0), 4),
                      "recall": round(recall_score(yt, yp, zero_division=0), 4),
                      "f1": round(f1_score(yt, yp, zero_division=0), 4),
                      "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                      "localizacion": f"{locd}/{len(man)}"})
    df_umbral = pd.DataFrame(filas); df_umbral.to_csv(OUT_DIR / "metricas_por_umbral_libri.csv", index=False, sep=";")
    matriz_confusion_fig(df_umbral, 0.50, OUT_DIR / "matriz_confusion_libri.png")
    mejor = df_umbral.loc[df_umbral["f1"].idxmax()]

    # modelo final con todos los datos (para la app)
    bal = balancear_train(df)
    modelo = RandomForestClassifier(n_estimators=C.N_ESTIMATORS, min_samples_leaf=C.MIN_SAMPLES_LEAF,
                                    random_state=C.RANDOM_STATE, n_jobs=-1)
    modelo.fit(bal[feats], bal["etiqueta_borde"])
    # best_threshold = umbral operativo del sistema (0,50, seleccionado en la
    # memoria evitando el punto degenerado de umbrales bajos); se guarda también
    # el mejor umbral por F1 out-of-fold como referencia.
    joblib.dump({"model": modelo, "feature_cols": feats, "meta_cols": sorted(META),
                 "best_threshold": 0.50, "best_threshold_f1": float(mejor["umbral"]),
                 "target": "etiqueta_borde"},
                Path(__file__).resolve().parent / "modelo_libri.joblib")

    txt = [
        "DETECTOR DE SPLICING CROSS-SOURCE (LibriSpeech) — validación POR HABLANTE",
        "=" * 74, "",
        f"Audios: {df['archivo_generado'].nunique()} | hablantes host: {df['id_hablante_base'].nunique()} | "
        f"ventanas: {len(df)}",
        f"Pliegues (GroupKFold por hablante): {n_folds}", "",
        f"AUC por archivo (independiente del umbral):  ROC-AUC = {roc_f:.3f} | PR-AUC = {ap_f:.3f}", "",
        "Capacidad por tipo de empalme (ROC-AUC vs limpios, 0,5 = azar):",
        pd.DataFrame(por_tipo).to_string(index=False), "",
        f"Mejor umbral por F1 (out-of-fold): {mejor['umbral']}  ->  "
        f"accuracy={mejor['accuracy']}  precision={mejor['precision']}  recall={mejor['recall']}  f1={mejor['f1']}  "
        f"localización={mejor['localizacion']}", "",
        "Métricas por umbral:", df_umbral.to_string(index=False), "",
        "Lectura: el ROC-AUC global y, sobre todo, el desglose por tipo indican si el",
        "sistema detecta empalmes con discontinuidad real (cross_speaker / con ruido) y",
        "cómo se comporta en el caso difícil (same_speaker).",
    ]
    (OUT_DIR / "resumen_libri.txt").write_text("\n".join(txt), encoding="utf-8")
    print("\n".join(txt))
    print(f"\nModelo guardado en reconstruccion/modelo_libri.joblib")
    print(f"Informes en {OUT_DIR}")


if __name__ == "__main__":
    main()
