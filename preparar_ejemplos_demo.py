# Proyecto desarrollado como parte del Trabajo de Fin de Estudios del Grado en
# Ingeniería Informática de UNIR (2026). Autor: Ángel Carlos Soler Encinas. Licencia MIT.
"""
Prepara los ejemplos precargados de la demo (app_tfg.py) en data/demo/.

Los ejemplos de la demo son empalmes cross-source de LibriSpeech, que son los que
el detector reformulado (modelo_libri) sí distingue. Para cada host elegido se
incluye: limpio, empalme de otra voz (cross_speaker) y empalme de otra voz con
ruido/entorno (cross_speaker_ruido). Se omite same_speaker (indetectable por diseño,
sección 6.8 de la memoria).

La selección de hosts es automática y se basa en el propio modelo: se eligen los
que presentan mayor separación entre el limpio (score bajo) y los manipulados
(score alto), de modo que la demo muestre el detector funcionando correctamente.

Escribe (sin tocar los CSV experimentales de data/processed ni data/manifests):
  - data/demo/audios/<12 wav>          (copiados de data/libri/audios)
  - data/demo/manifest_demo.csv        (ground truth de los elegidos)
  - data/demo/features_demo.csv        (ventanas precalculadas de los elegidos)

app_tfg.py usa data/demo/ automáticamente si existe.

Uso:  python3 preparar_ejemplos_demo.py [--n-hosts 4]
"""
from pathlib import Path
import argparse
import shutil

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
LIBRI = ROOT / "data" / "libri"
DEMO = ROOT / "data" / "demo"
MODEL_PATH = ROOT / "models" / "modelo_libri.joblib"
UMBRAL = 0.50
VARIANTES = ["clean", "cross_speaker", "cross_speaker_ruido"]


def variante_de(nombre: str) -> str:
    """Deriva la variante del sufijo del nombre de archivo."""
    s = nombre.replace(".wav", "")
    for t in ["cross_speaker_ruido", "cross_speaker", "same_speaker", "clean"]:
        if s.endswith("_" + t):
            return t
    return "otro"


def main(n_hosts: int) -> None:
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_cols = list(bundle["feature_cols"])

    lf = pd.read_csv(LIBRI / "window_features_libri.csv", sep=";")
    lm = pd.read_csv(LIBRI / "manifest_libri.csv", sep=";")

    # Tamper score por archivo = máximo score de ventana (igual que en la app).
    lf = lf.copy()
    lf["score"] = model.predict_proba(lf[feature_cols])[:, 1]
    tamper = lf.groupby("archivo_generado")["score"].max().rename("tamper").reset_index()
    tamper["variante"] = tamper["archivo_generado"].map(variante_de)
    tamper["base"] = tamper.apply(
        lambda r: r["archivo_generado"].replace("_" + r["variante"] + ".wav", ""), axis=1)

    piv = tamper.pivot_table(index="base", columns="variante", values="tamper")
    for t in VARIANTES:
        if t not in piv.columns:
            piv[t] = np.nan

    # Hosts con limpio por debajo del umbral, priorizando los que detectan ambos
    # empalmes y, después, la mayor separación limpio/manipulado.
    cand = piv[piv["clean"] < (UMBRAL - 0.02)].copy()
    cand["ambos_detectan"] = ((cand["cross_speaker"] >= UMBRAL)
                              & (cand["cross_speaker_ruido"] >= UMBRAL)).astype(int)
    cand["separacion"] = cand[["cross_speaker", "cross_speaker_ruido"]].mean(axis=1) - cand["clean"]
    cand = cand.sort_values(["ambos_detectan", "separacion"], ascending=[False, False])
    hosts = list(cand.head(n_hosts).index)

    print("Hosts elegidos (mayor separación limpio/manipulado):")
    print(cand.loc[hosts, VARIANTES + ["separacion"]].round(3).to_string())

    elegidos = [f"{h}_{t}.wav" for h in hosts for t in VARIANTES
                if (LIBRI / "audios" / f"{h}_{t}.wav").exists()]

    # Manifest de la demo con el esquema del manifest experimental (el de la app).
    demo_man_cols = pd.read_csv(ROOT / "data" / "manifests" / "splicing_manifest.csv",
                                sep=";", nrows=1).columns.tolist()
    filas = []
    for nombre in elegidos:
        m = lm[lm["archivo_generado"] == nombre].iloc[0]
        fila = {c: "" for c in demo_man_cols}
        for c in demo_man_cols:
            if c in m.index:
                fila[c] = m[c]
        fila["id_registro"] = nombre.replace(".wav", "")
        filas.append(fila)

    # Características precalculadas con el esquema del CSV experimental.
    demo_feat_cols = pd.read_csv(ROOT / "data" / "processed" / "window_features_borde.csv",
                                 sep=";", nrows=1).columns.tolist()
    sub = lf[lf["archivo_generado"].isin(elegidos)].copy()
    sub["id_registro"] = sub["archivo_generado"].str.replace(".wav", "", regex=False)
    sub["genero_base"] = ""
    if "etiqueta" not in sub.columns:
        sub["etiqueta"] = sub.get("etiqueta_borde", 0)
    sub["solape_manipulacion_s"] = 0.0
    for c in demo_feat_cols:
        if c not in sub.columns:
            sub[c] = 0

    # Escritura.
    (DEMO / "audios").mkdir(parents=True, exist_ok=True)
    for wav in (DEMO / "audios").glob("*.wav"):
        wav.unlink()
    for nombre in elegidos:
        shutil.copy(LIBRI / "audios" / nombre, DEMO / "audios" / nombre)
    pd.DataFrame(filas, columns=demo_man_cols).to_csv(DEMO / "manifest_demo.csv", sep=";", index=False)
    sub[demo_feat_cols].to_csv(DEMO / "features_demo.csv", sep=";", index=False)

    # Resumen de verificación.
    res = tamper[tamper["archivo_generado"].isin(elegidos)].copy()
    res["prediccion"] = np.where(res["tamper"] >= UMBRAL, "MANIPULADO", "LIMPIO")
    res = res.sort_values(["base", "variante"])
    print(f"\nAudios incluidos ({len(elegidos)}): {len(hosts)} hosts x {len(VARIANTES)} variantes")
    print(res[["archivo_generado", "variante", "tamper", "prediccion"]].round(4).to_string(index=False))
    limpios_ok = (res.loc[res["variante"] == "clean", "prediccion"] == "LIMPIO").all()
    ruido_ok = (res.loc[res["variante"] == "cross_speaker_ruido", "prediccion"] == "MANIPULADO").all()
    print("\n¿Todos los limpios -> LIMPIO?:", "SÍ" if limpios_ok else "NO (revisa)")
    print("¿Todos los cross_speaker_ruido -> MANIPULADO?:", "SÍ" if ruido_ok else "NO (revisa)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-hosts", type=int, default=4, help="nº de hosts (cada uno aporta 3 audios)")
    main(ap.parse_args().n_hosts)
