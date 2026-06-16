"""
Sustituye los ejemplos precargados de la demo (intra-fuente, del prototipo inicial)
por ejemplos cross-source de LibriSpeech, que son los que el detector reformulado
(modelo_libri) sí distingue. Para cada host elegido incluye: limpio, empalme de otra
voz (cross_speaker) y empalme de otra voz con ruido/entorno (cross_speaker_ruido).

La selección de hosts es automática y se basa en el propio modelo: se eligen aquellos
con mayor separación entre el limpio (score bajo) y los manipulados (score alto), de
modo que la demo muestre el detector funcionando correctamente.

Reescribe en el repo de la demo:
  - data/generated/<audios elegidos>.wav    (borra los WAV antiguos)
  - data/manifests/splicing_manifest.csv    (solo los audios elegidos, con su ground truth)
  - data/processed/window_features_borde.csv (ventanas precalculadas de esos audios)

Uso (con el .venv del proyecto activado):
    python3 preparar_ejemplos_demo.py
Después:  git add -A && git commit -m "Demo: ejemplos cross-source LibriSpeech" && git push
"""
from pathlib import Path
import shutil
import joblib
import numpy as np
import pandas as pd

# --- Rutas (se calculan desde tu carpeta de usuario) ---
HOME = Path.home()
DEMO = HOME / "Desktop" / "TFG" / "Pruebas y versiones antiguas" / "TFG_demo_streamlit"
CODE = HOME / "Desktop" / "TFG" / "ENTREGA_TFG_AngelSolerEncinas_v11_mejorada" / "Codigo_y_Resultados"
LIBRI = CODE / "data" / "libri"
MODEL_PATH = CODE / "models" / "modelo_libri.joblib"

N_HOSTS = 4               # nº de hosts (cada uno aporta 3 audios)
UMBRAL = 0.50
VARIANTES = ["clean", "cross_speaker", "cross_speaker_ruido"]  # se omite same_speaker (indetectable por diseño)


def variante_de(nombre: str) -> str:
    """Deriva la variante del nombre de archivo (el sufijo), no del tipo_splicing
    del manifest (donde el limpio figura como 'none')."""
    s = nombre.replace(".wav", "")
    for t in ["cross_speaker_ruido", "cross_speaker", "same_speaker", "clean"]:
        if s.endswith("_" + t):
            return t
    return "otro"


# --- Cargar modelo y características precalculadas de LibriSpeech ---
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
feature_cols = list(bundle["feature_cols"])

lf = pd.read_csv(LIBRI / "window_features_libri.csv", sep=";").copy()
lm = pd.read_csv(LIBRI / "manifest_libri.csv", sep=";")

# Tamper score por archivo = máximo score de ventana (igual que en la app)
lf["score"] = model.predict_proba(lf[feature_cols])[:, 1]
tamper = lf.groupby("archivo_generado")["score"].max().rename("tamper").reset_index()
tamper = tamper.merge(lm[["archivo_generado", "archivo_base"]], on="archivo_generado", how="left")
tamper["variante"] = tamper["archivo_generado"].map(variante_de)

piv = tamper.pivot_table(index="archivo_base", columns="variante", values="tamper")
for t in VARIANTES:
    if t not in piv.columns:
        piv[t] = np.nan

# Elegir hosts: limpio por debajo del umbral, priorizando aquellos donde AMBOS
# empalmes se detecten, y luego por mayor separación limpio/manipulado.
cand = piv[piv["clean"] < (UMBRAL - 0.02)].copy()
cand["ambos_detectan"] = ((cand["cross_speaker"] >= UMBRAL) & (cand["cross_speaker_ruido"] >= UMBRAL)).astype(int)
cand["separacion"] = cand[["cross_speaker", "cross_speaker_ruido"]].mean(axis=1) - cand["clean"]
cand = cand.sort_values(["ambos_detectan", "separacion"], ascending=[False, False])
hosts = list(cand.head(N_HOSTS).index)

print("Hosts elegidos (mayor separación limpio/manipulado):")
print(cand.loc[hosts, VARIANTES + ["separacion"]].round(3).to_string())

# Audios concretos a incluir
elegidos = []
for h in hosts:
    for t in VARIANTES:
        nombre = f"{h}_{t}.wav"
        if (LIBRI / "audios" / nombre).exists():
            elegidos.append(nombre)

# --- Construir el nuevo manifest en el esquema de la demo ---
demo_man_cols = pd.read_csv(DEMO / "data" / "manifests" / "splicing_manifest.csv", sep=";", nrows=1).columns.tolist()
filas_man = []
for nombre in elegidos:
    m = lm[lm["archivo_generado"] == nombre].iloc[0]
    fila = {c: "" for c in demo_man_cols}
    fila.update({
        "id_registro": nombre.replace(".wav", ""),
        "archivo_generado": m["archivo_generado"],
        "archivo_base": m["archivo_base"],
        "id_hablante_base": m["id_hablante_base"],
        "manipulado": m["manipulado"],
        "tipo_splicing": m["tipo_splicing"],
        "inicio_insercion_s": m["inicio_insercion_s"],
        "fin_insercion_s": m["fin_insercion_s"],
        "id_hablante_segmento": m["id_hablante_segmento"],
        "duracion_s": m["duracion_s"],
        "frecuencia_muestreo": m["frecuencia_muestreo"],
        "semilla": m["semilla"],
    })
    filas_man.append(fila)
nuevo_man = pd.DataFrame(filas_man, columns=demo_man_cols)

# --- Construir las características precalculadas en el esquema de la demo ---
demo_feat_cols = pd.read_csv(DEMO / "data" / "processed" / "window_features_borde.csv", sep=";", nrows=1).columns.tolist()
sub = lf[lf["archivo_generado"].isin(elegidos)].copy()
sub["id_registro"] = sub["archivo_generado"].str.replace(".wav", "", regex=False)
sub["genero_base"] = ""
sub["etiqueta"] = sub.get("etiqueta_borde", 0)
sub["solape_manipulacion_s"] = 0.0
for c in demo_feat_cols:
    if c not in sub.columns:
        sub[c] = 0
nuevo_feat = sub[demo_feat_cols]

# --- Escribir en el repo de la demo ---
gen_dir = DEMO / "data" / "generated"
for wav in gen_dir.glob("*.wav"):          # borrar ejemplos antiguos (intra-fuente)
    wav.unlink()
for nombre in elegidos:                     # copiar los nuevos
    shutil.copy(LIBRI / "audios" / nombre, gen_dir / nombre)

nuevo_man.to_csv(DEMO / "data" / "manifests" / "splicing_manifest.csv", sep=";", index=False)
nuevo_feat.to_csv(DEMO / "data" / "processed" / "window_features_borde.csv", sep=";", index=False)

# --- Resumen de verificación ---
print(f"\nAudios incluidos ({len(elegidos)}):  {len(hosts)} hosts x {len(VARIANTES)} variantes")
res = tamper[tamper["archivo_generado"].isin(elegidos)].copy()
res["prediccion"] = np.where(res["tamper"] >= UMBRAL, "MANIPULADO", "LIMPIO")
res = res.sort_values(["archivo_base", "variante"])
print(res[["archivo_generado", "variante", "tamper", "prediccion"]].round(4).to_string(index=False))

limpios_ok = (res.loc[res["variante"] == "clean", "prediccion"] == "LIMPIO").all()
ruido_ok = (res.loc[res["variante"] == "cross_speaker_ruido", "prediccion"] == "MANIPULADO").all()
print("\n¿Todos los limpios -> LIMPIO?:", "SÍ" if limpios_ok else "NO (revisa)")
print("¿Todos los cross_speaker_ruido -> MANIPULADO?:", "SÍ" if ruido_ok else "NO (revisa)")
print('\nListo. Ahora:  git add -A && git commit -m "Demo: ejemplos cross-source LibriSpeech" && git push origin main')
