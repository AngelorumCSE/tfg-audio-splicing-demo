from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import librosa
import librosa.display


ROOT = Path(__file__).parent

AUDIO_DIR = ROOT / "data" / "generated"
FEATURES_PATH = ROOT / "data" / "processed" / "window_features_borde.csv"
MANIFEST_PATH = ROOT / "data" / "manifests" / "splicing_manifest.csv"
MODEL_PATH = ROOT / "models" / "random_forest_borde.joblib"

DEFAULT_THRESHOLD = 0.50


st.set_page_config(
    page_title="Detector de splicing de audio",
    page_icon="🎧",
    layout="wide"
)


@st.cache_data
def cargar_features():
    return pd.read_csv(FEATURES_PATH, sep=";")


@st.cache_data
def cargar_manifest():
    return pd.read_csv(MANIFEST_PATH, sep=";")


@st.cache_resource
def cargar_modelo():
    bundle = joblib.load(MODEL_PATH)

    if isinstance(bundle, dict):
        model = bundle["model"]
        feature_cols = bundle["feature_cols"]
        target = bundle.get("target", "etiqueta_borde")
        return model, feature_cols, target, bundle

    raise ValueError("El modelo no está guardado como diccionario con 'model' y 'feature_cols'.")


def convertir_float(valor):
    if pd.isna(valor):
        return None
    texto = str(valor).strip()
    if texto == "":
        return None
    try:
        return float(texto)
    except ValueError:
        return None


def obtener_intervalos_predichos(df_audio, threshold):
    sospechosas = df_audio[df_audio["score_sospecha"] >= threshold].copy()

    if sospechosas.empty:
        return []

    sospechosas = sospechosas.sort_values("inicio_ventana_s")

    intervalos = []

    actual_inicio = None
    actual_fin = None
    actual_score = None

    for _, row in sospechosas.iterrows():
        ini = float(row["inicio_ventana_s"])
        fin = float(row["fin_ventana_s"])
        score = float(row["score_sospecha"])

        if actual_inicio is None:
            actual_inicio = ini
            actual_fin = fin
            actual_score = score
            continue

        if ini <= actual_fin:
            actual_fin = max(actual_fin, fin)
            actual_score = max(actual_score, score)
        else:
            intervalos.append({
                "inicio_s": actual_inicio,
                "fin_s": actual_fin,
                "score_maximo": actual_score
            })
            actual_inicio = ini
            actual_fin = fin
            actual_score = score

    intervalos.append({
        "inicio_s": actual_inicio,
        "fin_s": actual_fin,
        "score_maximo": actual_score
    })

    return intervalos


def hay_solape(intervalo_pred, inicio_gt, fin_gt):
    if inicio_gt is None or fin_gt is None:
        return False

    ini_pred = intervalo_pred["inicio_s"]
    fin_pred = intervalo_pred["fin_s"]

    return max(ini_pred, inicio_gt) <= min(fin_pred, fin_gt)


def generar_grafica_scores(df_audio, threshold, inicio_gt, fin_gt, intervalos_pred):
    fig, ax = plt.subplots(figsize=(13, 4))

    ax.plot(
        df_audio["centro_ventana_s"],
        df_audio["score_sospecha"],
        marker="o",
        label="Score de sospecha"
    )

    ax.axhline(
        threshold,
        linestyle="--",
        label=f"Umbral = {threshold:.2f}"
    )

    if inicio_gt is not None and fin_gt is not None:
        ax.axvspan(
            inicio_gt,
            fin_gt,
            alpha=0.20,
            label="Ground truth"
        )

    for i, intervalo in enumerate(intervalos_pred):
        ax.axvspan(
            intervalo["inicio_s"],
            intervalo["fin_s"],
            alpha=0.12,
            label="Intervalo predicho" if i == 0 else None
        )

    ax.set_title("Evolución temporal del score de sospecha")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Score de sospecha")
    ax.set_ylim(0, max(1.0, df_audio["score_sospecha"].max() + 0.05))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    return fig


def generar_espectrograma(audio_path, inicio_gt, fin_gt, intervalos_pred):
    """
    Genera el espectrograma de forma robusta.

    Esta versión evita errores cuando:
    - el audio es demasiado corto;
    - el audio está vacío o casi en silencio;
    - la STFT genera valores no finitos;
    - np.max devuelve 0 y librosa.amplitude_to_db no puede usarlo bien como referencia.
    """

    y, sr = librosa.load(audio_path, sr=None, mono=True)

    fig, ax = plt.subplots(figsize=(13, 4))

    if y is None or len(y) == 0:
        ax.set_title("Espectrograma no disponible")
        ax.text(
            0.5,
            0.5,
            "No se pudo generar el espectrograma porque el audio está vacío.",
            ha="center",
            va="center",
            transform=ax.transAxes
        )
        ax.set_axis_off()
        return fig

    y = np.asarray(y, dtype=np.float32)

    if not np.all(np.isfinite(y)):
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    # n_fft no puede ser mayor que la longitud del audio.
    n_fft = min(2048, len(y))

    # Para audios extremadamente cortos, evitamos valores inválidos.
    if n_fft < 32:
        ax.set_title("Espectrograma no disponible")
        ax.text(
            0.5,
            0.5,
            "El audio es demasiado corto para generar un espectrograma fiable.",
            ha="center",
            va="center",
            transform=ax.transAxes
        )
        ax.set_axis_off()
        return fig

    hop_length = max(1, n_fft // 4)

    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitud = np.abs(stft)

    max_magnitud = np.max(magnitud)

    if not np.isfinite(max_magnitud) or max_magnitud <= 0:
        max_magnitud = 1.0

    db = librosa.amplitude_to_db(magnitud, ref=max_magnitud)

    img = librosa.display.specshow(
        db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="hz",
        ax=ax
    )

    fig.colorbar(img, ax=ax, format="%+2.0f dB")

    if inicio_gt is not None and fin_gt is not None:
        ax.axvspan(
            inicio_gt,
            fin_gt,
            alpha=0.20,
            label="Ground truth"
        )

    for i, intervalo in enumerate(intervalos_pred):
        ax.axvspan(
            intervalo["inicio_s"],
            intervalo["fin_s"],
            alpha=0.12,
            label="Intervalo predicho" if i == 0 else None
        )

    ax.set_title("Espectrograma del audio")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Frecuencia (Hz)")

    if inicio_gt is not None or intervalos_pred:
        ax.legend(loc="best")

    return fig


def main():
    st.title("Detector de manipulación por splicing en audio")
    st.caption("Aplicación local de demostración del modelo desarrollado para el TFG.")

    rutas_obligatorias = {
        "Audios generados": AUDIO_DIR,
        "Características procesadas": FEATURES_PATH,
        "Manifest": MANIFEST_PATH,
        "Modelo": MODEL_PATH,
    }

    faltantes = [nombre for nombre, ruta in rutas_obligatorias.items() if not ruta.exists()]

    if faltantes:
        st.error("Faltan archivos o carpetas necesarios para ejecutar la aplicación.")
        for item in faltantes:
            st.write(f"- {item}: `{rutas_obligatorias[item]}`")
        st.stop()

    df = cargar_features()
    manifest = cargar_manifest()
    model, feature_cols, target, bundle = cargar_modelo()

    audios = sorted(df["archivo_generado"].unique())

    with st.sidebar:
        st.header("Configuración")

        audio_seleccionado = st.selectbox(
            "Selecciona un audio",
            audios
        )

        threshold = st.slider(
            "Umbral de decisión",
            min_value=0.30,
            max_value=0.75,
            value=DEFAULT_THRESHOLD,
            step=0.05
        )

        st.info(
            "El umbral 0.50 es el seleccionado en la evaluación final "
            "por ofrecer un equilibrio razonable entre precisión y recall."
        )

        st.write("Modelo cargado:")
        st.code(str(MODEL_PATH.name))

        st.write("Columnas usadas por el modelo:")
        st.code(str(len(feature_cols)))

    df_audio = df[df["archivo_generado"] == audio_seleccionado].copy()

    if df_audio.empty:
        st.error("No hay ventanas procesadas para este audio.")
        st.stop()

    missing_cols = [col for col in feature_cols if col not in df_audio.columns]
    if missing_cols:
        st.error("Faltan columnas necesarias para aplicar el modelo.")
        st.write(missing_cols)
        st.stop()

    df_audio = df_audio.sort_values("inicio_ventana_s").reset_index(drop=True)

    X = df_audio[feature_cols]
    df_audio["score_sospecha"] = model.predict_proba(X)[:, 1]
    df_audio["prediccion_ventana"] = (df_audio["score_sospecha"] >= threshold).astype(int)
    df_audio["centro_ventana_s"] = (
        df_audio["inicio_ventana_s"] + df_audio["fin_ventana_s"]
    ) / 2

    tamper_score = float(df_audio["score_sospecha"].max())
    prediccion_archivo = 1 if tamper_score >= threshold else 0

    manifest_audio = manifest[manifest["archivo_generado"] == audio_seleccionado]

    if manifest_audio.empty:
        manipulado_real = None
        tipo_splicing = "desconocido"
        inicio_gt = None
        fin_gt = None
    else:
        row_gt = manifest_audio.iloc[0]
        manipulado_real = int(row_gt["manipulado"])
        tipo_splicing = str(row_gt["tipo_splicing"])
        inicio_gt = convertir_float(row_gt["inicio_insercion_s"])
        fin_gt = convertir_float(row_gt["fin_insercion_s"])

    audio_path = AUDIO_DIR / audio_seleccionado

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(audio_seleccionado)

        if audio_path.exists():
            st.audio(audio_path.read_bytes(), format="audio/wav")
        else:
            st.warning("No se ha encontrado el archivo de audio en data/generated.")

    with col2:
        st.subheader("Resultado")

        if prediccion_archivo == 1:
            st.error("Predicción: audio sospechoso / manipulado")
        else:
            st.success("Predicción: audio limpio")

        st.metric("Tamper score", f"{tamper_score:.4f}")
        st.metric("Umbral aplicado", f"{threshold:.2f}")

        if manipulado_real is not None:
            if manipulado_real == 1:
                st.write(f"Ground truth: manipulado ({tipo_splicing})")
                st.write(f"Intervalo real: {inicio_gt:.3f}s - {fin_gt:.3f}s")
            else:
                st.write("Ground truth: audio limpio")
        else:
            st.write("Ground truth: no disponible")

    intervalos_pred = obtener_intervalos_predichos(df_audio, threshold)

    st.divider()

    st.subheader("Intervalos sospechosos detectados")

    if intervalos_pred:
        df_intervalos = pd.DataFrame(intervalos_pred)

        if inicio_gt is not None and fin_gt is not None:
            df_intervalos["solapa_ground_truth"] = df_intervalos.apply(
                lambda r: hay_solape(r, inicio_gt, fin_gt),
                axis=1
            )

        st.dataframe(df_intervalos, use_container_width=True)
    else:
        st.write("No se han detectado intervalos por encima del umbral seleccionado.")

    st.subheader("Curva temporal de sospecha")
    fig_scores = generar_grafica_scores(
        df_audio,
        threshold,
        inicio_gt,
        fin_gt,
        intervalos_pred
    )
    st.pyplot(fig_scores)

    st.subheader("Espectrograma")
    if audio_path.exists():
        try:
            fig_spec = generar_espectrograma(
                audio_path,
                inicio_gt,
                fin_gt,
                intervalos_pred
            )
            st.pyplot(fig_spec)
        except Exception as e:
            st.warning("No se ha podido generar el espectrograma para este audio.")
            st.code(str(e))

    st.subheader("Datos por ventana")

    columnas_mostrar = [
        "archivo_generado",
        "inicio_ventana_s",
        "fin_ventana_s",
        "centro_ventana_s",
        "score_sospecha",
        "prediccion_ventana",
    ]

    if target in df_audio.columns:
        columnas_mostrar.append(target)

    st.dataframe(
        df_audio[columnas_mostrar],
        use_container_width=True
    )

    csv_descarga = df_audio[columnas_mostrar].to_csv(index=False, sep=";").encode("utf-8")

    st.download_button(
        label="Descargar resultados de este audio en CSV",
        data=csv_descarga,
        file_name=f"resultado_{Path(audio_seleccionado).stem}.csv",
        mime="text/csv"
    )


if __name__ == "__main__":
    main()
