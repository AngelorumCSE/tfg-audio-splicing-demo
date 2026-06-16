"""
Aplicación Streamlit del prototipo de detección de manipulación por splicing.

La app mantiene los ejemplos precargados del TFG y añade un segundo modo de uso:
subir un audio externo para analizarlo con el mismo pipeline de inferencia.

Flujo de inferencia para audio externo:
1. Carga temporal del audio subido por el usuario.
2. Conversión a mono y remuestreo a 16 kHz, igual que en el entrenamiento.
3. Segmentación en ventanas de 1 segundo con salto de 0,5 segundos.
4. Extracción de características acústicas por ventana.
5. Cálculo de deltas respecto a la ventana anterior y siguiente.
6. Aplicación del modelo Random Forest entrenado.
7. Visualización de tamper score, intervalos sospechosos, curva temporal y espectrograma.

Nota metodológica:
- Para los audios precargados se conserva el ground truth del dataset generado.
- Para audios subidos por el usuario no existe ground truth, por lo que la app solo muestra
  la predicción del sistema y las zonas temporales que conviene revisar manualmente.
"""

from __future__ import annotations

from pathlib import Path
import os
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple

import joblib
import librosa
import librosa.display
import imageio_ffmpeg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st


ROOT = Path(__file__).parent

AUDIO_DIR = ROOT / "data" / "generated"
FEATURES_PATH = ROOT / "data" / "processed" / "window_features_borde.csv"
MANIFEST_PATH = ROOT / "data" / "manifests" / "splicing_manifest.csv"

# Modelo: se prefiere el detector reformulado cross-source (LibriSpeech) si está
# disponible; si no, se usa el modelo intra-fuente original. Ambos comparten el
# mismo formato (108 características), por lo que la inferencia es idéntica.
_MODELO_LIBRI = ROOT / "models" / "modelo_libri.joblib"
MODEL_PATH = _MODELO_LIBRI if _MODELO_LIBRI.exists() else ROOT / "models" / "random_forest_borde.joblib"

# Informe forense en PDF (opcional): se importa de src/. Si no está disponible,
# la app sigue funcionando sin el botón de descarga del informe.
import sys
sys.path.append(str(ROOT / "src"))
try:
    from informe_forense import construir_informe_forense_bytes, sha256_de_bytes
    INFORME_FORENSE_DISPONIBLE = True
except Exception:
    INFORME_FORENSE_DISPONIBLE = False

# Parámetros usados en el pipeline experimental del TFG.
SR_MODELO = 16000
VENTANA_S = 1.0
SALTO_S = 0.5
DEFAULT_THRESHOLD = 0.50

# Formatos aceptados por la demo. La carga de audios subidos se hace de forma
# robusta mediante FFmpeg cuando el códec no puede leerse directamente con soundfile.
EXTENSIONES_AUDIO = ["wav", "mp3", "m4a", "ogg", "flac"]

# Límites de seguridad para evitar que Streamlit Community Cloud se quede sin memoria.
# La demo pública está pensada para audios cortos o medios; para audios largos se
# recomienda ejecutar el proyecto localmente.
MAX_UPLOAD_MB = 20
MAX_DURATION_S = 120
FFMPEG_TIMEOUT_S = 90

# Columnas no usadas como variables predictoras. El resto de columnas numéricas
# del CSV de entrenamiento son las características del modelo.
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
    "centro_ventana_s",
    "etiqueta",
    "etiqueta_borde",
    "solape_manipulacion_s",
]


st.set_page_config(
    page_title="Detector de splicing de audio",
    page_icon="🎧",
    layout="wide",
)


@st.cache_data
def cargar_features() -> pd.DataFrame:
    """Carga las características precalculadas de los audios de ejemplo."""
    return pd.read_csv(FEATURES_PATH, sep=";")


@st.cache_data
def cargar_manifest() -> pd.DataFrame:
    """Carga el manifest con la información de ground truth de los ejemplos."""
    return pd.read_csv(MANIFEST_PATH, sep=";")


@st.cache_resource
def cargar_modelo():
    """Carga el modelo entrenado y su lista de columnas esperadas."""
    bundle = joblib.load(MODEL_PATH)

    if not isinstance(bundle, dict):
        raise ValueError(
            "El modelo no está guardado como diccionario con 'model' y 'feature_cols'."
        )

    model = bundle["model"]
    feature_cols = list(bundle["feature_cols"])
    target = bundle.get("target", "etiqueta_borde")
    return model, feature_cols, target, bundle


def convertir_float(valor) -> Optional[float]:
    """Convierte campos del manifest a float tratando valores vacíos."""
    if pd.isna(valor):
        return None
    texto = str(valor).strip()
    if texto == "":
        return None
    try:
        return float(texto)
    except ValueError:
        return None


def obtener_intervalos_predichos(df_audio: pd.DataFrame, threshold: float) -> List[Dict[str, float]]:
    """Agrupa ventanas consecutivas sospechosas en intervalos temporales."""
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

        # Las ventanas se solapan entre sí por diseño. Si una ventana sospechosa
        # empieza antes de que termine la anterior, se fusiona en un único intervalo.
        if ini <= actual_fin:
            actual_fin = max(actual_fin, fin)
            actual_score = max(actual_score, score)
        else:
            intervalos.append(
                {"inicio_s": actual_inicio, "fin_s": actual_fin, "score_maximo": actual_score}
            )
            actual_inicio = ini
            actual_fin = fin
            actual_score = score

    intervalos.append(
        {"inicio_s": actual_inicio, "fin_s": actual_fin, "score_maximo": actual_score}
    )
    return intervalos


def hay_solape(intervalo_pred: Dict[str, float], inicio_gt: Optional[float], fin_gt: Optional[float]) -> bool:
    """Comprueba si un intervalo predicho solapa con el intervalo real manipulado."""
    if inicio_gt is None or fin_gt is None:
        return False

    ini_pred = intervalo_pred["inicio_s"]
    fin_pred = intervalo_pred["fin_s"]
    return max(ini_pred, inicio_gt) <= min(fin_pred, fin_gt)


def extraer_features_ventana(y: np.ndarray, sr: int) -> Dict[str, float]:
    """Extrae las mismas características acústicas utilizadas durante el entrenamiento."""
    if len(y) == 0:
        raise ValueError("Ventana vacía")

    y = np.asarray(y, dtype=np.float32)
    if not np.all(np.isfinite(y)):
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    features: Dict[str, float] = {}

    # Coeficientes cepstrales en las frecuencias de Mel.
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f"mfcc_{i + 1}_mean"] = float(np.mean(mfcc[i]))
        features[f"mfcc_{i + 1}_std"] = float(np.std(mfcc[i]))

    # Energía RMS.
    rms = librosa.feature.rms(y=y)
    features["rms_mean"] = float(np.mean(rms))
    features["rms_std"] = float(np.std(rms))

    # Tasa de cruces por cero.
    zcr = librosa.feature.zero_crossing_rate(y)
    features["zcr_mean"] = float(np.mean(zcr))
    features["zcr_std"] = float(np.std(zcr))

    # Descriptores espectrales.
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features["spectral_centroid_mean"] = float(np.mean(centroid))
    features["spectral_centroid_std"] = float(np.std(centroid))

    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features["spectral_bandwidth_mean"] = float(np.mean(bandwidth))
    features["spectral_bandwidth_std"] = float(np.std(bandwidth))

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features["spectral_rolloff_mean"] = float(np.mean(rolloff))
    features["spectral_rolloff_std"] = float(np.std(rolloff))

    return features


def validar_archivo_subido(uploaded_file) -> Tuple[bool, str]:
    """Valida tamaño y extensión antes de intentar decodificar el audio."""
    if uploaded_file is None:
        return False, "No se ha recibido ningún archivo."

    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        return (
            False,
            f"El archivo pesa {size_mb:.1f} MB. En la demo web el límite es "
            f"{MAX_UPLOAD_MB} MB para evitar agotar la memoria del servidor. "
            "Recorta el audio, conviértelo a WAV/MP3 más corto o ejecútalo localmente."
        )

    suffix = Path(uploaded_file.name).suffix.lower().replace(".", "")
    if suffix not in EXTENSIONES_AUDIO:
        return (
            False,
            "Formato no admitido. Usa WAV, MP3, M4A, OGG o FLAC."
        )

    return True, ""


def obtener_ffmpeg_exe() -> str:
    """Devuelve una ruta a FFmpeg, usando el binario incluido por imageio-ffmpeg."""
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        # Último recurso: confiar en que el sistema tenga ffmpeg en PATH.
        return "ffmpeg"


def convertir_a_wav_con_ffmpeg(input_path: Path) -> Path:
    """
    Convierte un archivo de audio a WAV mono 16 kHz con FFmpeg.

    Esta conversión soluciona el problema típico de Streamlit Cloud con M4A/MP3:
    soundfile no soporta todos los códecs y audioread puede no encontrar backend.
    """
    output_path = input_path.with_suffix(".converted.wav")
    ffmpeg_exe = obtener_ffmpeg_exe()

    cmd = [
        ffmpeg_exe,
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-t", str(MAX_DURATION_S),
        "-i", str(input_path),
        "-vn",
        "-ac", "1",
        "-ar", str(SR_MODELO),
        "-f", "wav",
        str(output_path),
    ]

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=FFMPEG_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError(
            "El audio tarda demasiado en convertirse. Para la demo web usa un archivo "
            "más corto o de menor tamaño."
        ) from exc
    except subprocess.CalledProcessError as exc:
        mensaje = exc.stderr.decode("utf-8", errors="ignore").strip()
        raise ValueError(
            "No se ha podido decodificar este audio. Puede deberse a un códec no "
            "soportado o a un archivo dañado. Prueba a exportarlo como WAV o MP3."
            + (f"\nDetalle técnico: {mensaje}" if mensaje else "")
        ) from exc

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise ValueError("La conversión del audio no generó una señal válida.")

    return output_path


def leer_wav_seguro(wav_path: Path) -> Tuple[np.ndarray, int]:
    """Lee un WAV ya normalizado y garantiza mono float32 a 16 kHz."""
    y, sr = sf.read(wav_path, dtype="float32", always_2d=False)

    if y.ndim > 1:
        y = np.mean(y, axis=1).astype(np.float32)

    if sr != SR_MODELO:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR_MODELO)
        sr = SR_MODELO

    y = np.asarray(y, dtype=np.float32)
    if not np.all(np.isfinite(y)):
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    max_muestras = int(MAX_DURATION_S * sr)
    if len(y) > max_muestras:
        y = y[:max_muestras]

    return y, sr


def cargar_audio_desde_bytes(audio_bytes: bytes, suffix: str) -> Tuple[np.ndarray, int, Path]:
    """
    Guarda temporalmente el audio subido y lo carga de forma robusta.

    Primero se intenta una lectura directa para formatos sencillos. Si falla, se
    convierte con FFmpeg a WAV mono 16 kHz. Así se evitan errores NoBackendError
    en M4A/MP3 dentro de Streamlit Community Cloud.
    """
    suffix = suffix if suffix.startswith(".") else f".{suffix}"
    suffix = suffix.lower()

    tmp_path = None
    wav_convertido = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp_path = Path(tmp.name)

        # Para WAV/FLAC/OGG suele bastar soundfile/librosa. Para M4A/MP3 se fuerza
        # FFmpeg, porque en Streamlit Cloud audioread puede no tener backend.
        if suffix in {".m4a", ".mp3"}:
            wav_convertido = convertir_a_wav_con_ffmpeg(tmp_path)
            y, sr = leer_wav_seguro(wav_convertido)
        else:
            try:
                y, sr = librosa.load(
                    tmp_path,
                    sr=SR_MODELO,
                    mono=True,
                    duration=MAX_DURATION_S,
                )
            except Exception:
                wav_convertido = convertir_a_wav_con_ffmpeg(tmp_path)
                y, sr = leer_wav_seguro(wav_convertido)

        if y is None or len(y) == 0:
            raise ValueError("No se ha podido obtener señal de audio del archivo subido.")

        return y, sr, tmp_path

    finally:
        # El archivo convertido es auxiliar y puede eliminarse inmediatamente.
        if wav_convertido is not None:
            try:
                wav_convertido.unlink(missing_ok=True)
            except Exception:
                pass


@st.cache_data(show_spinner=False, max_entries=3)
def procesar_audio_subido(audio_bytes: bytes, filename: str) -> pd.DataFrame:
    """
    Construye el dataframe de características para un audio externo.

    Esta función está cacheada para evitar recalcular todas las características
    cada vez que el usuario mueve el slider de umbral.
    """
    suffix = Path(filename).suffix or ".wav"
    y, sr, tmp_path = cargar_audio_desde_bytes(audio_bytes, suffix)

    try:
        if y is None or len(y) == 0:
            raise ValueError("No se ha podido cargar señal de audio del archivo subido.")

        y = np.asarray(y, dtype=np.float32)
        if not np.all(np.isfinite(y)):
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        ventana_muestras = int(VENTANA_S * sr)
        salto_muestras = int(SALTO_S * sr)

        if ventana_muestras <= 0 or salto_muestras <= 0:
            raise ValueError("Parámetros de ventana no válidos.")

        # Si el audio dura menos de una ventana, se rellena con silencio para poder
        # generar una predicción. La app lo avisará indirectamente por duración.
        if len(y) < ventana_muestras:
            y = np.pad(y, (0, ventana_muestras - len(y)), mode="constant")

        registros = []
        total_muestras = len(y)

        for idx, inicio in enumerate(range(0, total_muestras - ventana_muestras + 1, salto_muestras)):
            fin = inicio + ventana_muestras
            inicio_s = inicio / sr
            fin_s = fin / sr
            ventana = y[inicio:fin]

            registro = {
                "archivo_generado": filename,
                "indice_ventana": idx,
                "inicio_ventana_s": round(inicio_s, 3),
                "fin_ventana_s": round(fin_s, 3),
                "centro_ventana_s": round((inicio_s + fin_s) / 2, 3),
            }
            registro.update(extraer_features_ventana(ventana, sr))
            registros.append(registro)

        df_audio = pd.DataFrame(registros)

        # Cálculo de diferencias absolutas con la ventana previa y posterior.
        base_feature_cols = [c for c in df_audio.columns if c not in META_COLS]
        base_feature_cols = [c for c in base_feature_cols if not c.startswith("delta_prev_") and not c.startswith("delta_next_")]

        df_audio = df_audio.sort_values("indice_ventana").reset_index(drop=True)
        delta_parts = []
        for col in base_feature_cols:
            prev = df_audio[col].shift(1)
            next_ = df_audio[col].shift(-1)
            delta_parts.append(
                pd.DataFrame(
                    {
                        f"delta_prev_{col}": (df_audio[col] - prev).abs(),
                        f"delta_next_{col}": (df_audio[col] - next_).abs(),
                    }
                )
            )

        if delta_parts:
            df_audio = pd.concat([df_audio] + delta_parts, axis=1)

        return df_audio.fillna(0)
    finally:
        # El archivo temporal se elimina tras extraer las características. Para el
        # reproductor y el espectrograma se usa de nuevo el contenido en memoria.
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def dataframe_ejemplo_precalculado(
    df: pd.DataFrame,
    audio_seleccionado: str,
    model,
    feature_cols: List[str],
    threshold: float,
) -> pd.DataFrame:
    """Obtiene las ventanas precalculadas de un audio de ejemplo y calcula scores."""
    df_audio = df[df["archivo_generado"] == audio_seleccionado].copy()

    if df_audio.empty:
        raise ValueError("No hay ventanas procesadas para este audio.")

    df_audio = df_audio.sort_values("inicio_ventana_s").reset_index(drop=True)

    missing_cols = [col for col in feature_cols if col not in df_audio.columns]
    if missing_cols:
        raise ValueError(f"Faltan columnas necesarias para aplicar el modelo: {missing_cols}")

    X = df_audio[feature_cols]
    df_audio["score_sospecha"] = model.predict_proba(X)[:, 1]
    df_audio["prediccion_ventana"] = (df_audio["score_sospecha"] >= threshold).astype(int)
    return df_audio


def dataframe_audio_subido_con_scores(
    df_audio: pd.DataFrame,
    model,
    feature_cols: List[str],
    threshold: float,
) -> pd.DataFrame:
    """Alinea columnas de un audio subido y calcula el score del modelo."""
    df_audio = df_audio.copy().sort_values("inicio_ventana_s").reset_index(drop=True)

    # Por robustez, si en el futuro el modelo espera alguna variable que no se
    # haya generado para un audio externo, se rellena con 0 y se mantiene el orden.
    for col in feature_cols:
        if col not in df_audio.columns:
            df_audio[col] = 0.0

    X = df_audio[feature_cols]
    df_audio["score_sospecha"] = model.predict_proba(X)[:, 1]
    df_audio["prediccion_ventana"] = (df_audio["score_sospecha"] >= threshold).astype(int)
    return df_audio


def generar_grafica_scores(
    df_audio: pd.DataFrame,
    threshold: float,
    inicio_gt: Optional[float],
    fin_gt: Optional[float],
    intervalos_pred: List[Dict[str, float]],
):
    """Dibuja la evolución temporal del score de sospecha."""
    fig, ax = plt.subplots(figsize=(13, 4))

    ax.plot(
        df_audio["centro_ventana_s"],
        df_audio["score_sospecha"],
        marker="o",
        label="Score de sospecha",
    )

    ax.axhline(threshold, linestyle="--", label=f"Umbral = {threshold:.2f}")

    if inicio_gt is not None and fin_gt is not None:
        ax.axvspan(inicio_gt, fin_gt, alpha=0.20, label="Ground truth")

    for i, intervalo in enumerate(intervalos_pred):
        ax.axvspan(
            intervalo["inicio_s"],
            intervalo["fin_s"],
            alpha=0.12,
            label="Intervalo predicho" if i == 0 else None,
        )

    ax.set_title("Evolución temporal del score de sospecha")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Score de sospecha")
    ax.set_ylim(0, max(1.0, float(df_audio["score_sospecha"].max()) + 0.05))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def generar_espectrograma_desde_path(
    audio_path: Path,
    inicio_gt: Optional[float],
    fin_gt: Optional[float],
    intervalos_pred: List[Dict[str, float]],
):
    """Genera el espectrograma desde una ruta de audio."""
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    return generar_espectrograma_desde_signal(y, sr, inicio_gt, fin_gt, intervalos_pred)


def generar_espectrograma_desde_bytes(
    audio_bytes: bytes,
    filename: str,
    inicio_gt: Optional[float],
    fin_gt: Optional[float],
    intervalos_pred: List[Dict[str, float]],
):
    """Genera el espectrograma de un archivo subido sin persistirlo en disco."""
    suffix = Path(filename).suffix or ".wav"
    y, sr, tmp_path = cargar_audio_desde_bytes(audio_bytes, suffix)
    try:
        return generar_espectrograma_desde_signal(y, sr, inicio_gt, fin_gt, intervalos_pred)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def generar_espectrograma_desde_signal(
    y: np.ndarray,
    sr: int,
    inicio_gt: Optional[float],
    fin_gt: Optional[float],
    intervalos_pred: List[Dict[str, float]],
):
    """Genera el espectrograma de forma robusta a partir de una señal de audio."""
    fig, ax = plt.subplots(figsize=(13, 4))

    if y is None or len(y) == 0:
        ax.set_title("Espectrograma no disponible")
        ax.text(
            0.5,
            0.5,
            "No se pudo generar el espectrograma porque el audio está vacío.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return fig

    y = np.asarray(y, dtype=np.float32)
    if not np.all(np.isfinite(y)):
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    n_fft = min(2048, len(y))
    if n_fft < 32:
        ax.set_title("Espectrograma no disponible")
        ax.text(
            0.5,
            0.5,
            "El audio es demasiado corto para generar un espectrograma fiable.",
            ha="center",
            va="center",
            transform=ax.transAxes,
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
        ax=ax,
    )

    fig.colorbar(img, ax=ax, format="%+2.0f dB")

    if inicio_gt is not None and fin_gt is not None:
        ax.axvspan(inicio_gt, fin_gt, alpha=0.20, label="Ground truth")

    for i, intervalo in enumerate(intervalos_pred):
        ax.axvspan(
            intervalo["inicio_s"],
            intervalo["fin_s"],
            alpha=0.12,
            label="Intervalo predicho" if i == 0 else None,
        )

    ax.set_title("Espectrograma del audio")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Frecuencia (Hz)")

    if inicio_gt is not None or intervalos_pred:
        ax.legend(loc="best")

    fig.tight_layout()
    return fig


def generar_grafica_onda(
    y: np.ndarray,
    sr: int,
    inicio_gt: Optional[float],
    fin_gt: Optional[float],
    intervalos_pred: List[Dict[str, float]],
):
    """Dibuja la forma de onda marcando ground truth e intervalos predichos."""
    fig, ax = plt.subplots(figsize=(13, 3))
    if y is None or len(y) == 0:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "Forma de onda no disponible.", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    # Submuestreo solo para dibujar (no afecta al análisis), para no saturar la figura.
    paso = max(1, len(y) // 8000)
    y_plot = y[::paso]
    t = np.arange(len(y_plot)) * paso / sr
    ax.plot(t, y_plot, lw=0.6, color="#1F6FB2")

    if inicio_gt is not None and fin_gt is not None:
        ax.axvspan(inicio_gt, fin_gt, alpha=0.20, color="green", label="Ground truth")
    for i, intervalo in enumerate(intervalos_pred):
        ax.axvspan(intervalo["inicio_s"], intervalo["fin_s"], alpha=0.15, color="red",
                   label="Intervalo predicho" if i == 0 else None)

    ax.set_title("Forma de onda con zonas marcadas")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Amplitud")
    ax.grid(True, alpha=0.3)
    if inicio_gt is not None or intervalos_pred:
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def modo_lote(model, feature_cols: List[str], threshold: float) -> None:
    """Procesa varios audios subidos y muestra una tabla-resumen descargable."""
    st.subheader("Procesamiento por lotes")
    st.caption(
        "Sube varios audios para obtener una tabla-resumen con la predicción y el tamper "
        "score de cada uno. Se aplican los mismos límites que en el modo individual."
    )
    archivos = st.file_uploader(
        "Sube uno o varios audios",
        type=EXTENSIONES_AUDIO,
        accept_multiple_files=True,
        help=f"Formatos: WAV, MP3, M4A, OGG, FLAC. Límite por archivo: {MAX_UPLOAD_MB} MB.",
    )
    if not archivos:
        st.info("Selecciona al menos un archivo para iniciar el análisis por lotes.")
        return

    filas = []
    barra = st.progress(0.0)
    for i, f in enumerate(archivos, start=1):
        registro = {"archivo": f.name, "estado": "OK"}
        try:
            ok, msg = validar_archivo_subido(f)
            if not ok:
                registro.update({"estado": msg, "prediccion": "-", "tamper_score": None,
                                 "intervalos": None})
            else:
                audio_bytes = f.getvalue()
                df_subido = procesar_audio_subido(audio_bytes, f.name)
                df_audio = dataframe_audio_subido_con_scores(df_subido, model, feature_cols, threshold)
                df_audio["centro_ventana_s"] = (df_audio["inicio_ventana_s"] + df_audio["fin_ventana_s"]) / 2
                tamper = float(df_audio["score_sospecha"].max())
                intervalos = obtener_intervalos_predichos(df_audio, threshold)
                registro.update({
                    "prediccion": "Sospechoso" if tamper >= threshold else "Limpio",
                    "tamper_score": round(tamper, 4),
                    "intervalos": len(intervalos),
                })
        except Exception as exc:  # un archivo problemático no debe abortar el lote
            registro.update({"estado": f"Error: {exc}", "prediccion": "-",
                             "tamper_score": None, "intervalos": None})
        filas.append(registro)
        barra.progress(i / len(archivos))

    df_resumen = pd.DataFrame(filas, columns=["archivo", "prediccion", "tamper_score",
                                              "intervalos", "estado"])
    st.dataframe(df_resumen, use_container_width=True)

    n_ok = int((df_resumen["prediccion"] != "-").sum())
    n_sosp = int((df_resumen["prediccion"] == "Sospechoso").sum())
    st.write(f"Analizados correctamente: {n_ok}/{len(archivos)}  ·  Marcados como sospechosos: {n_sosp}")

    csv = df_resumen.to_csv(index=False, sep=";").encode("utf-8")
    st.download_button("Descargar resumen del lote (CSV)", data=csv,
                       file_name="resumen_lote.csv", mime="text/csv")


def main() -> None:
    st.title("Detector de manipulación por splicing en audio")
    st.caption(
        "Aplicación de demostración del modelo desarrollado para el TFG. "
        "Permite analizar ejemplos precargados o subir un audio propio."
    )

    rutas_obligatorias = {
        "Carpeta de audios de ejemplo": AUDIO_DIR,
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

    # Solo se ofrecen en el selector los ejemplos cuyo WAV exista realmente.
    audios_con_features = set(df["archivo_generado"].unique())
    audios_con_fichero = {p.name for p in AUDIO_DIR.glob("*.wav")}
    audios_ejemplo = sorted(audios_con_features.intersection(audios_con_fichero))

    if not audios_ejemplo:
        st.error("No se han encontrado audios de ejemplo con características precalculadas.")
        st.stop()

    with st.sidebar:
        st.header("Configuración")

        modo = st.radio(
            "Modo de análisis",
            ["Ejemplos precargados", "Subir audio propio", "Procesar lote (varios audios)"],
            index=0,
        )

        audio_seleccionado = None
        uploaded_file = None

        if modo == "Ejemplos precargados":
            audio_seleccionado = st.selectbox("Selecciona un audio", audios_ejemplo)
        elif modo == "Procesar lote (varios audios)":
            st.caption("El análisis por lotes se realiza en el panel principal.")
        else:
            uploaded_file = st.file_uploader(
                "Sube un audio para analizarlo",
                type=EXTENSIONES_AUDIO,
                help=f"Formatos aceptados: WAV, MP3, M4A, OGG y FLAC. Límite recomendado en la demo web: {MAX_UPLOAD_MB} MB.",
            )
            st.caption(
                "Los audios subidos se procesan de forma temporal durante la sesión "
                "y no tienen ground truth asociado. Para proteger la demo web, se analiza "
                f"como máximo los primeros {MAX_DURATION_S} segundos."
            )

        threshold = st.slider(
            "Umbral de decisión",
            min_value=0.30,
            max_value=0.75,
            value=DEFAULT_THRESHOLD,
            step=0.05,
        )

        st.info(
            "El umbral 0.50 es el seleccionado en la evaluación final "
            "por ofrecer un equilibrio razonable entre precisión y recall."
        )

        st.write("Modelo cargado:")
        st.code(str(MODEL_PATH.name))
        st.write("Columnas usadas por el modelo:")
        st.code(str(len(feature_cols)))

    if modo == "Procesar lote (varios audios)":
        modo_lote(model, feature_cols, threshold)
        st.stop()

    if modo == "Subir audio propio" and uploaded_file is None:
        st.warning("Sube un archivo de audio para iniciar el análisis o cambia a los ejemplos precargados.")
        st.stop()

    inicio_gt = None
    fin_gt = None
    manipulado_real = None
    tipo_splicing = "desconocido"
    audio_path = None
    audio_bytes = None
    nombre_audio = ""

    try:
        if modo == "Ejemplos precargados":
            assert audio_seleccionado is not None
            nombre_audio = audio_seleccionado
            df_audio = dataframe_ejemplo_precalculado(df, audio_seleccionado, model, feature_cols, threshold)

            manifest_audio = manifest[manifest["archivo_generado"] == audio_seleccionado]
            if not manifest_audio.empty:
                row_gt = manifest_audio.iloc[0]
                manipulado_real = int(row_gt["manipulado"])
                tipo_splicing = str(row_gt["tipo_splicing"])
                inicio_gt = convertir_float(row_gt["inicio_insercion_s"])
                fin_gt = convertir_float(row_gt["fin_insercion_s"])

            audio_path = AUDIO_DIR / audio_seleccionado
        else:
            assert uploaded_file is not None
            nombre_audio = uploaded_file.name

            es_valido, mensaje_validacion = validar_archivo_subido(uploaded_file)
            if not es_valido:
                st.error(mensaje_validacion)
                st.stop()

            audio_bytes = uploaded_file.getvalue()

            with st.spinner("Extrayendo características del audio subido..."):
                df_subido = procesar_audio_subido(audio_bytes, uploaded_file.name)
                df_audio = dataframe_audio_subido_con_scores(df_subido, model, feature_cols, threshold)
    except Exception as exc:
        st.error("No se ha podido analizar el audio seleccionado.")
        st.warning(
            "El archivo puede tener un códec no soportado, estar dañado, durar demasiado "
            "o superar los recursos disponibles de la demo web. Prueba con un WAV/MP3 "
            "más corto o ejecuta la aplicación localmente."
        )
        with st.expander("Ver detalle técnico"):
            st.code(str(exc))
        st.stop()

    df_audio["centro_ventana_s"] = (df_audio["inicio_ventana_s"] + df_audio["fin_ventana_s"]) / 2

    tamper_score = float(df_audio["score_sospecha"].max())
    prediccion_archivo = 1 if tamper_score >= threshold else 0
    intervalos_pred = obtener_intervalos_predichos(df_audio, threshold)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(nombre_audio)
        if modo == "Ejemplos precargados" and audio_path is not None and audio_path.exists():
            st.audio(audio_path.read_bytes(), format="audio/wav")
        elif modo == "Subir audio propio" and audio_bytes is not None:
            st.audio(audio_bytes, format=f"audio/{Path(nombre_audio).suffix.replace('.', '') or 'wav'}")

    with col2:
        st.subheader("Resultado")
        if prediccion_archivo == 1:
            st.error("Predicción: audio sospechoso / manipulado")
        else:
            st.success("Predicción: audio limpio")

        st.metric("Tamper score", f"{tamper_score:.4f}")
        st.metric("Umbral aplicado", f"{threshold:.2f}")

        if modo == "Ejemplos precargados" and manipulado_real is not None:
            if manipulado_real == 1:
                st.write(f"Ground truth: manipulado ({tipo_splicing})")
                if inicio_gt is not None and fin_gt is not None:
                    st.write(f"Intervalo real: {inicio_gt:.3f}s - {fin_gt:.3f}s")
            else:
                st.write("Ground truth: audio limpio")
        else:
            st.write("Ground truth: no disponible para audio externo")
            st.caption("La predicción debe interpretarse como una ayuda de cribado, no como una prueba pericial automática.")

    st.divider()

    st.subheader("Intervalos sospechosos detectados")
    if intervalos_pred:
        df_intervalos = pd.DataFrame(intervalos_pred)
        if inicio_gt is not None and fin_gt is not None:
            df_intervalos["solapa_ground_truth"] = df_intervalos.apply(
                lambda r: hay_solape(r, inicio_gt, fin_gt), axis=1
            )
        st.dataframe(df_intervalos, use_container_width=True)
    else:
        st.write("No se han detectado intervalos por encima del umbral seleccionado.")

    st.subheader("Curva temporal de sospecha")
    fig_scores = generar_grafica_scores(df_audio, threshold, inicio_gt, fin_gt, intervalos_pred)
    st.pyplot(fig_scores)
    plt.close(fig_scores)

    st.subheader("Espectrograma")
    try:
        if modo == "Ejemplos precargados" and audio_path is not None and audio_path.exists():
            fig_spec = generar_espectrograma_desde_path(audio_path, inicio_gt, fin_gt, intervalos_pred)
        elif modo == "Subir audio propio" and audio_bytes is not None:
            fig_spec = generar_espectrograma_desde_bytes(audio_bytes, nombre_audio, inicio_gt, fin_gt, intervalos_pred)
        else:
            fig_spec = None

        if fig_spec is not None:
            st.pyplot(fig_spec)
            plt.close(fig_spec)
    except Exception as exc:
        st.warning("No se ha podido generar el espectrograma para este audio.")
        st.code(str(exc))

    # Señal de audio (carga robusta), reutilizada para la forma de onda y el informe forense.
    y_sig, sr_sig = None, None
    try:
        if modo == "Ejemplos precargados" and audio_path is not None and audio_path.exists():
            y_sig, sr_sig = librosa.load(audio_path, sr=SR_MODELO, mono=True, duration=MAX_DURATION_S)
        elif audio_bytes is not None:
            y_sig, sr_sig, _tmp = cargar_audio_desde_bytes(audio_bytes, Path(nombre_audio).suffix or ".wav")
            try:
                _tmp.unlink(missing_ok=True)
            except Exception:
                pass
    except Exception:
        y_sig, sr_sig = None, None

    if y_sig is not None and len(y_sig) > 0:
        st.subheader("Forma de onda")
        fig_onda = generar_grafica_onda(y_sig, sr_sig, inicio_gt, fin_gt, intervalos_pred)
        st.pyplot(fig_onda)
        plt.close(fig_onda)

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

    columnas_mostrar = [c for c in columnas_mostrar if c in df_audio.columns]
    st.dataframe(df_audio[columnas_mostrar], use_container_width=True)

    csv_descarga = df_audio[columnas_mostrar].to_csv(index=False, sep=";").encode("utf-8")
    st.download_button(
        label="Descargar resultados de este audio en CSV",
        data=csv_descarga,
        file_name=f"resultado_{Path(nombre_audio).stem}.csv",
        mime="text/csv",
    )

    if INFORME_FORENSE_DISPONIBLE:
        st.subheader("Informe forense (PDF)")
        try:
            bytes_para_hash = (
                audio_path.read_bytes()
                if (modo == "Ejemplos precargados" and audio_path is not None and audio_path.exists())
                else audio_bytes
            )
            informe_pdf = construir_informe_forense_bytes(
                nombre_archivo=nombre_audio,
                df_audio=df_audio,
                threshold=threshold,
                tamper_score=tamper_score,
                prediccion=prediccion_archivo,
                intervalos=intervalos_pred,
                audio_bytes=bytes_para_hash,
                duracion_s=(len(y_sig) / sr_sig) if (y_sig is not None and sr_sig) else None,
                sr=sr_sig or SR_MODELO,
                y=y_sig,
                inicio_gt=inicio_gt,
                fin_gt=fin_gt,
            )
            st.download_button(
                label="Descargar informe forense (PDF)",
                data=informe_pdf,
                file_name=f"informe_forense_{Path(nombre_audio).stem}.pdf",
                mime="application/pdf",
            )
            st.caption("El informe incluye el hash SHA-256 del archivo (cadena de custodia).")
        except Exception as exc:
            st.warning("No se ha podido generar el informe forense en PDF.")
            st.code(str(exc))


if __name__ == "__main__":
    main()
