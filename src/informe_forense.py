"""
Generación de un informe forense en PDF por audio analizado.

Pensado para la cadena de custodia: incluye el hash SHA-256 del archivo, los
metadatos, la predicción y el tamper score, la tabla de intervalos sospechosos,
la curva temporal de score, el espectrograma con las zonas marcadas y un descargo
legal. Se construye con matplotlib (PdfPages), sin dependencias externas.

Uso desde la app (Streamlit):
    pdf_bytes = construir_informe_forense_bytes(
        nombre_archivo=nombre, audio_bytes=audio_bytes, sr=sr, y=y,
        df_audio=df_audio, threshold=th, tamper_score=ts, prediccion=pred,
        intervalos=intervalos, inicio_gt=gi, fin_gt=gf)
    st.download_button("Descargar informe forense (PDF)", data=pdf_bytes, ...)
"""
from __future__ import annotations

import hashlib
import io
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

VERSION_HERRAMIENTA = "TFG splicing detector v11"
DESCARGO = (
    "Este informe es una ayuda de cribado generada automáticamente por un prototipo "
    "experimental. No constituye una prueba pericial concluyente. Para audios sin "
    "ground truth, el resultado debe ser verificado por un perito humano."
)


def sha256_de_bytes(data: bytes) -> str:
    """Hash SHA-256 (hexadecimal) del contenido del archivo, para cadena de custodia."""
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _pagina_texto(pdf: PdfPages, nombre_archivo: str, meta: Dict[str, str],
                  prediccion: int, tamper_score: float, threshold: float,
                  intervalos: List[Dict[str, float]],
                  inicio_gt: Optional[float], fin_gt: Optional[float]):
    fig = plt.figure(figsize=(8.27, 11.69))  # A4
    fig.subplots_adjust(left=0.08, right=0.92, top=0.95, bottom=0.06)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")

    y = 0.96
    ax.text(0.08, y, "Informe forense de análisis de audio", fontsize=18, fontweight="bold",
            color="#1F6FB2", transform=ax.transAxes); y -= 0.025
    ax.text(0.08, y, f"{VERSION_HERRAMIENTA}  ·  {datetime.now():%Y-%m-%d %H:%M:%S}",
            fontsize=9, color="#555555", transform=ax.transAxes); y -= 0.03
    ax.plot([0.08, 0.92], [y, y], color="#1F6FB2", lw=1.2, transform=ax.transAxes); y -= 0.03

    ax.text(0.08, y, "1. Metadatos del archivo", fontsize=13, fontweight="bold",
            transform=ax.transAxes); y -= 0.025
    filas = [("Nombre", nombre_archivo)] + list(meta.items())
    for k, v in filas:
        ax.text(0.10, y, f"{k}:", fontsize=10, fontweight="bold", transform=ax.transAxes)
        ax.text(0.34, y, str(v), fontsize=10, transform=ax.transAxes, wrap=True)
        y -= 0.022
    y -= 0.015

    ax.text(0.08, y, "2. Resultado del análisis", fontsize=13, fontweight="bold",
            transform=ax.transAxes); y -= 0.028
    etiqueta = "SOSPECHOSO / MANIPULADO" if prediccion == 1 else "LIMPIO"
    color = "#B00020" if prediccion == 1 else "#1B7F3B"
    ax.text(0.10, y, "Predicción:", fontsize=11, fontweight="bold", transform=ax.transAxes)
    ax.text(0.34, y, etiqueta, fontsize=11, fontweight="bold", color=color, transform=ax.transAxes)
    y -= 0.024
    ax.text(0.10, y, "Tamper score:", fontsize=10, fontweight="bold", transform=ax.transAxes)
    ax.text(0.34, y, f"{tamper_score:.4f}  (umbral = {threshold:.2f})", fontsize=10, transform=ax.transAxes)
    y -= 0.03

    ax.text(0.08, y, "3. Intervalos sospechosos detectados", fontsize=13, fontweight="bold",
            transform=ax.transAxes); y -= 0.026
    if intervalos:
        ax.text(0.10, y, "Inicio (s)", fontsize=9, fontweight="bold", transform=ax.transAxes)
        ax.text(0.28, y, "Fin (s)", fontsize=9, fontweight="bold", transform=ax.transAxes)
        ax.text(0.44, y, "Score máx.", fontsize=9, fontweight="bold", transform=ax.transAxes)
        if inicio_gt is not None:
            ax.text(0.64, y, "Solapa GT", fontsize=9, fontweight="bold", transform=ax.transAxes)
        y -= 0.02
        for iv in intervalos[:25]:
            ax.text(0.10, y, f"{iv['inicio_s']:.2f}", fontsize=9, transform=ax.transAxes)
            ax.text(0.28, y, f"{iv['fin_s']:.2f}", fontsize=9, transform=ax.transAxes)
            ax.text(0.44, y, f"{iv['score_maximo']:.3f}", fontsize=9, transform=ax.transAxes)
            if inicio_gt is not None and fin_gt is not None:
                solapa = max(iv["inicio_s"], inicio_gt) <= min(iv["fin_s"], fin_gt)
                ax.text(0.64, y, "Sí" if solapa else "No", fontsize=9, transform=ax.transAxes)
            y -= 0.02
    else:
        ax.text(0.10, y, "No se detectaron intervalos por encima del umbral.", fontsize=10,
                transform=ax.transAxes); y -= 0.02

    if inicio_gt is not None and fin_gt is not None:
        y -= 0.01
        ax.text(0.10, y, f"Ground truth (intervalo real): {inicio_gt:.2f}–{fin_gt:.2f} s",
                fontsize=9, style="italic", color="#555555", transform=ax.transAxes); y -= 0.02

    # Descargo legal al pie
    ax.text(0.08, 0.05, "Aviso legal", fontsize=10, fontweight="bold", transform=ax.transAxes)
    ax.text(0.08, 0.02, DESCARGO, fontsize=8, color="#555555", wrap=True, transform=ax.transAxes)
    pdf.savefig(fig)
    plt.close(fig)


def _pagina_graficas(pdf: PdfPages, df_audio, threshold: float,
                     intervalos, inicio_gt, fin_gt,
                     y: Optional[np.ndarray], sr: Optional[int]):
    fig, axes = plt.subplots(2, 1, figsize=(8.27, 11.69))
    fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.07, hspace=0.25)

    # Curva de score
    ax1 = axes[0]
    ax1.plot(df_audio["centro_ventana_s"], df_audio["score_sospecha"], marker="o", ms=3)
    ax1.axhline(threshold, ls="--", color="#888888", label=f"Umbral = {threshold:.2f}")
    if inicio_gt is not None and fin_gt is not None:
        ax1.axvspan(inicio_gt, fin_gt, alpha=0.18, color="green", label="Ground truth")
    for i, iv in enumerate(intervalos):
        ax1.axvspan(iv["inicio_s"], iv["fin_s"], alpha=0.15, color="red",
                    label="Intervalo predicho" if i == 0 else None)
    ax1.set_title("Curva temporal del score de sospecha")
    ax1.set_xlabel("Tiempo (s)"); ax1.set_ylabel("Score")
    ax1.set_ylim(0, max(1.0, float(df_audio["score_sospecha"].max()) + 0.05))
    ax1.grid(alpha=0.3); ax1.legend(loc="best", fontsize=8)

    # Espectrograma (si hay señal y librosa)
    ax2 = axes[1]
    dibujado = False
    if y is not None and sr and len(y) > 64:
        try:
            import librosa
            import librosa.display
            n_fft = min(2048, len(y)); hop = max(1, n_fft // 4)
            S = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)),
                                        ref=np.max)
            librosa.display.specshow(S, sr=sr, hop_length=hop, x_axis="time", y_axis="hz", ax=ax2)
            dibujado = True
        except Exception:
            dibujado = False
    if dibujado:
        if inicio_gt is not None and fin_gt is not None:
            ax2.axvspan(inicio_gt, fin_gt, alpha=0.18, color="green")
        for iv in intervalos:
            ax2.axvspan(iv["inicio_s"], iv["fin_s"], alpha=0.15, color="red")
        ax2.set_title("Espectrograma con zonas marcadas")
        ax2.set_xlabel("Tiempo (s)"); ax2.set_ylabel("Frecuencia (Hz)")
    else:
        ax2.axis("off")
        ax2.text(0.5, 0.5, "Espectrograma no disponible para este informe.",
                 ha="center", va="center", transform=ax2.transAxes)
    pdf.savefig(fig)
    plt.close(fig)


def construir_informe_forense_bytes(*, nombre_archivo: str, df_audio, threshold: float,
                                    tamper_score: float, prediccion: int,
                                    intervalos: List[Dict[str, float]],
                                    audio_bytes: Optional[bytes] = None,
                                    sha256: Optional[str] = None,
                                    duracion_s: Optional[float] = None,
                                    sr: Optional[int] = None,
                                    y: Optional[np.ndarray] = None,
                                    inicio_gt: Optional[float] = None,
                                    fin_gt: Optional[float] = None) -> bytes:
    """Devuelve los bytes de un PDF forense listo para descargar."""
    if sha256 is None and audio_bytes is not None:
        sha256 = sha256_de_bytes(audio_bytes)

    meta = {}
    if audio_bytes is not None:
        meta["Tamaño"] = f"{len(audio_bytes) / 1024:.1f} KB"
    if duracion_s is not None:
        meta["Duración"] = f"{duracion_s:.2f} s"
    if sr is not None:
        meta["Frecuencia de muestreo"] = f"{sr} Hz"
    meta["Ventanas analizadas"] = str(len(df_audio))
    meta["SHA-256"] = sha256 or "no disponible"

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        _pagina_texto(pdf, nombre_archivo, meta, prediccion, tamper_score, threshold,
                      intervalos, inicio_gt, fin_gt)
        _pagina_graficas(pdf, df_audio, threshold, intervalos, inicio_gt, fin_gt, y, sr)
    return buf.getvalue()
