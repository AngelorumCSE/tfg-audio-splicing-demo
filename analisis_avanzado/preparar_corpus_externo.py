# Proyecto desarrollado como parte del Trabajo de Fin de Estudios del Grado en
# Ingeniería Informática de UNIR (2026). Autor: Ángel Carlos Soler Encinas. Licencia MIT.
"""
Prepara un corpus externo para 05_validacion_externa.py.

Convierte todos los audios de una carpeta de origen (mp3, m4a, wav, ogg, flac…) a
WAV mono 16 kHz en data/externo, descartando los que duren menos del mínimo. Así
solo tienes que reunir clips de voces NO usadas en el TFG y ejecutar un comando.

Uso:
    cd Codigo_y_Resultados
    python3 analisis_avanzado/preparar_corpus_externo.py --origen ~/Descargas/voces --max 30

Después:
    python3 analisis_avanzado/05_validacion_externa.py --dir data/externo
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import wave
from pathlib import Path

EXT = {".mp3", ".m4a", ".wav", ".ogg", ".flac", ".aac", ".opus"}

try:
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG = "ffmpeg"


def duracion_wav(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origen", required=True, help="carpeta con los audios de origen")
    ap.add_argument("--destino", default="data/externo", help="carpeta de salida (WAV 16 kHz mono)")
    ap.add_argument("--min-dur", type=float, default=6.0, help="duración mínima en segundos")
    ap.add_argument("--max", type=int, default=30, help="número máximo de audios a preparar")
    args = ap.parse_args()

    origen = Path(args.origen).expanduser()
    destino = Path(args.destino)
    destino.mkdir(parents=True, exist_ok=True)
    if not origen.is_dir():
        print(f"No existe la carpeta de origen: {origen}"); sys.exit(1)

    fuentes = sorted(p for p in origen.iterdir() if p.suffix.lower() in EXT)
    if not fuentes:
        print(f"No se encontraron audios en {origen} (extensiones: {sorted(EXT)})"); sys.exit(1)

    n_ok = 0
    for src in fuentes:
        if n_ok >= args.max:
            break
        dst = destino / f"ext_{n_ok + 1:03d}.wav"
        try:
            subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(src),
                            "-ac", "1", "-ar", "16000", str(dst)], check=True)
        except Exception as exc:
            print(f"  [error] {src.name}: {exc}"); continue
        d = duracion_wav(dst)
        if d < args.min_dur:
            dst.unlink(missing_ok=True)
            print(f"  [descartado] {src.name}: dura {d:.1f}s (< {args.min_dur}s)")
            continue
        n_ok += 1
        print(f"  [ok] {src.name} -> {dst.name} ({d:.1f}s)")

    print(f"\nPreparados {n_ok} audios en {destino}")
    if n_ok == 0:
        print("Revisa que los audios duren al menos", args.min_dur, "segundos.")
    else:
        print("Ahora ejecuta: python3 analisis_avanzado/05_validacion_externa.py --dir", args.destino)


if __name__ == "__main__":
    main()
