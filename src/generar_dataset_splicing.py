# Proyecto desarrollado como parte del Trabajo de Fin de Estudios del Grado en
# Ingeniería Informática de UNIR (2026). Autor: Ángel Carlos Soler Encinas. Licencia MIT.
"""
Genera el dataset de splicing intra-fuente a partir de grabaciones propias: crea variantes
limpias y manipuladas (mismo audio / mismo hablante) y un manifiesto con el ground truth temporal.
"""
from pathlib import Path
import csv
import random
import wave
import shutil
import numpy as np

RAW_DIR = Path("raw_wav")
MANIFEST_ORIG = Path("data/manifests/audios_originales.csv")
OUT_DIR = Path("data/generated")
OUT_MANIFEST = Path("data/manifests/splicing_manifest.csv")

SEED = 42
TARGET_SR = 16000

random.seed(SEED)
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)


def detectar_delimitador(path: Path) -> str:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        primera = f.readline()
    return ";" if ";" in primera else ","


def leer_manifest_original(path: Path):
    delimiter = detectar_delimitador(path)
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        rows = list(reader)

    audios = []
    for i, row in enumerate(rows, start=1):
        utilizable = row.get("utilizable", "").strip().lower()
        if utilizable not in ("si", "sí", "yes", "y", "1", "true"):
            continue

        audios.append({
            "audio_id": f"A{i:03d}",
            "nombre_archivo": row["nombre_archivo"].strip(),
            "id_hablante": row["id_hablante"].strip(),
            "genero": row["genero"].strip(),
            "duracion_s": float(str(row["duracion_s"]).replace(",", ".")),
            "frecuencia_muestreo": int(row["frecuencia_muestreo"]),
            "canales": int(row["canales"]),
        })

    return audios


def leer_wav(path: Path):
    with wave.open(str(path), "rb") as w:
        canales = w.getnchannels()
        ancho = w.getsampwidth()
        sr = w.getframerate()
        frames = w.getnframes()
        data = w.readframes(frames)

    if canales != 1:
        raise ValueError(f"{path} no es mono.")
    if ancho != 2:
        raise ValueError(f"{path} no es PCM 16-bit.")
    if sr != TARGET_SR:
        raise ValueError(f"{path} no está a {TARGET_SR} Hz.")

    audio = np.frombuffer(data, dtype=np.int16).copy()
    return audio, sr


def escribir_wav(path: Path, audio: np.ndarray, sr: int):
    audio = np.asarray(audio, dtype=np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(audio.tobytes())


def elegir_fragmento(audio: np.ndarray, sr: int, min_s=1.0, max_s=2.5):
    duracion_s = len(audio) / sr

    if duracion_s < 6:
        raise ValueError("Audio demasiado corto para generar splicing.")

    longitud_s = random.uniform(min_s, min(max_s, duracion_s / 4))
    longitud_muestras = int(longitud_s * sr)

    margen_s = 1.0
    inicio_max_s = max(margen_s, duracion_s - longitud_s - margen_s)
    inicio_s = random.uniform(margen_s, inicio_max_s)

    inicio = int(inicio_s * sr)
    fin = inicio + longitud_muestras

    return inicio, fin


def elegir_punto_insercion(audio: np.ndarray, sr: int):
    duracion_s = len(audio) / sr
    insercion_s = random.uniform(1.0, max(1.1, duracion_s - 1.0))
    return int(insercion_s * sr)


def insertar_segmento(audio_base: np.ndarray, segmento: np.ndarray, pos: int):
    return np.concatenate([audio_base[:pos], segmento, audio_base[pos:]])


def segundos(muestras: int, sr: int) -> float:
    return round(muestras / sr, 3)


def nombre_seguro(stem: str):
    return (
        stem.replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace(":", "")
            .replace("/", "_")
    )


def main():
    audios = leer_manifest_original(MANIFEST_ORIG)

    por_hablante = {}
    for a in audios:
        por_hablante.setdefault(a["id_hablante"], []).append(a)

    filas = []

    for audio_info in audios:
        archivo_original = audio_info["nombre_archivo"]
        path_original = RAW_DIR / archivo_original

        if not path_original.exists():
            raise FileNotFoundError(f"No existe: {path_original}")

        audio_base, sr = leer_wav(path_original)
        base_name = nombre_seguro(Path(archivo_original).stem)

        # 1. Copia limpia normalizada
        clean_file = f"{audio_info['audio_id']}_{base_name}_clean.wav"
        clean_path = OUT_DIR / clean_file
        shutil.copy2(path_original, clean_path)

        filas.append({
            "id_registro": f"{audio_info['audio_id']}_clean",
            "archivo_generado": clean_file,
            "archivo_base": archivo_original,
            "id_hablante_base": audio_info["id_hablante"],
            "genero_base": audio_info["genero"],
            "manipulado": 0,
            "tipo_splicing": "none",
            "inicio_insercion_s": "",
            "fin_insercion_s": "",
            "archivo_segmento": "",
            "id_hablante_segmento": "",
            "inicio_segmento_s": "",
            "fin_segmento_s": "",
            "duracion_s": round(len(audio_base) / sr, 3),
            "frecuencia_muestreo": sr,
            "semilla": SEED,
        })

        # 2. Splicing dentro del mismo archivo
        src_ini, src_fin = elegir_fragmento(audio_base, sr)
        segmento = audio_base[src_ini:src_fin]
        insert_pos = elegir_punto_insercion(audio_base, sr)
        manipulado = insertar_segmento(audio_base, segmento, insert_pos)

        out_file = f"{audio_info['audio_id']}_{base_name}_splice_mismo_audio.wav"
        escribir_wav(OUT_DIR / out_file, manipulado, sr)

        filas.append({
            "id_registro": f"{audio_info['audio_id']}_splice_mismo_audio",
            "archivo_generado": out_file,
            "archivo_base": archivo_original,
            "id_hablante_base": audio_info["id_hablante"],
            "genero_base": audio_info["genero"],
            "manipulado": 1,
            "tipo_splicing": "mismo_audio",
            "inicio_insercion_s": segundos(insert_pos, sr),
            "fin_insercion_s": segundos(insert_pos + len(segmento), sr),
            "archivo_segmento": archivo_original,
            "id_hablante_segmento": audio_info["id_hablante"],
            "inicio_segmento_s": segundos(src_ini, sr),
            "fin_segmento_s": segundos(src_fin, sr),
            "duracion_s": round(len(manipulado) / sr, 3),
            "frecuencia_muestreo": sr,
            "semilla": SEED,
        })

        # 3. Splicing con otro audio del mismo hablante
        candidatos = [
            x for x in por_hablante[audio_info["id_hablante"]]
            if x["nombre_archivo"] != archivo_original
        ]

        if candidatos:
            fuente_info = random.choice(candidatos)
            fuente_path = RAW_DIR / fuente_info["nombre_archivo"]
            audio_fuente, sr_fuente = leer_wav(fuente_path)

            if sr_fuente != sr:
                raise ValueError("Frecuencias de muestreo distintas.")

            src_ini, src_fin = elegir_fragmento(audio_fuente, sr)
            segmento = audio_fuente[src_ini:src_fin]
            insert_pos = elegir_punto_insercion(audio_base, sr)
            manipulado = insertar_segmento(audio_base, segmento, insert_pos)

            out_file = f"{audio_info['audio_id']}_{base_name}_splice_mismo_hablante.wav"
            escribir_wav(OUT_DIR / out_file, manipulado, sr)

            filas.append({
                "id_registro": f"{audio_info['audio_id']}_splice_mismo_hablante",
                "archivo_generado": out_file,
                "archivo_base": archivo_original,
                "id_hablante_base": audio_info["id_hablante"],
                "genero_base": audio_info["genero"],
                "manipulado": 1,
                "tipo_splicing": "mismo_hablante",
                "inicio_insercion_s": segundos(insert_pos, sr),
                "fin_insercion_s": segundos(insert_pos + len(segmento), sr),
                "archivo_segmento": fuente_info["nombre_archivo"],
                "id_hablante_segmento": fuente_info["id_hablante"],
                "inicio_segmento_s": segundos(src_ini, sr),
                "fin_segmento_s": segundos(src_fin, sr),
                "duracion_s": round(len(manipulado) / sr, 3),
                "frecuencia_muestreo": sr,
                "semilla": SEED,
            })

    campos = [
        "id_registro",
        "archivo_generado",
        "archivo_base",
        "id_hablante_base",
        "genero_base",
        "manipulado",
        "tipo_splicing",
        "inicio_insercion_s",
        "fin_insercion_s",
        "archivo_segmento",
        "id_hablante_segmento",
        "inicio_segmento_s",
        "fin_segmento_s",
        "duracion_s",
        "frecuencia_muestreo",
        "semilla",
    ]

    with open(OUT_MANIFEST, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos, delimiter=";")
        writer.writeheader()
        writer.writerows(filas)

    print(f"Audios originales usados: {len(audios)}")
    print(f"Archivos generados: {len(filas)}")
    print(f"Carpeta de salida: {OUT_DIR}")
    print(f"Manifest generado: {OUT_MANIFEST}")


if __name__ == "__main__":
    main()
