# Proyecto desarrollado como parte del Trabajo de Fin de Estudios del Grado en
# Ingeniería Informática de UNIR (2026). Autor: Ángel Carlos Soler Encinas. Licencia MIT.
"""
Generador de dataset de splicing CROSS-SOURCE a partir de LibriSpeech dev-clean.

Corrige el problema de fondo del experimento original: los empalmes "del mismo
audio/hablante" apenas dejan discontinuidad acústica y, por tanto, no son
detectables. Aquí se generan empalmes en los que el fragmento insertado procede de
una FUENTE DISTINTA (otro hablante y, opcionalmente, otro entorno acústico), que es
el caso forense relevante y el que sí deja una huella detectable.

Diseño:
  - Hablantes divididos en dos grupos DISJUNTOS: "host" (los que se analizan) e
    "insert" (de los que se extraen los fragmentos). Así, al validar por hablante,
    se mide la detección sobre voces host nunca vistas.
  - Para cada host se construye un audio concatenando varias locuciones suyas y se
    generan variantes:
      * limpio                      (manipulado = 0)
      * splice_cross_speaker        (inserto de otro hablante; discontinuidad clara)
      * splice_cross_speaker_ruido  (inserto de otro hablante + ruido; otro entorno)
      * splice_same_speaker         (inserto de otra locución del MISMO host; difícil)
  - Se registra el ground truth temporal de cada inserción.

Estructura esperada de LibriSpeech: <dev-clean>/<id_hablante>/<id_capitulo>/*.flac

Uso:
    cd Codigo_y_Resultados
    python3 reconstruccion/01_generar_dataset_libri.py --libri RUTA/A/dev-clean \\
            --salida data/libri --n-host 30 --n-insert 10

Salidas:
    data/libri/audios/*.wav  y  data/libri/manifest_libri.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
import config as C

DUR_HOST_MIN, DUR_HOST_MAX = 25.0, 40.0   # s del audio host
FRAG_MIN, FRAG_MAX = 1.5, 2.5             # s del fragmento insertado
SNR_RUIDO = 12.0                          # dB del ruido para simular "otro entorno"
CAMPOS = ["archivo_generado", "archivo_base", "id_hablante_base", "id_hablante_segmento",
          "tipo_splicing", "manipulado", "inicio_insercion_s", "fin_insercion_s",
          "duracion_s", "frecuencia_muestreo", "semilla"]


def hablantes(libri_dir: Path):
    """Devuelve {id_hablante: [rutas .flac]} con material suficiente."""
    d = {}
    for spk in sorted(p for p in libri_dir.iterdir() if p.is_dir()):
        files = sorted(spk.rglob("*.flac"))
        if len(files) >= 3:
            d[spk.name] = files
    return d


def cargar(path, sr):
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32)


def normalizar(y, pico=0.9):
    m = float(np.max(np.abs(y))) if len(y) else 0.0
    return (y * (pico / m)).astype(np.float32) if m > 0 else y


def construir_host(files, sr, rng):
    objetivo = rng.uniform(DUR_HOST_MIN, DUR_HOST_MAX)
    segs, total = [], 0.0
    for f in files:
        y = cargar(f, sr)
        segs.append(y); total += len(y) / sr
        if total >= objetivo:
            break
    return np.concatenate(segs) if segs else None


def fragmento(y, sr, rng):
    dur = len(y) / sr
    if dur < FRAG_MIN + 0.5:
        return None
    L = rng.uniform(FRAG_MIN, min(FRAG_MAX, dur - 0.3))
    n = int(L * sr)
    ini = int(rng.uniform(0, len(y) - n))
    return y[ini:ini + n]


def add_ruido(seg, snr_db, rng):
    p = float(np.mean(seg ** 2)) + 1e-12
    ruido = rng.normal(0, np.sqrt(p / (10 ** (snr_db / 10))), len(seg))
    return (seg + ruido).astype(np.float32)


def insertar(host, seg, sr, rng):
    pos = int(rng.uniform(1.0, max(1.1, len(host) / sr - 1.0)) * sr)
    y = np.concatenate([host[:pos], seg, host[pos:]])
    return y.astype(np.float32), pos / sr, (pos + len(seg)) / sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--libri", required=True, help="ruta a la carpeta dev-clean de LibriSpeech")
    ap.add_argument("--salida", default="data/libri")
    ap.add_argument("--n-host", type=int, default=30)
    ap.add_argument("--n-insert", type=int, default=10)
    ap.add_argument("--seed", type=int, default=C.RANDOM_STATE)
    args = ap.parse_args()

    libri = Path(args.libri).expanduser()
    if not libri.is_dir():
        print(f"No existe la carpeta: {libri}"); sys.exit(1)

    spk = hablantes(libri)
    print(f"Hablantes encontrados con material suficiente: {len(spk)}")
    if len(spk) < args.n_host + args.n_insert:
        print(f"Hacen falta al menos {args.n_host + args.n_insert} hablantes; "
              f"reduce --n-host/--n-insert o usa un corpus mayor."); sys.exit(1)

    rng = np.random.default_rng(args.seed)
    ids = list(spk.keys()); rng.shuffle(ids)
    host_ids = ids[:args.n_host]
    insert_ids = ids[args.n_host:args.n_host + args.n_insert]

    out = Path(args.salida); (out / "audios").mkdir(parents=True, exist_ok=True)
    sr = C.SR
    filas = []

    def guardar(nombre, y, meta):
        y = normalizar(y)
        sf.write(out / "audios" / nombre, y, sr)
        meta["duracion_s"] = round(len(y) / sr, 3)
        meta["frecuencia_muestreo"] = sr
        meta["semilla"] = args.seed
        filas.append(meta)

    for i, h in enumerate(host_ids, start=1):
        host = construir_host(spk[h], sr, rng)
        if host is None or len(host) / sr < DUR_HOST_MIN * 0.6:
            print(f"  [omitido] hablante {h}: material insuficiente"); continue
        base = f"H{i:03d}_{h}"

        # 1) limpio
        guardar(f"{base}_clean.wav", host, {
            "archivo_generado": f"{base}_clean.wav", "archivo_base": base,
            "id_hablante_base": h, "id_hablante_segmento": "", "tipo_splicing": "none",
            "manipulado": 0, "inicio_insercion_s": "", "fin_insercion_s": ""})

        # 2) splice de OTRO hablante (cross-speaker)
        ins_id = insert_ids[rng.integers(0, len(insert_ids))]
        ins_audio = cargar(spk[ins_id][rng.integers(0, len(spk[ins_id]))], sr)
        seg = fragmento(ins_audio, sr, rng)
        if seg is not None:
            y, a, b = insertar(host, seg, sr, rng)
            guardar(f"{base}_cross_speaker.wav", y, {
                "archivo_generado": f"{base}_cross_speaker.wav", "archivo_base": base,
                "id_hablante_base": h, "id_hablante_segmento": ins_id,
                "tipo_splicing": "cross_speaker", "manipulado": 1,
                "inicio_insercion_s": round(a, 3), "fin_insercion_s": round(b, 3)})

            # 3) cross-speaker + ruido (otro entorno)
            y2, a2, b2 = insertar(host, add_ruido(seg, SNR_RUIDO, rng), sr, rng)
            guardar(f"{base}_cross_speaker_ruido.wav", y2, {
                "archivo_generado": f"{base}_cross_speaker_ruido.wav", "archivo_base": base,
                "id_hablante_base": h, "id_hablante_segmento": ins_id,
                "tipo_splicing": "cross_speaker_ruido", "manipulado": 1,
                "inicio_insercion_s": round(a2, 3), "fin_insercion_s": round(b2, 3)})

        # 4) splice del MISMO host, otra locución (difícil)
        if len(spk[h]) >= 2:
            otra = cargar(spk[h][-1], sr)
            seg2 = fragmento(otra, sr, rng)
            if seg2 is not None:
                y3, a3, b3 = insertar(host, seg2, sr, rng)
                guardar(f"{base}_same_speaker.wav", y3, {
                    "archivo_generado": f"{base}_same_speaker.wav", "archivo_base": base,
                    "id_hablante_base": h, "id_hablante_segmento": h,
                    "tipo_splicing": "same_speaker", "manipulado": 1,
                    "inicio_insercion_s": round(a3, 3), "fin_insercion_s": round(b3, 3)})
        print(f"  [{i}/{len(host_ids)}] host {h} -> variantes generadas")

    with open(out / "manifest_libri.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS, delimiter=";")
        w.writeheader(); w.writerows(filas)

    n_clean = sum(1 for r in filas if r["manipulado"] == 0)
    print(f"\nGenerados {len(filas)} audios ({n_clean} limpios, {len(filas) - n_clean} manipulados)")
    print(f"Hablantes host: {len(host_ids)} | hablantes de inserción (disjuntos): {len(insert_ids)}")
    print(f"Manifest -> {out / 'manifest_libri.csv'}")
    print("Ahora ejecuta: python3 reconstruccion/02_entrenar_evaluar_libri.py --datos", args.salida)


if __name__ == "__main__":
    main()
