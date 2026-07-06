# Proyecto desarrollado como parte del Trabajo de Fin de Estudios del Grado en
# Ingeniería Informática de UNIR (2026). Autor: Ángel Carlos Soler Encinas. Licencia MIT.
"""
Pruebas unitarias de la lógica de posprocesado (agrupación de intervalos,
solape e IoU). No requieren scikit-learn ni los datos del proyecto.

Ejecución:
    cd Codigo_y_Resultados
    python3 -m pytest tests/ -q          # con pytest
    python3 tests/test_posproceso.py     # sin pytest (runner mínimo incluido)
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from posproceso import agrupar_intervalos, balancear_train, hay_solape, iou  # noqa: E402


def _ventanas(scores, inicio0=0.0, paso=0.5, dur=1.0):
    """Construye un DataFrame de ventanas solapadas con los scores dados."""
    filas = []
    for i, s in enumerate(scores):
        ini = inicio0 + i * paso
        filas.append({"inicio_ventana_s": ini, "fin_ventana_s": ini + dur, "score_sospecha": s})
    return pd.DataFrame(filas)


def test_sin_positivos_devuelve_vacio():
    df = _ventanas([0.1, 0.2, 0.0])
    assert agrupar_intervalos(df, threshold=0.5) == []


def test_un_intervalo_contiguo():
    # cuatro ventanas seguidas por encima del umbral -> un único intervalo
    df = _ventanas([0.9, 0.8, 0.95, 0.7])
    iv = agrupar_intervalos(df, threshold=0.5)
    assert len(iv) == 1
    ini, fin, smax = iv[0]
    assert ini == 0.0 and fin == 2.5
    assert abs(smax - 0.95) < 1e-9


def test_dos_intervalos_separados():
    # bloque, hueco grande (varias ventanas bajas), otro bloque -> dos intervalos
    df = _ventanas([0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.9, 0.9])
    iv = agrupar_intervalos(df, threshold=0.5)
    assert len(iv) == 2


def test_descarta_intervalo_demasiado_corto():
    # una sola ventana positiva: duración 1.0 s >= 0.5 -> se conserva
    df = _ventanas([0.0, 0.9, 0.0])
    assert len(agrupar_intervalos(df, threshold=0.5, min_duration_s=0.5)) == 1
    # con duración mínima exigente (2 s) esa ventana se descarta
    assert agrupar_intervalos(df, threshold=0.5, min_duration_s=2.0) == []


def test_hay_solape():
    assert hay_solape((1.0, 2.0, 0.9), 1.5, 3.0) is True
    assert hay_solape((1.0, 2.0, 0.9), 2.5, 3.0) is False
    # contacto en un punto cuenta como solape (criterio permisivo)
    assert hay_solape((1.0, 2.0, 0.9), 2.0, 3.0) is True


def test_max_gap_fusiona_bloques_cercanos():
    # dos bloques separados por un hueco de 0.5 s (<= max_gap 0.75) -> se fusionan
    df = _ventanas([0.9, 0.0, 0.9])
    iv = agrupar_intervalos(df, threshold=0.5, max_gap_s=0.75)
    assert len(iv) == 1
    # con max_gap 0 -> quedan separados... pero la ventana intermedia solapa,
    # asi que se comprueba con ventanas no solapadas
    df2 = _ventanas([0.9, 0.0, 0.0, 0.9])
    assert len(agrupar_intervalos(df2, threshold=0.5, max_gap_s=0.0)) == 2


def test_iou_sin_solape_es_cero():
    assert iou((0.0, 1.0, 0.9), 2.0, 3.0) == 0.0


def test_balancear_train_ratio_y_reproducibilidad():
    import pandas as pd
    df = pd.DataFrame({
        "etiqueta_borde": [1] * 10 + [0] * 100,
        "x": range(110),
    })
    b1 = balancear_train(df, ratio=3, random_state=42)
    b2 = balancear_train(df, ratio=3, random_state=42)
    # conserva todos los positivos y limita negativos a 1:3
    assert (b1["etiqueta_borde"] == 1).sum() == 10
    assert (b1["etiqueta_borde"] == 0).sum() == 30
    # determinista con la misma semilla
    assert b1["x"].tolist() == b2["x"].tolist()


def test_iou():
    # predicho [1,3], real [2,4] -> intersección 1, unión 3 -> IoU = 1/3
    assert abs(iou((1.0, 3.0, 0.9), 2.0, 4.0) - 1 / 3) < 1e-9
    # sin solape -> IoU = 0
    assert iou((1.0, 2.0, 0.9), 5.0, 6.0) == 0.0
    # coincidencia total -> IoU = 1
    assert abs(iou((1.0, 3.0, 0.9), 1.0, 3.0) - 1.0) < 1e-9


def _run():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    fails = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            fails += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - fails}/{len(tests)} pruebas superadas.")
    return fails


if __name__ == "__main__":
    sys.exit(1 if _run() else 0)
