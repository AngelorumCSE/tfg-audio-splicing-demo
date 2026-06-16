"""
Configuración central del proyecto.

Reúne en un único lugar los parámetros que antes aparecían repetidos (números
"mágicos") en varios scripts: rutas, frecuencia de muestreo, geometría de
ventana, semilla, umbrales y parámetros del modelo. Centralizarlos mejora la
trazabilidad y evita inconsistencias entre el entrenamiento, la evaluación y la
aplicación.
"""
from pathlib import Path

# --- Rutas (relativas a la carpeta Codigo_y_Resultados) ---
DATA_DIR = Path("data")
GENERATED_DIR = DATA_DIR / "generated"
MANIFEST_CSV = DATA_DIR / "manifests" / "splicing_manifest.csv"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_BORDE_CSV = PROCESSED_DIR / "window_features_borde.csv"
MODELS_DIR = Path("models")
MODEL_BORDE = MODELS_DIR / "random_forest_borde.joblib"
REPORTS_DIR = Path("reports")

# --- Señal y ventaneo ---
SR = 16000          # frecuencia de muestreo común (Hz)
VENTANA_S = 1.0     # longitud de ventana (s)  -> L = SR * VENTANA_S
SALTO_S = 0.5       # salto entre ventanas (s) -> H = SR * SALTO_S
MARGEN_BORDE_S = 0.75   # tolerancia de la etiqueta de borde (s)

# --- Reproducibilidad y modelo ---
RANDOM_STATE = 42
RATIO_NEGATIVOS = 3     # submuestreo de negativos 1:3 en entrenamiento
N_ESTIMATORS = 500
MIN_SAMPLES_LEAF = 2

# --- Decisión / posprocesado ---
DEFAULT_THRESHOLD = 0.50
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
MAX_GAP_S = 0.75        # separación máxima para fusionar ventanas en un intervalo
MIN_DURATION_S = 0.50   # duración mínima de un intervalo sospechoso

# --- Columnas de metadatos (no predictoras) ---
META_COLS = [
    "id_registro", "archivo_generado", "archivo_base", "id_hablante_base",
    "genero_base", "tipo_splicing", "manipulado", "indice_ventana",
    "inicio_ventana_s", "fin_ventana_s", "centro_ventana_s", "etiqueta",
    "etiqueta_borde", "solape_manipulacion_s",
]
