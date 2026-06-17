"""
Produce el resumen final del modelo de borde: tabla de métricas por umbral, gráfica de evolución
y matriz de confusión en el umbral elegido (0,50).
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np

IN_CSV = Path("reports/evaluacion_umbrales_por_archivo_borde.csv")
OUT_DIR = Path("reports/final")
OUT_TXT = OUT_DIR / "resumen_final_modelo_borde.txt"
OUT_METRICAS = OUT_DIR / "metricas_por_umbral.png"
OUT_CONFUSION = OUT_DIR / "matriz_confusion_umbral_050.png"

UMBRAL_ELEGIDO = 0.50

OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(IN_CSV, sep=";")

# Resumen por umbral a partir del CSV de detalle
rows = []

for threshold, g in df.groupby("threshold"):
    y_true = g["manipulado_real"].astype(int)
    y_pred = g["detectado_archivo"].astype(int)

    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())

    accuracy = (tp + tn) / len(g)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    manipulados = g[g["manipulado_real"] == 1]
    localizados = pd.to_numeric(
        manipulados["acierto_localizacion"],
        errors="coerce"
    ).fillna(0).astype(int).sum()

    total_manipulados = len(manipulados)
    tasa_localizacion = localizados / total_manipulados if total_manipulados else 0

    rows.append({
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "localizados": localizados,
        "total_manipulados": total_manipulados,
        "tasa_localizacion": tasa_localizacion,
    })

resumen = pd.DataFrame(rows).sort_values("threshold")

elegido = resumen[resumen["threshold"].round(2) == UMBRAL_ELEGIDO].iloc[0]

# Gráfica de métricas por umbral
plt.figure(figsize=(10, 5))
plt.plot(resumen["threshold"], resumen["accuracy"], marker="o", label="Accuracy")
plt.plot(resumen["threshold"], resumen["precision"], marker="o", label="Precision")
plt.plot(resumen["threshold"], resumen["recall"], marker="o", label="Recall")
plt.plot(resumen["threshold"], resumen["f1"], marker="o", label="F1-score")
plt.axvline(UMBRAL_ELEGIDO, linestyle="--", label=f"Umbral elegido = {UMBRAL_ELEGIDO}")
plt.xlabel("Umbral")
plt.ylabel("Valor de la métrica")
plt.title("Evolución de métricas por umbral")
plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_METRICAS, dpi=200)
plt.close()

# Matriz de confusión del umbral elegido
g = df[df["threshold"].round(2) == UMBRAL_ELEGIDO].copy()
y_true = g["manipulado_real"].astype(int)
y_pred = g["detectado_archivo"].astype(int)

cm = np.array([
    [int(((y_true == 0) & (y_pred == 0)).sum()), int(((y_true == 0) & (y_pred == 1)).sum())],
    [int(((y_true == 1) & (y_pred == 0)).sum()), int(((y_true == 1) & (y_pred == 1)).sum())],
])

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Limpio", "Manipulado"]
)

disp.plot(values_format="d")
plt.title("Matriz de confusión - Umbral 0.50")
plt.tight_layout()
plt.savefig(OUT_CONFUSION, dpi=200)
plt.close()

texto = []
texto.append("Resumen final del modelo de detección de bordes")
texto.append("=" * 60)
texto.append("")
texto.append(f"Umbral seleccionado: {UMBRAL_ELEGIDO}")
texto.append("")
texto.append("Motivo de selección:")
texto.append(
    "Se selecciona el umbral 0.50 por ofrecer el mejor equilibrio general entre "
    "detección de audios manipulados y reducción de falsos positivos. Frente a "
    "umbrales más bajos, reduce notablemente los falsos positivos; frente a "
    "umbrales más altos, mantiene una mayor sensibilidad."
)
texto.append("")
texto.append("Métricas con el umbral seleccionado:")
texto.append(f"Accuracy:  {elegido['accuracy']:.4f}")
texto.append(f"Precision: {elegido['precision']:.4f}")
texto.append(f"Recall:    {elegido['recall']:.4f}")
texto.append(f"F1-score:  {elegido['f1']:.4f}")
texto.append("")
texto.append("Matriz de confusión:")
texto.append("[[TN FP]")
texto.append(" [FN TP]]")
texto.append(str(cm))
texto.append("")
texto.append("Interpretación:")
texto.append(f"- Verdaderos negativos, audios limpios correctamente clasificados: {int(elegido['tn'])}")
texto.append(f"- Falsos positivos, audios limpios marcados como sospechosos: {int(elegido['fp'])}")
texto.append(f"- Falsos negativos, audios manipulados no detectados: {int(elegido['fn'])}")
texto.append(f"- Verdaderos positivos, audios manipulados detectados: {int(elegido['tp'])}")
texto.append("")
texto.append("Localización temporal:")
texto.append(
    f"El sistema localizó correctamente la región manipulada en "
    f"{int(elegido['localizados'])}/{int(elegido['total_manipulados'])} audios manipulados."
)
texto.append(f"Tasa de localización: {elegido['tasa_localizacion']:.4f}")
texto.append("")
texto.append("Archivos generados:")
texto.append(f"- {OUT_METRICAS}")
texto.append(f"- {OUT_CONFUSION}")

OUT_TXT.write_text("\n".join(texto), encoding="utf-8")

print("\n".join(texto))
print()
print(f"Resumen guardado en: {OUT_TXT}")
print(f"Gráfica de métricas guardada en: {OUT_METRICAS}")
print(f"Matriz de confusión guardada en: {OUT_CONFUSION}")
