import numpy as np
import pandas as pd

from pathlib import Path
from scipy.special import ndtr
from scipy.optimize import minimize_scalar

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss
)

DATA = Path("data/ii_sem2026")

X = pd.read_csv(DATA / "X_train_Kaggle.csv")
y = pd.read_csv(DATA / "y_train_Kaggle.csv")[
    "alerta_desnutricion"
]

# Hipotesis basada en el generador publico.
# Las condiciones se reconstruyen con los datos observados.
# Algunos valores ya fueron alterados despues del etiquetado.

def riesgo_latente(df):

    edad = df["edad_meses"].clip(lower=1, upper=24)

    bajo_peso = (
        df["peso_kg"].div(edad) < 0.25
    ).astype(float)

    baja_talla = (
        df["talla_cm"].div(edad) < 3
    ).astype(float)

    sin_escolaridad = (
        df["nivel_educacion_madre"] == "Sin escolaridad"
    ).astype(float)

    amazonia = (
        df["region"] == "Amazonía"
    ).astype(float)

    edad_negativa = (
        df["edad_meses"] < 0
    ).astype(float)

    return (
        -5 * bajo_peso
        -3 * baja_talla
        -2 * sin_escolaridad
        -1 * amazonia
        -1.5 * edad_negativa
    ).to_numpy()

Xtr, Xval, ytr, yval = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=2027
)

r_tr = riesgo_latente(Xtr)
r_val = riesgo_latente(Xval)

# P(Y=1|X) = Phi(r(X)-c)

def probabilidades(r, c):
    return np.clip(
        ndtr(r - c),
        1e-8,
        1 - 1e-8
    )

def funcion_objetivo(c):
    p = probabilidades(r_tr, c)
    return log_loss(ytr, p, labels=[0, 1])

resultado = minimize_scalar(
    funcion_objetivo,
    bounds=(-2, 4),
    method="bounded"
)

assert resultado.success

c_estimado = resultado.x

prob = probabilidades(r_val, c_estimado)
pred = (prob >= 0.5).astype(int)

print("\n=== MODELO ESTOCASTICO PROBIT ===")
print("Umbral estimado c:", round(c_estimado, 6))
print("Maximo riesgo latente:", r_val.max())
print("Probabilidad maxima:", round(prob.max(), 6))
print("Probabilidad media:", round(prob.mean(), 6))

print("\n=== VALIDACION ===")
print("Accuracy:", accuracy_score(yval, pred))
print("ROC-AUC:", roc_auc_score(yval, prob))
print("Average Precision:",
      average_precision_score(yval, prob))
print("Brier Score:", brier_score_loss(yval, prob))
print("Log Loss:", log_loss(yval, prob))
print("Positivos predichos:", pred.sum())

baseline = np.zeros(len(yval), dtype=int)

print("\n=== BASELINE ===")
print("Accuracy:",
      accuracy_score(yval, baseline))

print("\n=== COMPROBACION MATEMATICA ===")

if c_estimado > 0 and r_val.max() <= 0:
    print("El umbral es positivo y r(X) nunca supera cero.")
    print("Todas las probabilidades estimadas son menores de 0.5.")
    print("La regla Probit optima predice clase 0.")
else:
    print("La hipotesis requiere investigacion adicional.")

print("\n=== TASAS SEGUN RIESGO LATENTE ===")

tabla = pd.DataFrame({
    "riesgo": r_val,
    "alerta": yval.to_numpy()
})

resumen = tabla.groupby("riesgo")["alerta"].agg(
    n="size",
    tasa="mean"
)

print(resumen.round(5).to_string())

print("\nEXPERIMENTO FINALIZADO")
print("No se ha realizado ninguna subida a Kaggle.")
