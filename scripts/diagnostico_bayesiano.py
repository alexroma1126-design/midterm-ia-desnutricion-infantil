import numpy as np
import pandas as pd
from pathlib import Path

from scipy.stats import beta
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    brier_score_loss
)

DATA = Path("data/ii_sem2026")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

X = pd.read_csv(DATA / "X_train_Kaggle.csv")
y = pd.read_csv(DATA / "y_train_Kaggle.csv")[
    "alerta_desnutricion"
]

columnas = ["region", "nivel_educacion_madre"]

Xtr, Xval, ytr, yval = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=2027
)

datos = Xtr[columnas].copy()
datos["objetivo"] = ytr.to_numpy()

tabla = (
    datos.groupby(columnas)["objetivo"]
    .agg(["sum", "count"])
    .reset_index()
)

# Distribucion previa Beta(1,9)
alpha = 1
beta_prior = 9

tabla["a_post"] = tabla["sum"] + alpha
tabla["b_post"] = (
    tabla["count"] - tabla["sum"] + beta_prior
)

tabla["prob_bayes"] = (
    tabla["a_post"] /
    (tabla["a_post"] + tabla["b_post"])
)

tabla["limite_superior_95"] = beta.ppf(
    0.975,
    tabla["a_post"],
    tabla["b_post"]
)

tabla = tabla.sort_values(
    "prob_bayes", ascending=False
)

print("\n=== PROBABILIDADES BAYESIANAS ===")
print(tabla[
    columnas + [
        "count",
        "sum",
        "prob_bayes",
        "limite_superior_95"
    ]
].round(5).to_string(index=False))

# Probabilidades estimadas solo con entrenamiento
mapa = {
    (r["region"], r["nivel_educacion_madre"]):
        r["prob_bayes"]
    for _, r in tabla.iterrows()
}

p_base = float(ytr.mean())

prob = np.array([
    mapa.get(
        (r, e), p_base
    )
    for r, e in zip(
        Xval["region"],
        Xval["nivel_educacion_madre"]
    )
])

pred = (prob >= 0.5).astype(int)

print("\n=== VALIDACION BAYESIANA ===")
print("Accuracy:", accuracy_score(yval, pred))
print("ROC-AUC:", roc_auc_score(yval, prob))
print("Brier Score:", brier_score_loss(yval, prob))

print("Probabilidad maxima estimada:", prob.max())
print("Registros predichos clase 1:", pred.sum())

print("\n=== INCERTIDUMBRE ===")
print(
    "Grupos cuyo limite superior supera 50%:",
    int((tabla["limite_superior_95"] > 0.5).sum())
)

tabla.to_csv(
    OUT / "diagnostico_bayesiano.csv",
    index=False
)

print("\nDIAGNOSTICO COMPLETADO")
