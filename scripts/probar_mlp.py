import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    recall_score
)

DATA = "data/ii_sem2026/"
SEED = 2026

X = pd.read_csv(DATA + "X_train_Kaggle.csv")
y = pd.read_csv(
    DATA + "y_train_Kaggle.csv"
)["alerta_desnutricion"]

# Misma particion que nuestro torneo.
X_dev, _, y_dev, _ = train_test_split(
    X, y,
    test_size=0.15,
    stratify=y,
    random_state=SEED
)

X_fit, X_val, y_fit, y_val = train_test_split(
    X_dev, y_dev,
    test_size=0.15 / 0.85,
    stratify=y_dev,
    random_state=SEED
)

numericas = ["edad_meses", "peso_kg", "talla_cm"]
categoricas = ["sexo", "region", "nivel_educacion_madre"]

preprocesador = ColumnTransformer([
    ("num", Pipeline([
        ("imputar", SimpleImputer(
            strategy="median",
            add_indicator=True
        )),
        ("escalar", StandardScaler())
    ]), numericas),

    ("cat", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ), categoricas)
])

modelo = Pipeline([
    ("preprocesador", preprocesador),

    ("mlp", MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        alpha=0.01,
        batch_size=256,
        learning_rate_init=0.001,
        max_iter=200,
        early_stopping=True,
        n_iter_no_change=15,
        random_state=SEED
    ))
])

print("Entrenando red neuronal...", flush=True)

modelo.fit(X_fit, y_fit)

proba = modelo.predict_proba(X_val)[:, 1]

print("\n=== DIAGNOSTICO MLP ===")
print("ROC-AUC:", roc_auc_score(y_val, proba))
print("Average Precision:", average_precision_score(y_val, proba))
print("Probabilidad maxima:", proba.max())
print("Probabilidad media:", proba.mean())

for umbral in [0.30, 0.40, 0.45, 0.50]:
    pred = (proba >= umbral).astype(int)

    print(
        f"Umbral={umbral:.2f} "
        f"Accuracy={accuracy_score(y_val,pred):.6f} "
        f"Recall={recall_score(y_val,pred,zero_division=0):.6f} "
        f"Positivos={pred.sum()}"
    )

print("\nBaseline:",
      accuracy_score(y_val, np.zeros(len(y_val))))

print("Experimento finalizado.")
