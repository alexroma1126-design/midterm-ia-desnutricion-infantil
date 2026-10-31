from pathlib import Path
import numpy as np
import pandas as pd

from scipy.stats import ks_2samp

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

DATA = Path("data/ii_sem2026")

train = pd.read_csv(DATA / "X_train_Kaggle.csv")
test = pd.read_csv(DATA / "X_test_Kaggle.csv")

NUM = ["edad_meses", "peso_kg", "talla_cm"]
CAT = ["sexo", "region", "nivel_educacion_madre"]

assert list(train.columns) == list(test.columns)

print("\n=== DISTRIBUCIONES NUMERICAS ===")

for col in NUM:
    a = train[col].dropna()
    b = test[col].dropna()

    ks = ks_2samp(a, b)

    print(f"\n{col}")
    print(f"Media train: {a.mean():.4f}")
    print(f"Media test:  {b.mean():.4f}")
    print(f"Nulos train: {train[col].isna().mean():.2%}")
    print(f"Nulos test:  {test[col].isna().mean():.2%}")
    print(f"KS estadistico: {ks.statistic:.5f}")
    print(f"KS p-value: {ks.pvalue:.6g}")

print("\n=== DISTRIBUCIONES CATEGORICAS ===")

for col in CAT:
    a = train[col].value_counts(normalize=True)
    b = test[col].value_counts(normalize=True)

    tabla = pd.concat(
        [a.rename("train"), b.rename("test")],
        axis=1
    ).fillna(0)

    tabla["diferencia_pp"] = (
        (tabla["test"] - tabla["train"]) * 100
    )

    print(f"\n{col}")
    print(tabla.round(4).to_string())

print("\n=== CLASIFICADOR DE CAMBIO DE DISTRIBUCION ===")

# Etiqueta de origen:
# 0 = entrenamiento, 1 = Kaggle test.
# No utilizamos las etiquetas de desnutricion.

X = pd.concat(
    [train, test],
    ignore_index=True
)

origen = np.concatenate([
    np.zeros(len(train), dtype=int),
    np.ones(len(test), dtype=int)
])

X_fit, X_val, y_fit, y_val = train_test_split(
    X,
    origen,
    test_size=0.20,
    random_state=2026,
    stratify=origen
)

preparar = ColumnTransformer([
    (
        "num",
        SimpleImputer(
            strategy="median",
            add_indicator=True
        ),
        NUM
    ),
    (
        "cat",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ),
        CAT
    )
])

modelo = Pipeline([
    ("preprocesamiento", preparar),
    ("clasificador", HistGradientBoostingClassifier(
        max_iter=100,
        max_leaf_nodes=15,
        min_samples_leaf=100,
        learning_rate=0.05,
        random_state=2026
    ))
])

modelo.fit(X_fit, y_fit)

prob = modelo.predict_proba(X_val)[:, 1]

auc = roc_auc_score(y_val, prob)

print("\n=== RESULTADO FINAL ===")
print(f"ROC-AUC para distinguir train/test: {auc:.6f}")

if auc < 0.55:
    print("No se detecta una separacion fuerte entre conjuntos.")
elif auc < 0.65:
    print("Existe cierta capacidad de distinguir los conjuntos.")
else:
    print("Se detecta una separacion importante entre conjuntos.")

print("\nAUDITORIA FINALIZADA")
print("No se realizo ningun envio a Kaggle.")
