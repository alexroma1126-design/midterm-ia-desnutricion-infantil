from pathlib import Path
import time
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

SEED = 42

DATA = Path("data/malnutrition")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

# 1. CARGAR DATOS

X = pd.read_csv(DATA / "X_train_Kaggle.csv")
y = pd.read_csv(
    DATA / "y_train_Kaggle.csv"
)["alerta_desnutricion"]

assert len(X) == len(y)
assert set(y.unique()) == {0, 1}

# 2. VARIABLES

numericas = [
    "edad_meses",
    "peso_kg",
    "talla_cm",
]

categoricas = [
    "sexo",
    "region",
    "nivel_educacion_madre",
]

# 3. PARTICION ESTRATIFICADA

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=SEED,
)

# 4. PREPROCESAMIENTO

prep_num = Pipeline([
    ("imputar", SimpleImputer(
        strategy="median",
        add_indicator=True
    )),
    ("escalar", StandardScaler()),
])

prep_cat = Pipeline([
    ("imputar", SimpleImputer(
        strategy="most_frequent"
    )),
    ("codificar", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )),
])

preprocesador = ColumnTransformer([
    ("numericas", prep_num, numericas),
    ("categoricas", prep_cat, categoricas),
])

# 5. MODELOS

modelos = {
    "Dummy": DummyClassifier(
        strategy="most_frequent"
    ),

    "LogisticRegression": LogisticRegression(
        max_iter=2000,
        random_state=SEED
    ),

    "RandomForest": RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=SEED
    ),

    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=31,
        random_state=SEED
    ),
}

# 6. ENTRENAMIENTO Y EVALUACION

resultados = []

print("\n=== COMPARACION DE MODELOS ===")

for nombre, modelo in modelos.items():

    pipeline = Pipeline([
        ("preprocesar", preprocesador),
        ("modelo", modelo)
    ])

    inicio = time.perf_counter()

    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_val)

    tiempo = time.perf_counter() - inicio

    resultado = {
        "modelo": nombre,
        "accuracy": accuracy_score(y_val, pred),
        "balanced_accuracy": balanced_accuracy_score(
            y_val, pred
        ),
        "precision_1": precision_score(
            y_val, pred, zero_division=0
        ),
        "recall_1": recall_score(
            y_val, pred, zero_division=0
        ),
        "f1_1": f1_score(
            y_val, pred, zero_division=0
        ),
        "tiempo_s": round(tiempo, 2),
    }

    resultados.append(resultado)

    print(f"\nMODELO: {nombre}")
    for clave, valor in resultado.items():
        if clave != "modelo":
            print(f"  {clave}: {valor:.4f}")

    print("Matriz de confusion:")
    print(confusion_matrix(
        y_val, pred, labels=[0, 1]
    ))

# 7. TABLA COMPARATIVA

tabla = pd.DataFrame(resultados)
tabla = tabla.sort_values(
    "accuracy", ascending=False
)

tabla.to_csv(
    OUT / "comparacion_modelos.csv",
    index=False
)

print("\n=== RANKING POR ACCURACY ===")
print(tabla.to_string(index=False))

print("\nResultados guardados en:")
print(OUT / "comparacion_modelos.csv")
