import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier
)
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    recall_score
)

SEED = 42
DATA = Path("data/malnutrition")

X = pd.read_csv(DATA / "X_train_Kaggle.csv")
y = pd.read_csv(DATA / "y_train_Kaggle.csv")[
    "alerta_desnutricion"
]

Xtr, Xval, ytr, yval = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=SEED
)

numeric = ["edad_meses", "peso_kg", "talla_cm"]
categorical = [
    "sexo", "region", "nivel_educacion_madre"
]

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(
        strategy="median",
        add_indicator=True
    )),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(
        strategy="most_frequent"
    )),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

preprocess = ColumnTransformer([
    ("num", num_pipeline, numeric),
    ("cat", cat_pipeline, categorical)
])

models = {
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
    )
}

for name, model in models.items():

    pipe = Pipeline([
        ("preprocess", preprocess),
        ("model", model)
    ])

    pipe.fit(Xtr, ytr)
    prob = pipe.predict_proba(Xval)[:, 1]

    print(f"\n========== {name} ==========")

    print("ROC-AUC:",
          round(roc_auc_score(yval, prob), 5))

    print("Average Precision:",
          round(average_precision_score(yval, prob), 5))

    print("Probabilidad minima:",
          round(prob.min(), 5))

    print("Probabilidad maxima:",
          round(prob.max(), 5))

    print("Percentiles 50, 90, 95, 99:")
    print(np.round(
        np.percentile(prob, [50, 90, 95, 99]), 5
    ))

    for threshold in [0.5, 0.3, 0.2]:
        pred = (prob >= threshold).astype(int)

        print(
            f"Umbral {threshold}: "
            f"positivos={pred.sum()}, "
            f"accuracy={accuracy_score(yval,pred):.5f}, "
            f"recall={recall_score(yval,pred):.5f}"
        )

print("\nDIAGNOSTICO COMPLETADO")
