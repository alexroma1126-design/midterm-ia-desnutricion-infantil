from pathlib import Path
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier
)

DATA = Path("data/ii_sem2026")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

X = pd.read_csv(DATA / "X_train_Kaggle.csv")
y = pd.read_csv(DATA / "y_train_Kaggle.csv")[
    "alerta_desnutricion"
]

numericas = ["edad_meses", "peso_kg", "talla_cm"]
categoricas = ["sexo", "region", "nivel_educacion_madre"]

preprocesamiento = ColumnTransformer([
    ("numericas", Pipeline([
        ("imputar", SimpleImputer(
            strategy="median", add_indicator=True
        )),
        ("escalar", StandardScaler())
    ]), numericas),

    ("categoricas", Pipeline([
        ("imputar", SimpleImputer(strategy="most_frequent")),
        ("codificar", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ]), categoricas)
])

modelos = {
    "Dummy": DummyClassifier(strategy="most_frequent"),

    "RandomForest": RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42
    ),

    "RF_Balanced": RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    ),

    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=31,
        random_state=42
    ),

    "HGB_Balanced": HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=31,
        class_weight="balanced",
        random_state=42
    )
}

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=2026
)

metricas = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

resultados = []

for nombre, modelo in modelos.items():
    print(f"\nEvaluando {nombre}...", flush=True)

    pipeline = Pipeline([
        ("preprocesamiento", preprocesamiento),
        ("clasificador", modelo)
    ])

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=metricas,
        n_jobs=1,
        error_score="raise"
    )

    fila = {"modelo": nombre}

    for metrica in metricas:
        valores = scores[f"test_{metrica}"]
        fila[metrica] = valores.mean()
        fila[f"{metrica}_std"] = valores.std()

    resultados.append(fila)

    print(
        f"Accuracy: {fila['accuracy']:.5f} "
        f"+/- {fila['accuracy_std']:.5f}"
    )
    print(f"Recall:   {fila['recall']:.5f}")
    print(f"F1:       {fila['f1']:.5f}")
    print(f"ROC-AUC:  {fila['roc_auc']:.5f}")

tabla = pd.DataFrame(resultados)
tabla = tabla.sort_values("accuracy", ascending=False)

print("\n========== RANKING FINAL CV ==========")
print(tabla[[
    "modelo", "accuracy", "accuracy_std",
    "balanced_accuracy", "recall", "f1", "roc_auc"
]].round(5).to_string(index=False))

tabla.to_csv(
    OUT / "comparacion_cv.csv",
    index=False
)

print("\nResultados guardados en outputs/comparacion_cv.csv")
