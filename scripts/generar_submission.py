from pathlib import Path
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

SEED = 42
DATA = Path("data/ii_sem2026")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

# Cargar archivos oficiales
X = pd.read_csv(DATA / "X_train_Kaggle.csv")
y = pd.read_csv(DATA / "y_train_Kaggle.csv")[
    "alerta_desnutricion"
]
X_test = pd.read_csv(DATA / "X_test_Kaggle.csv")
sample = pd.read_csv(
    DATA / "sample_submission_malnutrition.csv"
)

# Verificar integridad
assert len(X) == len(y)
assert list(X.columns) == list(X_test.columns)
assert len(X_test) == len(sample)
assert list(sample.columns) == ["ID", "alerta_desnutricion"]

numericas = ["edad_meses", "peso_kg", "talla_cm"]
categoricas = ["sexo", "region", "nivel_educacion_madre"]

# Mismo preprocesamiento utilizado en validacion
preprocesador = ColumnTransformer([
    ("num", Pipeline([
        ("imputar", SimpleImputer(
            strategy="median", add_indicator=True
        )),
        ("escalar", StandardScaler())
    ]), numericas),

    ("cat", Pipeline([
        ("imputar", SimpleImputer(
            strategy="most_frequent"
        )),
        ("codificar", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ]), categoricas)
])

# Random Forest seleccionado
modelo = Pipeline([
    ("preprocesamiento", preprocesador),
    ("clasificador", RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=SEED
    ))
])

print("Entrenando Random Forest...")
modelo.fit(X, y)

print("Generando predicciones...")
predicciones = modelo.predict(X_test)

# Preservar IDs y estructura oficial
submission = sample.copy()
submission["alerta_desnutricion"] = predicciones.astype(int)

assert submission["ID"].equals(sample["ID"])
assert submission["alerta_desnutricion"].isin([0, 1]).all()
assert not submission.isna().any().any()

ruta = OUT / "submission.csv"
submission.to_csv(ruta, index=False)

print("\n=== VERIFICACION FINAL ===")
print("Filas:", len(submission))
print("Columnas:", submission.columns.tolist())
print("\nDistribucion de predicciones:")
print(submission["alerta_desnutricion"].value_counts())
print("\nPrimeras filas:")
print(submission.head())
print("\nArchivo creado:", ruta)
