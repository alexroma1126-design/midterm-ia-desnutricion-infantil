from pathlib import Path
import nbformat as nbf

OUT = Path("midterm_ia.ipynb")
nb = nbf.v4.new_notebook()

def md(text):
    nb.cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    nb.cells.append(nbf.v4.new_code_cell(text))

md("""# Midterm — Predicción de alerta de desnutrición infantil

**Maestría en Ciencia de Datos — Fundamentos de Inteligencia Artificial**

## Objetivo
Predecir `alerta_desnutricion` a partir de seis variables.

**Métrica oficial: Accuracy.**

Se evalúan modelos simples y de ensamble, se analiza el desbalance
y se genera una predicción reproducible para Kaggle.
""")

code("""from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score, roc_auc_score, recall_score,
    f1_score, confusion_matrix, balanced_accuracy_score
)

DATA = Path("data/ii_sem2026")
OUTPUT = Path("outputs")
OUTPUT.mkdir(exist_ok=True)

X = pd.read_csv(DATA / "X_train_Kaggle.csv")
y = pd.read_csv(DATA / "y_train_Kaggle.csv")["alerta_desnutricion"]
X_test = pd.read_csv(DATA / "X_test_Kaggle.csv")
sample = pd.read_csv(DATA / "sample_submission_malnutrition.csv")

assert len(X) == len(y) == 70000
assert len(X_test) == len(sample) == 30000
assert list(X.columns) == list(X_test.columns)

print("Train:", X.shape, "Test:", X_test.shape)
print("Distribucion objetivo:")
print(y.value_counts(normalize=True).round(4))
""")

md("""## 1. Análisis exploratorio

El conjunto tiene un desbalance cercano a 90/10.
Por ello se compara cada modelo contra un clasificador
que predice siempre la clase mayoritaria.
""")

code("""display(X.describe(include="all").T)
display(X.isna().sum().to_frame("faltantes"))

print("Edades negativas:", (X["edad_meses"] < 0).sum())
print("Edades mayores de 24 meses:", (X["edad_meses"] > 24).sum())

fig, ax = plt.subplots(figsize=(6, 4))
y.value_counts().sort_index().plot.bar(ax=ax)
ax.set_title("Distribución de la variable objetivo")
ax.set_xlabel("Clase")
ax.set_ylabel("Observaciones")
plt.tight_layout()
plt.show()
""")

code("""eda = X.copy()
eda["alerta"] = y.to_numpy()

for columna in ["sexo", "region", "nivel_educacion_madre"]:
    print("\\n", columna)
    print(
        eda.groupby(columna)["alerta"]
        .agg(n="size", tasa_alerta="mean")
        .round(4)
    )
""")

md("""## 2. Preparación y prevención de fuga de información

La imputación y la codificación se ajustan **solo**
con los datos de entrenamiento gracias a `Pipeline`.
""")

code("""NUMERICAS = ["edad_meses", "peso_kg", "talla_cm"]
CATEGORICAS = ["sexo", "region", "nivel_educacion_madre"]

def preparar():
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputar", SimpleImputer(
                strategy="median", add_indicator=True
            )),
            ("escalar", StandardScaler())
        ]), NUMERICAS),
        ("cat", Pipeline([
            ("imputar", SimpleImputer(strategy="most_frequent")),
            ("codificar", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ))
        ]), CATEGORICAS)
    ])

X_fit, X_val, y_fit, y_val = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("Entrenamiento:", X_fit.shape)
print("Validación:", X_val.shape)
""")

md("""## 3. Modelos y evaluación

La elección debe considerar principalmente **accuracy**.

Se incluyen métricas adicionales para identificar si un modelo
ignora completamente la clase minoritaria.
""")

code("""configuraciones = {
    "Dummy": DummyClassifier(strategy="most_frequent"),
    "Regresión logística": LogisticRegression(max_iter=2000),
    "Random Forest": RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42
    ),
    "Gradient Boosting": HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=31,
        random_state=42
    )
}

resultados = []
matrices = {}

for nombre, clasificador in configuraciones.items():
    modelo = Pipeline([
        ("preprocesamiento", preparar()),
        ("modelo", clasificador)
    ])

    modelo.fit(X_fit, y_fit)
    pred = modelo.predict(X_val)
    prob = modelo.predict_proba(X_val)[:, 1]

    resultados.append({
        "modelo": nombre,
        "accuracy": accuracy_score(y_val, pred),
        "balanced_accuracy": balanced_accuracy_score(y_val, pred),
        "recall": recall_score(y_val, pred, zero_division=0),
        "f1": f1_score(y_val, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_val, prob),
        "positivos_predichos": int(pred.sum())
    })

    matrices[nombre] = confusion_matrix(
        y_val, pred, labels=[0, 1]
    )

resultados = pd.DataFrame(resultados).sort_values(
    ["accuracy", "roc_auc"], ascending=False
)

display(resultados.round(5))

print("Baseline clase mayoritaria:",
      round((y_val == 0).mean(), 6))
print("\\nMatriz de confusión Random Forest:")
print(matrices["Random Forest"])
""")

md("""## 4. Interpretación matemática

Para maximizar accuracy con costos de error iguales, la regla
bayesiana elige la clase de mayor probabilidad condicional.

Para Random Forest, la probabilidad positiva se obtiene
promediando las probabilidades estimadas por sus árboles:

$$
\\hat p(x)=\\frac{1}{B}\\sum_{b=1}^{B}\\hat p_b(x).
$$

La predicción positiva se produce cuando:

$$
\\hat p(x)>0.5.
$$

En los experimentos realizados, los modelos de ensamble
obtuvieron cierta discriminación (ROC-AUC alrededor de 0.65),
pero no superaron el accuracy del baseline. Esto es compatible
con el fuerte desbalance y el mecanismo estocástico sintético.

**Limitación:** un buen accuracy con recall cero no constituye
un detector clínico útil de alertas de desnutrición.
""")

md("""## 5. Entrenamiento final y archivo para Kaggle

Se utiliza Random Forest como modelo final por su rendimiento
validado, estabilidad y facilidad de interpretación matemática.

Su accuracy empata con el baseline, por lo que no se afirma
una superioridad predictiva inexistente.
""")

code("""modelo_final = Pipeline([
    ("preprocesamiento", preparar()),
    ("modelo", RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42
    ))
])

modelo_final.fit(X, y)
pred_test = modelo_final.predict(X_test).astype(int)

submission = sample.copy()
submission["alerta_desnutricion"] = pred_test

assert submission["ID"].equals(sample["ID"])
assert submission["alerta_desnutricion"].isin([0, 1]).all()
assert not submission.isna().any().any()

ruta = OUTPUT / "submission.csv"
submission.to_csv(ruta, index=False)

print("Archivo:", ruta)
print("Filas:", len(submission))
print("Predicciones:")
print(submission["alerta_desnutricion"].value_counts())
display(submission.head())
""")

md("""## 6. Conclusiones

1. El conjunto presenta un desbalance cercano al 90/10.
2. Se estableció un baseline explícito para accuracy.
3. Se compararon cuatro modelos con el mismo protocolo.
4. El modelo final no demostró una mejora en accuracy frente al baseline.
5. El resultado debe interpretarse junto al recall de la clase positiva.
6. Las predicciones se generan sin usar etiquetas del conjunto de prueba.

**Experimentos adicionales realizados en el proyecto:**
validación cruzada, CatBoost, LightGBM, Extra Trees,
MLP, modelo beta-binomial, Probit y análisis de umbrales.
""")

nb.metadata["kernelspec"] = {
    "display_name": "Python (Midterm IA)",
    "language": "python",
    "name": "midterm-ia"
}
nb.metadata["language_info"] = {"name": "python"}

nbf.write(nb, OUT)
print("Notebook creado:", OUT.resolve())
print("Celdas:", len(nb.cells))
