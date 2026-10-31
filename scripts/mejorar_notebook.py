from pathlib import Path
import nbformat as nbf

ruta = Path("midterm_ia.ipynb")
nb = nbf.read(ruta, as_version=4)

marca = "## 4.1. Validación cruzada"

if any(
    marca in c.source
    for c in nb.cells
    if c.cell_type == "markdown"
):
    print("Las secciones adicionales ya existen.")
    raise SystemExit(0)

nuevas = []

nuevas.append(nbf.v4.new_markdown_cell(
"""## 4.1. Validación cruzada estratificada

Evaluamos las configuraciones utilizando tres particiones
estratificadas y exactamente las mismas métricas.

**La selección prioriza accuracy**, que es la métrica oficial
de la competencia. Recall y ROC-AUC se presentan para
documentar el efecto del desbalance.

La validación cruzada sirve para comparar modelos; no
constituye un conjunto de prueba independiente."""
))

nuevas.append(nbf.v4.new_code_cell(
"""from sklearn.model_selection import StratifiedKFold, cross_validate

cv = StratifiedKFold(
    n_splits=3, shuffle=True, random_state=2026
)

modelos_cv = {
    "Dummy": DummyClassifier(strategy="most_frequent"),
    "Random Forest": RandomForestClassifier(
        n_estimators=120, max_depth=12,
        min_samples_leaf=5, n_jobs=-1, random_state=42
    ),
    "Random Forest balanceado": RandomForestClassifier(
        n_estimators=120, max_depth=12,
        min_samples_leaf=5, class_weight="balanced",
        n_jobs=-1, random_state=42
    ),
    "Gradient Boosting": HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.08,
        max_leaf_nodes=31, random_state=42
    )
}

filas_cv = []

for nombre, clasificador in modelos_cv.items():
    pipeline = Pipeline([
        ("preprocesamiento", preparar()),
        ("modelo", clasificador)
    ])

    scores = cross_validate(
        pipeline, X, y, cv=cv, n_jobs=1,
        scoring={
            "accuracy": "accuracy",
            "recall": "recall",
            "roc_auc": "roc_auc"
        },
        error_score="raise"
    )

    filas_cv.append({
        "modelo": nombre,
        "accuracy": scores["test_accuracy"].mean(),
        "accuracy_std": scores["test_accuracy"].std(),
        "recall": scores["test_recall"].mean(),
        "roc_auc": scores["test_roc_auc"].mean()
    })

tabla_cv = pd.DataFrame(filas_cv).sort_values(
    "accuracy", ascending=False
)

display(tabla_cv.round(5))
tabla_cv.to_csv(OUTPUT / "notebook_cv.csv", index=False)
"""))

nuevas.append(nbf.v4.new_markdown_cell(
"""## 4.2. Auditoría del umbral de clasificación

Para maximizar accuracy, una predicción positiva solo
conviene si su probabilidad condicional de pertenecer
a la clase 1 supera 0,5, suponiendo probabilidades
correctamente estimadas y costos iguales.

Estudiamos varios umbrales sobre la partición de
validación para comprobar si aumentar las predicciones
positivas produce más aciertos.

**Advertencia:** elegir un umbral después de observar
la validación puede sobreajustarlo. No se utiliza
este análisis para modificar las etiquetas ocultas
de Kaggle."""
))

nuevas.append(nbf.v4.new_code_cell(
"""modelo_umbral = Pipeline([
    ("preprocesamiento", preparar()),
    ("modelo", RandomForestClassifier(
        n_estimators=120, max_depth=12,
        min_samples_leaf=5, n_jobs=-1, random_state=42
    ))
])

modelo_umbral.fit(X_fit, y_fit)
prob_val = modelo_umbral.predict_proba(X_val)[:, 1]

comparacion_umbrales = []

for u in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
    pred_u = (prob_val >= u).astype(int)

    comparacion_umbrales.append({
        "umbral": u,
        "accuracy": accuracy_score(y_val, pred_u),
        "positivos_predichos": int(pred_u.sum()),
        "recall": recall_score(
            y_val, pred_u, zero_division=0
        )
    })

tabla_umbrales = pd.DataFrame(comparacion_umbrales)
baseline_accuracy = (y_val == 0).mean()

print("Accuracy baseline:", round(baseline_accuracy, 6))
display(tabla_umbrales.round(6))

tabla_umbrales.to_csv(
    OUTPUT / "notebook_umbrales.csv",
    index=False
)
"""))

# Insertar antes del entrenamiento final
indice = next(
    i for i, c in enumerate(nb.cells)
    if c.cell_type == "markdown"
    and "## 5. Entrenamiento final" in c.source
)

nb.cells[indice:indice] = nuevas

nbf.validate(nb)
nbf.write(nb, ruta)

print("Notebook actualizado:", ruta)
print("Total de celdas:", len(nb.cells))
