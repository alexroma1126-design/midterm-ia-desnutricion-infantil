from pathlib import Path
import time
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

SEED = 2026
DATA = Path("data/ii_sem2026")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

TARGET = "alerta_desnutricion"
CATS = ["sexo", "region", "nivel_educacion_madre"]

# ========================================
# 1. DATOS OFICIALES
# ========================================

X = pd.read_csv(DATA / "X_train_Kaggle.csv")
y = pd.read_csv(DATA / "y_train_Kaggle.csv")[TARGET]

X_test = pd.read_csv(DATA / "X_test_Kaggle.csv")
sample = pd.read_csv(
    DATA / "sample_submission_malnutrition.csv"
)

assert len(X) == len(y)
assert len(X_test) == len(sample)
assert list(X.columns) == list(X_test.columns)
assert list(sample.columns) == ["ID", TARGET]
assert set(y.unique()) == {0, 1}

print("\n=== DATASET OFICIAL ===")
print("Entrenamiento:", X.shape)
print("Kaggle Test:", X_test.shape)
print("Prevalencia:", round(y.mean(), 6))

# ========================================
# 2. INGENIERIA DE CARACTERISTICAS
# ========================================

def crear_features(df):
    d = df.copy()

    edad = d["edad_meses"].clip(lower=1, upper=24)
    talla_m = d["talla_cm"] / 100

    d["edad_negativa"] = (
        d["edad_meses"] < 0
    ).astype(int)

    d["edad_mayor_24"] = (
        d["edad_meses"] > 24
    ).astype(int)

    d["peso_faltante"] = (
        d["peso_kg"].isna()
    ).astype(int)

    d["talla_faltante"] = (
        d["talla_cm"].isna()
    ).astype(int)

    d["peso_por_edad"] = d["peso_kg"] / edad

    d["talla_por_edad"] = d["talla_cm"] / edad

    d["imc_sintetico"] = (
        d["peso_kg"] / talla_m.pow(2)
    )

    d["peso_por_talla"] = (
        d["peso_kg"] / d["talla_cm"]
    )

    d = d.replace([np.inf, -np.inf], np.nan)

    for col in CATS:
        d[col] = (
            d[col]
            .astype("string")
            .fillna("Desconocido")
            .astype(str)
        )

    return d

X = crear_features(X)
X_test = crear_features(X_test)

print("Variables despues de ingeniería:", X.shape[1])

# ========================================
# 3. DIVISION SIN FUGA DE INFORMACION
# ========================================

X_dev, X_check, y_dev, y_check = train_test_split(
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

print("\n=== PARTICIONES ===")
print("Entrenamiento:", len(X_fit))
print("Seleccion:", len(X_val))
print("Comprobacion:", len(X_check))

baseline = (y_val == 0).mean()

print("Baseline seleccion:", round(baseline, 6))

# ========================================
# 4. PREPROCESAMIENTO
# ========================================

def preprocesador():

    nums = [c for c in X.columns if c not in CATS]

    return ColumnTransformer([
        (
            "numericas",
            SimpleImputer(
                strategy="median",
                add_indicator=True
            ),
            nums
        ),
        (
            "categoricas",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            CATS
        )
    ])

# ========================================
# 5. MODELOS COMPETITIVOS
# ========================================

def construir_modelos():

    return {
        "RandomForest": Pipeline([
            ("prep", preprocesador()),
            ("modelo", RandomForestClassifier(
                n_estimators=180,
                max_depth=12,
                min_samples_leaf=5,
                n_jobs=-1,
                random_state=SEED
            ))
        ]),

        "ExtraTrees": Pipeline([
            ("prep", preprocesador()),
            ("modelo", ExtraTreesClassifier(
                n_estimators=250,
                min_samples_leaf=7,
                max_features=0.9,
                n_jobs=-1,
                random_state=SEED
            ))
        ]),

        "LightGBM_15": Pipeline([
            ("prep", preprocesador()),
            ("modelo", LGBMClassifier(
                n_estimators=500,
                learning_rate=0.035,
                num_leaves=15,
                min_child_samples=110,
                reg_lambda=5,
                verbosity=-1,
                n_jobs=4,
                random_state=SEED
            ))
        ]),

        "LightGBM_31": Pipeline([
            ("prep", preprocesador()),
            ("modelo", LGBMClassifier(
                n_estimators=500,
                learning_rate=0.035,
                num_leaves=31,
                min_child_samples=110,
                reg_lambda=5,
                verbosity=-1,
                n_jobs=4,
                random_state=SEED
            ))
        ]),

        "CatBoost_4": CatBoostClassifier(
            iterations=600,
            depth=4,
            learning_rate=0.04,
            l2_leaf_reg=5,
            loss_function="Logloss",
            random_seed=SEED,
            verbose=False,
            thread_count=4,
            allow_writing_files=False
        ),

        "CatBoost_6": CatBoostClassifier(
            iterations=600,
            depth=6,
            learning_rate=0.04,
            l2_leaf_reg=5,
            loss_function="Logloss",
            random_seed=SEED,
            verbose=False,
            thread_count=4,
            allow_writing_files=False
        )
    }

def entrenar(modelo, X_data, y_data):

    if isinstance(modelo, CatBoostClassifier):
        modelo.fit(
            X_data,
            y_data,
            cat_features=CATS,
            verbose=False
        )
    else:
        modelo.fit(X_data, y_data)

    return modelo

# ========================================
# 6. UMBRALES Y METRICAS
# ========================================

umbrales = np.round(
    np.arange(0.10, 0.901, 0.025), 3
)

def seleccionar_umbral(y_real, probabilidades):

    mejor_umbral = 0.5
    mejor_accuracy = -1

    for u in umbrales:

        pred = (probabilidades >= u).astype(int)
        acc = accuracy_score(y_real, pred)

        if acc > mejor_accuracy + 1e-12:
            mejor_accuracy = acc
            mejor_umbral = float(u)

        elif abs(acc - mejor_accuracy) <= 1e-12:
            if abs(u - 0.5) < abs(mejor_umbral - 0.5):
                mejor_umbral = float(u)

    return mejor_umbral

def evaluar(y_real, prob, umbral):

    pred = (prob >= umbral).astype(int)

    return {
        "accuracy": accuracy_score(y_real, pred),
        "balanced_accuracy": balanced_accuracy_score(
            y_real, pred
        ),
        "roc_auc": roc_auc_score(y_real, prob),
        "average_precision": average_precision_score(
            y_real, prob
        ),
        "recall": recall_score(
            y_real, pred, zero_division=0
        ),
        "f1": f1_score(
            y_real, pred, zero_division=0
        ),
        "positivos": int(pred.sum())
    }

# ========================================
# 7. TORNEO
# ========================================

modelos = construir_modelos()
resultados = []

print("\n========== TORNEO ==========")

for nombre, modelo in modelos.items():

    print(f"\nEntrenando {nombre}...", flush=True)

    inicio = time.perf_counter()

    entrenar(modelo, X_fit, y_fit)

    probabilidades = modelo.predict_proba(X_val)[:, 1]

    umbral = seleccionar_umbral(
        y_val, probabilidades
    )

    metricas = evaluar(
        y_val, probabilidades, umbral
    )

    metricas["modelo"] = nombre
    metricas["umbral"] = umbral
    metricas["tiempo_s"] = round(
        time.perf_counter() - inicio, 2
    )

    resultados.append(metricas)

    print("Accuracy:", round(metricas["accuracy"], 6))
    print("ROC-AUC:", round(metricas["roc_auc"], 6))
    print("F1:", round(metricas["f1"], 6))
    print("Umbral:", umbral)
    print("Positivos predichos:", metricas["positivos"])
    print("Tiempo:", metricas["tiempo_s"], "s")

# ========================================
# 8. RANKING DE SELECCION
# ========================================

tabla = pd.DataFrame(resultados)

tabla = tabla.sort_values(
    ["accuracy", "roc_auc"],
    ascending=[False, False]
).reset_index(drop=True)

tabla.to_csv(
    OUT / "torneo_modelos.csv",
    index=False
)

print("\n========== RANKING ==========")
print(tabla[[
    "modelo",
    "accuracy",
    "roc_auc",
    "average_precision",
    "recall",
    "f1",
    "umbral",
    "positivos"
]].round(5).to_string(index=False))

# ========================================
# 9. GANADOR Y COMPROBACION FINAL
# ========================================

ganador = tabla.iloc[0]

nombre_ganador = ganador["modelo"]
umbral_ganador = float(ganador["umbral"])

modelo_ganador = modelos[nombre_ganador]

prob_check = modelo_ganador.predict_proba(
    X_check
)[:, 1]

metricas_check = evaluar(
    y_check,
    prob_check,
    umbral_ganador
)

baseline_check = (y_check == 0).mean()

print("\n=== COMPROBACION FINAL SIN AJUSTES ===")
print("Modelo:", nombre_ganador)
print("Umbral:", umbral_ganador)

for metrica, valor in metricas_check.items():
    print(f"{metrica}: {valor}")

print("Baseline comprobacion:", baseline_check)
print(
    "Mejora frente a Dummy:",
    metricas_check["accuracy"] - baseline_check
)

print("Matriz de confusion:")
print(confusion_matrix(
    y_check,
    (prob_check >= umbral_ganador).astype(int),
    labels=[0, 1]
))

# ========================================
# 10. REENTRENAMIENTO FINAL
# ========================================

print("\nEntrenando candidato final...")

modelo_final = construir_modelos()[nombre_ganador]

entrenar(modelo_final, X, y)

prob_test = modelo_final.predict_proba(
    X_test
)[:, 1]

pred_test = (
    prob_test >= umbral_ganador
).astype(int)

submission = sample.copy()
submission[TARGET] = pred_test

assert len(submission) == 30000
assert submission["ID"].equals(sample["ID"])
assert submission[TARGET].isin([0, 1]).all()
assert submission.notna().all().all()

ruta = OUT / "submission_torneo.csv"

submission.to_csv(ruta, index=False)

print("\n=== ARCHIVO GENERADO ===")
print("Ruta:", ruta)
print("Modelo:", nombre_ganador)
print("Predicciones:")
print(submission[TARGET].value_counts())

if pred_test.sum() == 0:
    print(
        "AVISO: todas las predicciones son cero. "
        "No se demostro mejora en la clase minoritaria."
    )

print("\nTORNEO FINALIZADO")
print("No se ha enviado nada a Kaggle.")
