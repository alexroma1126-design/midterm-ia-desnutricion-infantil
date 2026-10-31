from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from lightgbm import LGBMClassifier

SEED = 2026
DATA = Path("data/ii_sem2026")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

CATS = ["sexo", "region", "nivel_educacion_madre"]

X = pd.read_csv(DATA / "X_train_Kaggle.csv")
y = pd.read_csv(DATA / "y_train_Kaggle.csv")[
    "alerta_desnutricion"
]

# Misma ingenieria de caracteristicas del torneo
def crear_features(df):
    d = df.copy()

    edad = d["edad_meses"].clip(lower=1, upper=24)
    talla_m = d["talla_cm"] / 100

    d["edad_negativa"] = (d["edad_meses"] < 0).astype(int)
    d["edad_mayor_24"] = (d["edad_meses"] > 24).astype(int)

    d["peso_faltante"] = d["peso_kg"].isna().astype(int)
    d["talla_faltante"] = d["talla_cm"].isna().astype(int)

    d["peso_por_edad"] = d["peso_kg"] / edad
    d["talla_por_edad"] = d["talla_cm"] / edad
    d["imc_sintetico"] = d["peso_kg"] / talla_m.pow(2)
    d["peso_por_talla"] = d["peso_kg"] / d["talla_cm"]

    d = d.replace([np.inf, -np.inf], np.nan)

    for col in CATS:
        d[col] = (
            d[col].astype("string")
            .fillna("Desconocido").astype(str)
        )

    return d

X = crear_features(X)

# Reproducimos las particiones del torneo
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

NUMS = [c for c in X.columns if c not in CATS]

def crear_pipeline(clasificador):
    prep = ColumnTransformer([
        ("num", SimpleImputer(
            strategy="median",
            add_indicator=True
        ), NUMS),
        ("cat", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ), CATS)
    ])

    return Pipeline([
        ("prep", prep),
        ("modelo", clasificador)
    ])

modelos = {
    "RandomForest": crear_pipeline(
        RandomForestClassifier(
            n_estimators=180,
            max_depth=12,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=SEED
        )
    ),

    "LightGBM": crear_pipeline(
        LGBMClassifier(
            n_estimators=500,
            learning_rate=0.035,
            num_leaves=15,
            min_child_samples=110,
            reg_lambda=5,
            verbosity=-1,
            n_jobs=4,
            random_state=SEED
        )
    )
}

probas_val = {}
probas_check = {}

for nombre, modelo in modelos.items():
    print(f"\nEntrenando {nombre}...", flush=True)

    modelo.fit(X_fit, y_fit)

    probas_val[nombre] = modelo.predict_proba(
        X_val
    )[:, 1]

    probas_check[nombre] = modelo.predict_proba(
        X_check
    )[:, 1]

# Ensemble simple sin entrenar sobre validacion
probas_val["Ensemble"] = (
    probas_val["RandomForest"] +
    probas_val["LightGBM"]
) / 2

probas_check["Ensemble"] = (
    probas_check["RandomForest"] +
    probas_check["LightGBM"]
) / 2

def analizar_ranking(y_real, probabilidades, nombre):
    etiquetas = np.asarray(y_real)
    n = len(etiquetas)

    orden = np.argsort(-probabilidades, kind="stable")
    reales_ordenados = etiquetas[orden]

    acumulados = np.cumsum(reales_ordenados)
    ks = np.arange(1, n + 1)

    # Aciertos ganados menos aciertos perdidos
    ganancias = 2 * acumulados - ks

    # Incluimos k=0: no cambiar ninguna prediccion
    ganancias = np.concatenate([[0], ganancias])

    k_mejor = int(np.argmax(ganancias))
    ganancia_mejor = int(ganancias[k_mejor])

    baseline = float(np.mean(etiquetas == 0))

    print(f"\n=== {nombre} ===")
    print("ROC-AUC:", round(
        roc_auc_score(y_real, probabilidades), 6
    ))
    print("Accuracy baseline:", round(baseline, 6))

    print("\n k     TP     FP    Precision   Accuracy")

    for k in [25, 50, 100, 200, 500, 1000, 2000]:
        if k > n:
            continue

        tp = int(acumulados[k - 1])
        fp = k - tp
        precision = tp / k
        accuracy = baseline + (tp - fp) / n

        print(
            f"{k:4d}  {tp:5d}  {fp:5d} "
            f"   {precision:.4f}    {accuracy:.6f}"
        )

    print("\nMejor k exploratorio:", k_mejor)
    print("Mejora maxima observada en aciertos:",
          ganancia_mejor)
    print(
        "Mejor accuracy exploratorio:",
        round(baseline + ganancia_mejor / n, 6)
    )

    return k_mejor, ganancia_mejor

resultados = {}

print("\n========== AUDITORIA DE ACCURACY ==========")

for nombre, probabilidades in probas_val.items():
    k, ganancia = analizar_ranking(
        y_val, probabilidades, nombre
    )

    resultados[nombre] = {
        "k": k,
        "ganancia": ganancia
    }

# Seleccion conservadora:
# exigimos al menos 10 aciertos netos adicionales
ganador = max(
    resultados,
    key=lambda nombre: resultados[nombre]["ganancia"]
)

mejora = resultados[ganador]["ganancia"]
k_ganador = resultados[ganador]["k"]

print("\n========== DECISION ==========")

if mejora < 10:
    print("No encontramos una mejora suficientemente")
    print("convincente para cambiar predicciones.")
    print("Seleccion conservadora: predecir todo cero.")
    k_ganador = 0
else:
    print("Candidato:", ganador)
    print("Ganancia en seleccion:", mejora)
    print("k seleccionado:", k_ganador)

# Comprobacion adicional.
# Esta particion ya fue consultada en el torneo
# anterior, por lo que NO es un test virgen.
y_real = np.asarray(y_check)
n = len(y_real)

pred = np.zeros(n, dtype=int)

if k_ganador > 0:
    fraccion = k_ganador / len(y_val)

    k_check = max(1, int(round(fraccion * n)))

    orden = np.argsort(
        -probas_check[ganador],
        kind="stable"
    )

    pred[orden[:k_check]] = 1

baseline_check = accuracy_score(
    y_real, np.zeros(n, dtype=int)
)

accuracy_check = accuracy_score(y_real, pred)

print("\n========== COMPROBACION ==========")
print("Baseline:", round(baseline_check, 6))
print("Accuracy candidato:", round(accuracy_check, 6))
print("Diferencia:", round(
    accuracy_check - baseline_check, 6
))
print("Predicciones positivas:", int(pred.sum()))
print("Positivos acertados:", int(
    ((pred == 1) & (y_real == 1)).sum()
))
print("Falsos positivos:", int(
    ((pred == 1) & (y_real == 0)).sum()
))

print("\nAUDITORIA TERMINADA")
print("No se envio ninguna prediccion a Kaggle.")
