from pathlib import Path
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, roc_auc_score

DATA = Path("data/ii_sem2026")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

X = pd.read_csv(DATA / "X_train_Kaggle.csv")
y = pd.read_csv(DATA / "y_train_Kaggle.csv")[
    "alerta_desnutricion"
].to_numpy()

def crear_features(df):
    d = df.copy()

    edad = d["edad_meses"].clip(1, 24)
    talla_m = d["talla_cm"] / 100

    d["peso_edad"] = d["peso_kg"] / edad
    d["talla_edad"] = d["talla_cm"] / edad
    d["imc"] = d["peso_kg"] / talla_m.pow(2)

    d["peso_bajo_relativo"] = (
        d["peso_edad"] < 0.25
    ).astype(int)

    d["talla_baja_relativa"] = (
        d["talla_edad"] < 3
    ).astype(int)

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

    return d.replace([np.inf, -np.inf], np.nan)

X = crear_features(X)

cat = ["sexo", "region", "nivel_educacion_madre"]
num = [c for c in X.columns if c not in cat]

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=2026
)

predicciones_oof = np.zeros(len(y))
filas = []

for fold, (idx_train, idx_val) in enumerate(
    cv.split(X, y), start=1
):
    preprocesador = ColumnTransformer([
        ("num", SimpleImputer(
            strategy="median",
            add_indicator=True
        ), num),

        ("cat", OneHotEncoder(
            handle_unknown="ignore"
        ), cat)
    ])

    modelo = Pipeline([
        ("preprocesamiento", preprocesador),
        ("clasificador", XGBClassifier(
            n_estimators=400,
            max_depth=3,
            learning_rate=0.03,
            min_child_weight=20,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=10,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=4,
            random_state=2026
        ))
    ])

    modelo.fit(X.iloc[idx_train], y[idx_train])

    prob = modelo.predict_proba(
        X.iloc[idx_val]
    )[:, 1]

    predicciones_oof[idx_val] = prob

    pred = (prob >= 0.5).astype(int)

    baseline = np.mean(y[idx_val] == 0)
    accuracy = accuracy_score(y[idx_val], pred)

    filas.append({
        "fold": fold,
        "baseline": baseline,
        "accuracy": accuracy,
        "mejora": accuracy - baseline,
        "positivos_predichos": int(pred.sum()),
        "roc_auc": roc_auc_score(y[idx_val], prob),
        "prob_max": float(prob.max())
    })

    print(
        f"Fold {fold}: "
        f"baseline={baseline:.6f}, "
        f"XGB={accuracy:.6f}, "
        f"mejora={accuracy-baseline:+.6f}, "
        f"positivos={pred.sum()}",
        flush=True
    )

tabla = pd.DataFrame(filas)

print("\n=== VALIDACION CRUZADA XGBOOST ===")
print(tabla.round(6).to_string(index=False))

pred_oof = (predicciones_oof >= 0.5).astype(int)

accuracy_global = accuracy_score(y, pred_oof)
baseline_global = np.mean(y == 0)

print("\n=== RESULTADO GLOBAL OOF ===")
print("Baseline:", round(baseline_global, 6))
print("XGBoost:", round(accuracy_global, 6))
print(
    "Mejora:",
    round(accuracy_global - baseline_global, 6)
)
print(
    "ROC-AUC:",
    round(roc_auc_score(y, predicciones_oof), 6)
)
print(
    "Probabilidad maxima:",
    round(predicciones_oof.max(), 6)
)
print(
    "Positivos predichos:",
    int(pred_oof.sum())
)

print("\n=== PRECISION EN LOS CASOS DE MAYOR RIESGO ===")
orden = np.argsort(-predicciones_oof)

for k in [25, 50, 100, 250, 500, 1000]:
    precision = y[orden[:k]].mean()
    print(f"Top {k}: {precision:.4%}")

tabla.to_csv(
    OUT / "validacion_xgboost.csv",
    index=False
)

print("\n=== DECISION PRELIMINAR ===")

mejora_pp = 100 * (accuracy_global - baseline_global)
folds_positivos = int((tabla["mejora"] > 0).sum())

if mejora_pp >= 0.10 and folds_positivos >= 4:
    print("CANDIDATO: estudiar la mejora con mas rigor.")
else:
    print("NO HAY EVIDENCIA SUFICIENTE PARA REEMPLAZAR EL BASELINE.")

print("\nNo se ha generado ni enviado una nueva submission.")
