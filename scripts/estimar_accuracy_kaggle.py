from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path("data/ii_sem2026")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

X = pd.read_csv(DATA / "X_train_Kaggle.csv")
y = pd.read_csv(DATA / "y_train_Kaggle.csv")[
    "alerta_desnutricion"
]
X_test = pd.read_csv(DATA / "X_test_Kaggle.csv")

GRUPOS = ["region", "nivel_educacion_madre"]

# Estimar probabilidades usando solo las etiquetas disponibles.
train = X[GRUPOS].copy()
train["alerta"] = y.to_numpy()

estadisticas = (
    train.groupby(GRUPOS)["alerta"]
    .agg(alertas="sum", n_train="size")
    .reset_index()
)

conteos_test = (
    X_test.groupby(GRUPOS)
    .size()
    .rename("n_test")
    .reset_index()
)

tabla = estadisticas.merge(
    conteos_test,
    on=GRUPOS,
    how="outer"
)

assert tabla["n_train"].notna().all()
assert tabla["n_test"].notna().all()

# Posterior beta-binomial con prior Beta(1,1).
a = tabla["alertas"] + 1
b = tabla["n_train"] - tabla["alertas"] + 1

tabla["prob_alerta"] = a / (a + b)

tabla["alertas_esperadas"] = (
    tabla["n_test"] * tabla["prob_alerta"]
)

# Varianza predictiva beta-binomial.
tabla["varianza_predictiva"] = (
    tabla["n_test"]
    * a * b
    * (a + b + tabla["n_test"])
    / ((a + b)**2 * (a + b + 1))
)

n = len(X_test)
alertas_estimadas = tabla["alertas_esperadas"].sum()
desviacion = np.sqrt(
    tabla["varianza_predictiva"].sum()
)

# Al predecir siempre 0, los aciertos
# son exactamente los registros sin alerta.
accuracy_esperado = 1 - alertas_estimadas / n

limite_inferior = (
    1 - (alertas_estimadas + 1.96 * desviacion) / n
)
limite_superior = (
    1 - (alertas_estimadas - 1.96 * desviacion) / n
)

print("\n=== ESTIMACION DE ACCURACY KAGGLE ===")

print(
    "Prevalencia observada train:",
    f"{y.mean():.4%}"
)
print(
    "Prevalencia estimada test:",
    f"{alertas_estimadas/n:.4%}"
)
print(
    "Alertas esperadas en test:",
    round(alertas_estimadas)
)
print(
    "Accuracy esperado (todo cero):",
    f"{accuracy_esperado:.4%}"
)

print(
    "Intervalo predictivo aproximado 95%:",
    f"[{limite_inferior:.4%}, {limite_superior:.4%}]"
)

print("\n=== GRUPOS DEL TEST ===")

print(
    tabla[
        GRUPOS + [
            "n_test",
            "prob_alerta",
            "alertas_esperadas"
        ]
    ].round(4).to_string(index=False)
)

print("\n=== VARIABILIDAD DEL LEADERBOARD ===")

p = alertas_estimadas / n

for nombre, n_parcial in [
    ("Publico 10%", 3000),
    ("Privado 90%", 27000)
]:
    margen = 1.96 * np.sqrt(
        p * (1 - p) / n_parcial
    )

    print(
        f"{nombre}: variacion ilustrativa +/- "
        f"{margen * 100:.3f} puntos porcentuales"
    )

tabla.to_csv(
    OUT / "estimacion_accuracy_kaggle.csv",
    index=False
)

print("\nANALISIS COMPLETADO")
print("Ningun archivo fue enviado a Kaggle.")
