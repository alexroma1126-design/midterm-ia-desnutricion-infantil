import pandas as pd
from pathlib import Path

DATA = Path("data/ii_sem2026")

X = pd.read_csv(DATA / "X_train_Kaggle.csv")
y = pd.read_csv(DATA / "y_train_Kaggle.csv")

df = X.copy()
df["alerta"] = y["alerta_desnutricion"].to_numpy()

print("\n=== ALERTAS POR CATEGORIA ===")

for col in ["sexo", "region", "nivel_educacion_madre"]:
    print(f"\nVARIABLE: {col}")

    tabla = df.groupby(col, dropna=False)["alerta"].agg(
        ["count", "mean"]
    )

    tabla["alerta_pct"] = tabla["mean"] * 100

    print(
        tabla[["count", "alerta_pct"]]
        .round(3).to_string()
    )

print("\n=== ALERTAS SEGUN EDAD ===")

df["grupo_edad"] = pd.cut(
    df["edad_meses"],
    bins=[-float("inf"), -1, 5, 11, 17, 24, float("inf")],
    labels=[
        "Negativa", "0-5 meses", "6-11 meses",
        "12-17 meses", "18-24 meses", ">24 meses"
    ]
)

tabla_edad = df.groupby(
    "grupo_edad", observed=True
)["alerta"].agg(["count", "mean"])

tabla_edad["alerta_pct"] = tabla_edad["mean"] * 100
print(tabla_edad[["count", "alerta_pct"]].round(3))

print("\n=== ALERTAS SEGUN VALORES FALTANTES ===")

for col in ["peso_kg", "talla_cm"]:
    tabla = df.groupby(
        df[col].isna()
    )["alerta"].agg(["count", "mean"])

    tabla["alerta_pct"] = tabla["mean"] * 100

    print(f"\n{col}:")
    print(tabla[["count", "alerta_pct"]].round(3))

print("\n=== ALERTAS SEGUN CUANTILES DE PESO Y TALLA ===")

for col in ["peso_kg", "talla_cm"]:
    grupos = pd.qcut(
        df[col], q=5, duplicates="drop"
    )

    tabla = df.groupby(
        grupos, observed=True
    )["alerta"].agg(["count", "mean"])

    tabla["alerta_pct"] = tabla["mean"] * 100

    print(f"\n{col}:")
    print(tabla[["count", "alerta_pct"]].round(3))
