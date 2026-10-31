from pathlib import Path
import pandas as pd

DATA = Path("data/ii_sem2026")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

X = pd.read_csv(DATA / "X_train_Kaggle.csv")
y = pd.read_csv(DATA / "y_train_Kaggle.csv")

df = X.copy()
df["alerta"] = y["alerta_desnutricion"].to_numpy()

print("\n=== INTERACCION REGION Y EDUCACION ===")

tabla = (
    df.groupby(["region", "nivel_educacion_madre"])["alerta"]
      .agg(n="size", alertas="sum", tasa="mean")
      .reset_index()
)

tabla["alerta_pct"] = tabla["tasa"] * 100

tabla = tabla.sort_values(
    "alerta_pct", ascending=False
)

print(
    tabla[
        ["region", "nivel_educacion_madre",
         "n", "alertas", "alerta_pct"]
    ].round(3).to_string(index=False)
)

print("\n=== GRUPOS CON MAS DEL 50% DE ALERTAS ===")

grupos = tabla[
    (tabla["tasa"] > 0.5) &
    (tabla["n"] >= 100)
]

if grupos.empty:
    print("Ningun grupo con al menos 100 registros supera el 50%.")
else:
    print(grupos.to_string(index=False))

print("\n=== MAYOR TASA OBSERVADA ===")
print(tabla.iloc[0].to_string())

tabla.to_csv(
    OUT / "interacciones_region_educacion.csv",
    index=False
)

print("\nAnalisis completado.")
