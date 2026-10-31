from pathlib import Path
import nbformat
import pandas as pd

archivo = Path("midterm_ia_ejecutado.ipynb")

nb = nbformat.read(archivo, as_version=4)
nbformat.validate(nb)

codigo = [c for c in nb.cells if c.cell_type == "code"]

ejecutadas = [
    c for c in codigo
    if c.execution_count is not None
]

errores = [
    (i + 1, o.get("evalue", ""))
    for i, c in enumerate(nb.cells)
    if c.cell_type == "code"
    for o in c.get("outputs", [])
    if o.output_type == "error"
]

problemas_latex = [
    i + 1
    for i, c in enumerate(nb.cells)
    if c.cell_type == "markdown"
    and any(
        ord(ch) < 32 and ch not in "\n\r\t"
        for ch in c.source
    )
]

print("=== AUDITORIA DEL NOTEBOOK ===")
print("Celdas totales:", len(nb.cells))
print("Celdas de codigo:", len(codigo))
print("Celdas ejecutadas:", len(ejecutadas))
print("Errores:", errores)
print("Posibles problemas LaTeX:", problemas_latex)

csv = Path("outputs/submission.csv")

if csv.exists():
    df = pd.read_csv(csv)
    sample = pd.read_csv(
        "data/ii_sem2026/sample_submission_malnutrition.csv"
    )

    print("\n=== AUDITORIA DE KAGGLE ===")
    print("Filas:", len(df))
    print("Columnas:", df.columns.tolist())
    print("IDs correctos:", df["ID"].equals(sample["ID"]))
    print("Valores faltantes:", int(df.isna().sum().sum()))
    print("Predicciones:")
    print(df["alerta_desnutricion"].value_counts())

print("\n=== RESULTADO ===")

if (
    len(codigo) == len(ejecutadas)
    and not errores
    and not problemas_latex
):
    print("NOTEBOOK VALIDADO")
else:
    print("REVISAR OBSERVACIONES ANTERIORES")
