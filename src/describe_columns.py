"""Descreve cada coluna dos contratos: preenchimento, valores distintos e exemplos.

Usa só os contratos de treino (publicados de janeiro a setembro de 2025), para não espreitar o teste.
Os resultados estão em docs/dados.md.

Uso:
    .venv/bin/python src/describe_columns.py
"""

import pandas as pd

import cpv_data as cd

cols = pd.read_parquet(cd.RAW_PARQUET / "contratos2025.parquet").columns.tolist()
df = cd.load_contracts(cols)
df = df[cd.period(df["data"]) == "treino"].drop_duplicates("idcontrato")
n = len(df)
print(f"Contratos de treino: {n}\n")

print("| Coluna | Preenchida | Valores distintos | Valores mais comuns |")
print("|---|---|---|---|")
for c in cols:
    s = df[c]
    filled = s.notna() & (s.str.strip() != "")
    vals = s[filled].str.replace(r"\s+", " ", regex=True).str.strip()
    top = vals.value_counts().head(3)
    ex = "; ".join(f"`{v[:50]}`{'…' if len(v) > 50 else ''} ({k})" for v, k in top.items())
    ex = ex.replace("|", "\\|")
    distinct = f"{vals.nunique():,}".replace(",", " ")
    print(f"| `{c}` | {filled.mean():.1%} | {distinct} | {ex} |")
