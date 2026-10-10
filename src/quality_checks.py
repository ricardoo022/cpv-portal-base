"""Verificações de qualidade nos Parquet de data/processed/raw/ (fase 2 do workflow.md).

Só conta. Os exemplos que mostra vêm apenas de contratos publicados antes de novembro de 2025,
para não espreitar o período de teste.

Uso:
    .venv/bin/python src/quality_checks.py 2025 2026
"""

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IN_DIR = ROOT / "data" / "processed" / "raw"
TEST_START = pd.Timestamp("2025-11-01")
CPV_CODE = re.compile(r"\b(\d{8})-\d\b")


def pct(n, total):
    return f"{n:>9,} ({100 * n / total:5.1f}%)".replace(",", " ")


def filled(s):
    return s.notna() & (s.str.strip() != "")


def main(years):
    files = sorted(IN_DIR.glob("contratos*.parquet"))
    if years:
        files = [f for f in files if int(f.stem.removeprefix("contratos")) in years]
    cols = ["idcontrato", "objectoContrato", "descContrato", "CPV", "dataPublicacao"]
    df = pd.concat([pd.read_parquet(f, columns=cols).assign(ficheiro=f.stem) for f in files], ignore_index=True)
    total = len(df)
    df["data"] = pd.to_datetime(df["dataPublicacao"], errors="coerce")
    train_period = df["data"] < TEST_START

    print(f"Contratos: {total:,}".replace(",", " "), "em", ", ".join(f.name for f in files))

    print("\n== Contratos repetidos (mesmo idcontrato)")
    print("  idcontrato repetido:", pct(df["idcontrato"].duplicated().sum(), total))
    dup = df[df["idcontrato"].duplicated(keep=False)]
    same = dup.groupby("idcontrato")[["objectoContrato", "CPV"]].nunique().max(axis=1) == 1
    print("  cópias com o mesmo texto e CPV:", f"{same.sum()} de {len(same)}")

    print("\n== Data de publicação")
    print("  sem data válida:    ", pct(df["data"].isna().sum(), total))
    by_year = df["data"].dt.year.value_counts().sort_index()
    for y, n in by_year.items():
        print(f"  publicados em {int(y)}:", pct(n, total))

    print("\n== Campos preenchidos")
    obj, desc, cpv = filled(df["objectoContrato"]), filled(df["descContrato"]), filled(df["CPV"])
    print("  objectoContrato:    ", pct(obj.sum(), total))
    print("  descContrato:       ", pct(desc.sum(), total))
    print("  CPV:                ", pct(cpv.sum(), total))
    print("  objecto + CPV:      ", pct((obj & cpv).sum(), total))
    norm = lambda s: s.fillna("").str.lower().str.split().str.join(" ")
    print("  descContrato igual ao objectoContrato:", pct((norm(df["objectoContrato"]) == norm(df["descContrato"])).sum(), total))

    print("\n== Códigos por célula de CPV (só células preenchidas)")
    codes = df.loc[cpv, "CPV"].map(CPV_CODE.findall)
    n_codes = codes.map(len)
    n_cpv = int(cpv.sum())
    for k, n in n_codes.clip(upper=5).value_counts().sort_index().items():
        label = f"{k} códigos" if k < 5 else "5 ou mais"
        print(f"  {label:<20}", pct(n, n_cpv))
    multi = codes[n_codes > 1]
    same_div = multi.map(lambda cs: len({c[:2] for c in cs}) == 1)
    print("  vários, mesma divisão:    ", pct(same_div.sum(), n_cpv))
    print("  vários, divisões diferentes:", pct((~same_div).sum(), n_cpv))
    no_code = n_codes == 0
    if no_code.any():
        ex = df.loc[cpv].loc[no_code & train_period[cpv], "CPV"].head(5).tolist()
        print("  exemplos sem código reconhecido:", ex)

    print("\n== Textos duplicados (objectoContrato, ignorando maiúsculas e espaços)")
    base = df.loc[obj & cpv].copy()
    base["texto"] = base["objectoContrato"].str.lower().str.split().str.join(" ")
    base["primeiro"] = base["CPV"].map(lambda s: (CPV_CODE.findall(s) or [None])[0])
    n_base = len(base)
    dup = base["texto"].duplicated(keep="first")
    print("  contratos com texto repetido:", pct(dup.sum(), n_base))
    print("  textos distintos:     ", pct(base["texto"].nunique(), n_base))
    per_text = base.groupby("texto")["primeiro"].nunique()
    conflict = per_text[per_text > 1]
    print("  textos repetidos com códigos diferentes:", f"{len(conflict):,}".replace(",", " "),
          f"(cobrem {base['texto'].isin(conflict.index).sum():,} contratos)".replace(",", " "))
    top = base.loc[train_period[base.index], "texto"].value_counts().head(5)
    print("  textos mais repetidos (antes de novembro de 2025):")
    for t, n in top.items():
        print(f"    {n:>6}  {t[:80]}")


if __name__ == "__main__":
    main({int(a) for a in sys.argv[1:]})
