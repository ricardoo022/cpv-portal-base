"""Converte os .xlsx de data/raw/ para Parquet em data/processed/raw/, sem transformar nada.

Lê linha a linha (openpyxl em modo read_only) e grava aos blocos, para não esgotar a RAM.
Todas as colunas ficam como texto; as conversões de tipos fazem-se depois.

Uso:
    .venv/bin/python src/convert_to_parquet.py            # todos os anos
    .venv/bin/python src/convert_to_parquet.py 2025 2026  # só estes anos
"""

import sys
import time
from datetime import datetime
from pathlib import Path

import openpyxl
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed" / "raw"
BATCH = 20_000


def to_text(v):
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.isoformat()
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def convert(xlsx, out):
    t0 = time.time()
    wb = openpyxl.load_workbook(xlsx, read_only=True)
    rows = wb.active.iter_rows(values_only=True)
    header = [str(h) for h in next(rows)]
    schema = pa.schema([(h, pa.string()) for h in header])

    tmp = out.with_suffix(".parquet.part")
    n = 0
    with pq.ParquetWriter(tmp, schema) as writer:
        batch = []
        for row in rows:
            batch.append([to_text(v) for v in row])
            if len(batch) == BATCH:
                writer.write_table(pa.Table.from_pylist([dict(zip(header, r)) for r in batch], schema))
                n += len(batch)
                batch = []
                print(f"\r  {xlsx.name}: {n} linhas", end="", flush=True)
        if batch:
            writer.write_table(pa.Table.from_pylist([dict(zip(header, r)) for r in batch], schema))
            n += len(batch)
    wb.close()
    tmp.replace(out)
    print(f"\r  {xlsx.name}: {n} linhas -> {out.relative_to(ROOT)} ({time.time() - t0:.0f}s)")


def main(years):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(RAW_DIR.glob("contratos*.xlsx"))
    if years:
        files = [f for f in files if int(f.stem.removeprefix("contratos")) in years]
    for f in files:
        convert(f, OUT_DIR / f"{f.stem}.parquet")


if __name__ == "__main__":
    main({int(a) for a in sys.argv[1:]})
