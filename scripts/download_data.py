"""Descarrega os ficheiros .xlsx de contratos do Portal BASE (dados.gov.pt) para data/raw/.

Uso:
    python3 scripts/download_data.py            # todos os anos
    python3 scripts/download_data.py 2025 2024  # só estes anos

Salta ficheiros que já existem com o tamanho indicado pela API.
"""

import json
import re
import sys
import urllib.request
from pathlib import Path

DATASET = "contratos-publicos-portal-base-impic-contratos-de-2012-a-2026"
API_URL = f"https://dados.gov.pt/api/1/datasets/{DATASET}/"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def list_resources():
    with urllib.request.urlopen(API_URL, timeout=60) as resp:
        data = json.load(resp)
    resources = {}
    for r in data["resources"]:
        m = re.fullmatch(r"contratos(\d{4})\.xlsx", r["title"])
        if m:
            resources[int(m.group(1))] = r
    return resources


def download(url, dest, expected_size):
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as f:
        done = 0
        while chunk := resp.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if expected_size:
                print(f"\r  {dest.name}: {done / 1e6:.1f} / {expected_size / 1e6:.1f} MB", end="", flush=True)
    print()
    if expected_size and tmp.stat().st_size != expected_size:
        raise RuntimeError(f"{dest.name}: tamanho {tmp.stat().st_size}, esperado {expected_size}")
    tmp.replace(dest)


def main(years):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    resources = list_resources()
    wanted = years or sorted(resources)
    missing = [y for y in wanted if y not in resources]
    if missing:
        sys.exit(f"Anos sem ficheiro no dataset: {missing}. Disponíveis: {sorted(resources)}")

    for year in wanted:
        r = resources[year]
        dest = RAW_DIR / r["title"]
        size = r.get("filesize")
        if dest.exists() and size and dest.stat().st_size == size:
            print(f"  {dest.name}: já existe, a saltar")
            continue
        download(r["url"], dest, size)


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]])
