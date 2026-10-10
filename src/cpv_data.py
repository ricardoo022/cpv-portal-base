"""Funções partilhadas para ler os contratos e extrair os códigos CPV.

Usadas pelo notebook de exploração e pelo script de preparação, para as duas partes
aplicarem exatamente as mesmas regras.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_PARQUET = ROOT / "data" / "processed" / "raw"

# Split por data de publicação (ver docs/workflow.md, fase 3)
TRAIN_START = pd.Timestamp("2022-01-01")  # decisão em notebooks/02_anos_treino.ipynb
VAL_START = pd.Timestamp("2025-10-01")
TEST_START = pd.Timestamp("2025-11-01")
YEAR_END = pd.Timestamp("2026-01-01")

CPV_PATTERN = r"(\d{8})-\d"
LEVEL_NAMES = {2: "divisão", 3: "grupo", 4: "classe", 5: "categoria"}


ALL_YEARS = range(2012, 2027)


def load_contracts(columns, years=ALL_YEARS, since=None, until=None):
    """Lê os Parquet dos anos pedidos e junta-os, com a data de publicação já convertida.

    Por defeito lê todos os ficheiros: o ficheiro de um ano é o ano de celebração do contrato,
    por isso os contratos publicados num ano estão espalhados por vários ficheiros.
    `since` e `until` filtram pela data de publicação ficheiro a ficheiro, para poupar memória.
    """
    cols = list(dict.fromkeys(["idcontrato", "dataPublicacao", *columns]))
    parts = []
    for y in years:
        part = pd.read_parquet(RAW_PARQUET / f"contratos{y}.parquet", columns=cols)
        part["data"] = pd.to_datetime(part["dataPublicacao"], errors="coerce")
        if since is not None:
            part = part[part["data"] >= since]
        if until is not None:
            part = part[part["data"] < until]
        part["ficheiro"] = y
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def first_code(cpv):
    """Primeiro código de 8 dígitos de cada célula de CPV ('90611000-3 - Serviços...' -> '90611000')."""
    return cpv.str.extract(CPV_PATTERN, expand=False)


def all_codes(cpv):
    """Todos os códigos de 8 dígitos de cada célula de CPV, como lista."""
    return cpv.fillna("").str.findall(CPV_PATTERN)


def code_level(code):
    """Número de dígitos significativos (antes dos zeros finais), no mínimo 2.

    '33000000' -> 2 (divisão), '33600000' -> 3 (grupo), '90911200' -> 6.
    É uma aproximação: assume que os zeros finais significam "nível acima".
    """
    return code.str.rstrip("0").str.len().clip(lower=2)


def level_name(level):
    return level.map(LEVEL_NAMES).fillna("mais detalhado (6-8)")


def normalize_text(s):
    """Minúsculas e espaços normalizados, para comparar textos."""
    return s.fillna("").str.lower().str.split().str.join(" ")


def period(data):
    """Período de cada contrato: 'treino', 'validação', 'teste', ou valor em falta (NaN) fora dos três."""
    out = pd.Series(None, index=data.index, dtype="object")
    out[(data >= TRAIN_START) & (data < VAL_START)] = "treino"
    out[(data >= VAL_START) & (data < TEST_START)] = "validação"
    out[(data >= TEST_START) & (data < YEAR_END)] = "teste"
    return out
