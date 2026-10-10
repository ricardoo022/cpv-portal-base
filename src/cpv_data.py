"""Funções partilhadas para ler os contratos e extrair os códigos CPV.

Usadas pelo notebook de exploração e pelo script de preparação, para as duas partes
aplicarem exatamente as mesmas regras.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_PARQUET = ROOT / "data" / "processed" / "raw"

# Split por data de publicação (ver docs/workflow.md, fase 3)
YEAR_START = pd.Timestamp("2025-01-01")
VAL_START = pd.Timestamp("2025-10-01")
TEST_START = pd.Timestamp("2025-11-01")
YEAR_END = pd.Timestamp("2026-01-01")

CPV_PATTERN = r"(\d{8})-\d"
LEVEL_NAMES = {2: "divisão", 3: "grupo", 4: "classe", 5: "categoria"}


def load_contracts(columns, years=(2025, 2026)):
    """Lê os Parquet dos anos pedidos e junta-os, com a data de publicação já convertida."""
    cols = list(dict.fromkeys(["idcontrato", "dataPublicacao", *columns]))
    df = pd.concat(
        [pd.read_parquet(RAW_PARQUET / f"contratos{y}.parquet", columns=cols) for y in years],
        ignore_index=True,
    )
    df["data"] = pd.to_datetime(df["dataPublicacao"], errors="coerce")
    return df


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
    """Período de cada contrato: 'treino', 'validação', 'teste' ou None (fora de 2025)."""
    out = pd.Series(None, index=data.index, dtype="object")
    out[(data >= YEAR_START) & (data < VAL_START)] = "treino"
    out[(data >= VAL_START) & (data < TEST_START)] = "validação"
    out[(data >= TEST_START) & (data < YEAR_END)] = "teste"
    return out
