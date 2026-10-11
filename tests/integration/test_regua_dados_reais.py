"""Teste de integração da régua "códigos mais comuns" com os dados reais preparados.

Confirma que os números escritos na secção 12 do docs/qualidade_dados.md continuam certos para o
treino e a validação que o `prepare.py` escreve. Se o `prepare.py` mudar e os números mudarem, o
teste falha e o doc tem de ser atualizado (com o notebook 03_regua.ipynb).

Os testes marcados `data` saltam se data/processed/ não existir (por exemplo, no CI). Nunca leem o
teste. O último teste não precisa de dados: confirma que o notebook da régua não abre o teste.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

import baselines as bl
import evaluation as ev

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
NOTEBOOK = ROOT / "notebooks" / "03_regua.ipynb"

# Secção 12 do docs/qualidade_dados.md
REGUA = ["33600000", "33140000", "33690000", "33100000", "98300000"]
TOP5 = {("todos", "codigo"): 0.263, ("todos", "classe"): 0.156,
        ("texto novo", "codigo"): 0.237, ("texto novo", "classe"): 0.148}

dados = pytest.mark.skipif(
    not (DATA / "treino.parquet").exists() or not (DATA / "validacao.parquet").exists(),
    reason="dados preparados ausentes (correr make preparar)",
)


@pytest.fixture(scope="module")
def treino():
    return pd.read_parquet(DATA / "treino.parquet", columns=["codigo", "n_contratos"])


@pytest.fixture(scope="module")
def validacao():
    return pd.read_parquet(DATA / "validacao.parquet", columns=["codigos", "visto_no_treino"])


@pytest.mark.data
@dados
@pytest.mark.parametrize("n_min", [5, 10, 20])
def test_regua_real_e_a_do_doc_com_qualquer_n(treino, n_min):
    assert bl.most_common_labels(treino["codigo"], n_min=n_min) == REGUA


@pytest.mark.data
@dados
def test_regua_por_contratos_e_diferente(treino):
    # Secção 12: contar contratos troca a 3.ª e a 4.ª etiqueta e põe 15800000 em 5.º
    assert bl.most_common_labels(treino["codigo"], treino["n_contratos"]) == [
        "33600000", "33140000", "33100000", "33690000", "15800000"]


@pytest.mark.data
@dados
def test_top5_da_regua_na_validacao_e_o_do_doc(validacao):
    sug = bl.suggest_most_common(REGUA, validacao.index)
    out = ev.report(sug, validacao["codigos"], validacao["visto_no_treino"])
    for (grupo, medida), esperado in TOP5.items():
        assert round(out.loc[(grupo, medida), "top5"], 3) == esperado, (grupo, medida)


def test_notebook_da_regua_nao_abre_o_teste():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    codigo = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    for proibido in ("teste.parquet", "TEST_START", 'FILES["teste"]', "YEAR_END"):
        assert proibido not in codigo, proibido
    assert "validacao.parquet" in codigo  # o notebook lê a validação, e só ela
