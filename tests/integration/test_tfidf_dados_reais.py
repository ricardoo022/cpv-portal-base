"""Teste de integração do TF-IDF (E4-H1) com os dados reais preparados.

Confirma que os números escritos na secção 13 do docs/qualidade_dados.md continuam certos para as
sugestões guardadas em metrics/e4_h1/ (gravadas pelo treino no pod) e a validação que o
`prepare.py` escreve. Se a preparação ou a avaliação mudarem, o teste falha e o doc tem de ser
atualizado (com o notebook 04_tfidf.ipynb).

Os testes marcados `data` saltam se data/processed/ não existir (por exemplo, no CI). Nunca leem o
teste. O último teste não precisa de dados: confirma que o notebook do TF-IDF não abre o teste.
"""

import json
import re
from pathlib import Path

import pandas as pd
import pytest

import evaluation as ev

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
SUGESTOES = ROOT / "metrics" / "e4_h1" / "sugestoes_validacao.parquet"
NOTEBOOK = ROOT / "notebooks" / "04_tfidf.ipynb"

# Secção 13 do docs/qualidade_dados.md: top-5 do TF-IDF do E4-H1
TOP5 = {("todos", "codigo"): 0.432, ("todos", "classe"): 0.379,
        ("texto novo", "codigo"): 0.385, ("texto novo", "classe"): 0.342}

dados = pytest.mark.skipif(
    not (DATA / "validacao.parquet").exists() or not SUGESTOES.exists(),
    reason="dados preparados ou sugestões do E4-H1 ausentes (correr make preparar)",
)


@pytest.fixture(scope="module")
def validacao():
    return pd.read_parquet(DATA / "validacao.parquet", columns=["idcontrato", "codigos", "visto_no_treino"])


@pytest.fixture(scope="module")
def sugestoes(validacao):
    sug = pd.read_parquet(SUGESTOES)
    sug["idcontrato"] = sug["idcontrato"].astype(str)
    return sug


@pytest.mark.data
@dados
def test_sugestoes_guardadas_sao_da_validacao_atual(validacao, sugestoes):
    # Uma linha por contrato, pela ordem do validacao.parquet, com 5 códigos distintos
    assert sugestoes["idcontrato"].tolist() == validacao["idcontrato"].astype(str).tolist()
    assert sugestoes["sugestoes"].map(lambda s: len(set(s)) == ev.TOP_K).all()


@pytest.mark.data
@dados
def test_top5_do_tfidf_na_validacao_e_o_do_doc(validacao, sugestoes):
    sug = pd.Series([list(s) for s in sugestoes["sugestoes"]], index=validacao.index)
    out = ev.report(sug, validacao["codigos"], validacao["visto_no_treino"])
    for (grupo, medida), esperado in TOP5.items():
        assert round(out.loc[(grupo, medida), "top5"], 3) == esperado, (grupo, medida)


def test_notebook_do_tfidf_nao_abre_o_teste():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    codigo = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    for proibido in ("teste", "TEST_START", "YEAR_END"):
        assert proibido not in codigo, proibido
    # Cada leitura de um ficheiro preparado é do treino ou da validação, escrita por extenso
    lidos = re.findall(r'DATA / "([^"]+)"', codigo)
    assert set(lidos) == {"treino.parquet", "validacao.parquet"}, lidos
