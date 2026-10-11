"""Teste de integração do spike E4-S4 (afinar o TF-IDF) com os dados reais preparados.

Confirma que a secção 14 do docs/qualidade_dados.md e a decisão 6 do E4 continuam a sair das
sugestões guardadas em metrics/e4_s4/ (gravadas pelos treinos no pod) e da validação que o
`prepare.py` escreve:
- os 16 ficheiros (4 alpha × 4 τ) são da validação atual;
- o treino com alpha = 0,0001 e τ = 0 repete o E4-H1 (o treino é repetível);
- cada comparação da tabela de escolha dá o vencedor escrito no doc, e o vencedor final é o
  valor por defeito em src/tfidf.py (`ALPHA`, `TAU`);
- o top-5 do escolhido é o da secção 14.

Os testes marcados `data` saltam se data/processed/ não existir (por exemplo, no CI). Nunca leem o
teste. O último teste não precisa de dados: confirma que o notebook da afinação não abre o teste.
"""

import json
import re
from pathlib import Path

import pandas as pd
import pytest

import comparison as cmp
import evaluation as ev
import tfidf

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
METRICS = ROOT / "metrics"
NOTEBOOK = ROOT / "notebooks" / "05_tfidf_afinacao.ipynb"

ALPHAS = ["1e-4", "1e-5", "1e-6", "1e-7"]
TAUS = [0.0, 0.25, 0.5, 1.0]

# Secção 14 do docs/qualidade_dados.md: (em vigor, candidato, quem fica), pela ordem da regra
ESCOLHA_ALPHA = [("1e-4", "1e-5", "1e-5"), ("1e-5", "1e-6", "1e-6"), ("1e-6", "1e-7", "1e-7")]
ESCOLHA_TAU = [(0.0, 0.25, 0.25), (0.25, 0.5, 0.25), (0.25, 1.0, 0.25)]
# Top-5 do escolhido (alpha 0,0000001, τ 0,25)
TOP5 = {("todos", "codigo"): 0.769, ("todos", "classe"): 0.829,
        ("texto novo", "codigo"): 0.735, ("texto novo", "classe"): 0.808}

dados = pytest.mark.skipif(
    not (DATA / "validacao.parquet").exists() or not (METRICS / "e4_s4" / "alpha_1e-7").exists(),
    reason="dados preparados ou sugestões do E4-S4 ausentes (correr make preparar)",
)


def ficheiro(alpha, tau):
    nome = "sugestoes_validacao.parquet" if tau == 0 else f"sugestoes_validacao_tau{tau:g}.parquet"
    return METRICS / "e4_s4" / f"alpha_{alpha}" / nome


@pytest.fixture(scope="module")
def validacao():
    return pd.read_parquet(DATA / "validacao.parquet", columns=["idcontrato", "codigos", "visto_no_treino"])


@pytest.fixture(scope="module")
def opcoes(validacao):
    """{(alpha, τ): Series de listas de 5 códigos}, alinhadas com a validação pelo idcontrato."""
    ids = validacao["idcontrato"].astype(str)
    out = {}
    for a in ALPHAS:
        for t in TAUS:
            sug = pd.read_parquet(ficheiro(a, t))
            sug["idcontrato"] = sug["idcontrato"].astype(str)
            s = sug.set_index("idcontrato")["sugestoes"].reindex(ids).map(list)
            s.index = validacao.index
            out[(a, t)] = s
    return out


def ganha(em_vigor, candidato, codigos):
    """A regra do E4-T1: 'b' se o candidato ganha, 'a' se fica a opção em vigor."""
    res = {}
    for medida in ("codigo", "classe"):
        a = cmp.weighted_mean(ev.hits(em_vigor, codigos, medida, ev.TOP_K))
        b = cmp.weighted_mean(ev.hits(candidato, codigos, medida, ev.TOP_K))
        res[medida] = cmp.paired_bootstrap(a, b, len(codigos))
    return cmp.choose(res["codigo"], res["classe"], default="a")


@pytest.mark.data
@dados
@pytest.mark.parametrize("alpha", ALPHAS)
def test_ficheiros_de_cada_alpha_sao_da_validacao_atual(validacao, alpha):
    ids = validacao["idcontrato"].astype(str).tolist()
    for t in TAUS:
        sug = pd.read_parquet(ficheiro(alpha, t))
        assert sug["idcontrato"].astype(str).tolist() == ids, (alpha, t)
        assert sug["sugestoes"].map(lambda s: len(set(s)) == ev.TOP_K).all(), (alpha, t)
    info = json.loads((METRICS / "e4_s4" / f"alpha_{alpha}" / "treino.json").read_text(encoding="utf-8"))
    assert info["alpha"] == float(alpha) and info["taus"] == TAUS and info["n_min"] == 10


@pytest.mark.data
@dados
def test_alpha_antigo_sem_correcao_repete_o_e4_h1(opcoes, validacao):
    e4h1 = pd.read_parquet(METRICS / "e4_h1" / "sugestoes_validacao.parquet")
    assert e4h1["sugestoes"].map(list).tolist() == opcoes[("1e-4", 0.0)].tolist()


@pytest.mark.data
@dados
def test_regra_de_escolha_da_os_vencedores_do_doc_e_o_valor_por_defeito(opcoes, validacao):
    codigos = validacao["codigos"]
    alpha = ESCOLHA_ALPHA[0][0]
    for em_vigor, candidato, fica in ESCOLHA_ALPHA:
        assert em_vigor == alpha
        r = ganha(opcoes[(alpha, 0.0)], opcoes[(candidato, 0.0)], codigos)
        alpha = candidato if r == "b" else alpha
        assert alpha == fica, (em_vigor, candidato)
    tau = ESCOLHA_TAU[0][0]
    for em_vigor, candidato, fica in ESCOLHA_TAU:
        assert em_vigor == tau
        r = ganha(opcoes[(alpha, tau)], opcoes[(alpha, candidato)], codigos)
        tau = candidato if r == "b" else tau
        assert tau == fica, (em_vigor, candidato)
    assert (float(alpha), tau) == (tfidf.ALPHA, tfidf.TAU)


@pytest.mark.data
@dados
def test_top5_do_escolhido_e_o_do_doc(opcoes, validacao):
    escolhido = opcoes[(f"{tfidf.ALPHA:.0e}".replace("e-0", "e-"), tfidf.TAU)]
    out = ev.report(escolhido, validacao["codigos"], validacao["visto_no_treino"])
    for (grupo, medida), esperado in TOP5.items():
        assert round(out.loc[(grupo, medida), "top5"], 3) == esperado, (grupo, medida)


def test_notebook_da_afinacao_nao_abre_o_teste():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    codigo = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    for proibido in ("teste", "TEST_START", "YEAR_END"):
        assert proibido not in codigo, proibido
    # Cada leitura de um ficheiro preparado é do treino ou da validação, escrita por extenso
    lidos = re.findall(r'DATA / "([^"]+)"', codigo)
    assert set(lidos) == {"treino.parquet", "validacao.parquet"}, lidos
