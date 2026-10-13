"""Teste de integração do spike E4-S5 (épocas e um alpha mais pequeno no TF-IDF) com os dados reais.

Confirma que a decisão do notebooks/06_tfidf_epocas.ipynb continua a sair das sugestões guardadas
em metrics/e4_s5/ (gravadas pelos treinos no pod), do experiencias/e4_s5.yaml e da validação que o
`prepare.py` escreve:
- todos os ficheiros de sugestões (incluindo os 24 da curva) são da validação atual, um por
  contrato, com 5 códigos distintos de 8 dígitos;
- cada `treino.json` (e o `curva.json`) diz o que o YAML pede: N, `alpha` e épocas;
- o treino com o `alpha` em vigor e a regra por defeito repete o E4-S4 (o treino é repetível);
- a regra de escolha, aplicada como no notebook, dá `alpha` 1e-7 e 24 épocas, e esses são os
  valores por defeito em src/tfidf.py (`ALPHA`, `TAU`, `EPOCAS`);
- o top-5 do escolhido é o do doc.

Os testes marcados `data` saltam se data/processed/ não existir (por exemplo, no CI). Nunca leem o
teste. O último teste não precisa de dados: confirma que o notebook 06 não abre o teste.
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

import comparison as cmp
import evaluation as ev
import tfidf

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "scripts"))
import correr_experiencia as ce  # noqa: E402

DATA = RAIZ / "data" / "processed"
METRICS = RAIZ / "metrics"
NOTEBOOK = RAIZ / "notebooks" / "06_tfidf_epocas.ipynb"
EXP = ce.ler(RAIZ / "experiencias" / "e4_s5.yaml")
SAIDA = RAIZ / EXP["saida"]

ALPHAS = [EXP["alpha"]["em_vigor"], *EXP["alpha"]["candidatos"]]   # o 1.º é o em vigor
TAUS = [EXP["tau"]["em_vigor"], *EXP["tau"]["candidatos"]]
EPOCAS = EXP["epocas"]["candidatos"]                                # em vigor: regra por defeito

# Resultado do notebook 06, escrito à mão: (em vigor, candidato, quem fica), pela ordem da regra.
# None é a regra por defeito do scikit-learn.
ESCOLHA_ALPHA = [(1e-7, 1e-8, 1e-7)]
ESCOLHA_EPOCAS = [(None, 3, None), (None, 6, None), (None, 12, 12), (12, 18, 12), (12, 24, 24)]
# Top-5 do escolhido (alpha 1e-7, τ 0,25, 24 épocas), em %, arredondado a 1 casa como no doc
TOP5 = {("todos", "codigo"): 78.3, ("todos", "classe"): 84.1,
        ("texto novo", "codigo"): 75.1, ("texto novo", "classe"): 82.2}

CODIGO = re.compile(r"\d{8}")

dados = pytest.mark.skipif(
    not (DATA / "validacao.parquet").exists() or not SAIDA.exists(),
    reason="dados preparados ou sugestões do E4-S5 ausentes (correr make preparar)",
)


def nome_tau(tau):
    return "sugestoes_validacao.parquet" if tau == 0 else f"sugestoes_validacao_tau{tau:g}.parquet"


def exp_em(saida):
    """O YAML com outra pasta de saída (para correr a escolha sobre uma cópia dos resultados)."""
    return EXP | {"saida": str(saida)}


def ler_sugestoes(path, validacao):
    """Series de listas de 5 códigos alinhada com a validação; falha se o ficheiro não for dela."""
    sug = pd.read_parquet(path, columns=["idcontrato", "sugestoes"])
    sug["idcontrato"] = sug["idcontrato"].astype(str)
    ids = validacao["idcontrato"].astype(str)
    assert not sug["idcontrato"].duplicated().any(), f"{path}: contratos repetidos"
    assert set(sug["idcontrato"]) == set(ids) and len(sug) == len(ids), \
        f"{path}: os contratos não são os da validação atual"
    s = sug.set_index("idcontrato")["sugestoes"].reindex(ids).map(list)
    s.index = validacao.index
    ok = s.map(lambda l: len(l) == ev.TOP_K and len(set(l)) == ev.TOP_K and all(CODIGO.fullmatch(c) for c in l))
    assert ok.all(), f"{path}: há contratos sem {ev.TOP_K} códigos distintos de 8 dígitos"
    return s


def ficheiros_de_sugestoes(saida):
    exp = exp_em(saida)
    out = [ce.pasta_alpha(exp, a) / nome_tau(t) for a in ALPHAS for t in TAUS]
    out += [ce.pasta_epocas(exp, k) / nome_tau(t) for k in EPOCAS for t in (0.0, tfidf.TAU)]
    out += [ce.pasta_curva(exp) / f"sugestoes_epoca_{e:02d}.parquet" for e in range(1, EXP["curva_epocas"] + 1)]
    return out


def duelo(em_vigor, candidato, codigos):
    """A regra do E4-T1, como no notebook: 'b' se o candidato ganha, 'a' se fica a opção em vigor."""
    res = {}
    for medida in ("codigo", "classe"):
        a = cmp.weighted_mean(ev.hits(em_vigor, codigos, medida, ev.TOP_K))
        b = cmp.weighted_mean(ev.hits(candidato, codigos, medida, ev.TOP_K))
        res[medida] = cmp.paired_bootstrap(a, b, len(codigos))
    return cmp.choose(res["codigo"], res["classe"], default="a")


def escolher(saida, validacao):
    """Os três passos do notebook 06 sobre os resultados em `saida`.

    Devolve (alpha, τ, épocas, duelos), com épocas None para a regra por defeito e duelos a lista
    de (em vigor, candidato, quem fica) de cada passo.
    """
    exp = exp_em(saida)
    codigos = validacao["codigos"]
    duelos = {"alpha": [], "epocas": []}
    # Passo 1, com τ = 0
    alpha = ALPHAS[0]
    for cand in ALPHAS[1:]:
        antes = alpha
        if duelo(ler_sugestoes(ce.pasta_alpha(exp, alpha) / nome_tau(0.0), validacao),
                 ler_sugestoes(ce.pasta_alpha(exp, cand) / nome_tau(0.0), validacao), codigos) == "b":
            alpha = cand
        duelos["alpha"].append((antes, cand, alpha))
    # Passo 2: só se o alpha mudar (no E4-S5 não mudou; o τ fica o do E4-S4)
    if alpha != ALPHAS[0]:
        pytest.fail("o alpha mudou: o passo 2 (τ) não está neste teste")
    tau = tfidf.TAU
    # Passo 3, com o alpha e o τ em vigor
    opcoes = {None: ce.pasta_alpha(exp, alpha) / nome_tau(tau)} | \
             {k: ce.pasta_epocas(exp, k) / nome_tau(tau) for k in EPOCAS}
    epocas = None
    for k in EPOCAS:
        antes = epocas
        if duelo(ler_sugestoes(opcoes[epocas], validacao), ler_sugestoes(opcoes[k], validacao), codigos) == "b":
            epocas = k
        duelos["epocas"].append((antes, k, epocas))
    return alpha, tau, epocas, duelos


@pytest.fixture(scope="module")
def validacao():
    return pd.read_parquet(DATA / "validacao.parquet", columns=["idcontrato", "codigos", "visto_no_treino"])


@pytest.fixture(scope="module")
def escolha(validacao):
    return escolher(SAIDA, validacao)


@pytest.mark.data
@dados
def test_todos_os_ficheiros_de_sugestoes_sao_da_validacao_atual(validacao):
    ficheiros = ficheiros_de_sugestoes(SAIDA)
    assert len(ficheiros) == len(ALPHAS) * len(TAUS) + 2 * len(EPOCAS) + EXP["curva_epocas"]
    for path in ficheiros:
        ler_sugestoes(path, validacao)


@pytest.mark.data
@dados
def test_treinos_gravados_sao_os_que_o_yaml_pede(validacao):
    n = len(validacao)
    for a in ALPHAS:
        info = json.loads((RAIZ / ce.pasta_alpha(EXP, a) / "treino.json").read_text(encoding="utf-8"))
        assert info["n_min"] == EXP["n_min"] and info["alpha"] == a, a
        assert info["taus"] == TAUS, a
        # Regra por defeito do scikit-learn: sem épocas fixas, com paragem pela tolerância
        assert info["epocas"] is None and info["tol"] == 0.001, a
        assert info["validacao"]["contratos"] == n, a
    for k in EPOCAS:
        info = json.loads((RAIZ / ce.pasta_epocas(EXP, k) / "treino.json").read_text(encoding="utf-8"))
        assert info["n_min"] == EXP["n_min"] and info["alpha"] == tfidf.ALPHA, k
        assert tfidf.TAU in info["taus"], k
        # Épocas fixas: sem paragem antecipada, todas as etiquetas fazem exatamente k épocas
        assert info["epocas"] == k and info["tol"] is None and info["n_iter"][0] == k, k
        assert info["validacao"]["contratos"] == n, k
    curva = json.loads((RAIZ / ce.pasta_curva(EXP) / "curva.json").read_text(encoding="utf-8"))
    assert curva["n_min"] == EXP["n_min"]
    assert (curva["alpha"], curva["tau"]) == (tfidf.ALPHA, tfidf.TAU)
    assert curva["epocas"] == EXP["curva_epocas"] == 24
    assert curva["validacao"]["contratos"] == n


@pytest.mark.data
@dados
@pytest.mark.parametrize("tau", TAUS)
def test_treino_em_vigor_repete_o_e4_s4(validacao, tau):
    e4s5 = ler_sugestoes(RAIZ / ce.pasta_alpha(EXP, ALPHAS[0]) / nome_tau(tau), validacao)
    e4s4 = ler_sugestoes(METRICS / "e4_s4" / "alpha_1e-7" / nome_tau(tau), validacao)
    assert (e4s5 == e4s4).all(), f"τ {tau:g}: {(e4s5 == e4s4).mean():.2%} iguais"


@pytest.mark.data
@dados
def test_regra_de_escolha_da_os_vencedores_do_notebook_e_o_valor_por_defeito(escolha):
    alpha, tau, epocas, duelos = escolha
    assert duelos["alpha"] == ESCOLHA_ALPHA
    assert duelos["epocas"] == ESCOLHA_EPOCAS
    assert (alpha, tau, epocas) == (tfidf.ALPHA, tfidf.TAU, tfidf.EPOCAS)
    assert (tfidf.ALPHA, tfidf.TAU, tfidf.EPOCAS) == (1e-7, 0.25, 24)


@pytest.mark.data
@dados
def test_top5_do_escolhido_e_o_do_doc(validacao):
    assert tfidf.EPOCAS in EPOCAS, f"as EPOCAS por defeito ({tfidf.EPOCAS}) não são nenhuma das testadas"
    escolhido = ler_sugestoes(RAIZ / ce.pasta_epocas(EXP, tfidf.EPOCAS) / nome_tau(tfidf.TAU), validacao)
    out = ev.report(escolhido, validacao["codigos"], validacao["visto_no_treino"])
    for (grupo, medida), esperado in TOP5.items():
        assert round(out.loc[(grupo, medida), "top5"] * 100, 1) == esperado, (grupo, medida)


def test_notebook_das_epocas_nao_abre_o_teste():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    codigo = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    for proibido in ("teste", "TEST_START", "YEAR_END"):
        assert proibido not in codigo, proibido
    # Cada leitura de um ficheiro preparado é do treino ou da validação, escrita por extenso
    lidos = re.findall(r'DATA / "([^"]+)"', codigo)
    assert set(lidos) == {"treino.parquet", "validacao.parquet"}, lidos
