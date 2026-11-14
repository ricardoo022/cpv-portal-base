"""Testes unitários a src/grelha_bert.py (leitura dos treinos da grelha do BERT, spike E5-S1).

Tudo com pastas e ficheiros feitos à mão em `tmp_path`: não precisa de dados nem do torch.
"""

import json
import math

import pandas as pd
import pytest

import grelha_bert as gb

SHA_VAL = "a" * 64


def resumo(**muda):
    """Um `treino.json` mínimo de um treino da grelha (albertina, 3e-5, semente 0), como o do pod."""
    r = {"modelo": {"chave": "albertina"}, "taxa_aprendizagem": 3e-05, "semente": 0, "epocas": 3,
         "n_min": 10, "taus": [0.0, 0.25, 0.5], "ensaio": False,
         "treino": {"ficheiro": "treino.parquet", "datas": ["2022-01-01", "2025-09-30"]},
         "validacao": {"ficheiro": "validacao.parquet", "datas": ["2025-10-01", "2025-10-31"]},
         "sha256": {"treino": "b" * 64, "validacao": SHA_VAL},
         "tempo_treinar_s": 7200.0, "gpu": "NVIDIA GeForce RTX 4090"}
    r.update(muda)
    return r


def verificar(r, **kw):
    args = dict(modelo="albertina", taxa=3e-5, semente=0, epocas=3, n_min=10, taus=[0.0, 0.25, 0.5],
                validacao_sha256=SHA_VAL)
    args.update(kw)
    return gb.verificar_treino(r, "pasta", **args)


# --- nomes das pastas e dos ficheiros ---

@pytest.mark.parametrize("taxa, txt", [(2e-5, "2e-5"), (3.0e-5, "3e-5"), (0.00005, "5e-5"), (1e-4, "1e-4"), (2.5e-5, "2.5e-5")])
def test_taxa_txt_sem_zero_no_expoente(taxa, txt):
    assert gb.taxa_txt(taxa) == txt


def test_pasta_treino_tem_modelo_taxa_e_semente():
    assert gb.pasta_treino("metrics/e5_s1", "bertimbau", 2e-5, 1).as_posix() == \
        "metrics/e5_s1/bertimbau_taxa_2e-5_semente_1"


@pytest.mark.parametrize("tau, nome", [(0.0, "sugestoes_validacao.parquet"),
                                       (0.25, "sugestoes_validacao_tau0.25.parquet"),
                                       (0.5, "sugestoes_validacao_tau0.5.parquet")])
def test_ficheiro_sugestoes_por_epoca_e_tau(tmp_path, tau, nome):
    assert gb.ficheiro_sugestoes(tmp_path, 2, tau) == tmp_path / "epoca_2" / nome


# --- verificação do treino.json ---

def test_treino_certo_passa_sem_avisos():
    assert verificar(resumo()) == []


@pytest.mark.parametrize("muda, texto", [
    ({"modelo": {"chave": "bertimbau"}}, "modelo"),
    ({"taxa_aprendizagem": 2e-05}, "taxa"),
    ({"semente": 1}, "semente"),
    ({"epocas": 2}, "épocas"),
    ({"n_min": 5}, "N = 5"),
    ({"taus": [0.0, 0.25]}, "não gravou"),
    ({"ensaio": True}, "ensaio"),
    ({"treino": {"ficheiro": "teste.parquet", "datas": ["2022-01-01", "2025-09-30"]}}, "teste"),
    ({"validacao": {"ficheiro": "teste.parquet", "datas": ["2025-10-01", "2025-10-31"]}}, "teste"),
    ({"treino": {"ficheiro": "treino.parquet", "datas": ["2019-01-01", "2025-09-30"]}}, "janela"),
    ({"treino": {"ficheiro": "treino.parquet", "datas": ["2022-01-01", "2025-10-03"]}}, "antes da validação"),
    ({"validacao": {"ficheiro": "validacao.parquet", "datas": ["2025-10-01", "2025-11-02"]}}, "teste"),
])
def test_treino_errado_da_erro(muda, texto):
    with pytest.raises(ValueError, match=texto):
        verificar(resumo(**muda))


def test_validacao_com_outro_sha_da_erro():
    with pytest.raises(ValueError, match="SHA-256"):
        verificar(resumo(), validacao_sha256="c" * 64)


def test_sem_sha_da_validacao_e_so_aviso():
    avisos = verificar(resumo(sha256={}))
    assert len(avisos) == 1 and "SHA-256" in avisos[0]


# --- sugestões ---

VAL_IDS = pd.Series(["10", "20", "30"], index=[7, 8, 9], name="idcontrato")
CINCO = ["11111111", "22222222", "33333333", "44444444", "55555555"]


def grava(tmp_path, ids, sugestoes):
    path = tmp_path / "s.parquet"
    pd.DataFrame({"idcontrato": ids, "sugestoes": sugestoes}).to_parquet(path, index=False)
    return path


def test_sugestoes_alinhadas_pelo_idcontrato_e_nao_pela_posicao(tmp_path):
    outro = ["99999999"] + CINCO[1:]
    path = grava(tmp_path, ["30", "10", "20"], [outro, CINCO, CINCO])  # outra ordem
    s = gb.ler_sugestoes(path, VAL_IDS)
    assert list(s.index) == [7, 8, 9]
    assert s.loc[9] == outro and s.loc[7] == CINCO


@pytest.mark.parametrize("ids", [["10", "20"], ["10", "20", "30", "30"], ["10", "20", "40"]])
def test_sugestoes_de_outros_contratos_dao_erro(tmp_path, ids):
    path = grava(tmp_path, ids, [CINCO] * len(ids))
    with pytest.raises(ValueError, match="contratos"):
        gb.ler_sugestoes(path, VAL_IDS)


@pytest.mark.parametrize("mau", [CINCO[:4], CINCO[:4] + ["11111111"], CINCO[:4] + ["123"]])
def test_sugestoes_sem_cinco_codigos_distintos_dao_erro(tmp_path, mau):
    path = grava(tmp_path, ["10", "20", "30"], [CINCO, mau, CINCO])
    with pytest.raises(ValueError, match="5 códigos"):
        gb.ler_sugestoes(path, VAL_IDS)


# --- custo e perda ---

def test_custo_com_preco_do_pod_info():
    c = gb.custo(resumo(tempo_treinar_s=6458.35), {"pod": {"preco_por_hora_usd": 0.89}})
    assert c["horas_gpu"] == pytest.approx(1.794, abs=1e-3)
    assert c["dolares"] == pytest.approx(1.60, abs=0.01)


def test_custo_sem_pod_info_da_nan_nos_dolares():
    c = gb.custo(resumo(), None)
    assert c["horas_gpu"] == 2.0 and math.isnan(c["dolares"])


def test_pod_info_da_pasta_do_treino_antes_da_de_cima(tmp_path):
    treino = tmp_path / "albertina_taxa_3e-5_semente_0"
    treino.mkdir()
    assert gb.ler_pod_info(treino) is None
    (tmp_path / "pod_info.json").write_text(json.dumps({"pod": {"preco_por_hora_usd": 0.5}}))
    assert gb.ler_pod_info(treino)["pod"]["preco_por_hora_usd"] == 0.5
    (treino / "pod_info.json").write_text(json.dumps({"pod": {"preco_por_hora_usd": 0.9}}))
    assert gb.ler_pod_info(treino)["pod"]["preco_por_hora_usd"] == 0.9


def test_perda_validacao_da_epoca_e_nan_se_faltar(tmp_path):
    (tmp_path / "epoca_1").mkdir()
    (tmp_path / "epoca_1" / "epoca.json").write_text(json.dumps({"epoca": 1, "perda_validacao": 2.5}))
    (tmp_path / "epoca_2").mkdir()
    (tmp_path / "epoca_2" / "epoca.json").write_text(json.dumps({"epoca": 2}))  # antes do E5-T7
    assert gb.perda_validacao(tmp_path, 1) == 2.5
    assert math.isnan(gb.perda_validacao(tmp_path, 2))
    assert math.isnan(gb.perda_validacao(tmp_path, 3))


# --- controlo e extremos ---

@pytest.mark.parametrize("valor, passa", [(76.98, True), (77.48, True), (76.48, True),
                                          (77.49, False), (76.47, False)])
def test_controlo_dentro_da_tolerancia_para_os_dois_lados(valor, passa):
    ok, dif = gb.controlo(valor, 76.98, 0.5)
    assert ok is passa
    assert dif == pytest.approx(valor - 76.98)


TAXAS, EPOCAS, TAUS = [2e-5, 3e-5, 5e-5], [1, 2, 3], [0.0, 0.25, 0.5]


def test_opcao_do_meio_nao_esta_em_extremos():
    assert gb.nos_extremos((3e-5, 2, 0.25), TAXAS, EPOCAS, TAUS) == []


def test_primeira_epoca_nao_conta_como_extremo():
    assert gb.nos_extremos((3e-5, 1, 0.25), TAXAS, EPOCAS, TAUS) == []


@pytest.mark.parametrize("opcao, esperado", [
    ((2e-5, 2, 0.25), ["taxa 2e-5 é a mais baixa"]),
    ((5e-5, 2, 0.25), ["taxa 5e-5 é a mais alta"]),
    ((3e-5, 3, 0.25), ["época 3 é a última"]),
    ((3e-5, 2, 0.0), ["τ 0 é o mais baixo"]),
    ((3e-5, 2, 0.5), ["τ 0.5 é o mais alto"]),
    ((5e-5, 3, 0.5), ["taxa 5e-5 é a mais alta", "época 3 é a última", "τ 0.5 é o mais alto"]),
])
def test_extremos_de_cada_eixo(opcao, esperado):
    assert gb.nos_extremos(opcao, TAXAS, EPOCAS, TAUS) == esperado
