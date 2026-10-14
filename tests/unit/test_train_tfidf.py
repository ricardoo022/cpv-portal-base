"""Testes unitários a src/train_tfidf.py: nomes dos ficheiros de sugestões por τ (spike E4-S4),
épocas fixas (spike E4-S5) e a entidade compradora junta ao texto (spike E4-S3).

O notebook 05 e os docs leem estes nomes exatos, por isso o teste escreve-os à mão em vez de os
pedir à própria função.
"""

import sys

import pandas as pd
import pytest

import tfidf
import train_tfidf as tt


@pytest.mark.parametrize("tau, nome", [
    (0.0, "sugestoes_validacao.parquet"),          # sem correção: o nome do E4-H1
    (0, "sugestoes_validacao.parquet"),
    (0.25, "sugestoes_validacao_tau0.25.parquet"),
    (0.5, "sugestoes_validacao_tau0.5.parquet"),
    (1.0, "sugestoes_validacao_tau1.parquet"),     # sem ".0"
])
def test_nome_do_ficheiro_de_sugestoes_por_tau(tau, nome):
    assert tt.suggestions_file(tau) == nome


# --- Spike E4-S5: épocas fixas (`tfidf.sgd_epochs`, que o `run` usa) ---

def test_sem_epocas_fica_a_regra_por_defeito_do_sklearn():
    assert tfidf.sgd_epochs(None) == {}


@pytest.mark.parametrize("k", [1, 3, 24])
def test_epocas_fixas_desligam_a_paragem_automatica(k):
    assert tfidf.sgd_epochs(k) == {"max_iter": k, "tol": None}


@pytest.mark.parametrize("epocas", [0, -3, 2.5, True, "12"])
def test_epocas_invalidas_dao_erro_antes_de_ler_os_ficheiros(tmp_path, epocas):
    # Os ficheiros não existem: o erro tem de ser o das épocas, não o de ficheiro em falta
    with pytest.raises(ValueError, match="epocas"):
        tt.run(tmp_path / "treino.parquet", tmp_path / "validacao.parquet", tmp_path / "saida", epocas=epocas)
    assert not (tmp_path / "saida").exists()


@pytest.mark.parametrize("qual", ["treino", "validacao"])
def test_caminho_do_teste_e_recusado_tambem_com_epocas(tmp_path, qual):
    paths = {"treino": tmp_path / "treino.parquet", "validacao": tmp_path / "validacao.parquet"}
    paths[qual] = tmp_path / "teste.parquet"
    with pytest.raises(ValueError, match="teste"):
        tt.run(paths["treino"], paths["validacao"], tmp_path / "saida", epocas=12)
    assert not (tmp_path / "saida").exists()


# --- Spike E4-S3: a entidade compradora junta ao texto de entrada (`with_entity`) ---

def test_with_entity_junta_a_entidade_no_inicio_do_texto():
    texto = pd.Series(["Limpeza de escolas", "Papel A4"])
    ent = pd.Series(["Câmara Municipal de Lisboa", "Junta de Freguesia"])
    assert tt.with_entity(texto, ent).tolist() == [
        "Câmara Municipal de Lisboa | Limpeza de escolas",
        "Junta de Freguesia | Papel A4",
    ]


@pytest.mark.parametrize("ent", [None, "", "   "])
def test_with_entity_com_entidade_vazia_ou_em_falta_fica_so_o_texto(ent):
    texto = pd.Series(["Limpeza de escolas", "Papel A4"])
    assert tt.with_entity(texto, pd.Series([ent, "Junta de Freguesia"])).tolist() == [
        "Limpeza de escolas", "Junta de Freguesia | Papel A4",
    ]


def test_with_entity_limpa_os_espacos_da_entidade_category():
    # A adjudicante chega do Parquet como `category`; os espaços limpos-se como no build_text
    texto = pd.Series(["Limpeza"])
    ent = pd.Series(["Câmara   Municipal de Lisboa"], dtype="category")
    assert tt.with_entity(texto, ent).tolist() == ["Câmara Municipal de Lisboa | Limpeza"]


def test_with_entity_mantem_o_indice_para_o_tfidf_alinhar_com_o_codigo():
    texto = pd.Series(["Limpeza", "Papel"], index=[7, 3])
    ent = pd.Series(["Entidade A", "Entidade B"], index=[7, 3])
    r = tt.with_entity(texto, ent)
    assert r.index.tolist() == [7, 3]
    assert r.tolist() == ["Entidade A | Limpeza", "Entidade B | Papel"]


@pytest.mark.parametrize("args, esperado", [
    ([], False),                      # por defeito: o texto da decisão 5, sem a entidade
    (["--com-entidade"], True),
])
def test_comando_passa_a_entidade_ao_run(monkeypatch, tmp_path, args, esperado):
    recebido = {}
    monkeypatch.setattr(tt, "run", lambda *a, **kw: recebido.update(kw))
    monkeypatch.setattr(sys, "argv", ["train_tfidf.py", "--sem-modelo", "--saida", str(tmp_path), *args])
    tt.main()
    assert recebido["com_entidade"] is esperado


# --- Recusa do teste pelas datas (E4-T5) ---

def _datas(tmp_path, *datas):
    path = tmp_path / "conjunto.parquet"
    pd.DataFrame({"data": pd.to_datetime(list(datas))}).to_parquet(path, index=False)
    return path


@pytest.mark.parametrize("until, ultima", [(tt.VAL_START, "2025-10-01"), (tt.TEST_START, "2025-11-01")])
def test_contrato_no_proprio_dia_do_limite_da_erro(tmp_path, until, ultima):
    # A coluna `data` é um dia à meia-noite: um contrato de 1 de outubro (ou de novembro) é igual ao limite
    with pytest.raises(ValueError, match="teste"):
        tt.check_dates(_datas(tmp_path, "2024-01-01", ultima), until)


@pytest.mark.parametrize("until, ultima", [(tt.VAL_START, "2025-09-30"), (tt.TEST_START, "2025-10-31")])
def test_contratos_ate_a_vespera_do_limite_passam(tmp_path, until, ultima):
    tt.check_dates(_datas(tmp_path, "2024-01-01", ultima), until)
