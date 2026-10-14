"""Testes unitários às funções partilhadas de src/cpv_data.py."""

import pandas as pd
import pytest

import cpv_data as cd


def test_first_code_tira_o_codigo_da_descricao():
    cpv = pd.Series(["90611000-3 - Serviços de limpeza de ruas"])
    assert cd.first_code(cpv).tolist() == ["90611000"]


def test_first_code_com_varios_codigos_fica_o_primeiro():
    cpv = pd.Series(["90611000-3 - Limpeza de ruas\n77311000-3 - Manutenção de jardins"])
    assert cd.first_code(cpv).tolist() == ["90611000"]


def test_first_code_sem_codigo_da_vazio():
    cpv = pd.Series(["sem código", None])
    assert cd.first_code(cpv).isna().all()


def test_all_codes_devolve_todos_por_ordem():
    cpv = pd.Series(["90611000-3 - Limpeza\n77311000-3 - Jardins", "33600000-6 - Farmacêuticos"])
    assert cd.all_codes(cpv).tolist() == [["90611000", "77311000"], ["33600000"]]


def test_all_codes_vazio_ou_none_da_lista_vazia():
    assert cd.all_codes(pd.Series(["", None])).tolist() == [[], []]


@pytest.mark.parametrize("codigo, nivel", [
    ("33000000", 2),  # divisão
    ("03000000", 2),  # divisão que começa por 0
    ("33600000", 3),  # grupo
    ("45210000", 4),  # classe
    ("90911000", 5),  # categoria
    ("90911200", 6),
    ("33141110", 7),
    ("33141111", 8),
])
def test_code_level(codigo, nivel):
    assert cd.code_level(pd.Series([codigo])).iloc[0] == nivel


def test_level_name():
    nomes = cd.level_name(pd.Series([2, 3, 4, 5, 6, 8])).tolist()
    assert nomes == ["divisão", "grupo", "classe", "categoria", "mais detalhado (6-8)", "mais detalhado (6-8)"]


def test_normalize_text_ignora_maiusculas_e_espacos():
    s = pd.Series(["  Serviços   de LIMPEZA\n", "serviços de limpeza", None])
    assert cd.normalize_text(s).tolist() == ["serviços de limpeza", "serviços de limpeza", ""]


@pytest.mark.parametrize("data, periodo", [
    ("2021-12-31", None),
    ("2022-01-01", "treino"),
    ("2025-09-30", "treino"),
    ("2025-10-01", "validação"),
    ("2025-10-31", "validação"),
    ("2025-11-01", "teste"),
    ("2025-12-31", "teste"),
    ("2026-01-01", None),
])
def test_period_nas_fronteiras(data, periodo):
    resultado = cd.period(pd.Series(pd.to_datetime([data]))).iloc[0]
    if periodo is None:
        assert pd.isna(resultado)  # fora dos três períodos: valor em falta
    else:
        assert resultado == periodo


# --- Início do treino indicado (E4-T4) ---

@pytest.mark.parametrize("data, periodo", [
    ("2018-12-31", None),
    ("2019-01-01", "treino"),
    ("2021-12-31", "treino"),
    ("2025-09-30", "treino"),
    ("2025-10-01", "validação"),
    ("2025-11-01", "teste"),
    ("2026-01-01", None),
])
def test_period_com_outro_inicio_do_treino_so_muda_o_treino(data, periodo):
    resultado = cd.period(pd.Series(pd.to_datetime([data])), train_start=pd.Timestamp("2019-01-01")).iloc[0]
    if periodo is None:
        assert pd.isna(resultado)
    else:
        assert resultado == periodo


def test_period_sem_inicio_do_treino_usa_train_start(monkeypatch):
    # O valor por defeito é lido de cd.TRAIN_START quando se chama, não copiado para outro sítio:
    # mudar a constante (só aqui, no teste) muda o período por defeito.
    datas = pd.Series(pd.to_datetime(["2021-12-31", "2022-01-01", "2023-06-30"]))
    assert cd.period(datas, train_start=None).tolist()[1:] == ["treino", "treino"]
    assert pd.isna(cd.period(datas, train_start=None).iloc[0])
    monkeypatch.setattr(cd, "TRAIN_START", pd.Timestamp("2023-01-01"))
    resultado = cd.period(datas)
    assert resultado.isna().tolist() == [True, True, False]
