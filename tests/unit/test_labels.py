"""Testes unitários a src/labels.py (códigos raros sobem na árvore)."""

import pandas as pd
import pytest

import labels as lb


def codes(spec):
    """{'90919300': 3, ...} -> Series com cada código repetido esse número de vezes."""
    return pd.Series([c for c, n in spec.items() for _ in range(n)])


def label_of(codigo, label, original):
    return set(label[codigo == original])


def test_to_level():
    s = pd.Series(["90911200"])
    assert lb.to_level(s, 4).item() == "90910000"
    assert lb.to_level(s, 3).item() == "90900000"


def test_codigo_frequente_fica_igual():
    c = codes({"90919300": 3})
    assert lb.collapse_rare_codes(c, n_min=3).tolist() == ["90919300"] * 3


def test_codigo_raro_sobe_para_a_classe():
    # 90911200 tem 1 exemplo; a classe 9091 junta-o a 2 exemplos de 90911100 (também raro) -> 3
    c = codes({"90911200": 1, "90911100": 2})
    label = lb.collapse_rare_codes(c, n_min=3)
    assert label_of(c, label, "90911200") == {"90910000"}
    assert label_of(c, label, "90911100") == {"90910000"}


def test_codigo_raro_junta_se_a_classe_ja_existente():
    # Um contrato já escolheu a classe 90910000; os raros juntam-se a ela
    c = codes({"90910000": 2, "90911200": 1})
    label = lb.collapse_rare_codes(c, n_min=3)
    assert set(label) == {"90910000"}


def test_classe_rara_sobe_para_o_grupo():
    # Classes 9091 e 9092 raras sozinhas, mas o grupo 909 junta 3
    c = codes({"90911200": 1, "90921000": 2})
    label = lb.collapse_rare_codes(c, n_min=3)
    assert set(label) == {"90900000"}


def test_grupo_raro_sai():
    c = codes({"90919300": 3, "45233200": 1})
    label = lb.collapse_rare_codes(c, n_min=3)
    assert label_of(c, label, "90919300") == {"90919300"}
    assert label[c == "45233200"].isna().all()


def test_nunca_sobe_ate_a_divisao():
    # Grupos diferentes da mesma divisão, todos raros: saem em vez de virar 90000000
    c = codes({"90911200": 1, "90511000": 1, "90611000": 1})
    label = lb.collapse_rare_codes(c, n_min=3)
    assert label.isna().all()


def test_codigo_frequente_nao_e_afetado_pelos_raros_da_mesma_classe():
    c = codes({"90919300": 5, "90911200": 1})
    label = lb.collapse_rare_codes(c, n_min=3)
    assert label_of(c, label, "90919300") == {"90919300"}
    assert label[c == "90911200"].isna().all()  # 9091 só teria 1, 909 também -> sai


@pytest.mark.parametrize("n_min", [1, 2, 5, 10])
def test_todas_as_etiquetas_que_ficam_tem_pelo_menos_n_exemplos(n_min):
    c = codes({"90919300": 12, "90911200": 4, "90911100": 3, "90921000": 2, "45233200": 7, "45233100": 1})
    label = lb.collapse_rare_codes(c, n_min=n_min)
    assert (label.value_counts() >= n_min).all()


def test_n_por_defeito_e_10():
    # Sem n_min: 10 exemplos chegam para ficar igual, 9 não chegam e sobem para a classe
    c = codes({"90919300": 10, "90911200": 9, "90911100": 1})
    label = lb.collapse_rare_codes(c)
    assert label_of(c, label, "90919300") == {"90919300"}
    assert label_of(c, label, "90911200") == {"90910000"}


def test_n_igual_a_1_nao_muda_nada():
    c = codes({"90919300": 2, "90911200": 1})
    assert lb.collapse_rare_codes(c, n_min=1).tolist() == c.tolist()


def test_codigo_ao_nivel_da_divisao_no_treino_da_erro():
    # Não devia acontecer (o prepare.py tira-os); se acontecer, a função recusa
    with pytest.raises(RuntimeError, match="divisão"):
        lb.collapse_rare_codes(codes({"90000000": 3}), n_min=3)


def test_summary_conta_iguais_subidas_e_saidas():
    c = codes({"90919300": 3, "90911200": 1, "90911100": 2, "45233200": 1})
    s = lb.summary(c, lb.collapse_rare_codes(c, n_min=3))
    assert s.to_dict() == {"exemplos": 7, "iguais": 3, "subiram": 3, "saíram": 1, "etiquetas distintas": 2}
