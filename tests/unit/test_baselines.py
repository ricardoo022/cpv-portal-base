"""Testes unitários a src/baselines.py (régua "códigos mais comuns")."""

import pandas as pd
import pytest

import baselines as bl
import evaluation as ev


def codes(spec):
    """{'90919300': 3, ...} -> Series com cada código repetido esse número de vezes."""
    return pd.Series([c for c, n in spec.items() for _ in range(n)])


SEIS = {"11111111": 6, "22222222": 5, "33333333": 4, "44444444": 3, "55555555": 2, "66666666": 1}


def test_as_5_mais_comuns_por_ordem():
    # Com N = 1 nenhum código sobe; o 66666666 (1 exemplo) é o 6.º e fica de fora
    assert bl.most_common_labels(codes(SEIS), n_min=1) == ["11111111", "22222222", "33333333", "44444444", "55555555"]


def test_conta_depois_da_regra_dos_codigos_raros():
    # Com N = 3: 90911100 (2) e 90911200 (2) são raros e sobem para a classe 90910000, que fica com 4
    # e passa à frente do 33333333 (3). Sem a regra, 90911100 e 90911200 entravam e 90910000 não.
    c = codes({"11111111": 9, "22222222": 8, "90911100": 2, "90911200": 2,
               "33333333": 3, "44444444": 3, "55555555": 3})
    assert bl.most_common_labels(c, n_min=3) == ["11111111", "22222222", "90910000", "33333333", "44444444"]


def test_exemplos_que_saem_nao_contam():
    # Com N = 3, 66666666, 77777777 e 88888888 (2 exemplos cada) são raros, e as suas classes e grupos
    # também: os 6 exemplos saem. Juntos seriam o "valor" mais comum (6) se a régua os contasse.
    c = codes({"11111111": 5, "22222222": 5, "33333333": 4, "44444444": 3, "55555555": 3,
               "66666666": 2, "77777777": 2, "88888888": 2})
    top = bl.most_common_labels(c, n_min=3)
    assert top == ["11111111", "22222222", "33333333", "44444444", "55555555"]


def test_empate_fica_o_menor_codigo_seja_qual_for_a_ordem():
    # 22222222 e 11111111 empatam com 4; 55555555, 33333333, 44444444 e 66666666 empatam com 2
    spec = {"22222222": 4, "11111111": 4, "55555555": 2, "33333333": 2, "66666666": 2, "44444444": 2}
    esperado = ["11111111", "22222222", "33333333", "44444444", "55555555"]
    c = codes(spec)
    assert bl.most_common_labels(c, n_min=1) == esperado
    assert bl.most_common_labels(c.iloc[::-1].reset_index(drop=True), n_min=1) == esperado


def test_conta_contratos_e_nao_exemplos():
    # Por exemplos: 11111111 (6) à frente de 22222222 (5). Mas cada exemplo de 22222222 tem 3 contratos
    # (15) e cada um de 11111111 só 1 (6): por contratos, 22222222 passa à frente.
    c = codes(SEIS)
    n = pd.Series([3 if x == "22222222" else 1 for x in c], index=c.index)
    assert bl.most_common_labels(c, n, n_min=1) == ["22222222", "11111111", "33333333", "44444444", "55555555"]
    assert bl.most_common_labels(c, n_min=1)[:2] == ["11111111", "22222222"]  # sem pesos, cada exemplo vale 1


def test_regra_dos_raros_conta_exemplos_nao_contratos():
    # N = 3: 66666666 tem 2 exemplos (raro, sai) mas 100 contratos. Se a regra contasse contratos,
    # ficava e era o 1.º; como conta exemplos (como o modelo), sai e não aparece.
    c = codes({"11111111": 5, "22222222": 5, "33333333": 4, "44444444": 3, "55555555": 3, "66666666": 2})
    n = pd.Series([50 if x == "66666666" else 1 for x in c], index=c.index)
    assert "66666666" not in bl.most_common_labels(c, n, n_min=3)


def test_exemplos_que_saem_nao_somam_contratos():
    # Os exemplos que saem (66666666, 77777777, 88888888) têm muitos contratos; não podem contar em nada
    c = codes({"11111111": 5, "22222222": 5, "33333333": 4, "44444444": 3, "55555555": 3,
               "66666666": 2, "77777777": 2, "88888888": 2})
    n = pd.Series([40 if x[0] in "678" else 1 for x in c], index=c.index)
    assert bl.most_common_labels(c, n, n_min=3) == ["11111111", "22222222", "33333333", "44444444", "55555555"]


def test_indices_diferentes_da_erro():
    c = codes(SEIS)
    with pytest.raises(ValueError, match="índices diferentes"):
        bl.most_common_labels(c, pd.Series(1, index=c.index + 100), n_min=1)


def test_k_escolhe_quantas_sugestoes():
    assert bl.most_common_labels(codes(SEIS), n_min=1, k=3) == ["11111111", "22222222", "33333333"]


def test_menos_de_5_etiquetas_da_erro():
    with pytest.raises(ValueError, match="só tem 4 etiquetas"):
        bl.most_common_labels(codes({"11111111": 3, "22222222": 3, "33333333": 3, "44444444": 3}), n_min=3)


def test_sugere_o_mesmo_a_todos_os_contratos():
    idx = pd.Index([10, 20, 30])
    sugg = bl.suggest_most_common(["1", "2", "3", "4", "5"], idx)
    assert sugg.index.equals(idx)
    assert sugg.tolist() == [["1", "2", "3", "4", "5"]] * 3
    sugg[10].append("x")  # cada contrato tem a sua lista: mudar uma não muda as outras
    assert sugg[20] == ["1", "2", "3", "4", "5"]


def test_regua_avaliada_com_a_avaliacao_comum():
    # Treino: 11111111 (6), 22222222 (5), 33333333 (4), 44444444 (3), 55555555 (2) -> régua com N = 1.
    # Validação, 3 contratos:
    #   a) 22222222            -> acerta no código no top-5 (2.ª sugestão), não no top-1
    #   b) 11119999            -> não acerta o código; acerta a classe 1111 no top-1
    #   c) 45000000            -> só divisão: não conta no código nem na classe
    top = bl.most_common_labels(codes(SEIS), n_min=1)
    codigos = pd.Series(["22222222", "11119999", "45000000"], index=["a", "b", "c"])
    out = ev.evaluate(bl.suggest_most_common(top, codigos.index), codigos)
    assert out.loc["codigo", "contratos"] == 2
    assert out.loc["codigo", "top1"] == 0.0
    assert out.loc["codigo", "top5"] == 0.5
    assert out.loc["classe", "top1"] == 0.5
    assert out.loc["classe", "top5"] == 1.0
    assert out.loc["so_divisao", "contratos"] == 1
