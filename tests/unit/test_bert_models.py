"""Testes unitários a src/bert_models.py (os modelos do E5 e o comprimento escolhido no E5-T1)."""

import re

import bert_models as bm


def test_os_dois_modelos_com_nome_e_revisao_fixa():
    # Os nomes escritos à mão: são os da secção 20 do qualidade_dados.md, que o treino vai usar.
    assert {m: v["nome"] for m, v in bm.MODELOS.items()} == {
        "bertimbau": "neuralmind/bert-base-portuguese-cased",
        "albertina": "PORTULAN/albertina-100m-portuguese-ptpt-encoder",
    }
    # Uma revisão é um commit (40 caracteres hexadecimais), não um ramo como "main", que pode mudar.
    for v in bm.MODELOS.values():
        assert re.fullmatch(r"[0-9a-f]{40}", v["revisao"])


def test_regra_do_comprimento_e_a_do_backlog():
    assert bm.CANDIDATOS == (128, 256, 512)
    assert bm.QUOTA == 0.99


def test_comprimento_escolhido_para_cada_modelo_e_um_candidato():
    assert set(bm.COMPRIMENTO_MAXIMO) == set(bm.MODELOS)
    assert all(n in bm.CANDIDATOS for n in bm.COMPRIMENTO_MAXIMO.values())
    # O valor do notebook 11 (secção 20 do qualidade_dados.md).
    assert bm.COMPRIMENTO_MAXIMO == {"bertimbau": 256, "albertina": 256}
