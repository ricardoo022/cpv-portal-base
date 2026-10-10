"""Testes unitários a src/evaluation.py (top-k e F1 macro nas três medidas).

Os contratos são inventados; o resultado esperado está calculado à mão no comentário de cada teste.
"""

import numpy as np
import pandas as pd
import pytest

import evaluation as ev

OUTRAS = ["11111111", "22222222", "44444444", "55555555"]  # sugestões que não acertam em nada


def sugg(*rows):
    """Cada linha: as primeiras sugestões, completadas com OUTRAS até 5."""
    return pd.Series([list(r) + OUTRAS[: 5 - len(r)] for r in rows])


def cods(*rows):
    return pd.Series(list(rows))


def h(suggestions, codigos, measure, k=1):
    return ev.hits(suggestions, codigos, measure, k).tolist()


# --- acertos por medida -------------------------------------------------------------------------

def test_etiqueta_de_grupo_acerta_com_qualquer_sugestao_do_grupo():
    # 33600000 é nível 3: 33651100 começa por 336 -> acerta no código; 3365 != 3360 mas a classe
    # não conta (nível 3 < 4) -> NaN; divisão 33 -> acerta. 33100000 não começa por 336 -> falha.
    c = cods("33600000", "33600000")
    s = sugg(["33651100"], ["33100000"])
    assert h(s, c, "codigo") == [1.0, 0.0]
    assert np.isnan(h(s, c, "classe")).all()
    assert h(s, c, "divisao") == [1.0, 1.0]


def test_etiqueta_de_categoria_5_digitos():
    # 90911000 é nível 5 (9091 1): 90911200 começa por 90911 -> acerta; 90912000 não -> falha
    # no código, mas tem a mesma classe 9091 -> acerta na classe.
    c = cods("90911000", "90911000")
    s = sugg(["90911200"], ["90912000"])
    assert h(s, c, "codigo") == [1.0, 0.0]
    assert h(s, c, "classe") == [1.0, 1.0]
    assert h(s, c, "divisao") == [1.0, 1.0]


def test_etiqueta_de_6_ou_mais_digitos_acerta_o_codigo_e_os_filhos():
    # 90911200 é nível 6: compara os 6 dígitos 909112. Acertam 90911200 e um código mais fundo
    # (90911210, mesma regra do grupo); falham o irmão 90911300 e o pai 90911000.
    c = cods("90911200", "90911200", "90911200", "90911200")
    s = sugg(["90911200"], ["90911210"], ["90911300"], ["90911000"])
    assert h(s, c, "codigo") == [1.0, 1.0, 0.0, 0.0]
    assert h(s, c, "classe") == [1.0, 1.0, 1.0, 1.0]


def test_codigo_de_8_digitos_exige_igualdade():
    c = cods("45233221")
    assert h(sugg(["45233221"]), c, "codigo") == [1.0]
    assert h(sugg(["45233222"]), c, "codigo") == [0.0]


def test_divisao_e_classe_no_mesmo_contrato():
    # "33000000 33690000": conta na classe por causa do 2.º código.
    # Sugestão 33100000: classe 3310 != 3369 -> falha na classe; 33000000 aceita qualquer 33 ->
    # acerta no código e na divisão.
    # Sugestão 33691000: começa por 3369 -> acerta na classe (e nas outras duas).
    c = cods("33000000 33690000", "33000000 33690000")
    s = sugg(["33100000"], ["33691000"])
    assert h(s, c, "classe") == [0.0, 1.0]
    assert h(s, c, "codigo") == [1.0, 1.0]
    assert h(s, c, "divisao") == [1.0, 1.0]


def test_acerta_em_qualquer_codigo_do_contrato_nao_so_na_etiqueta():
    # Etiqueta 90911200, 2.º código 77311000. Sugestão 77311000 acerta pelo 2.º código em todas.
    c = cods("90911200 77311000")
    s = sugg(["77311000"])
    assert h(s, c, "codigo") == [1.0]
    assert h(s, c, "classe") == [1.0]
    assert h(s, c, "divisao") == [1.0]


def test_contrato_so_com_divisao_nao_conta_na_classe():
    # 45000000 (nível 2): NaN na classe; conta no código e na divisão (qualquer 45 acerta).
    c = cods("45000000", "45000000 33000000")
    s = sugg(["45233221"], ["33600000"])
    assert np.isnan(h(s, c, "classe")).all()
    assert h(s, c, "codigo") == [1.0, 1.0]
    assert h(s, c, "divisao") == [1.0, 1.0]


def test_classe_usa_4_digitos_e_nao_3():
    # Etiqueta 90911200 (classe 9091). Sugestão 90921000 tem o mesmo grupo 909 mas classe 9092 -> falha.
    c = cods("90911200")
    assert h(sugg(["90921000"]), c, "classe") == [0.0]


def test_top1_contra_top5():
    # Acerto na 3.ª sugestão: top-1 falha, top-3 e top-5 acertam. Acerto na 6.ª não conta.
    c = cods("90911200", "90911200")
    s = pd.Series([
        ["11111111", "22222222", "90911200", "44444444", "55555555"],
        ["11111111", "22222222", "44444444", "55555555", "66666666", "90911200"],
    ])
    assert h(s, c, "codigo", k=1) == [0.0, 0.0]
    assert h(s, c, "codigo", k=3) == [1.0, 0.0]
    assert h(s, c, "codigo", k=5) == [1.0, 0.0]


def test_hits_mantem_o_indice():
    c = pd.Series(["90911200", "45000000"], index=[10, 7])
    s = pd.Series([["90911200"] + OUTRAS, ["45233221"] + OUTRAS], index=[10, 7])
    out = ev.hits(s, c, "classe", 5)
    assert out.index.tolist() == [10, 7]
    assert out[10] == 1.0 and np.isnan(out[7])


# --- validação das entradas ---------------------------------------------------------------------

def test_indices_diferentes_da_erro():
    c = pd.Series(["90911200"], index=[0])
    s = pd.Series([["90911200"] + OUTRAS], index=[1])
    with pytest.raises(ValueError, match="índices"):
        ev.hits(s, c, "codigo", 1)


def test_sugestao_que_nao_e_codigo_da_erro():
    with pytest.raises(ValueError, match="8 dígitos"):
        ev.hits(sugg(["9091120"]), cods("90911200"), "codigo", 5)


def test_menos_de_k_sugestoes_da_erro():
    s = pd.Series([["90911200", "11111111"]])
    with pytest.raises(ValueError, match="menos de 5"):
        ev.hits(s, cods("90911200"), "codigo", 5)


def test_codigos_mal_formados_dao_erro():
    with pytest.raises(ValueError, match="8 dígitos"):
        ev.hits(sugg(["90911200"]), cods("9091-1200"), "codigo", 1)


def test_medida_desconhecida_da_erro():
    with pytest.raises(ValueError, match="medida"):
        ev.hits(sugg(["90911200"]), cods("90911200"), "grupo", 1)


# --- pares para o F1 ----------------------------------------------------------------------------

def test_f1_pairs_codigo_corta_a_sugestao_ao_nivel_da_etiqueta():
    # Etiqueta 33600000 (nível 3), sugestão 33651100 -> 336 + zeros = 33600000.
    # Etiqueta 90911200 (nível 6), sugestão 90911210 -> 909112 + zeros = 90911200.
    # Só conta a 1.ª sugestão e só o 1.º código: o 2.º código 77311000 não entra.
    c = cods("33600000", "90911200 77311000")
    s = sugg(["33651100"], ["90911210"])
    p = ev.f1_pairs(s, c, "codigo")
    assert p["verdadeiro"].tolist() == ["33600000", "90911200"]
    assert p["previsto"].tolist() == ["33600000", "90911200"]


def test_f1_pairs_classe_so_com_etiqueta_de_nivel_4_ou_mais():
    # 1.º contrato: etiqueta 33000000 (divisão) -> NaN, mesmo com o 2.º código de classe.
    # 2.º: etiqueta 90911200 -> ("9091", "4523").
    c = cods("33000000 33690000", "90911200")
    s = sugg(["33691000"], ["45233221"])
    p = ev.f1_pairs(s, c, "classe")
    assert p.iloc[0].isna().all()
    assert p.iloc[1].tolist() == ["9091", "4523"]


def test_f1_pairs_divisao():
    p = ev.f1_pairs(sugg(["45233221"]), cods("90911200"), "divisao")
    assert p.iloc[0].tolist() == ["90", "45"]


# --- F1 macro -----------------------------------------------------------------------------------

def test_f1_macro_a_mao():
    # verdadeiro: A A A B B ; previsto: A A B B C
    # A: TP=2, FP=0, FN=1 -> 4/5 = 0.8
    # B: TP=1, FP=1 (3.ª linha), FN=1 (última) -> 2/4 = 0.5
    # C não é verdadeiro de ninguém: não entra na média. Média = (0.8 + 0.5)/2 = 0.65
    v = ["A", "A", "A", "B", "B"]
    p = ["A", "A", "B", "B", "C"]
    assert ev.f1_macro(v, p) == pytest.approx(0.65)


def test_f1_macro_ignora_linhas_com_nan():
    # A linha com NaN não conta: fica A->A (F1 de A = 1).
    assert ev.f1_macro(["A", np.nan], ["A", "B"]) == pytest.approx(1.0)


def test_f1_macro_peso_zero_tira_a_etiqueta_da_media():
    # Com peso 0 nas linhas de B, só A fica na média: A TP=2, FN=1 -> 0.8.
    # A 3.ª linha (A->B) tem peso 1 e continua a ser FN de A.
    v = ["A", "A", "A", "B", "B"]
    p = ["A", "A", "B", "B", "C"]
    assert ev.f1_macro(v, p, weights=[1, 1, 1, 0, 0]) == pytest.approx(0.8)


def test_f1_macro_peso_2_igual_a_duplicar_a_linha():
    # Peso 2 na última linha (B->C) = escrever essa linha duas vezes.
    # A: TP=2, FN=1 -> 0.8; B: TP=1, FP=1, FN=2 -> 2/5 = 0.4; média 0.6
    v = ["A", "A", "A", "B", "B"]
    p = ["A", "A", "B", "B", "C"]
    com_peso = ev.f1_macro(v, p, weights=[1, 1, 1, 1, 2])
    duplicado = ev.f1_macro(v + ["B"], p + ["C"])
    assert com_peso == pytest.approx(0.6)
    assert com_peso == pytest.approx(duplicado)


def test_f1_macro_peso_zero_numa_linha_de_falso_positivo():
    # Peso 0 na 3.ª linha (A->B): A fica TP=2 FN=0 -> 1; B TP=1, FP=0, FN=1 -> 2/3.
    v = ["A", "A", "A", "B", "B"]
    p = ["A", "A", "B", "B", "C"]
    assert ev.f1_macro(v, p, weights=[1, 1, 0, 1, 1]) == pytest.approx((1 + 2 / 3) / 2)


def test_f1_macro_pesos_negativos_ou_tamanho_errado_dao_erro():
    with pytest.raises(ValueError):
        ev.f1_macro(["A"], ["A"], weights=[-1])
    with pytest.raises(ValueError):
        ev.f1_macro(["A", "B"], ["A", "B"], weights=[1])


# --- evaluate e report --------------------------------------------------------------------------

def contratos_exemplo():
    # 0: 33600000           sugestões 33651100, ...            (grupo)
    # 1: 90911200           sugestões 90921000, 90911200, ...  (acerta só no top-5 no código)
    # 2: 45000000           sugestões 45233221, ...            (só divisão: não conta na classe)
    # 3: 33000000 33690000  sugestões 33100000, ...
    c = cods("33600000", "90911200", "45000000", "33000000 33690000")
    s = sugg(["33651100"], ["90921000", "90911200"], ["45233221"], ["33100000"])
    return s, c


def test_evaluate_contagens_e_taxas():
    # codigo: contam 4. top-1: 0 acerta (336), 1 falha, 2 acerta (45), 3 acerta (33) -> 3/4.
    #         top-5: o 1 acerta na 2.ª sugestão -> 4/4.
    #         F1 (verdadeiro -> previsto): 33600000->33600000, 90911200->90921000,
    #         45000000->45000000, 33000000->33000000 -> três etiquetas com F1 1, 90911200 com 0 -> 0.75
    # classe: contam 2 (o 1 e o 3; o 0 é nível 3, o 2 é divisão).
    #         top-1: 1 -> 9092 != 9091 falha; 3 -> 3310 != 3369 falha -> 0. top-5: 1 acerta -> 1/2.
    #         F1: só o 1 tem etiqueta de nível >= 4: 9091 -> 9092 -> 0.
    # divisao: contam 4; top-1 e top-5 = 1; F1: 33->33, 90->90, 45->45, 33->33 -> 1.
    s, c = contratos_exemplo()
    out = ev.evaluate(s, c)
    assert out.index.tolist() == list(ev.MEASURES)
    assert out.columns.tolist() == ["contratos", "top1", "top5", "f1_macro"]
    assert out["contratos"].tolist() == [4, 2, 4]
    assert out.loc["codigo", ["top1", "top5", "f1_macro"]].tolist() == pytest.approx([0.75, 1.0, 0.75])
    assert out.loc["classe", ["top1", "top5", "f1_macro"]].tolist() == pytest.approx([0.0, 0.5, 0.0])
    assert out.loc["divisao", ["top1", "top5", "f1_macro"]].tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_report_separa_texto_novo():
    # Texto novo = contratos 1 e 2.
    # codigo: contam 2; top-1: 1 falha, 2 acerta -> 0.5; top-5: 1.0.
    # classe: conta só o 1 -> top-1 0, top-5 1.
    s, c = contratos_exemplo()
    visto = pd.Series([True, False, False, True])
    out = ev.report(s, c, visto)
    assert out.index.names == ["grupo", "medida"]
    assert out.index.get_level_values("grupo").unique().tolist() == ["todos", "texto novo"]
    assert out.loc[("todos", "codigo"), "contratos"] == 4
    novo = out.loc["texto novo"]
    assert novo["contratos"].tolist() == [2, 1, 2]
    assert novo.loc["codigo", ["top1", "top5"]].tolist() == pytest.approx([0.5, 1.0])
    assert novo.loc["classe", ["top1", "top5"]].tolist() == pytest.approx([0.0, 1.0])


def test_report_indice_diferente_da_erro():
    s, c = contratos_exemplo()
    with pytest.raises(ValueError, match="índices"):
        ev.report(s, c, pd.Series([True, False, False, True], index=[9, 8, 7, 6]))


def test_report_visto_no_treino_em_falta_da_erro():
    s, c = contratos_exemplo()
    with pytest.raises(ValueError, match="em falta"):
        ev.report(s, c, pd.Series([True, None, False, True]))
