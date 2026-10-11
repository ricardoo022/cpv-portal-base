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

def test_etiqueta_de_grupo_so_acerta_o_codigo_igual():
    # 33600000 é nível 3. 33600000 -> acerta. 33651100 é mais específica (do mesmo grupo) -> erro
    # no código (decisão 8); 33100000 -> erro. A classe não conta (nível 3 < 4) -> NaN.
    # Divisão 33 -> acertam as três.
    c = cods("33600000", "33600000", "33600000")
    s = sugg(["33600000"], ["33651100"], ["33100000"])
    assert h(s, c, "codigo") == [1.0, 0.0, 0.0]
    assert np.isnan(h(s, c, "classe")).all()
    assert h(s, c, "divisao") == [1.0, 1.0, 1.0]


def test_etiqueta_de_categoria_5_digitos():
    # 90911000 é nível 5. 90911000 -> acerta. 90911200 (filho) e 90912000 (outra categoria) ->
    # erro no código, mas as três têm a classe 9091 -> acertam na classe.
    c = cods("90911000", "90911000", "90911000")
    s = sugg(["90911000"], ["90911200"], ["90912000"])
    assert h(s, c, "codigo") == [1.0, 0.0, 0.0]
    assert h(s, c, "classe") == [1.0, 1.0, 1.0]
    assert h(s, c, "divisao") == [1.0, 1.0, 1.0]


def test_etiqueta_de_6_ou_mais_digitos_filho_irmao_e_pai_sao_erro():
    # 90911200 é nível 6. Só 90911200 acerta; o filho 90911210 (sobre-especialização), o irmão
    # 90911300 e o pai 90911000 são erro no código, mas todos acertam na classe 9091.
    c = cods("90911200", "90911200", "90911200", "90911200")
    s = sugg(["90911200"], ["90911210"], ["90911300"], ["90911000"])
    assert h(s, c, "codigo") == [1.0, 0.0, 0.0, 0.0]
    assert h(s, c, "classe") == [1.0, 1.0, 1.0, 1.0]


def test_codigo_de_8_digitos_exige_igualdade():
    c = cods("45233221")
    assert h(sugg(["45233221"]), c, "codigo") == [1.0]
    assert h(sugg(["45233222"]), c, "codigo") == [0.0]


def test_divisao_e_classe_no_mesmo_contrato():
    # "33000000 33690000": conta na classe e no código completo por causa do 2.º código.
    # 33100000: não é 33690000 -> erro no código; classe 3310 != 3369 -> erro; divisão 33 -> acerta.
    # 33691000: classe 3369 -> acerta na classe e na divisão; não é 33690000 -> erro no código.
    # 33690000: igual ao 2.º código -> acerta nas três.
    # 33000000: igual ao 1.º código, mas é uma divisão, que não conta no código completo -> erro;
    #           classe 3300 != 3369 -> erro; divisão -> acerta.
    c = cods(*["33000000 33690000"] * 4)
    s = sugg(["33100000"], ["33691000"], ["33690000"], ["33000000"])
    assert h(s, c, "classe") == [0.0, 1.0, 1.0, 0.0]
    assert h(s, c, "codigo") == [0.0, 0.0, 1.0, 0.0]
    assert h(s, c, "divisao") == [1.0, 1.0, 1.0, 1.0]


def test_acerta_em_qualquer_codigo_do_contrato_nao_so_na_etiqueta():
    # Etiqueta 90911200, 2.º código 77311000. Sugestão 77311000 acerta pelo 2.º código em todas.
    c = cods("90911200 77311000")
    s = sugg(["77311000"])
    assert h(s, c, "codigo") == [1.0]
    assert h(s, c, "classe") == [1.0]
    assert h(s, c, "divisao") == [1.0]


def test_contrato_so_com_divisao_nao_conta_na_classe():
    # Só códigos de divisão: NaN na classe e no código completo, mesmo com a sugestão 45000000;
    # contam na divisão (qualquer 45, ou 33 no 2.º contrato).
    c = cods("45000000", "45000000 33000000", "45000000")
    s = sugg(["45233221"], ["33600000"], ["45000000"])
    assert np.isnan(h(s, c, "classe")).all()
    assert np.isnan(h(s, c, "codigo")).all()
    assert h(s, c, "divisao") == [1.0, 1.0, 1.0]


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

def test_f1_pairs_codigo_usa_os_codigos_inteiros():
    # A sugestão não é cortada ao nível da etiqueta: 33651100 fica 33651100 (erro contra 33600000).
    # Só conta a 1.ª sugestão e só o 1.º código: o 2.º código 77311000 não entra.
    c = cods("33600000", "90911200 77311000")
    s = sugg(["33651100"], ["90911200"])
    p = ev.f1_pairs(s, c, "codigo")
    assert p["verdadeiro"].tolist() == ["33600000", "90911200"]
    assert p["previsto"].tolist() == ["33651100", "90911200"]


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
    # codigo: contam 3 (o 2 só tem a divisão 45000000; o 3 conta pelo 33690000).
    #         top-1: nenhuma 1.ª sugestão é igual a um código -> 0/3.
    #         top-5: o 1 tem 90911200 na 2.ª sugestão -> 1/3.
    #         F1: só etiquetas abaixo da divisão: 33600000->33651100, 90911200->90921000 -> 0
    # classe: contam 2 (o 1 e o 3; o 0 é nível 3, o 2 é divisão).
    #         top-1: 1 -> 9092 != 9091 falha; 3 -> 3310 != 3369 falha -> 0. top-5: 1 acerta -> 1/2.
    #         F1: só o 1 tem etiqueta de nível >= 4: 9091 -> 9092 -> 0.
    # divisao: contam 4; top-1 e top-5 = 1; F1: 33->33, 90->90, 45->45, 33->33 -> 1.
    # so_divisao: só o 2 (45000000); 45233221 acerta a divisão -> top-1, top-5 e F1 = 1.
    s, c = contratos_exemplo()
    out = ev.evaluate(s, c)
    assert out.index.tolist() == [*ev.MEASURES, "so_divisao"]
    assert out.columns.tolist() == ["contratos", "top1", "top5", "f1_macro"]
    assert out["contratos"].tolist() == [3, 2, 4, 1]
    assert out.loc["codigo", ["top1", "top5", "f1_macro"]].tolist() == pytest.approx([0.0, 1 / 3, 0.0])
    assert out.loc["classe", ["top1", "top5", "f1_macro"]].tolist() == pytest.approx([0.0, 0.5, 0.0])
    assert out.loc["divisao", ["top1", "top5", "f1_macro"]].tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert out.loc["so_divisao", ["top1", "top5", "f1_macro"]].tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_report_separa_texto_novo():
    # Texto novo = contratos 1 e 2.
    # codigo: conta só o 1 (o 2 é só divisão); top-1 falha -> 0; top-5 acerta na 2.ª sugestão -> 1.
    # so_divisao: o 2.
    # classe: conta só o 1 -> top-1 0, top-5 1.
    s, c = contratos_exemplo()
    visto = pd.Series([True, False, False, True])
    out = ev.report(s, c, visto)
    assert out.index.names == ["grupo", "medida"]
    assert out.index.get_level_values("grupo").unique().tolist() == ["todos", "texto novo"]
    assert out.loc[("todos", "codigo"), "contratos"] == 3
    novo = out.loc["texto novo"]
    assert novo["contratos"].tolist() == [1, 1, 2, 1]
    assert novo.loc["codigo", ["top1", "top5"]].tolist() == pytest.approx([0.0, 1.0])
    assert novo.loc["classe", ["top1", "top5"]].tolist() == pytest.approx([0.0, 1.0])


def test_report_indice_diferente_da_erro():
    s, c = contratos_exemplo()
    with pytest.raises(ValueError, match="índices"):
        ev.report(s, c, pd.Series([True, False, False, True], index=[9, 8, 7, 6]))


def test_report_visto_no_treino_em_falta_da_erro():
    s, c = contratos_exemplo()
    with pytest.raises(ValueError, match="em falta"):
        ev.report(s, c, pd.Series([True, None, False, True]))


# --- medidas hierárquicas (Kiritchenko et al., 2005; artigo CPV de 2024) ------------------------

def test_hierarchical_scores_conta_os_nos_em_comum():
    # Etiqueta 45233200 (nível 6): nós 45, 452, 4523, 45233, 452332 -> 5.
    # 45233223 (nível 8, 7 nós, sobre-especialização): partilha os 5.
    # 45200000 (o avô, nível 3, 2 nós): partilha 45 e 452 -> 2.
    # 45233200 (igual): 5. 33600000 (outra divisão, 2 nós): 0.
    c = cods("45233200", "45233200", "45233200", "45233200")
    s = sugg(["45233223"], ["45200000"], ["45233200"], ["33600000"])
    sc = ev.hierarchical_scores(s, c)
    assert sc["comuns"].tolist() == [5, 2, 5, 0]
    assert sc["nos_sugestao"].tolist() == [7, 2, 5, 2]
    assert sc["nos_etiqueta"].tolist() == [5, 5, 5, 5]


def test_hierarchical_scores_codigos_iguais_genericos():
    # 45200000 contra 45200000: os dígitos são todos iguais, mas o nível é 3 -> 2 nós em comum, não 7.
    # Etiqueta de divisão 33000000 (1 nó) contra 33600000: partilham 33 -> 1.
    sc = ev.hierarchical_scores(sugg(["45200000"], ["33600000"]), cods("45200000", "33000000"))
    assert sc["comuns"].tolist() == [2, 1]
    assert sc["nos_etiqueta"].tolist() == [2, 1]


def test_hierarchical_scores_so_1a_sugestao_e_1o_codigo():
    # A 2.ª sugestão (igual à etiqueta) e o 2.º código (igual à 1.ª sugestão) não entram.
    sc = ev.hierarchical_scores(sugg(["77311000", "45233200"]), cods("45233200 77311000"))
    assert sc["comuns"].tolist() == [0]


def test_hierarchical_sobre_especializacao_e_sugestao_vaga():
    # Filho: precisão 5/7, recall 5/5. Avô: precisão 2/2, recall 2/5.
    # Micro: comuns 5+2 = 7; nós da sugestão 7+2 = 9; nós da etiqueta 5+5 = 10 -> P 7/9, R 7/10.
    out = ev.hierarchical(sugg(["45233223"], ["45200000"]), cods("45233200", "45233200"))
    p, r = 7 / 9, 7 / 10
    assert out["precisao_micro"] == pytest.approx(p)
    assert out["recall_micro"] == pytest.approx(r)
    assert out["f1_micro"] == pytest.approx(2 * p * r / (p + r))


def test_hierarchical_macro_faz_media_por_etiqueta():
    # Etiqueta 33600000: 3 contratos certos (2 nós cada) -> P 1, R 1.
    # Etiqueta 45233200: 1 contrato com o avô 45200000 -> P 2/2 = 1, R 2/5 = 0.4.
    # Macro: P = (1 + 1) / 2 = 1; R = (1 + 0.4) / 2 = 0.7.
    # Micro: comuns 2+2+2+2 = 8; nós sugestão 2·3 + 2 = 8; nós etiqueta 2·3 + 5 = 11 -> R 8/11.
    c = cods("33600000", "33600000", "33600000", "45233200")
    s = sugg(["33600000"], ["33600000"], ["33600000"], ["45200000"])
    out = ev.hierarchical(s, c)
    assert out["precisao_macro"] == pytest.approx(1.0)
    assert out["recall_macro"] == pytest.approx(0.7)
    assert out["f1_macro"] == pytest.approx(2 * 0.7 / 1.7)
    assert out["recall_micro"] == pytest.approx(8 / 11)


def test_hierarchical_tudo_errado_da_zero():
    out = ev.hierarchical(sugg(["33600000"]), cods("45233200"))
    assert out.tolist() == [0.0] * 6


def test_hierarchical_report_separa_texto_novo():
    # Texto novo = só o 2.º contrato (o avô): P 1, R 2/5.
    c = cods("45233200", "45233200")
    s = sugg(["45233200"], ["45200000"])
    out = ev.hierarchical_report(s, c, pd.Series([True, False]))
    assert out.index.tolist() == ["todos", "texto novo"]
    assert out.loc["texto novo", ["precisao_micro", "recall_micro"]].tolist() == pytest.approx([1.0, 0.4])
    assert out.loc["todos", "recall_micro"] == pytest.approx(7 / 10)


def test_contrato_so_com_divisao_nao_conta_no_codigo_completo():
    # O modelo nunca sugere divisões (decisão 8), por isso um contrato só com 45000000 não se
    # pode acertar no código completo: conta só na divisão, como já acontecia na classe.
    # A 90911200 -> sugestão 90911200: acerta no código, na classe e na divisão.
    # B 33690000 -> 33600000: falha no código e na classe (3360 != 3369); acerta na divisão.
    # C 45000000 -> 45233200 e D 45000000 -> 45210000: NaN no código e na classe; acertam na divisão.
    c = cods("90911200", "33690000", "45000000", "45000000")
    s = sugg(["90911200"], ["33600000"], ["45233200"], ["45210000"])
    assert h(s, c, "codigo")[:2] == [1.0, 0.0]
    assert np.isnan(h(s, c, "codigo")[2:]).all()
    assert h(s, c, "divisao") == [1.0, 1.0, 1.0, 1.0]
    out = ev.evaluate(s, c)
    assert out["contratos"].tolist() == [2, 2, 4]  # código completo: 1 em 2 = 50%, não 1 em 4
    assert out.loc["codigo", "top1"] == pytest.approx(0.5)
    # F1 do código: só as etiquetas abaixo da divisão (A e B); C e D ficam de fora
    p = ev.f1_pairs(s, c, "codigo")
    assert p.iloc[2:].isna().all().all()


def test_contrato_so_com_divisao_nao_conta_no_codigo_completo():
    # O modelo nunca sugere divisões (decisão 8), por isso um contrato só com 45000000 não se
    # pode acertar no código completo: conta só na divisão, como já acontecia na classe.
    # A 90911200 -> 90911200: acerta no código, na classe e na divisão.
    # B 33690000 -> 33600000: falha no código e na classe (3360 != 3369); acerta na divisão.
    # C 45000000 -> 45233200 e D 45000000 -> 45210000: NaN no código e na classe; acertam na divisão.
    c = cods("90911200", "33690000", "45000000", "45000000")
    s = sugg(["90911200"], ["33600000"], ["45233200"], ["45210000"])
    assert h(s, c, "codigo")[:2] == [1.0, 0.0]
    assert np.isnan(h(s, c, "codigo")[2:]).all()
    assert h(s, c, "divisao") == [1.0, 1.0, 1.0, 1.0]
    # F1 do código: C e D ficam de fora (etiqueta ao nível da divisão)
    assert ev.f1_pairs(s, c, "codigo").iloc[2:].isna().all().all()

    out = ev.evaluate(s, c)
    # código completo: 1 em 2 = 50% (não 1 em 4); a linha so_divisao mostra C e D à parte
    assert out.loc[["codigo", "classe", "divisao", "so_divisao"], "contratos"].tolist() == [2, 2, 4, 2]
    assert out.loc["codigo", "top1"] == pytest.approx(0.5)
    assert out.loc["so_divisao", ["top1", "top5"]].tolist() == pytest.approx([1.0, 1.0])


def test_so_divisao_conta_so_contratos_sem_codigo_abaixo_da_divisao():
    # "33000000 33690000" tem um código da classe -> conta no código completo, não em so_divisao.
    # "45000000 33000000" só tem divisões -> so_divisao; a sugestão 33100000 acerta a divisão 33
    # (2.º código) no top-1. "45000000" com 33100000 e depois 45210000 -> falha no top-1,
    # acerta no top-5. so_divisao: top-1 1/2, top-5 2/2.
    c = cods("33000000 33690000", "45000000 33000000", "45000000")
    s = sugg(["33690000"], ["33100000"], ["33100000", "45210000"])
    out = ev.evaluate(s, c)
    assert out.loc["codigo", "contratos"] == 1
    assert out.loc["so_divisao", "contratos"] == 2
    assert out.loc["so_divisao", ["top1", "top5"]].tolist() == pytest.approx([0.5, 1.0])
