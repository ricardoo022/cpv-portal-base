"""Testes unitários a src/comparison.py (bootstrap emparelhado e regra de escolha)."""

import math

import numpy as np
import pytest

import comparison as cmp


def hits(n, correct):
    """Vetor 1.0/0.0 de tamanho n com 1.0 nas posições `correct` (contratos acertados)."""
    v = np.zeros(n)
    v[list(correct)] = 1.0
    return v


def compare(a, b, **kw):
    return cmp.paired_bootstrap(cmp.weighted_mean(a), cmp.weighted_mean(b), len(a), **kw)


def result(a_melhor, b_melhor):
    """Resultado de paired_bootstrap feito à mão (só os campos que `choose` usa)."""
    return {"a_melhor": a_melhor, "b_melhor": b_melhor,
            "conta": max(a_melhor, b_melhor) >= cmp.THRESHOLD}


CONTA_A = result(0.99, 0.01)
CONTA_B = result(0.01, 0.99)
NAO_CONTA = result(0.60, 0.30)


# --- amostras ---

def test_amostras_sao_pesos_de_n_contratos():
    amostras = list(cmp.bootstrap_weights(50, n_boot=20, seed=0))
    assert len(amostras) == 20
    assert all(len(w) == 50 and w.sum() == 50 and (w >= 0).all() for w in amostras)


def test_mesma_semente_mesmo_resultado():
    a, b = hits(200, range(120)), hits(200, range(10, 125))
    assert compare(a, b, seed=7) == compare(a, b, seed=7)
    w1 = list(cmp.bootstrap_weights(200, n_boot=5, seed=7))
    w2 = list(cmp.bootstrap_weights(200, n_boot=5, seed=7))
    assert all((x == y).all() for x, y in zip(w1, w2))


def test_semente_diferente_amostras_diferentes():
    w1 = next(cmp.bootstrap_weights(200, seed=0))
    w2 = next(cmp.bootstrap_weights(200, seed=1))
    assert not (w1 == w2).all()


# --- paired_bootstrap ---

def test_opcoes_iguais_nao_contam():
    a = hits(200, range(0, 200, 3))
    r = compare(a, a.copy())
    assert r["diferenca"] == 0
    assert r["a_melhor"] == r["b_melhor"] == 0
    assert r["conta"] is False


def test_valores_nos_dados_todos():
    r = compare(hits(10, range(7)), hits(10, range(4)))
    assert r["a"] == pytest.approx(0.7)
    assert r["b"] == pytest.approx(0.4)
    assert r["diferenca"] == pytest.approx(0.3)


def test_a_claramente_melhor_conta():
    r = compare(hits(200, range(180)), hits(200, range(20)))  # 90% contra 10%
    assert r["conta"] is True
    assert r["a_melhor"] >= 0.95
    assert r["b_melhor"] == 0


def test_b_claramente_melhor_conta_do_lado_de_b():
    r = compare(hits(200, range(20)), hits(200, range(180)))
    assert r["conta"] is True
    assert r["b_melhor"] >= 0.95


def test_um_contrato_de_diferenca_em_100_nao_conta():
    # Só o contrato 50 separa A de B; fica fora de ~37% das amostras -> diferença 0
    r = compare(hits(100, range(51)), hits(100, range(50)))
    assert r["b_melhor"] == 0
    assert 0 < r["a_melhor"] < 0.95
    assert r["conta"] is False


def test_limiar_e_95_por_cento():
    # A acerta mais 5 contratos e B mais 1: com a semente 0, A é melhor em 94,6% das amostras
    a = hits(100, range(5))
    b = hits(100, [5])
    r = compare(a, b)
    assert 0.90 <= r["a_melhor"] < 0.95
    assert r["conta"] is False
    # Com mais 1 contrato a favor de A já passa os 95%
    r = compare(hits(100, range(6)), hits(100, [6]))
    assert r["a_melhor"] >= 0.95
    assert r["conta"] is True


def test_exatamente_95_por_cento_conta():
    # A é melhor em 19 das 20 amostras (95%); a última chamada é a dos dados todos
    valores = iter([1.0] * 19 + [0.0] + [1.0])
    r = cmp.paired_bootstrap(lambda w: next(valores), lambda w: 0.5, 10, n_boot=20)
    assert r["a_melhor"] == pytest.approx(0.95)
    assert r["conta"] is True


def test_emparelhar_deteta_diferenca_que_amostras_independentes_nao_detetam():
    # A e B acertam os mesmos 500 contratos em 1 000; A acerta mais 15 onde B falha
    n = 1000
    b = hits(n, range(500))
    a = hits(n, range(515))
    stat_a, stat_b = cmp.weighted_mean(a), cmp.weighted_mean(b)

    # Sem emparelhar: amostras independentes para A e B (sementes diferentes)
    independent = [
        stat_a(wa) - stat_b(wb)
        for wa, wb in zip(cmp.bootstrap_weights(n, seed=0), cmp.bootstrap_weights(n, seed=1))
    ]
    assert np.mean(np.array(independent) > 0) < 0.95

    # Emparelhado: a mesma amostra para as duas opções
    r = cmp.paired_bootstrap(stat_a, stat_b, n)
    assert r["conta"] is True
    assert r["a_melhor"] >= 0.95


def test_amostra_com_nan_nao_conta_para_nenhum_lado():
    # Uma métrica que dá sempre NaN nunca dá vitória a ninguém
    r = cmp.paired_bootstrap(lambda w: float("nan"), lambda w: 0.5, 10, n_boot=50)
    assert r["a_melhor"] == r["b_melhor"] == 0
    assert r["conta"] is False


# --- weighted_mean ---

def test_weighted_mean_respeita_pesos_e_ignora_nan():
    f = cmp.weighted_mean([1.0, 0.0, np.nan, 1.0])
    # (2·1 + 1·0 + 0·1) / (2 + 1 + 0); o contrato NaN não conta, nem com peso 5
    assert f(np.array([2, 1, 5, 0])) == pytest.approx(2 / 3)
    assert f(np.ones(4, dtype=int)) == pytest.approx(2 / 3)


def test_weighted_mean_sem_contratos_validos_da_nan():
    f = cmp.weighted_mean([np.nan, 1.0])
    assert math.isnan(f(np.array([3, 0])))


# --- choose: um teste por ramo da regra ---

def test_ganha_no_codigo_completo():
    assert cmp.choose(CONTA_A, NAO_CONTA, default="b") == "a"
    assert cmp.choose(CONTA_B, NAO_CONTA, default="a") == "b"


def test_ganha_no_codigo_e_na_classe():
    assert cmp.choose(CONTA_A, CONTA_A, default="b") == "a"


def test_ganha_no_codigo_mas_piora_a_classe_e_eliminada():
    assert cmp.choose(CONTA_A, CONTA_B, default="a") == "b"
    assert cmp.choose(CONTA_B, CONTA_A, default="b") == "a"


def test_empate_no_codigo_desempata_a_classe():
    assert cmp.choose(NAO_CONTA, CONTA_A, default="b") == "a"
    assert cmp.choose(NAO_CONTA, CONTA_B, default="a") == "b"


@pytest.mark.parametrize("default", ["a", "b"])
def test_empate_em_tudo_fica_o_default(default):
    assert cmp.choose(NAO_CONTA, NAO_CONTA, default=default) == default


def test_default_invalido():
    with pytest.raises(ValueError):
        cmp.choose(NAO_CONTA, NAO_CONTA, default="c")


def test_choose_com_resultados_reais():
    # A acerta mais no código completo, mas na classe B é claramente melhor -> B
    n = 200
    codigo = compare(hits(n, range(150)), hits(n, range(50)))
    classe = compare(hits(n, range(60)), hits(n, range(190)))
    assert cmp.choose(codigo, classe, default="a") == "b"


# --- compare: liga a avaliação (src/evaluation.py) ao bootstrap ---

def contratos_e_sugestoes():
    """4 contratos; A acerta em todos no código completo, B só na divisão (mesma divisão, outra classe)."""
    import pandas as pd
    codigos = pd.Series(["90911200", "45210000", "33600000", "79710000"] * 10)
    a = codigos.map(lambda c: [c] + ["03000000"] * 4)
    b = codigos.map(lambda c: [c[:2] + "999999"] + ["03000000"] * 4)
    return codigos, a, b


def test_compare_opcoes_iguais_nada_conta():
    codigos, a, _ = contratos_e_sugestoes()
    comp = cmp.compare(a, a, codigos, n_boot=50)
    assert len(comp) == 9  # 3 medidas × (top1, top5, f1_macro)
    assert (comp["diferenca"] == 0).all()
    assert not comp["conta"].any()


def test_compare_a_melhor_no_codigo_e_na_classe_ganha_a():
    # Código completo e classe: A acerta 40/40, B 0/40 -> conta. Divisão: ambos 40/40 -> não conta.
    codigos, a, b = contratos_e_sugestoes()
    comp = cmp.compare(a, b, codigos, n_boot=50)
    assert comp.loc[("codigo", "top5"), "diferenca"] == 1.0
    assert comp.loc[("classe", "top5"), "conta"]
    assert comp.loc[("divisao", "top5"), "diferenca"] == 0
    assert not comp.loc[("divisao", "top5"), "conta"]
    assert cmp.choose_from(comp, default="b") == "a"
    assert cmp.choose_from(cmp.compare(b, a, codigos, n_boot=50), default="a") == "b"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_f1_macro_stat_igual_a_f1_macro_da_avaliacao(seed):
    # A versão rápida do bootstrap tem de dar o mesmo que evaluation.f1_macro, com e sem pesos
    import pandas as pd
    import evaluation as ev
    rng = np.random.default_rng(seed)
    true = pd.Series(rng.choice(["a", "b", "c", "d", None], 60))
    pred = pd.Series(rng.choice(["a", "b", "c", "x", None], 60))
    stat = cmp.f1_macro_stat(true, pred)
    assert stat(np.ones(60)) == pytest.approx(ev.f1_macro(true, pred))
    for w in cmp.bootstrap_weights(60, n_boot=20, seed=seed):
        assert stat(w) == pytest.approx(ev.f1_macro(true, pred, w), nan_ok=True)


def test_compare_top5_usa_as_5_sugestoes():
    # A tem o código certo só na 2.ª sugestão; B nunca: top-1 empata (0 - 0), top-5 dá 1 - 0
    codigos, _, b = contratos_e_sugestoes()
    a = codigos.map(lambda c: ["03000000", c] + ["03000000"] * 3)
    comp = cmp.compare(a, b, codigos, n_boot=50)
    assert comp.loc[("codigo", "top1"), "diferenca"] == 0
    assert comp.loc[("codigo", "top5"), "diferenca"] == 1.0


def test_choose_from_le_o_top5_e_nao_o_top1():
    # O top-1 diz B nas duas medidas, mas a regra só olha para o top-5: código conta para A -> A
    import pandas as pd
    comp = pd.DataFrame.from_dict({
        ("codigo", "top1"): CONTA_B, ("codigo", "top5"): CONTA_A,
        ("classe", "top1"): CONTA_B, ("classe", "top5"): NAO_CONTA,
    }, orient="index")
    assert cmp.choose_from(comp, default="b") == "a"


def test_compare_usa_a_semente_pedida():
    # B falha 3 contratos em 40: a fração de amostras em que A ganha depende das amostras tiradas
    codigos, a, _ = contratos_e_sugestoes()
    b = a.copy()
    b.iloc[:3] = [["03000000"] * 5] * 3
    um = cmp.compare(a, b, codigos, seed=0)
    assert um.equals(cmp.compare(a, b, codigos, seed=0))
    assert not um["a_melhor"].equals(cmp.compare(a, b, codigos, seed=1)["a_melhor"])
