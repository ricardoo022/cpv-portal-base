"""Comparar duas opções de modelo (A e B) nos mesmos contratos de validação.

Bootstrap emparelhado: tira-se `N_BOOT` vezes, à sorte e com repetição, uma amostra dos n
contratos, sempre a mesma amostra para A e para B, e calcula-se a diferença (A − B) em cada uma.
A diferença só conta se ficar do mesmo lado (> 0, ou < 0) em pelo menos `THRESHOLD` das amostras.
Amostras com diferença exatamente 0 não contam para nenhum lado.

Cada amostra é um vetor de pesos: quantas vezes cada contrato saiu. Assim as métricas recebem
os pesos em vez de uma cópia dos dados, e a mesma semente dá as mesmas amostras para todas as
métricas de uma comparação (emparelhada também entre métricas).
"""

import numpy as np
import pandas as pd

import evaluation as ev

N_BOOT = 1000
THRESHOLD = 0.95


def bootstrap_weights(n, n_boot=N_BOOT, seed=0):
    """Gera, uma a uma, `n_boot` amostras de n contratos como pesos inteiros (tamanho n).

    Ex.: n = 3, sorteio [0, 0, 2] -> pesos [2, 0, 1]. Uma amostra de cada vez poupa memória
    (n_boot × n inteiros de uma vez seriam ~190 MB com n = 24 000).
    """
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        yield np.bincount(rng.integers(0, n, n), minlength=n)


def weighted_mean(values):
    """f(pesos) = média de `values` com esses pesos, ignorando NaN (contrato que não conta).

    `values` é um valor por contrato (ex.: 1.0 acertou no top-5, 0.0 falhou). Devolve NaN se
    nenhum contrato válido tiver peso.
    """
    values = np.asarray(values, dtype=float)
    valid = ~np.isnan(values)
    filled = np.where(valid, values, 0.0)
    valid = valid.astype(float)

    def stat(weights):
        total = weights @ valid
        return float(weights @ filled / total) if total > 0 else float("nan")

    return stat


def f1_macro_stat(verdadeiro, previsto):
    """f(pesos) = `evaluation.f1_macro` com esses pesos, mas rápida o bastante para 1 000 amostras.

    Os códigos passam a números uma só vez; cada amostra só faz contagens com np.bincount.
    """
    verdadeiro, previsto = pd.Series(verdadeiro), pd.Series(previsto)
    keep = (verdadeiro.notna() & previsto.notna()).to_numpy()
    codes, uniques = pd.factorize(pd.concat([verdadeiro[keep], previsto[keep]]))
    n = int(keep.sum())
    true, pred = codes[:n], codes[n:]
    same = true == pred
    size = len(uniques)

    def stat(weights):
        w = np.asarray(weights, dtype=float)[keep]
        support = np.bincount(true, w, size)  # TP + FN
        predicted = np.bincount(pred, w, size)  # TP + FP
        tp = np.bincount(true[same], w[same], size)
        labels = support > 0  # só as etiquetas verdadeiras com peso
        if not labels.any():
            return float("nan")
        return float(np.mean(2 * tp[labels] / (support[labels] + predicted[labels])))

    return stat


def paired_bootstrap(stat_a, stat_b, n, n_boot=N_BOOT, seed=0):
    """Compara A e B com as mesmas amostras; devolve o resumo da comparação.

    `stat_a` e `stat_b` recebem os pesos de uma amostra e devolvem um número (ex.: top-5).
    Uma amostra em que uma das métricas dê NaN não conta para nenhum lado.
    """
    a_wins = b_wins = 0
    for weights in bootstrap_weights(n, n_boot, seed):
        diff = stat_a(weights) - stat_b(weights)  # a mesma amostra para as duas opções
        if diff > 0:
            a_wins += 1
        elif diff < 0:
            b_wins += 1
        # diff == 0 (ou NaN) não conta para nenhum lado

    ones = np.ones(n, dtype=np.int64)
    a, b = stat_a(ones), stat_b(ones)
    a_melhor, b_melhor = a_wins / n_boot, b_wins / n_boot
    return {
        "a": a,
        "b": b,
        "diferenca": a - b,
        "a_melhor": a_melhor,
        "b_melhor": b_melhor,
        "conta": max(a_melhor, b_melhor) >= THRESHOLD,
    }


def _winner(result):
    """'a' ou 'b' se a diferença conta; None se não conta."""
    if not result["conta"]:
        return None
    return "a" if result["a_melhor"] > result["b_melhor"] else "b"


def _other(option):
    return "b" if option == "a" else "a"


def choose(codigo_top5, classe_top5, default):
    """Aplica a regra de escolha do projeto e devolve 'a' ou 'b'.

    `codigo_top5` e `classe_top5` são resultados de `paired_bootstrap` (top-5 no código
    completo e top-5 na classe); `default` é a opção que fica se não houver diferença que conte.
    """
    if default not in ("a", "b"):
        raise ValueError(f"default tem de ser 'a' ou 'b', não {default!r}")

    # 1. Guarda: a opção que piora a classe, com diferença que conte, é eliminada
    classe_winner = _winner(classe_top5)
    eliminated = _other(classe_winner) if classe_winner else None

    # 2. Ganha a melhor no código completo, se a diferença contar e não tiver sido eliminada
    codigo_winner = _winner(codigo_top5)
    if codigo_winner is not None:
        if codigo_winner != eliminated:
            return codigo_winner
        return _other(eliminated)  # ganhou no código mas piorou a classe: fica a outra

    # 3. Empate no código completo: desempata a classe, se essa diferença contar
    if classe_winner is not None:
        return classe_winner

    # 4. Empate em tudo: fica a opção por defeito
    return default


def _stats(suggestions, codigos):
    """Por (medida, métrica), a função f(pesos) dessa métrica para uma opção."""
    out = {}
    for m in ev.MEASURES:
        for k, name in ((1, "top1"), (ev.TOP_K, "top5")):
            out[m, name] = weighted_mean(ev.hits(suggestions, codigos, m, k))
        pairs = ev.f1_pairs(suggestions, codigos, m)
        out[m, "f1_macro"] = f1_macro_stat(pairs["verdadeiro"], pairs["previsto"])
    return out


def compare(suggestions_a, suggestions_b, codigos, n_boot=N_BOOT, seed=0):
    """Compara as sugestões de A e B nos mesmos contratos, em cada medida e métrica.

    Devolve uma linha por (medida, métrica) com as colunas de `paired_bootstrap`. Todas as linhas
    usam a mesma semente, logo as mesmas amostras. Para ver só o texto novo, passa os contratos
    já filtrados.
    """
    stats_a, stats_b = _stats(suggestions_a, codigos), _stats(suggestions_b, codigos)
    rows = {key: paired_bootstrap(stats_a[key], stats_b[key], len(codigos), n_boot, seed) for key in stats_a}
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.names = ["medida", "metrica"]
    return out


def choose_from(comparison, default):
    """`choose` aplicado à tabela de `compare`."""
    return choose(comparison.loc[("codigo", "top5")], comparison.loc[("classe", "top5")], default)
