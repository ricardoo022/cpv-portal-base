"""Avaliação comum a todos os modelos: top-1, top-5 e F1 macro em três medidas.

Cada contrato tem `codigos` (um ou mais códigos de 8 dígitos separados por espaço; o 1.º é a
etiqueta) e o modelo dá uma lista ordenada de sugestões (códigos de 8 dígitos).

Medidas:
- "codigo": a sugestão é igual ao código (33600000 -> só 33600000; 33651100 é mais específica
  e conta como erro, decisão 8). Só contam os contratos com algum código abaixo da divisão
  (nível >= 3), e só esses códigos servem para acertar: o modelo nunca sugere divisões
  (tiradas do treino, decisão 8), por isso um contrato só com 45000000 seria sempre erro.
- "classe": os mesmos 4 primeiros dígitos. Só contam os contratos com algum código de
  nível >= 4, e só esses códigos servem para acertar.
- "divisao": os mesmos 2 primeiros dígitos. Contam todos os contratos.

Os contratos que só têm códigos de divisão avaliam-se à parte, na linha "so_divisao" de
`evaluate`: quantos são e quantas vezes o modelo acerta a divisão.

Um contrato acerta no top-k se alguma das k primeiras sugestões acertar em algum dos seus códigos.
O F1 macro usa só a 1.ª sugestão contra a etiqueta (1.º código).

As medidas hierárquicas (`hierarchical`) dão meio acerto a uma sugestão perto da etiqueta; só se
mostram, não entram na regra de comparação.
"""

import re

import numpy as np
import pandas as pd

from cpv_data import code_level

MEASURES = ("codigo", "classe", "divisao")
TOP_K = 5
HIERARCHICAL = ("precisao_micro", "recall_micro", "f1_micro", "precisao_macro", "recall_macro", "f1_macro")

# Por medida: (nível mínimo do código para contar, dígitos comparados)
_RULES = {"codigo": (3, 8), "classe": (4, 4), "divisao": (2, 2)}

_CODE = re.compile(r"\d{8}")
_CODIGOS = re.compile(r"\s*\d{8}(\s+\d{8})*\s*")  # 1+ códigos separados por espaço


def _check_measure(measure):
    if measure not in _RULES:
        raise ValueError(f"medida desconhecida: {measure!r} (válidas: {MEASURES})")


def _check_codigos(codigos):
    bad = [c for c in codigos if not isinstance(c, str) or not _CODIGOS.fullmatch(c)]
    if bad:
        raise ValueError(f"`codigos` com valores que não são códigos de 8 dígitos: {bad[:3]}")


def _check_suggestions(suggestions, codigos, k):
    if not suggestions.index.equals(codigos.index):
        raise ValueError("`suggestions` e `codigos` têm índices diferentes")
    for s in suggestions:
        if not isinstance(s, (list, tuple)):
            raise ValueError(f"cada sugestão tem de ser uma lista ou tuplo de códigos, não {type(s).__name__}")
        if len(s) < k:
            raise ValueError(f"há contratos com menos de {k} sugestões: {list(s)}")
        bad = [c for c in s[:k] if not isinstance(c, str) or not _CODE.fullmatch(c)]
        if bad:
            raise ValueError(f"sugestões que não são códigos de 8 dígitos: {bad[:3]}")


def _prefixes(codigos, measure):
    """Por contrato, os prefixos que uma sugestão tem de ter para acertar (lista vazia = não conta)."""
    min_level, digits = _RULES[measure]
    long = codigos.reset_index(drop=True).str.split().explode()  # índice = posição do contrato
    level = code_level(long)
    out = [[] for _ in range(len(codigos))]
    for pos, c, lv in zip(long.index, long, level):
        if lv >= min_level:
            out[pos].append(c[:digits])
    return out


def hits(suggestions, codigos, measure, k):
    """1.0 se alguma das k primeiras sugestões acerta, 0.0 se não, NaN se o contrato não conta."""
    _check_measure(measure)
    _check_codigos(codigos)
    _check_suggestions(suggestions, codigos, k)
    return _hits(suggestions, _prefixes(codigos, measure), k)


def _hits(suggestions, prefixes, k):
    out = [
        float(any(s.startswith(p) for s in sugg[:k] for p in prefs)) if prefs else np.nan
        for sugg, prefs in zip(suggestions, prefixes)
    ]
    return pd.Series(out, index=suggestions.index, dtype=float)


def f1_pairs(suggestions, codigos, measure):
    """Pares (verdadeiro, previsto) da 1.ª sugestão contra a etiqueta, no nível da medida.

    "codigo": os códigos inteiros, só etiquetas de nível >= 3. "classe": só etiquetas de
    nível >= 4, com 4 dígitos. "divisao": 2 dígitos.
    """
    _check_measure(measure)
    _check_codigos(codigos)
    _check_suggestions(suggestions, codigos, 1)
    return _f1_pairs(suggestions, codigos, measure)


def _f1_pairs(suggestions, codigos, measure):
    label = codigos.str.split().str[0]
    first = suggestions.map(lambda s: s[0])
    level = code_level(label)
    if measure == "codigo":
        counts = level >= 3
        true, pred = label.where(counts), first.where(counts)
    elif measure == "classe":
        counts = level >= 4
        true = label.str[:4].where(counts)
        pred = first.str[:4].where(counts)
    else:
        true, pred = label.str[:2], first.str[:2]
    return pd.DataFrame({"verdadeiro": true, "previsto": pred}, index=codigos.index)


def f1_macro(verdadeiro, previsto, weights=None):
    """Média do F1 de cada etiqueta verdadeira; linhas com NaN são ignoradas.

    `weights` (inteiros >= 0, por exemplo multiplicidades de um bootstrap) multiplica cada linha.
    Só entram na média as etiquetas verdadeiras com peso total > 0. Um valor previsto que não é
    etiqueta verdadeira de ninguém não cria classe, mas conta como FN da etiqueta verdadeira.
    """
    true = pd.Series(verdadeiro).reset_index(drop=True)
    pred = pd.Series(previsto).reset_index(drop=True)
    if len(true) != len(pred):
        raise ValueError("`verdadeiro` e `previsto` têm tamanhos diferentes")
    if weights is None:
        w = pd.Series(1.0, index=true.index)
    else:
        w = pd.Series(np.asarray(weights, dtype=float))
        if len(w) != len(true):
            raise ValueError("`weights` tem um tamanho diferente de `verdadeiro`")
        if (w < 0).any() or w.isna().any():
            raise ValueError("`weights` tem de ser >= 0")
    keep = true.notna() & pred.notna()
    true, pred, w = true[keep], pred[keep], w[keep]

    support = w.groupby(true).sum()  # TP + FN
    support = support[support > 0]
    if support.empty:
        return np.nan
    predicted = w.groupby(pred).sum().reindex(support.index, fill_value=0)  # TP + FP
    tp = w[true == pred].groupby(true[true == pred]).sum().reindex(support.index, fill_value=0)
    # 2·TP / (2·TP + FP + FN) = 2·TP / ((TP + FN) + (TP + FP))
    return float((2 * tp / (support + predicted)).mean())


def _division_only(suggestions, codigos):
    """Contratos só com códigos de divisão: quantos são e como o modelo acerta a divisão neles."""
    only = codigos.str.split().map(lambda cs: all(c.endswith("000000") for c in cs))
    sugg, cods_ = suggestions[only], codigos[only]
    prefixes = _prefixes(cods_, "divisao")
    h1, h5 = _hits(sugg, prefixes, 1), _hits(sugg, prefixes, TOP_K)
    pairs = _f1_pairs(sugg, cods_, "divisao")
    return {
        "contratos": int(only.sum()),
        "top1": h1.mean(),
        "top5": h5.mean(),
        "f1_macro": f1_macro(pairs["verdadeiro"], pairs["previsto"]),
    }


def evaluate(suggestions, codigos):
    """Uma linha por medida: contratos que contam, top-1, top-5 e F1 macro.

    Mais a linha "so_divisao": os contratos que só têm códigos de divisão (não contam no código
    completo nem na classe), com o top-1, o top-5 e o F1 macro na divisão.
    """
    _check_codigos(codigos)
    _check_suggestions(suggestions, codigos, TOP_K)
    rows = {}
    for m in MEASURES:
        prefixes = _prefixes(codigos, m)
        h1 = _hits(suggestions, prefixes, 1)
        h5 = _hits(suggestions, prefixes, TOP_K)
        pairs = _f1_pairs(suggestions, codigos, m)
        rows[m] = {
            "contratos": int(h5.notna().sum()),
            "top1": h1.mean(),
            "top5": h5.mean(),
            "f1_macro": f1_macro(pairs["verdadeiro"], pairs["previsto"]),
        }
    rows["so_divisao"] = _division_only(suggestions, codigos)
    out = pd.DataFrame.from_dict(rows, orient="index")
    out["contratos"] = out["contratos"].astype(int)
    out.index.name = "medida"
    return out


def _common_nodes(a, b, level_a, level_b):
    """Nós em comum entre dois códigos, contando do nível 2 até ao mais baixo dos dois.

    Os nós de um código são os seus prefixos de 2 até ao nível dele: 45233200 -> 45, 452, 4523,
    45233, 452332. Com 45233223 partilha os 5; com 45200000 partilha 2 (45, 452).
    """
    prefix = 0
    for x, y in zip(a, b):
        if x != y:
            break
        prefix += 1
    return max(0, min(prefix, level_a, level_b) - 1)


def hierarchical_scores(suggestions, codigos):
    """Por contrato, a 1.ª sugestão contra a etiqueta: nós em comum, nós de cada uma e a etiqueta."""
    _check_codigos(codigos)
    _check_suggestions(suggestions, codigos, 1)
    label = codigos.str.split().str[0]
    first = suggestions.map(lambda s: s[0])
    lv_label, lv_first = code_level(label), code_level(first)
    common = [_common_nodes(a, b, la, lb) for a, b, la, lb in zip(first, label, lv_first, lv_label)]
    return pd.DataFrame({
        "etiqueta": label,
        "comuns": common,
        "nos_sugestao": lv_first - 1,
        "nos_etiqueta": lv_label - 1,
    }, index=codigos.index)


def _f1(p, r):
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def hierarchical(suggestions, codigos):
    """Precisão, recall e F1 hierárquicos da 1.ª sugestão (Kiritchenko et al., 2005).

    Micro: somam-se os nós de todos os contratos antes de dividir. Macro (como no artigo
    "Zero-Shot Hierarchical Classification on the CPV Taxonomy", 2024): a precisão e o recall
    de cada contrato fazem-se média por etiqueta, e depois média sobre as etiquetas.
    """
    sc = hierarchical_scores(suggestions, codigos)
    if sc.empty:
        return pd.Series(np.nan, index=HIERARCHICAL)
    p_micro = sc["comuns"].sum() / sc["nos_sugestao"].sum()
    r_micro = sc["comuns"].sum() / sc["nos_etiqueta"].sum()
    per_label = pd.DataFrame({
        "p": sc["comuns"] / sc["nos_sugestao"],
        "r": sc["comuns"] / sc["nos_etiqueta"],
    }).groupby(sc["etiqueta"]).mean()
    p_macro, r_macro = per_label["p"].mean(), per_label["r"].mean()
    return pd.Series([p_micro, r_micro, _f1(p_micro, r_micro), p_macro, r_macro, _f1(p_macro, r_macro)],
                     index=HIERARCHICAL, dtype=float)


def _by_group(fn, suggestions, codigos, visto_no_treino):
    """`fn` para todos os contratos e só para os de texto novo (não visto no treino)."""
    if not visto_no_treino.index.equals(codigos.index):
        raise ValueError("`visto_no_treino` e `codigos` têm índices diferentes")
    if visto_no_treino.isna().any():
        raise ValueError("`visto_no_treino` tem valores em falta")
    new = ~visto_no_treino.astype(bool)
    return {"todos": fn(suggestions, codigos), "texto novo": fn(suggestions[new], codigos[new])}


def report(suggestions, codigos, visto_no_treino):
    """`evaluate` para todos os contratos e só para os de texto novo, com MultiIndex (grupo, medida)."""
    return pd.concat(_by_group(evaluate, suggestions, codigos, visto_no_treino), names=["grupo", "medida"])


def hierarchical_report(suggestions, codigos, visto_no_treino):
    """`hierarchical` para todos os contratos e só para os de texto novo: uma linha por grupo."""
    out = pd.DataFrame(_by_group(hierarchical, suggestions, codigos, visto_no_treino)).T
    out.index.name = "grupo"
    return out
