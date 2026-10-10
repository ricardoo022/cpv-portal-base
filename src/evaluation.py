"""Avaliação comum a todos os modelos: top-1, top-5 e F1 macro em três medidas.

Cada contrato tem `codigos` (um ou mais códigos de 8 dígitos separados por espaço; o 1.º é a
etiqueta) e o modelo dá uma lista ordenada de sugestões (códigos de 8 dígitos).

Medidas:
- "codigo": a sugestão tem os mesmos dígitos que o código até ao nível dele
  (33600000 -> começa por 336; 90911200 -> começa por 909112). Contam todos os contratos.
- "classe": os mesmos 4 primeiros dígitos. Só contam os contratos com algum código de
  nível >= 4, e só esses códigos servem para acertar.
- "divisao": os mesmos 2 primeiros dígitos. Contam todos os contratos.

Um contrato acerta no top-k se alguma das k primeiras sugestões acertar em algum dos seus códigos.
O F1 macro usa só a 1.ª sugestão contra a etiqueta (1.º código).
"""

import re

import numpy as np
import pandas as pd

from cpv_data import code_level

MEASURES = ("codigo", "classe", "divisao")
TOP_K = 5

# Por medida: (nível mínimo do código para contar, dígitos comparados; None = o nível do código)
_RULES = {"codigo": (2, None), "classe": (4, 4), "divisao": (2, 2)}

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
            out[pos].append(c[: lv if digits is None else digits])
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

    "codigo": a sugestão é cortada ao nível da etiqueta e completada com zeros
    (etiqueta 33600000, sugestão 33651100 -> 33600000). "classe": só etiquetas de nível >= 4.
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
        true = label
        pred = pd.Series([s[:n].ljust(8, "0") for s, n in zip(first, level)], index=codigos.index)
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


def evaluate(suggestions, codigos):
    """Uma linha por medida: contratos que contam, top-1, top-5 e F1 macro."""
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
    out = pd.DataFrame.from_dict(rows, orient="index")
    out["contratos"] = out["contratos"].astype(int)
    out.index.name = "medida"
    return out


def report(suggestions, codigos, visto_no_treino):
    """`evaluate` para todos os contratos e só para os de texto novo (não visto no treino)."""
    if not visto_no_treino.index.equals(codigos.index):
        raise ValueError("`visto_no_treino` e `codigos` têm índices diferentes")
    if visto_no_treino.isna().any():
        raise ValueError("`visto_no_treino` tem valores em falta")
    new = ~visto_no_treino.astype(bool)
    parts = {
        "todos": evaluate(suggestions, codigos),
        "texto novo": evaluate(suggestions[new], codigos[new]),
    }
    return pd.concat(parts, names=["grupo", "medida"])
