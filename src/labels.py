"""Etiquetas de treino: códigos raros sobem na árvore do CPV (back-off hierárquico).

Um código com menos de `n_min` exemplos no treino sobe para a classe (4 dígitos); se a classe
também for rara, sobe para o grupo (3 dígitos); se o grupo também for raro, o exemplo sai.
Nunca se sobe até à divisão (2 dígitos): esses códigos já foram tirados do treino (decisão 8).

As contagens fazem-se só com o treino. Contar também com a validação seria fuga de dados.
"""

import pandas as pd

DEFAULT_N_MIN = 10  # até a fase 4 escolher entre 5, 10 e 20


def to_level(codigo, digits):
    """'90911200' -> '90910000' (4 dígitos) ou '90900000' (3 dígitos)."""
    return codigo.str[:digits].str.ljust(8, "0")


def _rare(label, n_min):
    return label.map(label.value_counts()).lt(n_min)


def collapse_rare_codes(codigo, n_min=DEFAULT_N_MIN):
    """Etiqueta de treino de cada exemplo, ou valor em falta (NaN) se o exemplo sai.

    `codigo` é a coluna de códigos de 8 dígitos do treino; `n_min` é o N.
    """
    label = codigo.copy()
    for digits in (4, 3):  # código raro -> classe; classe rara -> grupo
        label = label.where(~_rare(label, n_min), to_level(label, digits))
    label = label.where(~_rare(label, n_min))  # grupo raro -> sai

    counts = label.value_counts()
    if not (counts >= n_min).all():
        raise RuntimeError("ficou uma etiqueta com menos de n_min exemplos")
    if label.dropna().str.endswith("000000").any():
        raise RuntimeError("uma etiqueta subiu até à divisão")
    return label


def summary(codigo, label):
    """Quantos exemplos ficaram iguais, subiram ou saíram."""
    return pd.Series({
        "exemplos": len(codigo),
        "iguais": int((label == codigo).sum()),
        "subiram": int((label.notna() & (label != codigo)).sum()),
        "saíram": int(label.isna().sum()),
        "etiquetas distintas": int(label.nunique()),
    })
