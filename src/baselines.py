"""Régua "códigos mais comuns": sugerir sempre as mesmas 5 etiquetas, sem ler o texto.

É o mínimo que qualquer modelo tem de bater (E4-T2). As etiquetas contam-se no treino depois da
regra dos códigos raros (`labels.collapse_rare_codes`), com o mesmo N que o modelo usa, para a
régua e o modelo escolherem entre as mesmas etiquetas.

Cada exemplo de treino (uma linha do `treino.parquet`, um por texto) pesa o seu `n_contratos`,
os contratos que tinham esse texto. Assim a régua conta contratos, como a validação, que guarda
todos os contratos com as repetições (decidido a 11 de outubro de 2026). Esperava-se uma régua mais
exigente; na validação ficou 0,35 pontos abaixo da que conta exemplos (`qualidade_dados.md`,
secção 12 e decisão 10). A regra dos códigos raros continua a contar exemplos, como no modelo.
"""

import pandas as pd

from evaluation import TOP_K
from labels import DEFAULT_N_MIN, collapse_rare_codes


def most_common_labels(codigo, n_contratos=None, n_min=DEFAULT_N_MIN, k=TOP_K):
    """As k etiquetas de treino com mais contratos, da mais para a menos frequente.

    `codigo` é a coluna de códigos de 8 dígitos do treino e `n_contratos` o peso de cada exemplo
    (sem ela, cada exemplo vale 1). Empate: o menor código primeiro, para o resultado não depender
    da ordem das linhas.
    """
    weight = pd.Series(1, index=codigo.index) if n_contratos is None else n_contratos
    if not weight.index.equals(codigo.index):
        raise ValueError("`n_contratos` e `codigo` têm índices diferentes")
    label = collapse_rare_codes(codigo, n_min=n_min)
    keep = label.notna()
    counts = weight[keep].groupby(label[keep]).sum().rename_axis("etiqueta").reset_index(name="n")
    counts = counts.sort_values(["n", "etiqueta"], ascending=[False, True])
    top = counts["etiqueta"].head(k).tolist()
    if len(top) < k:
        raise ValueError(f"o treino só tem {len(top)} etiquetas depois da regra dos códigos raros; são precisas {k}")
    return top


def suggest_most_common(top, index):
    """As mesmas sugestões `top` para cada contrato de `index` (formato da `evaluation`)."""
    return pd.Series([list(top) for _ in range(len(index))], index=index, dtype=object)
