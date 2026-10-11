"""Régua "códigos mais comuns": sugerir sempre as mesmas 5 etiquetas, sem ler o texto.

É o mínimo que qualquer modelo tem de bater (E4-T2). As etiquetas contam-se no treino depois da
regra dos códigos raros (`labels.collapse_rare_codes`), com o mesmo N que o modelo usa, para a
régua e o modelo escolherem entre as mesmas etiquetas.

A régua conta exemplos de treino (uma linha do `treino.parquet`, um por texto), como a regra dos
códigos raros. Também se pode contar contratos, passando o `n_contratos` como peso; na validação
essa régua ficou mais fácil de bater, por isso a régua do projeto é a que conta exemplos
(`qualidade_dados.md`, secção 12 e decisão 10).
"""

import pandas as pd

from evaluation import TOP_K
from labels import DEFAULT_N_MIN, collapse_rare_codes


def most_common_labels(codigo, n_contratos=None, n_min=DEFAULT_N_MIN, k=TOP_K):
    """As k etiquetas de treino mais frequentes, da mais para a menos frequente.

    `codigo` é a coluna de códigos de 8 dígitos do treino. Sem `n_contratos`, cada exemplo vale 1
    (a régua do projeto); com ela, cada exemplo pesa os seus contratos (só para comparar). Empate: o menor código primeiro, para o resultado não depender
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
