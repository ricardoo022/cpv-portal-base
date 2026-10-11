"""Testes unitários a src/train_tfidf.py (spike E4-S4): nomes dos ficheiros de sugestões por τ.

O notebook 05 e os docs leem estes nomes exatos, por isso o teste escreve-os à mão em vez de os
pedir à própria função.
"""

import pytest

import train_tfidf as tt


@pytest.mark.parametrize("tau, nome", [
    (0.0, "sugestoes_validacao.parquet"),          # sem correção: o nome do E4-H1
    (0, "sugestoes_validacao.parquet"),
    (0.25, "sugestoes_validacao_tau0.25.parquet"),
    (0.5, "sugestoes_validacao_tau0.5.parquet"),
    (1.0, "sugestoes_validacao_tau1.parquet"),     # sem ".0"
])
def test_nome_do_ficheiro_de_sugestoes_por_tau(tau, nome):
    assert tt.suggestions_file(tau) == nome
