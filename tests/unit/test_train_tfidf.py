"""Testes unitários a src/train_tfidf.py: nomes dos ficheiros de sugestões por τ (spike E4-S4)
e épocas fixas (spike E4-S5).

O notebook 05 e os docs leem estes nomes exatos, por isso o teste escreve-os à mão em vez de os
pedir à própria função.
"""

import pytest

import tfidf
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


# --- Spike E4-S5: épocas fixas (`tfidf.sgd_epochs`, que o `run` usa) ---

def test_sem_epocas_fica_a_regra_por_defeito_do_sklearn():
    assert tfidf.sgd_epochs(None) == {}


@pytest.mark.parametrize("k", [1, 3, 24])
def test_epocas_fixas_desligam_a_paragem_automatica(k):
    assert tfidf.sgd_epochs(k) == {"max_iter": k, "tol": None}


@pytest.mark.parametrize("epocas", [0, -3, 2.5, True, "12"])
def test_epocas_invalidas_dao_erro_antes_de_ler_os_ficheiros(tmp_path, epocas):
    # Os ficheiros não existem: o erro tem de ser o das épocas, não o de ficheiro em falta
    with pytest.raises(ValueError, match="epocas"):
        tt.run(tmp_path / "treino.parquet", tmp_path / "validacao.parquet", tmp_path / "saida", epocas=epocas)
    assert not (tmp_path / "saida").exists()


@pytest.mark.parametrize("qual", ["treino", "validacao"])
def test_caminho_do_teste_e_recusado_tambem_com_epocas(tmp_path, qual):
    paths = {"treino": tmp_path / "treino.parquet", "validacao": tmp_path / "validacao.parquet"}
    paths[qual] = tmp_path / "teste.parquet"
    with pytest.raises(ValueError, match="teste"):
        tt.run(paths["treino"], paths["validacao"], tmp_path / "saida", epocas=12)
    assert not (tmp_path / "saida").exists()
