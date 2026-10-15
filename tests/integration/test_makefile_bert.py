"""Os atalhos do Makefile para o BERT (E5-T2): `treinar-bert` e `verificar-gpu`.

São os comandos que se escrevem no pod; um erro aqui (outra pasta de saída, a verificação da
GPU que não falha) só se veria lá, a pagar. `make -n` mostra o comando sem o correr.
"""

import subprocess
import sys
from pathlib import Path

import pytest
import torch

RAIZ = Path(__file__).resolve().parents[2]


def _make(*args, ver_so=False):
    cmd = ["make", "--no-print-directory", *(["-n"] if ver_so else []), *args]
    return subprocess.run(cmd, cwd=RAIZ, capture_output=True, text=True, timeout=300)


@pytest.mark.parametrize("args, modelo, saida", [
    ((), "bertimbau", "metrics/bert/bertimbau"),
    (("MODELO=albertina",), "albertina", "metrics/bert/albertina"),
    (("MODELO=albertina", "SAIDA=metrics/bert/outra"), "albertina", "metrics/bert/outra"),
])
def test_treinar_bert_passa_o_modelo_e_a_pasta_de_saida(args, modelo, saida):
    r = _make("treinar-bert", *args, ver_so=True)
    assert r.returncode == 0, r.stderr
    assert f"src/train_bert.py --modelo {modelo} --saida {saida}" in r.stdout


def test_treinar_bert_passa_os_outros_argumentos():
    r = _make("treinar-bert", "ARGS=--retomar --epocas 2", ver_so=True)
    assert r.stdout.rstrip().endswith("--retomar --epocas 2")


def test_treinar_bert_nao_muda_a_saida_do_preparar_janela():
    # Os dois alvos usam SAIDA; o valor por defeito do BERT não pode entrar no preparar-janela
    r = _make("preparar-janela", "INICIO=2019-01-01", ver_so=True)
    assert r.stdout.rstrip().endswith("--inicio-treino 2019-01-01 --saida")


@pytest.mark.skipif(torch.cuda.is_available(), reason="há GPU: a verificação passa")
def test_verificar_gpu_falha_sem_gpu():
    # O Python que corre os testes: no CI não há .venv, as bibliotecas estão no Python do sistema
    r = _make("verificar-gpu", f"PY={sys.executable}")
    assert r.returncode != 0, r.stdout
    assert "GPU" in r.stderr, f"código {r.returncode}: {r.stderr}"
