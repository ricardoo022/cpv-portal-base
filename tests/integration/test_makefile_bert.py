"""Os atalhos do Makefile para o BERT (E5-T2 a E5-T4): `treinar-bert`, `verificar-gpu`, `registar-bert`,
`sugerir-bert` e `sinal-decorar`.

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


def test_registar_bert_passa_a_pasta_e_os_outros_argumentos():
    r = _make("registar-bert", "PASTA=metrics/bert/bertimbau", "ARGS=--nome primeiro", ver_so=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.rstrip().endswith("src/registo_bert.py --pasta metrics/bert/bertimbau --nome primeiro")


@pytest.mark.parametrize("args, base, dispositivo", [
    ((), "bertimbau", "cpu"),
    (("MODELO=albertina", "DISPOSITIVO=cuda"), "albertina", "cuda"),
])
def test_sugerir_bert_passa_o_checkpoint_o_modelo_de_base_e_o_dispositivo(args, base, dispositivo):
    ck = "models/bert/bertimbau/checkpoints/checkpoint-10"
    r = _make("sugerir-bert", f"CHECKPOINT={ck}", *args, "ARGS=--tau 0 \"papel A4\"", ver_so=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.rstrip().endswith(f"src/suggest.py --bert {ck} --base {base} --dispositivo {dispositivo} "
                                      f"--tau 0 \"papel A4\"")


@pytest.mark.parametrize("args, modelo, saida", [
    ((), "bertimbau", "metrics/bert/sinal_decorar_bertimbau"),
    (("MODELO=albertina",), "albertina", "metrics/bert/sinal_decorar_albertina"),
    (("MODELO=albertina", "SAIDA=metrics/bert/outra"), "albertina", "metrics/bert/outra"),
])
def test_sinal_decorar_passa_o_modelo_a_pasta_a_taxa_e_os_outros_argumentos(args, modelo, saida):
    r = _make("sinal-decorar", *args, "TAXA=1e-4", "ARGS=--experiencia experiencias/e5_t4.yaml", ver_so=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.rstrip().endswith(f"scripts/sinal_decorar.py --modelo {modelo} --saida {saida} "
                                      "--taxa-aprendizagem 1e-4 --experiencia experiencias/e5_t4.yaml")


def test_sinal_decorar_sem_taxa_falha_sem_correr():
    # A taxa do sinal fica escrita no YAML antes de correr: sem ela, o atalho não adivinha uma
    r = _make("sinal-decorar", ver_so=True)
    assert r.returncode != 0 and "TAXA" in r.stderr
    assert "sinal_decorar.py" not in r.stdout
