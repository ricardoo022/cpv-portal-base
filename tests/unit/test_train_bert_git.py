"""Testes unitários do que o `train_bert.py` grava no treino.json para o registo no MLflow (E5-T3).

O `read_git` corre com um `run` falso: nenhum teste executa o git verdadeiro (e o
tests/conftest.py substitui o `train_bert.git_info` em todos os testes). O `file_sha256` lê o
ficheiro por blocos de 1 MiB: o teste usa um ficheiro maior, para passar por mais de um bloco.
"""

import hashlib
import subprocess

import pytest

import train_bert as tb


def _run_falso(respostas):
    """Um `subprocess.run` falso: `respostas` dá (código de saída, saída) para cada comando do git."""
    chamadas = []

    def run(cmd, cwd=None, capture_output=False, text=False):
        chamadas.append((tuple(cmd), cwd, capture_output, text))
        codigo, saida = respostas[tuple(cmd[1:])]
        return subprocess.CompletedProcess(cmd, codigo, stdout=saida, stderr="")

    return run, chamadas


SHA = "a" * 40


def test_read_git_sem_alteracoes(tmp_path):
    run, chamadas = _run_falso({("rev-parse", "HEAD"): (0, SHA + "\n"), ("status", "--porcelain", "--untracked-files=no"): (0, "")})
    assert tb.read_git(tmp_path, run=run) == {"commit": SHA, "alteracoes_por_gravar": False}
    # Os dois comandos correm na raiz pedida e com a saída capturada como texto. O status ignora os
    # ficheiros que o git não segue: a pasta --saida aparece a meio do treino e, se contasse, a
    # retoma depois de uma queda era sempre recusada (alterações por gravar de False a True)
    assert sorted(c[0] for c in chamadas) == [("git", "rev-parse", "HEAD"), ("git", "status", "--porcelain", "--untracked-files=no")]
    assert all(c[1:] == (tmp_path, True, True) for c in chamadas)


def test_read_git_com_alteracoes_por_gravar(tmp_path):
    run, _ = _run_falso({("rev-parse", "HEAD"): (0, SHA + "\n"),
                         ("status", "--porcelain", "--untracked-files=no"): (0, " M src/train_bert.py\n")})
    assert tb.read_git(tmp_path, run=run) == {"commit": SHA, "alteracoes_por_gravar": True}


def test_read_git_sem_git_instalado_da_nones(tmp_path):
    def run(*a, **kw):
        raise FileNotFoundError("git")

    assert tb.read_git(tmp_path, run=run) == {"commit": None, "alteracoes_por_gravar": None}


@pytest.mark.parametrize("falha", [("rev-parse", "HEAD"), ("status", "--porcelain", "--untracked-files=no")])
def test_read_git_com_erro_do_git_da_nones(tmp_path, falha):
    # Por exemplo, a pasta não é um repositório (o git sai com 128)
    respostas = {("rev-parse", "HEAD"): (0, SHA + "\n"), ("status", "--porcelain", "--untracked-files=no"): (0, "")}
    respostas[falha] = (128, "")
    run, _ = _run_falso(respostas)
    assert tb.read_git(tmp_path, run=run) == {"commit": None, "alteracoes_por_gravar": None}


def test_file_sha256_igual_ao_do_hashlib_num_ficheiro_de_varios_blocos(tmp_path):
    dados = bytes(range(256)) * (5 * 4096 + 7)  # ~5 MiB, que não acaba num bloco inteiro
    f = tmp_path / "dados.bin"
    f.write_bytes(dados)
    assert len(dados) > 1 << 20
    assert tb.file_sha256(f) == hashlib.sha256(dados).hexdigest()
