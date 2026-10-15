"""As listas de requisitos do E5-T2: versões fixas, o torch fora do requirements.txt e o CI.

O portátil e o CI instalam o `requirements-cpu.txt` (torch só para processador) e o pod o
`requirements-gpu.txt` (torch com CUDA). As duas listas incluem o `requirements.txt` e têm de
fixar as mesmas versões, para o código testado no CI ser o que treina no pod.
"""

import re
from importlib.metadata import version
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parents[2]
NOVAS = ("torch", "accelerate")


def _pins(nome):
    """{biblioteca: versão} das linhas `nome==versão` de um ficheiro de requisitos."""
    linhas = (RAIZ / nome).read_text(encoding="utf-8").splitlines()
    return dict(re.match(r"([A-Za-z0-9_.-]+)==(\S+)$", l).groups() for l in linhas if "==" in l)


def _linhas(nome):
    return [l.strip() for l in (RAIZ / nome).read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")]


def test_requirements_txt_nao_tem_torch_e_tudo_tem_versao_fixa():
    # Sem torch: o pip instalaria a versão com CUDA (vários GB) também no CI
    assert not set(NOVAS) & {b.lower() for b in _pins("requirements.txt")}
    assert all("==" in l for l in _linhas("requirements.txt"))


def test_listas_cpu_e_gpu_incluem_o_requirements_e_fixam_as_mesmas_versoes():
    cpu, gpu = _pins("requirements-cpu.txt"), _pins("requirements-gpu.txt")
    for nome in ("requirements-cpu.txt", "requirements-gpu.txt"):
        assert "-r requirements.txt" in _linhas(nome)
        # Cada linha que instala alguma coisa tem a versão fixa
        assert all("==" in l for l in _linhas(nome) if not l.startswith("-"))
    assert set(cpu) == set(gpu) == set(NOVAS)
    assert cpu["torch"] == gpu["torch"] + "+cpu"  # o mesmo torch, só muda a variante
    assert cpu["accelerate"] == gpu["accelerate"]


def test_torch_do_processador_vem_do_indice_do_pytorch():
    assert "--extra-index-url https://download.pytorch.org/whl/cpu" in _linhas("requirements-cpu.txt")
    assert not any(l.startswith("--") for l in _linhas("requirements-gpu.txt"))


def test_versoes_instaladas_sao_as_fixadas():
    # No CI e no portátil instala-se a lista do processador; no pod, a da GPU (o torch sem "+cpu")
    cpu = _pins("requirements-cpu.txt")
    assert version("torch").split("+")[0] == cpu["torch"].split("+")[0]
    assert version("accelerate") == cpu["accelerate"]


def test_ci_instala_a_lista_do_processador():
    ci = yaml.safe_load((RAIZ / ".github" / "workflows" / "testes.yml").read_text(encoding="utf-8"))
    comandos = [p.get("run", "") for p in ci["jobs"]["pytest"]["steps"]]
    assert any("pip install -r requirements-cpu.txt" in c for c in comandos)
    assert not any(re.search(r"-r requirements\.txt\b", c) for c in comandos)
