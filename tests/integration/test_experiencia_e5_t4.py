"""O YAML pré-registado do E5-T4 (`experiencias/e5_t4.yaml`) e os comandos que ele manda correr.

O YAML só regista: quem decide o treino são as flags do comando. Se o `comando` de uma secção
passar outra taxa, outro lote ou outro número de passos do que os valores escritos nessa secção, o
treino pago corre com uma receita e o MLflow regista outra. Aqui cada `comando` passa pelo
Makefile (`make -n`) e pelo `argparse` do script verdadeiro (com o `run` substituído, para não
treinar), e os argumentos que chegam ao `run` comparam-se com os valores do YAML.
"""

import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

RAIZ = Path(__file__).resolve().parents[2]
YAML = RAIZ / "experiencias" / "e5_t4.yaml"
sys.path.insert(0, str(RAIZ / "scripts"))

import bert_models  # noqa: E402
import sinal_decorar  # noqa: E402
import train_bert  # noqa: E402
from cpv_data import MAX_YAML  # noqa: E402


@pytest.fixture(scope="module")
def exp():
    return yaml.safe_load(YAML.read_text(encoding="utf-8"))


def _argumentos_do_run(comando, modulo, monkeypatch, devolve):
    """Os argumentos com que o `comando` (um `make ...` do YAML) chamaria o `run` do `modulo`."""
    partes = shlex.split(comando)
    assert partes[0] == "make"
    r = subprocess.run(["make", "--no-print-directory", "-n", *partes[1:]], cwd=RAIZ, capture_output=True,
                       text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    linha = shlex.split(r.stdout.strip().splitlines()[-1])
    script = Path(modulo.__file__).name
    assert Path(linha[1]).name == script, linha
    visto = {}

    def falso_run(*args, **kwargs):
        visto["args"], visto["kwargs"] = args, kwargs
        return devolve

    monkeypatch.setattr(modulo, "run", falso_run)
    modulo.main(linha[2:])
    return visto["args"], visto["kwargs"]


def test_yaml_cabe_no_limite_e_le_se_como_no_treino(exp):
    assert len(YAML.read_text(encoding="utf-8")) < MAX_YAML
    assert train_bert.read_experiment(YAML)["sha256"]
    assert exp["modelo"]["revisao"] == bert_models.MODELOS["bertimbau"]["revisao"]
    assert exp["modelo"]["max_length"] == bert_models.COMPRIMENTO_MAXIMO["bertimbau"]


def test_comando_do_treino_completo_passa_os_valores_do_yaml(exp, monkeypatch):
    t = exp["treino_completo"]
    _, kw = _argumentos_do_run(t["comando"], train_bert, monkeypatch, devolve=None)
    assert kw["modelo"] == exp["modelo"]["nome_curto"]
    assert kw["taxa_aprendizagem"] == t["taxa_aprendizagem"]
    assert (kw["lote"], kw["epocas"], kw["semente"]) == (t["lote"], t["epocas"], t["semente"])
    assert (kw["aquecimento"], kw["decaimento_pesos"]) == (t["aquecimento"], t["decaimento_pesos"])
    assert kw["n_min"] == exp["n_min"]
    assert list(kw["taus"]) == exp["taus"]
    assert kw["max_length"] == exp["modelo"]["max_length"]
    assert Path(kw["experiencia"]).resolve() == YAML
    assert kw["max_passos"] is None  # um treino completo, não um ensaio


def test_comando_do_ensaio_mede_a_receita_do_treino_completo(exp, monkeypatch):
    # O custo medido só serve para o treino completo se a configuração for a mesma
    t, e = exp["treino_completo"], exp["ensaio_custo"]
    args, kw = _argumentos_do_run(e["comando"], train_bert, monkeypatch, devolve=None)
    args_completo, _ = _argumentos_do_run(t["comando"], train_bert, monkeypatch, devolve=None)
    # Outra pasta de saída: o ensaio recusa uma com o treino.json do treino a sério, e vice-versa
    assert Path(args[2]) != Path(args_completo[2])
    assert kw["max_passos"] == e["passos"]
    assert kw["lote"] == e["lote"] == t["lote"]
    assert kw["max_length"] == e["max_length"] == exp["modelo"]["max_length"]
    assert kw["taxa_aprendizagem"] == t["taxa_aprendizagem"]
    assert (kw["aquecimento"], kw["decaimento_pesos"]) == (t["aquecimento"], t["decaimento_pesos"])
    assert kw["n_min"] == exp["n_min"]  # dá o número de etiquetas, o tamanho da camada de saída
    assert kw["modelo"] == exp["modelo"]["nome_curto"]
    assert Path(kw["experiencia"]).resolve() == YAML


def test_comando_do_sinal_decorar_passa_os_valores_do_yaml(exp, monkeypatch):
    s = exp["sinal_decorar"]
    fingido = {"acertos": 100, "contratos": 100, "criterio_acertos": 95, "passou": True}
    _, kw = _argumentos_do_run(s["comando"], sinal_decorar, monkeypatch, devolve=fingido)
    assert kw["taxa_aprendizagem"] == s["taxa_aprendizagem"]
    assert (kw["n"], kw["epocas"]) == (s["n_contratos"], s["epocas_max"])
    assert kw["semente"] == s["semente"] == s["semente_amostra"]  # o script usa uma só semente
    assert (kw["lote"], kw["aquecimento"], kw["decaimento_pesos"]) == (s["lote"], s["aquecimento"],
                                                                      s["decaimento_pesos"])
    assert kw["n_min"] == exp["n_min"]
    assert kw["max_length"] == exp["modelo"]["max_length"]
    assert kw["modelo"] == exp["modelo"]["nome_curto"]
    assert Path(kw["experiencia"]).resolve() == YAML
