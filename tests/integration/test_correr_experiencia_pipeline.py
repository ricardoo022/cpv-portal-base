"""Teste de integração de scripts/correr_experiencia.py (spike E4-S5): YAML -> treinos -> ficheiros.

Corre as duas fases com um treino e uma validação minúsculos, através dos scripts reais
(`src/train_tfidf.py` e `scripts/curva_epocas.py`). Guarda a ligação entre o YAML e os dois
scripts: uma opção com nome trocado só se descobria no pod, a pagar.
"""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[2]
SCRIPT = RAIZ / "scripts" / "correr_experiencia.py"

EXP = """spike: teste
n_min: 5
saida: {saida}
alpha: {{em_vigor: 1.0e-5, candidatos: [1.0e-6]}}
tau: {{em_vigor: 0.0, candidatos: [0.5]}}
epocas: {{em_vigor: null, candidatos: [2, 3]}}
curva_epocas: 2
"""


@pytest.fixture(scope="module")
def corrido(tmp_path_factory, temas_decorar):
    base = tmp_path_factory.mktemp("experiencia")
    # Cada código em pelo menos 5 exemplos: com N = 5 nenhum sobe para a classe.
    linhas = [(c, t) for c, textos in temas_decorar.items() for t in textos for _ in range(3)]
    pd.DataFrame({"data": pd.Timestamp("2024-03-01"), "texto": [t for _, t in linhas],
                  "codigo": [c for c, _ in linhas]}).to_parquet(base / "treino.parquet", index=False)
    pd.DataFrame({"idcontrato": ["v1", "v2", "v3"], "data": pd.Timestamp("2025-10-02"),
                  "texto": ["Limpeza das escolas", "Asfalto novo", "Papel"]}
                 ).to_parquet(base / "validacao.parquet", index=False)
    saida = base / "saida"
    exp = base / "exp.yaml"
    exp.write_text(EXP.format(saida=saida), encoding="utf-8")
    dados = ["--treino", str(base / "treino.parquet"), "--validacao", str(base / "validacao.parquet")]
    fases = {}
    for fase, extra in (("alpha", []), ("epocas", ["--alpha", "1e-5", "--tau", "0.5"])):
        fases[fase] = subprocess.run(
            [sys.executable, str(SCRIPT), str(exp), fase, *extra, "--paralelo", "2", "--n-jobs", "1", *dados],
            cwd=RAIZ, capture_output=True, text=True)
    return saida, fases


def test_as_duas_fases_acabam_sem_erros(corrido):
    _, fases = corrido
    for fase, r in fases.items():
        assert r.returncode == 0, f"{fase}: {r.stdout}\n{r.stderr}"
        assert "falharam: nenhum" in r.stdout


def test_fase_alpha_grava_um_treino_por_alpha_com_um_ficheiro_por_tau(corrido):
    saida, _ = corrido
    for alpha in ("1e-05", "1e-06"):
        pasta = saida / f"alpha_{alpha}"
        info = json.loads((pasta / "treino.json").read_text(encoding="utf-8"))
        assert info["alpha"] == float(alpha)
        assert info["epocas"] is None and info["n_min"] == 5
        assert info["treino"]["datas"] == ["2024-03-01", "2024-03-01"]  # o treino pedido, não o real
        assert (pasta / "sugestoes_validacao.parquet").exists()
        assert (pasta / "sugestoes_validacao_tau0.5.parquet").exists()


def test_fase_epocas_grava_um_treino_por_numero_de_epocas_e_a_curva(corrido):
    saida, _ = corrido
    for k in (2, 3):
        info = json.loads((saida / f"epocas_{k:02d}" / "treino.json").read_text(encoding="utf-8"))
        assert info["epocas"] == k and info["n_iter"] == [k] and info["alpha"] == 1e-5
        assert (saida / f"epocas_{k:02d}" / "sugestoes_validacao_tau0.5.parquet").exists()
    curva = json.loads((saida / "curva" / "curva.json").read_text(encoding="utf-8"))
    assert curva["epocas"] == 2 and curva["alpha"] == 1e-5 and curva["tau"] == 0.5
    assert sorted(p.name for p in (saida / "curva").glob("sugestoes_epoca_*.parquet")) == \
        ["sugestoes_epoca_01.parquet", "sugestoes_epoca_02.parquet"]


def test_cada_treino_tem_o_seu_log_com_inicio_e_fim(corrido):
    saida, _ = corrido
    nomes = ["alpha_1e-05", "alpha_1e-06", "epocas_02", "epocas_03", "curva"]
    for nome in nomes:
        log = (saida / f"pod_{nome}.log").read_text(encoding="utf-8")
        assert log.startswith(f"INICIO {nome}") and f"FIM {nome}" in log and "codigo=0" in log


def test_treino_que_falha_faz_o_comando_falhar(tmp_path):
    exp = tmp_path / "exp.yaml"
    exp.write_text(EXP.format(saida=tmp_path / "saida"), encoding="utf-8")
    r = subprocess.run([sys.executable, str(SCRIPT), str(exp), "alpha", "--n-jobs", "1",
                        "--treino", str(tmp_path / "nao_existe.parquet"),
                        "--validacao", str(tmp_path / "nao_existe.parquet")],
                       cwd=RAIZ, capture_output=True, text=True)
    assert r.returncode == 1
    assert "falharam: ['alpha_1e-05', 'alpha_1e-06']" in r.stdout
