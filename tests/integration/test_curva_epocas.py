"""Teste de integração do script da curva por época (spike E4-S5, `scripts/curva_epocas.py`).

Cria um treino e uma validação minúsculos em `tmp_path` (os temas de `TEMAS_DECORAR`) e corre o
script com N = 1. Cada texto repete-se em 2 a 8 contratos, conforme o código: pelo menos 2 para
passar o `min_df=2`, e um número diferente por código para as frequências das etiquetas serem
desiguais e a correção das etiquetas frequentes (τ) mudar alguma sugestão. Verifica:
- um ficheiro de sugestões por época, uma linha por contrato da validação pela mesma ordem, 5
  códigos distintos das etiquetas do treino;
- a 1.ª época é igual a um `tfidf.train(..., epocas=1)`: numa só época a ordem dos
  exemplos do `partial_fit` e do `fit` é a mesma, por isso o modelo e as sugestões coincidem;
- o passo da aprendizagem continua de época para época (`t_` = 1 + épocas × exemplos);
- correr duas vezes dá os mesmos ficheiros;
- o teste (`teste.parquet`) é recusado.
"""

import importlib.util
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import tfidf
from labels import collapse_rare_codes

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "curva_epocas.py"
N_MIN = 1
EPOCAS = 3

VALIDACAO = [
    ("v-limpeza", "Limpeza das escolas"),                # texto visto no treino
    ("v-estradas", "Asfalto para estradas novas"),
    ("v-viaturas", "Viaturas elétricas para a câmara"),
    ("v-seguro", "Seguro de acidentes pessoais"),
    ("v-nada", "zzpalavra sem nada em comum"),           # nenhum termo do vocabulário
    ("v-cantina", "Refeições e almoços na cantina"),
]


def _carregar_script():
    spec = importlib.util.spec_from_file_location("curva_epocas", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ce = _carregar_script()


@pytest.fixture(scope="module")
def dados(tmp_path_factory, temas_decorar):
    base = tmp_path_factory.mktemp("curva")
    linhas = [(codigo, texto) for i, (codigo, textos) in enumerate(temas_decorar.items())
              for texto in textos for _ in range(2 + i)]
    treino = pd.DataFrame({
        "data": pd.Timestamp("2024-03-01"),
        "texto": [t for _, t in linhas],
        "codigo": [c for c, _ in linhas],
    })
    val = pd.DataFrame({
        "idcontrato": [i for i, _ in VALIDACAO],
        "data": [pd.Timestamp(datetime(2025, 10, d)) for d in range(1, len(VALIDACAO) + 1)],
        "texto": [t for _, t in VALIDACAO],
    })
    paths = {"treino": base / "treino.parquet", "validacao": base / "validacao.parquet"}
    treino.to_parquet(paths["treino"], index=False)
    val.to_parquet(paths["validacao"], index=False)
    return {**paths, "base": base, "treino_df": treino, "val_df": val}


def _correr(dados, saida, *extra):
    ce.main(["--treino", str(dados["treino"]), "--validacao", str(dados["validacao"]),
             "--saida", str(saida), "--n-min", str(N_MIN), "--epocas", str(EPOCAS), *extra])


@pytest.fixture(scope="module")
def corrido(dados):
    saida = dados["base"] / "saida"
    _correr(dados, saida)
    return saida


def _sugestoes(saida, epoca):
    return pd.read_parquet(saida / ce.suggestions_file(epoca))


# --- Um ficheiro por época, no formato do train_tfidf.py ---

def test_um_ficheiro_por_epoca_pela_ordem_da_validacao(corrido, dados):
    assert sorted(p.name for p in corrido.glob("sugestoes_epoca_*.parquet")) == [
        f"sugestoes_epoca_{e:02d}.parquet" for e in range(1, EPOCAS + 1)]
    etiquetas = set(collapse_rare_codes(dados["treino_df"]["codigo"], N_MIN).dropna())
    for e in range(1, EPOCAS + 1):
        sug = _sugestoes(corrido, e)
        assert list(sug.columns) == ["idcontrato", "sugestoes"]
        assert sug["idcontrato"].tolist() == dados["val_df"]["idcontrato"].tolist()
        for s in sug["sugestoes"]:
            s = list(s)
            assert len(s) == 5 and len(set(s)) == 5
            assert all(isinstance(c, str) for c in s)
            assert set(s) <= etiquetas


def test_curva_json(corrido, dados):
    with open(corrido / ce.RESUMO, encoding="utf-8") as f:
        resumo = json.load(f)
    assert resumo["alpha"] == tfidf.ALPHA and resumo["tau"] == tfidf.TAU
    assert resumo["n_min"] == N_MIN and resumo["epocas"] == EPOCAS
    assert resumo["validacao"]["contratos"] == len(VALIDACAO)
    assert resumo["validacao"]["datas"] == ["2025-10-01", f"2025-10-{len(VALIDACAO):02d}"]
    assert resumo["etiquetas"] == len(dados["treino_df"]["codigo"].unique())
    assert [t["epoca"] for t in resumo["tempos"]] == list(range(1, EPOCAS + 1))
    assert all({"treino_s", "sugestoes_s"} <= set(t) for t in resumo["tempos"])
    assert "partial_fit" in resumo["nota"]
    # O passo continua de época para época: t_ começa em 1 e soma os exemplos de cada época
    assert resumo["t_final"] == 1 + EPOCAS * len(dados["treino_df"])


# --- A 1.ª época é um fit de uma época ---

@pytest.mark.parametrize("tau", [0.0, tfidf.TAU])
def test_epoca_1_igual_a_um_fit_de_uma_epoca(dados, tmp_path, tau):
    clf = ce.run(dados["treino"], dados["validacao"], tmp_path, epocas=1, tau=tau, n_min=N_MIN)
    tr, val = dados["treino_df"], dados["val_df"]
    model = tfidf.train(tr["texto"], tr["codigo"], n_min=N_MIN, epocas=1)
    ref = model.named_steps["clf"]
    np.testing.assert_array_equal(clf.classes_, ref.classes_)
    np.testing.assert_array_equal(clf.coef_, ref.coef_)
    np.testing.assert_array_equal(clf.intercept_, ref.intercept_)
    esperado = tfidf.suggest(model, val["texto"], tau=tau)
    if tau != 0:  # os dados distinguem τ: sem isto, um script que ignorasse o τ passava
        assert esperado.tolist() != tfidf.suggest(model, val["texto"], tau=0.0).tolist()
    assert [list(s) for s in _sugestoes(tmp_path, 1)["sugestoes"]] == esperado.tolist()


# --- Repetível ---

def test_correr_duas_vezes_da_os_mesmos_ficheiros(corrido, dados):
    outra = dados["base"] / "outra"
    _correr(dados, outra)
    for e in range(1, EPOCAS + 1):
        pd.testing.assert_frame_equal(_sugestoes(corrido, e), _sugestoes(outra, e))


# --- O teste nunca se lê ---

@pytest.mark.parametrize("qual", ["treino", "validacao"])
def test_caminho_do_teste_da_erro(dados, tmp_path, qual):
    teste = tmp_path / "teste.parquet"
    teste.write_bytes(Path(dados[qual]).read_bytes())
    args = {"treino": dados["treino"], "validacao": dados["validacao"], qual: teste}
    with pytest.raises(ValueError, match="teste"):
        ce.main(["--treino", str(args["treino"]), "--validacao", str(args["validacao"]),
                 "--saida", str(tmp_path / "saida"), "--n-min", str(N_MIN), "--epocas", "1"])
    assert not (tmp_path / "saida").exists()


# --- Validação de argumentos ---

@pytest.mark.parametrize("epocas", [0, -1])
def test_epocas_invalidas_dao_erro(dados, tmp_path, epocas):
    with pytest.raises(ValueError, match="--epocas tem de ser >= 1"):
        ce.run(dados["treino"], dados["validacao"], tmp_path / "saida", epocas=epocas, n_min=N_MIN)
    assert not (tmp_path / "saida").exists()


@pytest.mark.parametrize("tau", [-0.1, -1.0])
def test_tau_negativo_da_erro(dados, tmp_path, tau):
    with pytest.raises(ValueError, match="τ tem de ser >= 0"):
        ce.run(dados["treino"], dados["validacao"], tmp_path / "saida", epocas=1, tau=tau, n_min=N_MIN)
    assert not (tmp_path / "saida").exists()

