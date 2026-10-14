"""Teste de integração do portão de qualidade (E4-T5): dos .xlsx inventados ao campeão gravado.

Corre a preparação e o treino do TF-IDF (como em test_tfidf_pipeline.py) sobre os contratos da
régua (tests/conftest.py) e passa a pasta do treino ao portão (`model_gate.run`), com o campeão e
o registo do MLflow em pastas temporárias. Verifica que:
- sem campeão, o modelo passa a campeão, com as métricas do modelo por defeito (τ escolhido);
- o mesmo modelo outra vez passa (empate) e o ficheiro fica igual;
- um campeão melhor recusa o modelo, e o ficheiro do campeão não muda;
- cada avaliação fica um run no MLflow, com o resultado do portão;
- o teste (`teste.parquet`) é recusado.
"""

import json
import sys

import mlflow
import pandas as pd
import pytest
from mlflow.models.evaluation.validation import ModelValidationFailedException

import evaluation as ev
import model_gate as mg
import train_tfidf as tt


@pytest.fixture(scope="module")
def treino(tmp_path_factory, run_pipeline, contratos_regua):
    base = tmp_path_factory.mktemp("portao")
    run_pipeline(base, contratos_regua)
    paths = {"treino": base / "out" / "treino.parquet", "validacao": base / "out" / "validacao.parquet",
             "pasta": base / "metrics"}
    tt.run(paths["treino"], paths["validacao"], paths["pasta"], min_df=1)
    return paths


@pytest.fixture
def portao(treino, tmp_path):
    """Corre o portão com o campeão e o MLflow em `tmp_path`; devolve (função, caminho do campeão, uri)."""
    campeao, uri = tmp_path / "validacao.json", f"sqlite:///{tmp_path / 'mlflow.db'}"

    def correr(pasta=treino["pasta"], val=treino["validacao"]):
        return mg.run(pasta, val, campeao_path=campeao, tracking_uri=uri)
    return correr, campeao, uri


def _runs(uri):
    mlflow.set_tracking_uri(uri)
    return mlflow.search_runs(experiment_names=["portao_validacao"], order_by=["start_time ASC"])


def test_sem_campeao_o_modelo_passa_a_campeao(treino, portao):
    correr, campeao, uri = portao
    gravado = correr()
    assert json.loads(campeao.read_text(encoding="utf-8")) == gravado
    assert gravado["conjunto"] == "validacao"
    assert gravado["periodo"] == ["2025-10-02", "2025-10-05"]
    assert set(gravado["metricas"]) == set(mg.METRICAS)
    assert _runs(uri)["tags.portao"].tolist() == ["aceite"]


def test_run_do_mlflow_leva_as_metricas_os_parametros_e_o_custo_do_treino(treino, portao):
    correr, _, uri = portao
    gravado = correr()
    resumo = json.loads((treino["pasta"] / tt.RESUMO).read_text(encoding="utf-8"))
    run = _runs(uri).iloc[0]
    for m, v in gravado["metricas"].items():
        assert run[f"metrics.{m}"] == pytest.approx(v)
    assert run["params.n_min"] == str(resumo["n_min"])
    assert run["params.epocas"] == str(resumo["epocas"])
    assert run["params.tau"] == str(tt.tfidf.TAU)
    assert run["params.conjunto"] == "validacao"
    assert run["metrics.treino_vocabulario"] == resumo["vocabulario"]
    assert run["metrics.treino_etiquetas"] == resumo["etiquetas"]


def test_avalia_o_ficheiro_do_modelo_por_defeito_e_nao_o_de_tau_0(treino, portao):
    correr, _, _ = portao
    gravado = correr()
    assert gravado["sugestoes"].endswith(tt.suggestions_file(tt.tfidf.TAU))
    val = pd.read_parquet(treino["validacao"])
    sug = pd.read_parquet(treino["pasta"] / tt.suggestions_file(tt.tfidf.TAU))
    report = ev.report(pd.Series([list(s) for s in sug["sugestoes"]], index=val.index),
                       val["codigos"], val["visto_no_treino"])
    assert gravado["metricas"] == mg.gate_metrics(report)


def test_mesmo_modelo_outra_vez_passa_e_o_ficheiro_fica_igual(portao):
    correr, campeao, uri = portao
    correr()
    antes = campeao.read_bytes()
    correr()
    assert campeao.read_bytes() == antes
    assert _runs(uri)["tags.portao"].tolist() == ["aceite", "aceite"]


@pytest.mark.parametrize("metrica", mg.METRICAS)
def test_campeao_melhor_recusa_o_modelo_e_nao_muda_o_ficheiro(portao, metrica):
    correr, campeao, uri = portao
    gravado = correr()
    gravado["metricas"][metrica] += 0.001  # o campeão passa a ser um bocadinho melhor nesta métrica
    campeao.write_text(json.dumps(gravado), encoding="utf-8")
    antes = campeao.read_bytes()
    with pytest.raises(ModelValidationFailedException, match=metrica):
        correr()
    assert campeao.read_bytes() == antes
    runs = _runs(uri)
    assert runs["tags.portao"].tolist() == ["aceite", "recusado"]
    assert runs["status"].tolist() == ["FINISHED", "FAILED"]


def test_campeao_de_outro_conjunto_da_erro(portao):
    correr, campeao, _ = portao
    gravado = correr()
    gravado["conjunto"] = "teste"
    campeao.write_text(json.dumps(gravado), encoding="utf-8")
    with pytest.raises(mg.GateError, match="validação"):
        correr()


def test_teste_e_recusado_e_nada_se_grava(treino, portao, tmp_path):
    correr, campeao, _ = portao
    teste = tmp_path / "teste.parquet"
    teste.write_bytes(treino["validacao"].read_bytes())
    with pytest.raises(ValueError, match="teste"):
        correr(val=teste)
    assert not campeao.exists()


def test_sugestoes_por_outra_ordem_dao_erro(treino, portao, tmp_path):
    correr, campeao, _ = portao
    pasta = tmp_path / "baralhado"
    pasta.mkdir()
    (pasta / tt.RESUMO).write_bytes((treino["pasta"] / tt.RESUMO).read_bytes())
    nome = tt.suggestions_file(tt.tfidf.TAU)
    pd.read_parquet(treino["pasta"] / nome).iloc[::-1].to_parquet(pasta / nome, index=False)
    with pytest.raises(mg.GateError, match="ordem"):
        correr(pasta=pasta)
    assert not campeao.exists()


def test_treino_sem_modelo_por_defeito_da_erro(treino, portao, tmp_path):
    correr, _, _ = portao
    pasta = tmp_path / "antigo"
    pasta.mkdir()
    resumo = json.loads((treino["pasta"] / tt.RESUMO).read_text(encoding="utf-8"))
    del resumo["ficheiro_do_modelo_por_defeito"]  # como o treino.json do E4-H1
    (pasta / tt.RESUMO).write_text(json.dumps(resumo), encoding="utf-8")
    with pytest.raises(mg.GateError, match="modelo por defeito"):
        correr(pasta=pasta)


def test_comando_grava_o_campeao_e_o_run(treino, tmp_path, monkeypatch):
    campeao, uri = tmp_path / "validacao.json", f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setattr(sys, "argv", ["model_gate.py", "--pasta", str(treino["pasta"]),
                                      "--validacao", str(treino["validacao"]),
                                      "--campeao", str(campeao), "--tracking", uri])
    mg.main()
    assert json.loads(campeao.read_text(encoding="utf-8"))["modelo"] == "tfidf"
    assert len(_runs(uri)) == 1


def test_campeao_avaliado_noutra_validacao_da_erro_e_nao_muda_o_ficheiro(portao):
    correr, campeao, _ = portao
    gravado = correr()
    gravado["contratos"] += 1  # como se a validação tivesse mudado depois de o campeão ser avaliado
    campeao.write_text(json.dumps(gravado), encoding="utf-8")
    antes = campeao.read_bytes()
    with pytest.raises(mg.GateError, match="contratos"):
        correr()
    assert campeao.read_bytes() == antes


def test_copia_do_teste_com_outro_nome_e_recusada_pelas_datas(treino, portao, tmp_path):
    correr, campeao, _ = portao
    copia = tmp_path / "outubro.parquet"
    copia.write_bytes((treino["validacao"].parent / "teste.parquet").read_bytes())
    with pytest.raises(ValueError, match="teste"):
        correr(val=copia)
    assert not campeao.exists()
