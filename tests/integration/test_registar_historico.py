"""Teste de integração de scripts/registar_historico.py: treinos guardados -> runs no MLflow.

Prepara contratos inventados, treina o TF-IDF numa pasta como as de `metrics/` (um spike, uma
opção, dois τ), junta uma curva por época e um treino com outra janela, e regista tudo num
MLflow temporário. Verifica que há um run por ficheiro de sugestões com as métricas da avaliação
comum, que a curva fica com uma métrica por época, que a régua entra, que o treino de outra
janela usa a validação dessa janela (ou fica de fora se ela não existir) e que correr outra vez
não duplica.
"""

import importlib.util
import json
import shutil
from pathlib import Path

import mlflow
import pandas as pd
import pytest

import evaluation as ev
import train_tfidf as tt

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "registar_historico.py"


@pytest.fixture(scope="module")
def rh():
    spec = importlib.util.spec_from_file_location("registar_historico", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def guardado(tmp_path_factory, run_pipeline, contratos_regua):
    base = tmp_path_factory.mktemp("historico")
    run_pipeline(base, contratos_regua)
    dados = base / "out"
    metrics = base / "metrics"
    opcao = metrics / "e4_xx" / "opcao_a"
    tt.run(dados / "treino.parquet", dados / "validacao.parquet", opcao, min_df=1)
    # Curva por época: duas "épocas" com os ficheiros de τ = 0 e τ escolhido
    curva = metrics / "e4_s5" / "curva"
    curva.mkdir(parents=True)
    shutil.copy(opcao / tt.SUGESTOES, curva / "sugestoes_epoca_01.parquet")
    shutil.copy(opcao / tt.suggestions_file(tt.tfidf.TAU), curva / "sugestoes_epoca_02.parquet")
    (curva / "curva.json").write_text(json.dumps({"n_min": 10, "alpha": 1e-7, "tau": 0.25, "epocas": 2}))
    # Outra janela (desde 2019): o mesmo treino, com outras datas no treino.json
    janela = metrics / "e4_yy" / "desde_2019"
    shutil.copytree(opcao, janela)
    resumo = json.loads((janela / tt.RESUMO).read_text(encoding="utf-8"))
    resumo["treino"]["datas"][0] = "2019-01-02"
    (janela / tt.RESUMO).write_text(json.dumps(resumo), encoding="utf-8")
    return {"base": base, "dados": dados, "metrics": metrics, "opcao": opcao}


def _registar(rh, guardado, tmp_path, janelas=None):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    novos = rh.run(guardado["metrics"], guardado["dados"], janelas or tmp_path / "sem_janelas", uri, verbose=False)
    mlflow.set_tracking_uri(uri)
    runs = mlflow.search_runs(experiment_names=[rh.EXPERIENCIA]).set_index("tags.mlflow.runName")
    return novos, runs, uri


def test_um_run_por_ficheiro_de_sugestoes_mais_a_regua_e_a_curva(rh, guardado, tmp_path):
    novos, runs, _ = _registar(rh, guardado, tmp_path)
    assert sorted(novos) == sorted(["e4_xx/opcao_a/tau0", f"e4_xx/opcao_a/tau{tt.tfidf.TAU:g}", "regua", "e4_s5/curva"])
    assert runs.loc["e4_xx/opcao_a/tau0", "params.tau"] == "0.0"
    assert runs.loc["e4_xx/opcao_a/tau0", "params.spike"] == "e4_xx"


def test_metricas_sao_as_da_avaliacao_comum(rh, guardado, tmp_path):
    _, runs, _ = _registar(rh, guardado, tmp_path)
    val = pd.read_parquet(guardado["dados"] / "validacao.parquet")
    nome = tt.suggestions_file(tt.tfidf.TAU)
    sug = pd.read_parquet(guardado["opcao"] / nome)
    rep = ev.report(pd.Series([list(s) for s in sug["sugestoes"]], index=val.index), val["codigos"], val["visto_no_treino"])
    run = runs.loc[f"e4_xx/opcao_a/tau{tt.tfidf.TAU:g}"]
    for (grupo, medida), g in [(("todos", "codigo"), "todos"), (("texto novo", "classe"), "texto_novo")]:
        assert run[f"metrics.top5_{medida}_{g}"] == pytest.approx(rep.loc[(grupo, medida), "top5"])
        assert run[f"metrics.top1_{medida}_{g}"] == pytest.approx(rep.loc[(grupo, medida), "top1"])


def test_curva_tem_uma_metrica_por_epoca(rh, guardado, tmp_path):
    _, runs, uri = _registar(rh, guardado, tmp_path)
    hist = mlflow.MlflowClient(uri).get_metric_history(runs.loc["e4_s5/curva", "run_id"], "top5_codigo_todos")
    assert sorted(m.step for m in hist) == [1, 2]


def test_outra_janela_fica_de_fora_sem_a_sua_validacao_e_entra_com_ela(rh, guardado, tmp_path):
    novos, _, _ = _registar(rh, guardado, tmp_path / "sem")
    assert not any(n.startswith("e4_yy") for n in novos)
    janelas = tmp_path / "janelas"
    (janelas / "2019").mkdir(parents=True)
    # A validação da janela: os mesmos contratos, com todos os textos marcados como vistos
    val = pd.read_parquet(guardado["dados"] / "validacao.parquet")
    val.assign(visto_no_treino=True).to_parquet(janelas / "2019" / "validacao.parquet", index=False)
    novos, runs, _ = _registar(rh, guardado, tmp_path / "com", janelas)
    run = runs.loc["e4_yy/desde_2019/tau0"]
    assert run["params.treino_inicio"] == "2019-01-02"
    assert run["metrics.contratos_codigo_texto_novo"] == 0  # leu a validação da janela, não a de 2022


def test_correr_outra_vez_nao_duplica(rh, guardado, tmp_path):
    _registar(rh, guardado, tmp_path)
    novos, runs, _ = _registar(rh, guardado, tmp_path)
    assert novos == []
    assert len(runs) == 4


def test_sugestoes_por_outra_ordem_dao_erro(rh, guardado, tmp_path):
    val = pd.read_parquet(guardado["dados"] / "validacao.parquet", columns=rh.COLUNAS_VAL)
    baralhado = tmp_path / "baralhado.parquet"
    pd.read_parquet(guardado["opcao"] / tt.SUGESTOES).iloc[::-1].to_parquet(baralhado, index=False)
    with pytest.raises(rh.mg.GateError, match="ordem"):
        rh.evaluate_file(baralhado, val)


def test_cada_metrica_do_relatorio_vai_para_o_seu_nome(rh):
    # Relatório feito à mão, com um número diferente em cada célula, no formato do `ev.report`
    linhas = [(g, m) for g in ("todos", "texto novo") for m in (*ev.MEASURES, "so_divisao")]
    rep = pd.DataFrame([{"contratos": 10 * i + 1, "top1": i / 100, "top5": i / 100 + 0.5, "f1_macro": i / 100 + 0.25}
                        for i in range(len(linhas))],
                       index=pd.MultiIndex.from_tuples(linhas, names=["grupo", "medida"]))
    out = rh.report_metrics(rep)
    assert len(out) == 2 * len(ev.MEASURES) * 4  # sem a linha so_divisao
    for i, (g, m) in enumerate(linhas):
        if m == "so_divisao":
            continue
        nome = f"{m}_{'todos' if g == 'todos' else 'texto_novo'}"
        assert out[f"top1_{nome}"] == i / 100
        assert out[f"top5_{nome}"] == i / 100 + 0.5
        assert out[f"f1_macro_{nome}"] == i / 100 + 0.25
        assert out[f"contratos_{nome}"] == 10 * i + 1


def test_ficheiro_de_sugestoes_sem_tau_no_nome_fica_de_fora(rh, guardado, tmp_path):
    estranho = guardado["opcao"] / "sugestoes_validacao_velho.parquet"
    shutil.copy(guardado["opcao"] / tt.SUGESTOES, estranho)
    try:
        novos, _, _ = _registar(rh, guardado, tmp_path)
    finally:
        estranho.unlink()
    assert not any("velho" in n for n in novos)
    assert "e4_xx/opcao_a/tau0" in novos
