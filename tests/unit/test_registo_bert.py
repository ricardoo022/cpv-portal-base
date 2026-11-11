"""Testes do registo de um treino do BERT no MLflow (E5-T3), sem dados reais: correm no CI.

Cada teste faz à mão, num `tmp_path`, uma validação minúscula (4 contratos, 2 de texto novo), o
`treino.parquet` ao lado (o `check_window` lê-o) e a pasta `--saida` de um treino com 2 épocas,
com sugestões escolhidas para se saber o top-1 e o top-5 de cada época. O MLflow escreve sempre
num `mlflow.db` temporário, nunca no da raiz: um só para o módulo (criar a base de dados do MLflow
demora uns segundos), com uma experiência por teste, para os testes não se verem uns aos outros.
"""

import json
import shutil

import mlflow
import pandas as pd
import pytest
from mlflow import MlflowClient

import model_gate as mg
import registo_bert as rb
from cpv_data import file_sha256

# Os 4 contratos da validação: a e b com texto visto no treino, c e d com texto novo
IDS = ["a", "b", "c", "d"]
CODIGOS = ["45233200", "33141100", "09134100", "72212000"]
VISTO = [True, True, False, False]
OUTROS = ["03111000", "03112000", "03113000", "03114000", "03115000"]  # nenhum acerta, nem na classe

# Sugestões por época e por τ: para cada contrato, (posição do código certo, ou None, e um código
# só da mesma classe, ou None). Valores à mão por baixo.
SUG = {
    1: {"sugestoes_validacao_tau0.25.parquet": [(3, None), (None, None), (None, None), (None, None)],
        "sugestoes_validacao.parquet": [(None, None), (None, None), (None, None), (None, None)]},
    2: {"sugestoes_validacao_tau0.25.parquet": [(1, None), (1, None), (2, None), (None, "72210000")],
        "sugestoes_validacao.parquet": [(1, None), (1, None), (1, None), (5, None)]},
}
# τ = 0,25 (o modelo por defeito)
TOP5_CODIGO_TODOS = {1: 0.25, 2: 0.75}
TOP1_CODIGO_TODOS = {1: 0.0, 2: 0.5}
TOP5_CODIGO_NOVO = {1: 0.0, 2: 0.5}
TOP5_CLASSE_TODOS = {1: 0.25, 2: 1.0}
# τ = 0
TOP5_CODIGO_TODOS_TAU0 = {1: 0.0, 2: 1.0}
TOP1_CODIGO_TODOS_TAU0 = {1: 0.0, 2: 0.75}

PERDA = {1: 2.5, 2: 1.25}
PERDA_VALIDACAO = {1: 3.0, 2: 3.5}  # sobe com o top-5 a melhorar: a perda é só informação (E5-T7)
N_CONTRATOS_PERDA = {1: 3, 2: 3}  # um dos 4 contratos com código fora das etiquetas não entra na média
TREINO_INICIO = "2022-01-03"
YAML = "spike: e5_s1\nopcoes: [bertimbau]\n"


def _sugestoes(linhas):
    out = []
    for codigo, (pos, classe) in zip(CODIGOS, linhas):
        s = list(OUTROS)
        if classe is not None:
            s[0] = classe
        if pos is not None:
            s[pos - 1] = codigo
        out.append(s)
    return pd.DataFrame({"idcontrato": IDS, "sugestoes": out})


def fazer_validacao(pasta, visto=VISTO):
    pasta.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"idcontrato": IDS, "data": pd.to_datetime(["2025-10-01", "2025-10-10", "2025-10-20", "2025-10-31"]),
                  "codigos": CODIGOS, "visto_no_treino": visto}).to_parquet(pasta / "validacao.parquet", index=False)
    pd.DataFrame({"data": pd.to_datetime([TREINO_INICIO, "2025-09-30"])}).to_parquet(pasta / "treino.parquet", index=False)
    return pasta / "validacao.parquet"


def resumo_de(val_path, **muda):
    por_epoca = [{"epoca": k, "checkpoint": f"/w/ck/checkpoint-{10 * k}", "pontuacoes": f"epoca_{k}.npy",
                  "segundos_desde_o_inicio": 100.0 * k, "perda_treino": PERDA[k],
                  "perda_validacao": PERDA_VALIDACAO[k], "n_contratos_perda": N_CONTRATOS_PERDA[k]} for k in (1, 2)]
    r = {
        "data": "2026-10-15T10:00:00",
        "modelo": {"chave": "bertimbau", "nome": "neuralmind/bert-base-portuguese-cased", "revisao": "abc123"},
        "max_length": 256, "n_min": 10, "taus": [0.0, 0.25],
        "ficheiros": {"sugestoes_validacao.parquet": 0.0, "sugestoes_validacao_tau0.25.parquet": 0.25},
        "ficheiro_do_modelo_por_defeito": "sugestoes_validacao_tau0.25.parquet",
        "treino": {"ficheiro": "treino.parquet", "datas": [TREINO_INICIO, "2025-09-30"]},
        "validacao": {"ficheiro": "validacao.parquet", "datas": ["2025-10-01", "2025-10-31"], "contratos": 4},
        "exemplos_usados": 900, "exemplos_saidos": 12, "etiquetas": 30,
        "taxa_aprendizagem": 2e-5, "lote": 32, "epocas": 2, "aquecimento": 0.1, "decaimento_pesos": 0.01,
        "semente": 0, "dispositivo": "cuda", "gpu": "NVIDIA A40", "n_gpus": 1,
        "checkpoints": "/w/ck", "pontuacoes": "/w/pt", "por_epoca": por_epoca,
        "retomado_depois_da_epoca": None, "tempo_treinar_s": 210.0, "pico_rss_mb": 4000, "nucleos": 8,
        "versoes": {"python": "3.10.12", "torch": "2.14.1"},
        "sha256": {"treino": file_sha256(val_path.parent / "treino.parquet"), "validacao": file_sha256(val_path)},
        "git": {"commit": "d836321", "alteracoes_por_gravar": False},
        "experiencia": {"ficheiro": "experiencias/e5_s1.yaml", "sha256": "e" * 64, "conteudo": YAML},
    }
    r.update(muda)
    return r


def fazer_treino(pasta, val_path, sug=SUG, **muda):
    pasta.mkdir(parents=True)
    resumo = resumo_de(val_path, **muda)
    for k, ficheiros in sug.items():
        (pasta / f"epoca_{k}").mkdir()
        for nome, linhas in ficheiros.items():
            _sugestoes(linhas).to_parquet(pasta / f"epoca_{k}" / nome, index=False)
    for nome in sug[max(sug)]:
        shutil.copyfile(pasta / f"epoca_{max(sug)}" / nome, pasta / nome)
    (pasta / "treino.json").write_text(json.dumps(resumo), encoding="utf-8")
    return pasta


@pytest.fixture(scope="module")
def uri(tmp_path_factory):
    return f"sqlite:///{tmp_path_factory.mktemp('mlflow') / 'mlflow.db'}"


@pytest.fixture
def base(tmp_path, uri, request):
    val = fazer_validacao(tmp_path / "dados")
    return {"val": val, "treino": fazer_treino(tmp_path / "bertimbau_lr2e-5", val), "uri": uri,
            "exp": request.node.name}


def registar(base, pasta=None, val=None):
    return rb.log_training(pasta or base["treino"], val or base["val"], tracking_uri=base["uri"],
                           experiencia=base["exp"])


def _runs(uri, experiencia):
    mlflow.set_tracking_uri(uri)
    exp = mlflow.get_experiment_by_name(experiencia)
    return [] if exp is None else MlflowClient(uri).search_runs([exp.experiment_id])


# --- O run e o que leva ---

def test_regista_parametros_tags_e_metricas_do_treino_json(base):
    run_id = registar(base)
    run = MlflowClient(base["uri"]).get_run(run_id)
    assert run.info.run_name == "bertimbau_lr2e-5"
    assert run.data.params == {
        "modelo_chave": "bertimbau", "modelo_nome": "neuralmind/bert-base-portuguese-cased",
        "modelo_revisao": "abc123", "max_length": "256", "n_min": "10", "taus": "0 0.25",
        "taxa_aprendizagem": "2e-05", "lote": "32", "epocas": "2", "aquecimento": "0.1",
        "decaimento_pesos": "0.01", "semente": "0", "dispositivo": "cuda", "gpu": "NVIDIA A40", "n_gpus": "1",
        "treino_inicio": TREINO_INICIO, "treino_fim": "2025-09-30",
        "treino_sha256": file_sha256(base["val"].parent / "treino.parquet"),
        "validacao_sha256": file_sha256(base["val"]), "git_commit": "d836321",
        "git_alteracoes_por_gravar": "False", "experiencia_ficheiro": "experiencias/e5_s1.yaml",
        "experiencia_sha256": "e" * 64, "retomado_depois_da_epoca": "nenhum", "tau_por_defeito": "0.25",
        "pasta": str(base["treino"].resolve()), "versao_python": "3.10.12", "versao_torch": "2.14.1",
    }
    assert run.data.tags["treino_json_sha256"] == file_sha256(base["treino"] / "treino.json")
    assert run.data.tags["modelo"] == "bertimbau"
    assert run.data.tags["experiencia_yaml"] == YAML
    m = run.data.metrics  # o último valor de cada métrica (a época 2)
    assert {"treino_tempo_treinar_s": 210.0, "treino_pico_rss_mb": 4000.0, "treino_exemplos_usados": 900.0,
            "treino_exemplos_saidos": 12.0, "treino_etiquetas": 30.0,
            "treino_segundos_por_epoca": 100.0}.items() <= m.items()
    nomes = {f"{top}_{med}_{g}{suf}" for top in ("top1", "top5") for med in ("codigo", "classe")
             for g in ("todos", "texto_novo") for suf in ("", "_tau0")}
    assert set(m) == nomes | {"perda_treino", "perda_validacao", "n_contratos_perda",
                              "treino_tempo_treinar_s", "treino_pico_rss_mb",
                              "treino_exemplos_usados", "treino_exemplos_saidos", "treino_etiquetas",
                              "treino_segundos_por_epoca"}
    assert set(mg.METRICAS) <= set(m)  # o modelo por defeito tem os nomes do portão


def test_campos_em_falta_ficam_nenhum(base, tmp_path):
    pasta = fazer_treino(tmp_path / "t", base["val"], gpu=None, git={"commit": None, "alteracoes_por_gravar": None},
                         experiencia=None)
    run = MlflowClient(base["uri"]).get_run(registar(base, pasta))
    for p in ("gpu", "git_commit", "git_alteracoes_por_gravar", "experiencia_ficheiro", "experiencia_sha256"):
        assert run.data.params[p] == "nenhum"
    assert run.data.tags["experiencia_yaml"] == "nenhum"


def test_segundos_por_epoca_contam_desde_a_retoma(tmp_path):
    # Retomado depois da época 1: a época 2 durou os 100 s do seu `segundos_desde_o_inicio`
    val = fazer_validacao(tmp_path / "dados")
    resumo = resumo_de(val, retomado_depois_da_epoca=1)
    resumo["por_epoca"][1]["segundos_desde_o_inicio"] = 100.0
    assert rb.training_cost(resumo)["treino_segundos_por_epoca"] == 100.0


def test_custo_leva_o_tempo_por_passo_e_a_memoria_da_gpu(tmp_path):
    # Campos do E5-T4; no processador a memória da GPU é None e não dá métrica
    val = fazer_validacao(tmp_path / "dados")
    custo = rb.training_cost(resumo_de(val, segundos_por_passo=0.25, passos_por_epoca=14931,
                                       memoria_gpu_mb={"pico_alocada": 9000, "pico_reservada": 11000}))
    assert {"treino_segundos_por_passo": 0.25, "treino_passos_por_epoca": 14931.0,
            "treino_memoria_gpu_pico_alocada_mb": 9000.0,
            "treino_memoria_gpu_pico_reservada_mb": 11000.0}.items() <= custo.items()
    assert not any("memoria" in k for k in rb.training_cost(resumo_de(val, memoria_gpu_mb=None)))


def test_ensaio_de_custo_recusa_sem_run(base, tmp_path):
    # Um ensaio (--max-passos) na tabela dos treinos passaria por um treino a sério
    pasta = fazer_treino(tmp_path / "ensaio", base["val"], ensaio=True)
    with pytest.raises(rb.RegistoError, match="ensaio de custo"):
        registar(base, pasta)
    assert _runs(base["uri"], base["exp"]) == []


# --- A curva por época ---

def _historia(uri, run_id, metrica):
    return {h.step: h.value for h in MlflowClient(uri).get_metric_history(run_id, metrica)}


def test_curva_tem_uma_entrada_por_epoca_com_o_passo_da_epoca(base):
    run_id = registar(base)
    uri = base["uri"]
    assert _historia(uri, run_id, "top5_codigo_todos") == TOP5_CODIGO_TODOS
    assert _historia(uri, run_id, "top1_codigo_todos") == TOP1_CODIGO_TODOS
    assert _historia(uri, run_id, "top5_codigo_texto_novo") == TOP5_CODIGO_NOVO
    assert _historia(uri, run_id, "top5_classe_todos") == TOP5_CLASSE_TODOS
    assert _historia(uri, run_id, "top5_codigo_todos_tau0") == TOP5_CODIGO_TODOS_TAU0
    assert _historia(uri, run_id, "top1_codigo_todos_tau0") == TOP1_CODIGO_TODOS_TAU0
    assert _historia(uri, run_id, "perda_treino") == PERDA
    assert _historia(uri, run_id, "perda_validacao") == PERDA_VALIDACAO
    assert _historia(uri, run_id, "n_contratos_perda") == N_CONTRATOS_PERDA


def test_treino_sem_perda_na_validacao_regista_sem_ela(base, tmp_path):
    # Os treinos do E5-T4 e do E5-T5 são anteriores ao E5-T7: o `por_epoca` não tem estes campos
    val = base["val"]
    resumo = resumo_de(val)
    for e in resumo["por_epoca"]:
        del e["perda_validacao"], e["n_contratos_perda"]
    pasta = fazer_treino(tmp_path / "antigo", val, por_epoca=resumo["por_epoca"])
    run_id = registar(base, pasta)
    m = MlflowClient(base["uri"]).get_run(run_id).data.metrics
    assert "perda_validacao" not in m and "n_contratos_perda" not in m
    assert _historia(base["uri"], run_id, "perda_treino") == PERDA  # o resto da curva fica igual
    assert _historia(base["uri"], run_id, "top5_codigo_todos") == TOP5_CODIGO_TODOS


def test_perda_validacao_none_nao_da_metrica(tmp_path):
    # Nenhum contrato entrou na média (todos com códigos fora das etiquetas): o treino grava null
    val = fazer_validacao(tmp_path / "dados")
    pasta = fazer_treino(tmp_path / "t", val)
    resumo = json.loads((pasta / "treino.json").read_text(encoding="utf-8"))
    for e in resumo["por_epoca"]:
        e["perda_validacao"], e["n_contratos_perda"] = None, 0
    por_epoca = rb.epoch_metrics(pasta, resumo, pd.read_parquet(val, columns=rb.COLUNAS_VAL))
    assert all("perda_validacao" not in m and m["n_contratos_perda"] == 0.0 for m in por_epoca.values())


# --- Vários treinos na mesma experiência, sem duplicar ---

def test_dois_treinos_ficam_na_mesma_tabela_com_as_mesmas_colunas(base, tmp_path):
    outro = fazer_treino(tmp_path / "albertina", base["val"], semente=1,
                         modelo={"chave": "albertina", "nome": "PORTULAN/albertina", "revisao": "def456"})
    registar(base)
    registar(base, outro)
    mlflow.set_tracking_uri(base["uri"])
    tabela = mlflow.search_runs(experiment_names=[base["exp"]])
    assert sorted(tabela["tags.mlflow.runName"]) == ["albertina", "bertimbau_lr2e-5"]
    params = [c for c in tabela.columns if c.startswith("params.")]
    assert tabela[params].notna().all().all()  # as mesmas colunas, preenchidas nos dois
    assert sorted(tabela["params.modelo_chave"]) == ["albertina", "bertimbau"]
    assert sorted(tabela["params.semente"]) == ["0", "1"]


def test_registar_outra_vez_nao_duplica(base):
    assert registar(base) is not None
    assert registar(base) is None
    assert len(_runs(base["uri"], base["exp"])) == 1


def test_run_que_rebentou_a_meio_nao_conta(base):
    mlflow.set_tracking_uri(base["uri"])
    mlflow.set_experiment(base["exp"])
    with mlflow.start_run(run_name="partido") as partido:
        mlflow.set_tag("treino_json_sha256", file_sha256(base["treino"] / "treino.json"))
    MlflowClient(base["uri"]).set_terminated(partido.info.run_id, status="FAILED")
    assert registar(base) is not None
    assert len(_runs(base["uri"], base["exp"])) == 2


# --- O que se recusa (e não deixa run nenhum) ---

def test_validacao_com_outro_conteudo_recusa(base, tmp_path):
    outra = fazer_validacao(tmp_path / "outra", visto=[True, False, False, False])
    with pytest.raises(rb.RegistoError, match="não é a validação que o pod usou"):
        registar(base, val=outra)
    assert _runs(base["uri"], base["exp"]) == []


def test_treino_com_outro_conteudo_recusa(base):
    # O treino.parquet do portátil não é o do pod (preparado outra vez com outras regras, por exemplo)
    treino = base["val"].parent / "treino.parquet"
    pd.DataFrame({"data": pd.to_datetime([TREINO_INICIO, "2025-09-29"])}).to_parquet(treino, index=False)
    with pytest.raises(rb.RegistoError, match="não é o treino que o pod usou"):
        registar(base)
    assert _runs(base["uri"], base["exp"]) == []


def test_validacao_chamada_teste_recusa(base):
    teste = base["val"].parent / "teste.parquet"
    shutil.copyfile(base["val"], teste)
    with pytest.raises(ValueError, match="o conjunto de teste não se lê"):
        registar(base, val=teste)
    assert _runs(base["uri"], base["exp"]) == []


def test_treino_json_sem_sha256_recusa(base):
    path = base["treino"] / "treino.json"
    resumo = json.loads(path.read_text(encoding="utf-8"))
    del resumo["sha256"]
    path.write_text(json.dumps(resumo), encoding="utf-8")
    with pytest.raises(rb.RegistoError, match="não tem o SHA-256 da validação"):
        registar(base)
    assert _runs(base["uri"], base["exp"]) == []


def test_sugestoes_por_outra_ordem_recusa(base):
    path = base["treino"] / "epoca_2" / "sugestoes_validacao.parquet"
    pd.read_parquet(path).iloc[::-1].to_parquet(path, index=False)
    with pytest.raises(rb.RegistoError, match="pela mesma ordem"):
        registar(base)
    assert _runs(base["uri"], base["exp"]) == []


def test_treino_que_entra_na_validacao_recusa(base, tmp_path):
    pasta = fazer_treino(tmp_path / "t", base["val"], treino={"ficheiro": "treino.parquet",
                                                              "datas": [TREINO_INICIO, "2025-10-05"]})
    with pytest.raises(mg.GateError, match="não acaba antes da validação"):
        registar(base, pasta)
    assert _runs(base["uri"], base["exp"]) == []


def test_treino_de_outra_janela_recusa(base, tmp_path):
    # O texto novo (visto_no_treino) é o da janela da validação: um treino desde 2019 avaliado com
    # esta validação contava como novos textos que viu
    pasta = fazer_treino(tmp_path / "t", base["val"], treino={"ficheiro": "treino.parquet",
                                                              "datas": ["2019-01-02", "2025-09-30"]})
    with pytest.raises(mg.GateError, match="usar a validação da janela"):
        registar(base, pasta)
    assert _runs(base["uri"], base["exp"]) == []


def test_yaml_maior_que_uma_tag_do_mlflow_recusa_sem_run(base, tmp_path):
    grande = {"ficheiro": "experiencias/x.yaml", "sha256": "e" * 64, "conteudo": "a: 1\n" * 2000}  # 10 000 carateres
    pasta = fazer_treino(tmp_path / "t", base["val"], experiencia=grande)
    with pytest.raises(rb.RegistoError, match="8000"):
        registar(base, pasta)
    assert _runs(base["uri"], base["exp"]) == []


def test_epoca_sem_ficheiro_de_sugestoes_recusa_sem_run(base):
    # Por exemplo, a pasta copiada do pod a meio do rsync
    (base["treino"] / "epoca_1" / "sugestoes_validacao.parquet").unlink()
    with pytest.raises(rb.RegistoError, match="falta"):
        registar(base)
    assert _runs(base["uri"], base["exp"]) == []


def test_pasta_sem_treino_json_recusa(base, tmp_path):
    (tmp_path / "vazia").mkdir()
    with pytest.raises(rb.RegistoError, match="não é a saída de um treino"):
        registar(base, tmp_path / "vazia")
    assert _runs(base["uri"], base["exp"]) == []


# --- Linha de comandos ---

def test_main_regista_e_depois_diz_que_ja_estava(base, capsys):
    argv = ["--pasta", str(base["treino"]), "--validacao", str(base["val"]), "--tracking", base["uri"],
            "--experiencia", base["exp"], "--nome", "pela_linha"]
    rb.main(argv)
    assert "Registado" in capsys.readouterr().out
    rb.main(argv)
    assert "Já estava registado" in capsys.readouterr().out
    runs = _runs(base["uri"], base["exp"])
    assert [r.info.run_name for r in runs] == ["pela_linha"]
