"""Teste de integração do E4-H1: da preparação ao modelo TF-IDF e às sugestões.

Gera .xlsx inventados, corre a conversão e a preparação (como em test_pipeline.py), treina o
modelo com `train_tfidf.run` sobre o treino e a validação que a preparação escreveu, e verifica
as sugestões gravadas, o resumo do treino e o comando de sugestões (`suggest.py`).

O teste mais importante é o da fuga de dados (critério de aceitação do E4-H1): nenhum contrato
da validação entra no treino do modelo nem nas contagens do N.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

import evaluation as ev
import suggest as sg
import train_tfidf as tt
from labels import collapse_rare_codes

# N = 6: com estes contratos, 33141400 (5 exemplos no treino) é raro, a classe 3314 só com ele
# também, e o exemplo sai. Se a validação (mais 3 contratos 33141400) entrasse nas contagens,
# o 33141400 passava a etiqueta. Ficam 7 etiquetas, chega para 5 sugestões.
N_MIN = 6
SO_VALIDACAO = "zzpalavravalidacao"  # palavra que só aparece nos textos da validação

VALIDACAO = [
    (2025, "v-limpeza", datetime(2025, 10, 2), "Limpeza 0", None, None, "90919300-5 - Limpeza"),  # texto visto
    (2025, "v-papel", datetime(2025, 10, 3), f"Papel {SO_VALIDACAO} A4", None, None, "30197630-1 - Papel"),
    *[(2025, f"v-luvas-{i}", datetime(2025, 10, 4), f"Luvas {SO_VALIDACAO} {i}", None, None, "33141400-0 - Luvas")
      for i in range(3)],
    (2025, "v-obras", datetime(2025, 10, 5), f"Obras {SO_VALIDACAO}", None, None, "45000000-0 - Obras"),
]
TESTE = [(2025, "t-1", datetime(2025, 11, 5), "Papel novo", None, None, "30197630-1 - Papel")]


@pytest.fixture(scope="module")
def treinado(tmp_path_factory, run_pipeline, contratos_regua):
    # O treino dos contratos da régua (tests/conftest.py), com a validação e o teste daqui
    treino = [c for c in contratos_regua if c[2] < datetime(2025, 10, 1)]
    base = tmp_path_factory.mktemp("tfidf")
    run_pipeline(base, treino + VALIDACAO + TESTE)
    paths = {"treino": base / "out" / "treino.parquet", "validacao": base / "out" / "validacao.parquet",
             "saida": base / "metrics", "modelo": base / "models" / "tfidf.joblib"}
    model = tt.run(paths["treino"], paths["validacao"], paths["saida"], model_path=paths["modelo"],
                   n_min=N_MIN, min_df=1)
    return {"model": model, **paths,
            "treino_df": pd.read_parquet(paths["treino"]), "val_df": pd.read_parquet(paths["validacao"]),
            "sugestoes": pd.read_parquet(paths["saida"] / tt.SUGESTOES)}


# --- Sugestões da validação ---

def test_uma_linha_por_contrato_da_validacao_pela_mesma_ordem(treinado):
    sug, val = treinado["sugestoes"], treinado["val_df"]
    assert list(sug.columns) == ["idcontrato", "sugestoes"]
    assert sug["idcontrato"].tolist() == val["idcontrato"].tolist()
    assert len(val) == len(VALIDACAO)


def test_cinco_codigos_distintos_de_8_digitos_do_treino_sem_divisoes(treinado):
    classes = set(treinado["model"].classes_)
    for s in treinado["sugestoes"]["sugestoes"]:
        s = list(s)
        assert len(s) == 5 and len(set(s)) == 5
        assert all(isinstance(c, str) and len(c) == 8 and c.isdigit() for c in s)
        assert set(s) <= classes
        assert not any(c.endswith("000000") for c in s)


def test_sugestoes_avaliam_se_com_a_avaliacao_comum(treinado):
    val = treinado["val_df"]
    sug = pd.Series(treinado["sugestoes"]["sugestoes"].map(list).to_numpy(), index=val.index)
    out = ev.report(sug, val["codigos"], val["visto_no_treino"])
    # Divisão: os 6 contratos da validação; o v-obras (só 45000000) fica em so_divisao
    assert out.loc[("todos", "divisao"), "contratos"] == len(VALIDACAO)
    assert out.loc[("todos", "so_divisao"), "contratos"] == 1
    # O texto visto "Limpeza 0" tem de dar 90919300 em 1.º
    assert treinado["sugestoes"].set_index("idcontrato").loc["v-limpeza", "sugestoes"][0] == "90919300"


# --- Fuga de dados: a validação não entra no treino (critério de aceitação do E4-H1) ---

def test_palavra_so_da_validacao_nao_esta_no_vocabulario(treinado):
    vocab = treinado["model"].named_steps["tfidf"].vocabulary_
    assert treinado["val_df"]["texto"].str.contains(SO_VALIDACAO).sum() == 5  # está mesmo na validação
    assert not any(SO_VALIDACAO in termo for termo in vocab)


def test_etiquetas_contadas_so_no_treino(treinado):
    esperado = collapse_rare_codes(treinado["treino_df"]["codigo"], N_MIN).dropna().unique()
    assert sorted(treinado["model"].classes_) == sorted(esperado)
    # Com a validação nas contagens, o 33141400 teria 8 exemplos (>= N) e seria etiqueta
    assert "33141400" not in treinado["model"].classes_


def test_mudar_a_validacao_nao_muda_o_modelo(treinado, tmp_path):
    outra = treinado["val_df"].copy()
    outra["texto"] = [f"outra palavra {i}" for i in range(len(outra))]
    outra["codigo"] = "33141400"
    outra["codigos"] = "33141400"
    outra.to_parquet(tmp_path / "validacao.parquet", index=False)
    model2 = tt.run(treinado["treino"], tmp_path / "validacao.parquet", tmp_path / "saida", n_min=N_MIN, min_df=1)
    m1, m2 = treinado["model"], model2
    assert m1.named_steps["tfidf"].vocabulary_ == m2.named_steps["tfidf"].vocabulary_
    np.testing.assert_array_equal(m1.named_steps["clf"].coef_, m2.named_steps["clf"].coef_)
    textos = treinado["treino_df"]["texto"]
    assert tt.tfidf.suggest(m1, textos).tolist() == tt.tfidf.suggest(m2, textos).tolist()


# --- O teste nunca se lê ---

@pytest.mark.parametrize("qual", ["treino", "validacao"])
def test_caminho_do_teste_da_erro(treinado, tmp_path, qual):
    teste = tmp_path / "teste.parquet"
    teste.write_bytes(Path(treinado[qual]).read_bytes())
    paths = {"treino": treinado["treino"], "validacao": treinado["validacao"], qual: teste}
    with pytest.raises(ValueError, match="teste"):
        tt.run(paths["treino"], paths["validacao"], tmp_path / "saida", n_min=N_MIN, min_df=1)
    assert not (tmp_path / "saida" / tt.SUGESTOES).exists()


# --- Resumo do treino ---

def test_treino_json_tem_as_chaves_esperadas(treinado):
    with open(treinado["saida"] / tt.RESUMO, encoding="utf-8") as f:
        resumo = json.load(f)
    assert set(resumo) == {
        "data", "n_min", "min_df", "alpha", "taus", "ficheiros", "ficheiro_do_modelo_por_defeito", "treino", "validacao", "exemplos_usados", "exemplos_saidos",
        "etiquetas", "vocabulario", "epocas", "tol", "max_iter", "n_iter", "convergiu", "tempo_vetorizar_treinar_s",
        "tempo_sugerir_s", "pico_rss_mb", "nucleos", "versoes", "coef_bytes"}
    tr = treinado["treino_df"]
    assert resumo["n_min"] == N_MIN and resumo["min_df"] == 1
    # por defeito, os valores escolhidos no E4-S4, e também τ = 0 para comparar
    assert resumo["alpha"] == tt.tfidf.ALPHA == 1e-7 and resumo["taus"] == [0.0, tt.tfidf.TAU] == [0.0, 0.25]
    assert resumo["ficheiros"] == {"sugestoes_validacao.parquet": 0.0, "sugestoes_validacao_tau0.25.parquet": 0.25}
    assert resumo["ficheiro_do_modelo_por_defeito"] == "sugestoes_validacao_tau0.25.parquet"
    assert resumo["treino"]["ficheiro"] == "treino.parquet"
    assert resumo["validacao"]["ficheiro"] == "validacao.parquet"
    assert resumo["treino"]["datas"] == [str(tr["data"].min().date()), str(tr["data"].max().date())]
    assert resumo["validacao"]["datas"] == ["2025-10-02", "2025-10-05"]
    assert resumo["exemplos_usados"] + resumo["exemplos_saidos"] == len(tr)
    assert resumo["exemplos_saidos"] == 5  # os 5 exemplos de 33141400
    assert resumo["etiquetas"] == len(treinado["model"].classes_) == 7
    assert set(resumo["versoes"]) == {"python", "sklearn", "numpy", "scipy", "pandas"}


def test_modelo_gravado_da_as_mesmas_sugestoes(treinado):
    gravado = joblib.load(treinado["modelo"])
    textos = treinado["val_df"]["texto"]
    # o ficheiro principal é o sem correção (τ = 0); o do τ escolhido tem o seu próprio nome
    assert tt.tfidf.suggest(gravado, textos, tau=0.0).tolist() == treinado["sugestoes"]["sugestoes"].map(list).tolist()
    com_tau = pd.read_parquet(treinado["saida"] / tt.suggestions_file(tt.tfidf.TAU))
    assert tt.tfidf.suggest(gravado, textos).tolist() == com_tau["sugestoes"].map(list).tolist()


def test_comando_de_treino_grava_o_modelo_onde_o_comando_de_sugestoes_o_procura(treinado, tmp_path, monkeypatch):
    # O `make sugerir` lê models/tfidf_n10.joblib: o treino, sem --modelo, tem de o gravar aí
    monkeypatch.setattr(tt, "RAIZ", tmp_path)
    monkeypatch.setattr(sys, "argv", ["train_tfidf.py", "--treino", str(treinado["treino"]),
                                      "--validacao", str(treinado["validacao"]), "--n-min", "10",
                                      "--saida", str(tmp_path / "saida")])
    tt.main()
    assert (tmp_path / "models" / "tfidf_n10.joblib").exists()
    makefile = (Path(tt.__file__).parents[1] / "Makefile").read_text(encoding="utf-8")
    assert "--modelo models/tfidf_n10.joblib" in makefile


# --- Spike E4-S4: alpha e correção das etiquetas frequentes ---

def test_alpha_e_taus_gravam_um_ficheiro_por_tau(treinado, tmp_path):
    taus = [0.0, 0.5, 1.0]
    model = tt.run(treinado["treino"], treinado["validacao"], tmp_path, n_min=N_MIN, min_df=1,
                   alpha=1e-5, taus=taus)
    assert model.named_steps["clf"].alpha == 1e-5
    with open(tmp_path / tt.RESUMO, encoding="utf-8") as f:
        resumo = json.load(f)
    assert resumo["alpha"] == 1e-5 and resumo["taus"] == taus
    tr = pd.read_parquet(treinado["treino"], columns=["codigo"])["codigo"]
    prior = tt.tfidf.log_prior(tr, model.classes_, n_min=N_MIN)
    val = treinado["val_df"]
    for tau in taus:
        lido = pd.read_parquet(tmp_path / tt.suggestions_file(tau))
        assert lido["idcontrato"].tolist() == val["idcontrato"].astype(str).tolist()
        esperado = tt.tfidf.suggest(model, val["texto"], tau=tau, prior=prior)
        assert lido["sugestoes"].map(list).tolist() == esperado.tolist()
    assert tt.suggestions_file(0.0) == tt.SUGESTOES
    assert not (tmp_path / "sugestoes_validacao_tau0.parquet").exists()


def test_treino_por_defeito_grava_so_o_ficheiro_sem_correcao_e_o_do_tau_escolhido(treinado):
    gravados = sorted(p.name for p in treinado["saida"].glob("sugestoes_validacao*.parquet"))
    assert gravados == ["sugestoes_validacao.parquet", "sugestoes_validacao_tau0.25.parquet"]


def test_mudar_a_validacao_nao_muda_as_frequencias_nem_as_sugestoes_corrigidas(treinado, tmp_path):
    # A correção usa a frequência de cada etiqueta: tem de vir só do treino
    outra = treinado["val_df"].copy()
    outra["codigo"] = "33141400"
    outra["codigos"] = "33141400"
    outra.to_parquet(tmp_path / "validacao.parquet", index=False)
    model2 = tt.run(treinado["treino"], tmp_path / "validacao.parquet", tmp_path / "saida", n_min=N_MIN, min_df=1)
    np.testing.assert_array_equal(treinado["model"].log_prior_, model2.log_prior_)
    textos = treinado["treino_df"]["texto"]
    tau = tt.tfidf.TAU
    assert (tt.tfidf.suggest(treinado["model"], textos, tau=tau).tolist()
            == tt.tfidf.suggest(model2, textos, tau=tau).tolist())


def test_comando_de_treino_com_alpha_taus_e_sem_modelo(treinado, tmp_path):
    saida, modelo = tmp_path / "saida", tmp_path / "models" / "m.joblib"
    subprocess.run([sys.executable, str(Path(tt.__file__)), "--treino", str(treinado["treino"]),
                    "--validacao", str(treinado["validacao"]), "--n-min", "5", "--saida", str(saida),
                    "--modelo", str(modelo), "--sem-modelo", "--alpha", "0.001", "--taus", "0", "0.5"],
                   capture_output=True, text=True, check=True)
    assert not modelo.exists()  # --sem-modelo: nada gravado, nem com --modelo
    with open(saida / tt.RESUMO, encoding="utf-8") as f:
        resumo = json.load(f)
    assert resumo["alpha"] == 0.001 and resumo["taus"] == [0.0, 0.5] and resumo["n_min"] == 5
    assert sorted(p.name for p in saida.glob("sugestoes_validacao*.parquet")) == [
        "sugestoes_validacao.parquet", "sugestoes_validacao_tau0.5.parquet"]


def test_comando_de_treino_por_defeito_grava_o_modelo(treinado, tmp_path):
    modelo = tmp_path / "models" / "m.joblib"
    subprocess.run([sys.executable, str(Path(tt.__file__)), "--treino", str(treinado["treino"]),
                    "--validacao", str(treinado["validacao"]), "--n-min", "5", "--saida", str(tmp_path / "s"),
                    "--modelo", str(modelo)], capture_output=True, text=True, check=True)
    gravado = joblib.load(modelo)
    assert gravado.named_steps["clf"].alpha == tt.tfidf.ALPHA


# --- Comando de sugestões ---

def _blocos(texto):
    return [b for b in texto.strip().split("\n\n") if b]


def test_mesma_descricao_duas_vezes_da_o_mesmo(treinado):
    d = "limpeza de escolas"
    out = sg.format_suggestions(treinado["model"], [d, d])
    a, b = _blocos(out)
    assert a == b
    assert sg.format_suggestions(treinado["model"], [d]) == sg.format_suggestions(treinado["model"], [d])


def test_comando_sem_descricoes_mostra_os_3_exemplos_com_5_codigos(treinado):
    r = subprocess.run([sys.executable, str(Path(tt.__file__).with_name("suggest.py")),
                        "--modelo", str(treinado["modelo"])], capture_output=True, text=True, check=True)
    blocos = _blocos(r.stdout)
    assert len(blocos) == 3
    assert "serviços de limpeza de três escolas básicas durante 12 meses" in sg.EXEMPLOS
    for exemplo, bloco in zip(sg.EXEMPLOS, blocos):
        linhas = bloco.splitlines()
        assert linhas[0] == f"Descrição: {exemplo}"
        codigos = [l.split()[1] for l in linhas[1:]]
        assert [l.split()[0] for l in linhas[1:]] == ["1.", "2.", "3.", "4.", "5."]
        assert len(set(codigos)) == 5 and set(codigos) <= set(treinado["model"].classes_)


def test_probabilidades_por_ordem_decrescente_sem_correcao(treinado):
    out = sg.format_suggestions(treinado["model"], ["papel de impressão"], tau=0.0)
    probs = [float(l.split("probabilidade ")[1].rstrip(")")) for l in out.splitlines()[1:] if l]
    assert len(probs) == 5 and probs == sorted(probs, reverse=True)


def test_comando_ordena_com_o_tau_escolhido(treinado):
    textos = ["limpeza de escolas", "papel de impressão", "obras na estrada"]
    out = sg.format_suggestions(treinado["model"], textos)
    codigos = [[l.split()[1] for l in b.splitlines()[1:]] for b in _blocos(out)]
    assert codigos == tt.tfidf.suggest(treinado["model"], textos, tau=tt.tfidf.TAU).tolist()


def test_modelo_gravado_leva_as_frequencias_do_treino(treinado):
    gravado = joblib.load(treinado["modelo"])
    tr = treinado["treino_df"]["codigo"]
    np.testing.assert_allclose(gravado.log_prior_, tt.tfidf.log_prior(tr, gravado.classes_, n_min=N_MIN))


def test_comando_de_sugestoes_aceita_outro_tau(treinado):
    textos = ["limpeza de escolas", "papel de impressão", "obras na estrada", "medicamentos"]
    forte = tt.tfidf.suggest(treinado["model"], textos, tau=5.0).tolist()
    # pré-condição: com τ = 5 a ordem muda, senão o teste não distinguia os dois τ
    assert forte != tt.tfidf.suggest(treinado["model"], textos, tau=tt.tfidf.TAU).tolist()
    r = subprocess.run([sys.executable, str(Path(tt.__file__).with_name("suggest.py")),
                        "--modelo", str(treinado["modelo"]), "--tau", "5", *textos],
                       capture_output=True, text=True, check=True)
    codigos = [[l.split()[1] for l in b.splitlines()[1:]] for b in _blocos(r.stdout)]
    assert codigos == forte


def test_treino_por_defeito_grava_numa_pasta_propria_e_nao_no_registo_do_e4_h1():
    # metrics/e4_h1/ e metrics/e4_s4/ são o registo fixo de cada item; um treino novo não os apaga
    assert tt.SAIDA == Path(tt.__file__).parents[1] / "metrics" / "tfidf"


# --- E4-T3: testes ao treino, do .xlsx ao modelo gravado ---

@pytest.mark.parametrize("n_min", [N_MIN, 10])
def test_sugestoes_gravadas_sao_as_etiquetas_do_treino_gravado(treinado, tmp_path, n_min):
    # As etiquetas contam-se aqui a partir do treino.parquet que a preparação escreveu, não do
    # `classes_` do modelo, que viria do mesmo erro. Com N = 10 os dois códigos da classe 3314
    # sobem juntos para 33140000 (tests/conftest.py): apanha um treino com os códigos originais.
    treino_df = treinado["treino_df"]
    permitidas = set(collapse_rare_codes(treino_df["codigo"], n_min).dropna())
    if n_min == N_MIN:
        sugestoes = treinado["sugestoes"]
    else:
        assert "33140000" in permitidas and "33141100" not in permitidas  # pré-condição
        tt.run(treinado["treino"], treinado["validacao"], tmp_path, n_min=n_min, min_df=1)
        sugestoes = pd.read_parquet(tmp_path / tt.SUGESTOES)
    for s in sugestoes["sugestoes"]:
        assert set(s) <= permitidas


def test_treino_minusculo_pelo_pipeline_acerta_todos_os_textos_no_top_1(tmp_path, run_pipeline, temas_decorar):
    # Do .xlsx ao modelo gravado, com N = 1 (sem a regra dos códigos raros) e a regularização
    # fraca do modelo: o modelo gravado tem de acertar no top-1 de cada texto do treino.parquet.
    # Se não acertar, há um erro no caminho (textos e etiquetas desalinhados, exemplos perdidos,
    # N ignorado, coluna errada, modelo gravado diferente do treinado), mesmo sem dar erro.
    # Cada contrato tem um 2.º CPV: a etiqueta é o 1.º (coluna `codigo`), nunca a lista `codigos`.
    treino = [(2024, f"d-{c}-{i}", datetime(2024, 3, 1), t, None, None, f"{c}-0 - Tema\n15800000-0 - Outro")
              for c, textos in temas_decorar.items() for i, t in enumerate(textos)]
    outros = [(2025, "v-1", datetime(2025, 10, 2), "Papel e canetas", None, None, "30192000-0 - Papel"),
              (2025, "t-1", datetime(2025, 11, 5), "Papel novo", None, None, "30192000-0 - Papel")]
    run_pipeline(tmp_path, treino + outros)
    out = tmp_path / "out"
    modelo = tmp_path / "models" / "tfidf.joblib"
    tt.run(out / "treino.parquet", out / "validacao.parquet", tmp_path / "metrics", model_path=modelo,
           n_min=1, min_df=1)
    treino_df = pd.read_parquet(out / "treino.parquet")
    # pré-condições: a preparação não juntou nem tirou textos, e os contratos têm 2 códigos
    assert len(treino_df) == len(treino)
    assert (treino_df["codigos"] != treino_df["codigo"]).all()
    # τ = 0: ordem só pela pontuação; com τ > 0 a correção das etiquetas frequentes mexia na ordem
    top1 = tt.tfidf.suggest(joblib.load(modelo), treino_df["texto"], tau=0.0).str[0]
    errados = top1[top1 != treino_df["codigo"]]
    assert errados.empty, f"top-1 errado em {len(errados)} de {len(treino_df)} textos: {errados.to_dict()}"


# --- Spike E4-S5: número fixo de épocas, sem paragem automática ---

def _resumo(pasta):
    with open(Path(pasta) / tt.RESUMO, encoding="utf-8") as f:
        return json.load(f)


def test_treino_por_defeito_grava_as_epocas_escolhidas_no_e4_s5(treinado):
    # Sem `epocas`, o treino usa as épocas fixas escolhidas no E4-S5, sem paragem automática
    resumo = _resumo(treinado["saida"])
    assert resumo["epocas"] == tt.tfidf.EPOCAS
    assert resumo["tol"] is None
    assert resumo["max_iter"] == tt.tfidf.EPOCAS
    assert resumo["n_iter"] == [tt.tfidf.EPOCAS]
    clf = treinado["model"].named_steps["clf"]
    assert clf.tol is None and clf.max_iter == clf.n_iter_ == tt.tfidf.EPOCAS


def test_epocas_none_grava_a_regra_de_paragem_do_sklearn(treinado, tmp_path):
    # epocas=None: o treino de antes do E4-S5, tol = 1e-3 e paragem automática
    model = tt.run(treinado["treino"], treinado["validacao"], tmp_path, n_min=N_MIN, min_df=1,
                   epocas=None, n_jobs=1)
    resumo = _resumo(tmp_path)
    assert resumo["epocas"] is None
    assert resumo["tol"] == 0.001
    assert resumo["max_iter"] == 1000
    clf = model.named_steps["clf"]
    assert clf.tol == 1e-3 and clf.max_iter == 1000


@pytest.mark.parametrize("epocas", [3, 12])
def test_epocas_fixas_treinam_exatamente_essas_epocas(treinado, tmp_path, epocas):
    model = tt.run(treinado["treino"], treinado["validacao"], tmp_path, n_min=N_MIN, min_df=1,
                   epocas=epocas, n_jobs=1)
    resumo = _resumo(tmp_path)
    assert resumo["epocas"] == epocas
    assert resumo["tol"] is None  # sem paragem automática
    assert resumo["max_iter"] == epocas
    # O n_iter do sklearn é o máximo das etiquetas; com tol = None todas correm as k épocas
    assert resumo["n_iter"] == [epocas]
    assert resumo["convergiu"] is True  # sem tol, o sklearn não avisa de falta de convergência
    clf = model.named_steps["clf"]
    assert clf.tol is None and clf.max_iter == epocas and clf.n_iter_ == epocas
    # As épocas fixas não mexem no resto: o alpha continua o do E4-S4
    assert clf.alpha == resumo["alpha"] == tt.tfidf.ALPHA


def test_comando_de_treino_com_epocas_e_n_jobs(treinado, tmp_path):
    saida = tmp_path / "saida"
    subprocess.run([sys.executable, str(Path(tt.__file__)), "--treino", str(treinado["treino"]),
                    "--validacao", str(treinado["validacao"]), "--n-min", "5", "--saida", str(saida),
                    "--sem-modelo", "--epocas", "4", "--n-jobs", "2"],
                   capture_output=True, text=True, check=True)
    resumo = _resumo(saida)
    assert resumo["epocas"] == 4 and resumo["tol"] is None and resumo["max_iter"] == 4
    assert resumo["n_iter"] == [4]
    assert (saida / tt.SUGESTOES).exists()


@pytest.mark.parametrize("args, esperado", [
    ([], {"epocas": tt.tfidf.EPOCAS, "n_jobs": -1}),
    (["--epocas", "12", "--n-jobs", "8"], {"epocas": 12, "n_jobs": 8}),
    (["--regra-por-defeito"], {"epocas": None, "n_jobs": -1}),
])
def test_comando_de_treino_passa_epocas_e_n_jobs_ao_run(monkeypatch, tmp_path, args, esperado):
    # O n_jobs não fica no treino.json: verifica-se o que o comando entrega ao `run`
    recebido = {}
    monkeypatch.setattr(tt, "run", lambda *a, **kw: recebido.update(kw))
    monkeypatch.setattr(sys, "argv", ["train_tfidf.py", "--sem-modelo", "--saida", str(tmp_path), *args])
    tt.main()
    assert {k: recebido[k] for k in esperado} == esperado


@pytest.mark.parametrize("args, epocas, tol", [
    ([], tt.tfidf.EPOCAS, None),               # sem --epocas: as do E4-S5
    (["--regra-por-defeito"], None, 0.001),    # a regra de paragem do sklearn
])
def test_comando_de_treino_grava_as_epocas_por_defeito_ou_a_regra_do_sklearn(treinado, tmp_path,
                                                                             args, epocas, tol):
    saida = tmp_path / "saida"
    subprocess.run([sys.executable, str(Path(tt.__file__)), "--treino", str(treinado["treino"]),
                    "--validacao", str(treinado["validacao"]), "--n-min", "5", "--saida", str(saida),
                    "--sem-modelo", "--n-jobs", "1", *args],
                   capture_output=True, text=True, check=True)
    resumo = _resumo(saida)
    assert resumo["epocas"] == epocas and resumo["tol"] == tol
    if epocas is None:
        assert resumo["max_iter"] == 1000
    else:
        assert resumo["max_iter"] == epocas and resumo["n_iter"] == [epocas]


def test_comando_de_treino_com_epocas_e_regra_por_defeito_juntos_da_erro(tmp_path):
    # Dizem coisas contrárias: o argparse recusa antes de ler qualquer ficheiro. O treino e a
    # validação apontam para ficheiros que não existem: se o argparse aceitasse, o comando falhava
    # na leitura, em vez de treinar com os dados reais de data/processed/ (que não cabem no portátil).
    r = subprocess.run([sys.executable, str(Path(tt.__file__)), "--saida", str(tmp_path / "saida"),
                        "--treino", str(tmp_path / "treino.parquet"),
                        "--validacao", str(tmp_path / "validacao.parquet"),
                        "--sem-modelo", "--epocas", "3", "--regra-por-defeito"],
                       capture_output=True, text=True)
    assert r.returncode == 2
    assert "not allowed with argument" in r.stderr
    assert not (tmp_path / "saida").exists()


def test_n_jobs_nao_muda_o_modelo_com_epocas_fixas(treinado, tmp_path):
    # No pod correm vários treinos ao mesmo tempo, cada um com menos núcleos: o modelo é o mesmo
    m1 = tt.run(treinado["treino"], treinado["validacao"], tmp_path / "a", n_min=N_MIN, min_df=1,
                epocas=5, n_jobs=1)
    m2 = tt.run(treinado["treino"], treinado["validacao"], tmp_path / "b", n_min=N_MIN, min_df=1,
                epocas=5, n_jobs=2)
    np.testing.assert_array_equal(m1.named_steps["clf"].coef_, m2.named_steps["clf"].coef_)
    # e o n_jobs pedido chega mesmo ao classificador
    assert (m1.named_steps["clf"].n_jobs, m2.named_steps["clf"].n_jobs) == (1, 2)


@pytest.mark.parametrize("epocas", [3, 24])
def test_treino_minusculo_com_epocas_fixas_acerta_todos_os_textos_no_top_1(tmp_path, run_pipeline,
                                                                           temas_decorar, epocas):
    # O mesmo espírito do teste do E4-T3 acima, mas pelo caminho das épocas fixas: textos bem
    # distintos, N = 1 e a regularização fraca do modelo; o top-1 de cada texto do treino tem de
    # ser a sua etiqueta. Guarda a ligação textos-etiquetas e o alpha neste caminho, não o número
    # de épocas certo (passa com qualquer k).
    treino = [(2024, f"d-{c}-{i}", datetime(2024, 3, 1), t, None, None, f"{c}-0 - Tema")
              for c, textos in temas_decorar.items() for i, t in enumerate(textos)]
    outros = [(2025, "v-1", datetime(2025, 10, 2), "Papel e canetas", None, None, "30192000-0 - Papel"),
              (2025, "t-1", datetime(2025, 11, 5), "Papel novo", None, None, "30192000-0 - Papel")]
    run_pipeline(tmp_path, treino + outros)
    out = tmp_path / "out"
    model = tt.run(out / "treino.parquet", out / "validacao.parquet", tmp_path / "metrics",
                   n_min=1, min_df=1, epocas=epocas, n_jobs=1)
    treino_df = pd.read_parquet(out / "treino.parquet")
    assert len(treino_df) == len(treino)  # pré-condição: a preparação não juntou textos
    assert model.named_steps["clf"].n_iter_ == epocas
    top1 = tt.tfidf.suggest(model, treino_df["texto"], tau=0.0).str[0]
    errados = top1[top1 != treino_df["codigo"]]
    assert errados.empty, f"top-1 errado em {len(errados)} de {len(treino_df)} textos: {errados.to_dict()}"
