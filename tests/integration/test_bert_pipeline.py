"""Testes de integração do E5-T2: dos .xlsx inventados ao treino do BERT e às sugestões gravadas.

Gera .xlsx inventados, corre a conversão e a preparação (`run_pipeline`, tests/conftest.py) e
tuna um modelo minúsculo de pesos aleatórios, construído aqui (`make_tiny_model`), com
`train_bert.run` no processador. Nunca se descarrega o BERTimbau nem o Albertina.

Estes testes guardam a ligação entre textos, etiquetas e sugestões (textos e etiquetas
alinhados, a tradução número -> código, a ordem das linhas da validação, o τ, os ficheiros por
época e os checkpoints), não a qualidade do modelo: um modelo minúsculo com textos tão distintos
decora-os com quase qualquer configuração, por isso o CI não apanha hiperparâmetros maus (como
os testes de sobreajuste do TF-IDF, E4-T3). Cada teste diz que critério de aceitação do E5-T2 guarda.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer

import bert
import evaluation as ev
import model_gate as mg
import tfidf
import train_bert as tb
import train_tfidf as tt
from conftest import TEMAS_DECORAR, make_tiny_model
from labels import collapse_rare_codes

# --- Treino principal: frequências diferentes por etiqueta (para o τ mudar a ordem) ---
# N = 2: os dois códigos da classe 3314, com 1 exemplo cada, sobem juntos para 33140000; o
# 34110000 (1 exemplo, sozinho na classe e no grupo) sai. Ficam 6 etiquetas.
N_MIN = 2
TEMAS = {
    "90911200": ["Limpeza das escolas", "Lavagem dos pavilhões escolares", "Limpeza de janelas",
                 "Lavagem de pavimentos"],
    "33600000": ["Medicamentos hospitalares", "Antibióticos e analgésicos", "Vacinas para a gripe"],
    "45233200": ["Pavimentação rodoviária", "Asfalto para estradas"],
    "30192000": ["Papel e canetas", "Tinteiros para impressoras"],
    "15800000": ["Refeições na cantina", "Almoços para idosos"],
    "33141100": ["Seringas descartáveis"],
    "33141400": ["Luvas cirúrgicas"],
    "34110000": ["Automóveis ligeiros"],
}
TREINO = [(2024, f"d-{c}-{i}", datetime(2024, 3, 1 + i), t, None, None, f"{c}-0 - Tema")
          for c, textos in TEMAS.items() for i, t in enumerate(textos)]
# Validação por uma ordem diferente da do treino; um texto novo e um contrato só com a divisão
VALIDACAO = [
    (2025, "v-papel", datetime(2025, 10, 2), "Papel e canetas", None, None, "30192000-0 - Papel"),
    (2025, "v-novo", datetime(2025, 10, 3), "Viaturas elétricas", None, None, "34110000-0 - Viaturas"),
    (2025, "v-limpeza", datetime(2025, 10, 4), "Limpeza das escolas", None, None, "90911200-0 - Limpeza"),
    (2025, "v-seringas", datetime(2025, 10, 5), "Seringas descartáveis", None, None, "33141100-0 - Seringas"),
    (2025, "v-obras", datetime(2025, 10, 6), "Obras novas", None, None, "45000000-0 - Obras"),
]
TESTE = [(2025, "t-1", datetime(2025, 11, 5), "Papel novo", None, None, "30192000-0 - Papel")]

EPOCAS = 3
TREINO_RAPIDO = dict(n_min=N_MIN, max_length=32, taxa_aprendizagem=3e-3, lote=2, epocas=EPOCAS,
                     aquecimento=0.0, dispositivo="cpu")


def _treinar(base, treino_path, val_path, modelo, **kw):
    """`train_bert.run` com as pastas dos checkpoints e das pontuações dentro de `base`."""
    args = {"checkpoints_dir": base / "checkpoints", "pontuacoes_dir": base / "pontuacoes", **TREINO_RAPIDO, **kw}
    return tb.run(treino_path, val_path, base / "saida", modelo=modelo, **args)


@pytest.fixture(scope="module")
def treinado(tmp_path_factory, run_pipeline):
    base = tmp_path_factory.mktemp("bert")
    run_pipeline(base, TREINO + VALIDACAO + TESTE)
    paths = {"treino": base / "out" / "treino.parquet", "validacao": base / "out" / "validacao.parquet",
             "modelo": make_tiny_model(base / "modelo", [c[3] for c in TREINO])}
    out = _treinar(base, paths["treino"], paths["validacao"], paths["modelo"])
    return {**paths, "out": out, "saida": base / "saida", "checkpoints": base / "checkpoints",
            "pontuacoes": base / "pontuacoes", "treino_df": pd.read_parquet(paths["treino"]),
            "val_df": pd.read_parquet(paths["validacao"]),
            "resumo": json.loads((base / "saida" / tt.RESUMO).read_text(encoding="utf-8"))}


def _ficheiros_de_sugestoes(saida):
    """Todos os ficheiros de sugestões gravados: os da raiz e os de cada época."""
    return sorted(saida.glob("sugestoes_validacao*.parquet")) + sorted(saida.glob("epoca_*/sugestoes_validacao*.parquet"))


def _classes(treinado):
    return json.loads((treinado["pontuacoes"] / tb.ETIQUETAS).read_text(encoding="utf-8"))


# --- Critério 1: com 5 contratos de temas bem distintos, top-1 certo em todos ---

@pytest.mark.parametrize("arquitetura", ["bert", "deberta"])
def test_treino_minusculo_pelo_pipeline_acerta_os_5_contratos_no_top_1(tmp_path, run_pipeline, arquitetura):
    # 5 contratos, um por código (n_min = 1), dos temas do E4-T3. A validação tem os mesmos textos
    # por outra ordem: o top-1 do ficheiro gravado tem de ser o código de cada contrato. Apanha
    # etiquetas desalinhadas dos textos, a tradução número -> código trocada e linhas da
    # validação fora de ordem. "deberta" é a arquitetura do Albertina, "bert" a do BERTimbau.
    temas = list(TEMAS_DECORAR.items())[:5]
    treino = [(2024, f"d-{c}", datetime(2024, 3, 1), textos[0], None, None, f"{c}-0 - Tema") for c, textos in temas]
    val = [(2025, f"v-{c}", datetime(2025, 10, 2 + i), textos[0], None, None, f"{c}-0 - Tema")
           for i, (c, textos) in enumerate(reversed(temas))]
    run_pipeline(tmp_path, treino + val + TESTE)
    modelo = make_tiny_model(tmp_path / "modelo", [c[3] for c in treino], arquitetura)
    out = _treinar(tmp_path, tmp_path / "out" / "treino.parquet", tmp_path / "out" / "validacao.parquet", modelo,
                   n_min=1, taxa_aprendizagem=3e-3, lote=1, epocas=20)

    val_df = pd.read_parquet(tmp_path / "out" / "validacao.parquet")
    sug = pd.read_parquet(tmp_path / "saida" / tt.SUGESTOES)
    assert sug["idcontrato"].tolist() == val_df["idcontrato"].tolist()
    top1 = sug["sugestoes"].str[0].tolist()
    assert top1 == val_df["codigo"].tolist(), f"top-1 {top1} contra {val_df['codigo'].tolist()}"
    # e os textos do treino, pontuados pelo modelo devolvido
    treino_df = pd.read_parquet(tmp_path / "out" / "treino.parquet")
    pont = bert.scores(out["model"], out["tokenizer"], treino_df["texto"], 32)
    assert bert.suggest_from_scores(pont, out["classes"], k=1) == [[c] for c in treino_df["codigo"]]


# --- Critério 2: 5 sugestões, só etiquetas do treino, nunca divisões; a avaliação lê-as ---

def test_sugestoes_sempre_5_do_treino_sem_divisoes_pela_ordem_da_validacao(treinado):
    classes = set(_classes(treinado))
    ficheiros = _ficheiros_de_sugestoes(treinado["saida"])
    assert len(ficheiros) == 2 * (EPOCAS + 1)  # τ = 0 e τ = 0,25, na raiz e em cada época
    for f in ficheiros:
        sug = pd.read_parquet(f)
        assert list(sug.columns) == ["idcontrato", "sugestoes"]
        assert sug["idcontrato"].tolist() == treinado["val_df"]["idcontrato"].tolist()
        for s in sug["sugestoes"]:
            s = list(s)
            assert len(s) == 5 and len(set(s)) == 5
            assert all(isinstance(c, str) and len(c) == 8 and c.isdigit() for c in s)
            assert set(s) <= classes
            assert not any(c.endswith("000000") for c in s)


def test_etiquetas_contadas_so_no_treino_e_por_ordem_crescente(treinado):
    esperado = sorted(collapse_rare_codes(treinado["treino_df"]["codigo"], N_MIN).dropna().unique())
    assert _classes(treinado) == esperado == list(treinado["out"]["classes"])
    # A classe 3314 (dois códigos raros) é etiqueta; o 34110000 (raro e sozinho) saiu
    assert "33140000" in esperado and "34110000" not in esperado
    assert treinado["resumo"]["exemplos_saidos"] == 1
    assert treinado["resumo"]["exemplos_usados"] == len(treinado["treino_df"]) - 1


def test_sugestoes_avaliam_se_com_a_avaliacao_comum_e_com_o_portao(treinado):
    val = treinado["val_df"]
    sug = pd.read_parquet(treinado["saida"] / tt.SUGESTOES)
    sugestoes = pd.Series(sug["sugestoes"].map(list).to_numpy(), index=val.index)
    out = ev.report(sugestoes, val["codigos"], val["visto_no_treino"])
    assert out.loc[("todos", "divisao"), "contratos"] == len(VALIDACAO)
    assert out.loc[("todos", "so_divisao"), "contratos"] == 1  # o v-obras
    # O portão (E4-T5) lê a pasta do treino como a do TF-IDF, sem mudanças: o ficheiro do τ escolhido
    block, resumo = mg.evaluate_run(treinado["saida"], treinado["validacao"])
    mg.check_block(block)
    assert block["sugestoes"].endswith(tt.suggestions_file(tfidf.TAU))
    assert block["contratos"] == len(VALIDACAO)


# --- Critério 3: dois treinos no processador com a mesma semente dão o mesmo ---

def test_mesma_semente_da_exatamente_as_mesmas_sugestoes_e_pontuacoes(treinado, tmp_path):
    _treinar(tmp_path, treinado["treino"], treinado["validacao"], treinado["modelo"])
    for f in _ficheiros_de_sugestoes(treinado["saida"]):
        outro = tmp_path / "saida" / f.relative_to(treinado["saida"])
        assert pd.read_parquet(outro)["sugestoes"].map(list).tolist() == \
            pd.read_parquet(f)["sugestoes"].map(list).tolist(), f
    for k in range(1, EPOCAS + 1):
        np.testing.assert_array_equal(np.load(tmp_path / "pontuacoes" / tb.scores_file(k)),
                                      np.load(treinado["pontuacoes"] / tb.scores_file(k)))


def test_outra_semente_da_outras_pontuacoes(treinado, tmp_path):
    # Pré-condição do teste acima: a semente chega mesmo ao treino (senão a igualdade nada provava)
    _treinar(tmp_path, treinado["treino"], treinado["validacao"], treinado["modelo"], semente=1)
    a = np.load(tmp_path / "pontuacoes" / tb.scores_file(EPOCAS))
    b = np.load(treinado["pontuacoes"] / tb.scores_file(EPOCAS))
    assert not np.allclose(a, b)


# --- Critério 4: as verificações do E4-T5 recusam o teste pelo nome e pelas datas ---

def _nada_gravado(base):
    assert not (base / "saida").exists()
    assert not (base / "checkpoints").exists()
    assert not (base / "pontuacoes").exists()


@pytest.mark.parametrize("qual", ["treino", "validacao"])
def test_ficheiro_chamado_teste_e_recusado_antes_de_treinar(treinado, tmp_path, qual):
    teste = tmp_path / "teste.parquet"
    teste.write_bytes(Path(treinado[qual]).read_bytes())
    paths = {"treino": treinado["treino"], "validacao": treinado["validacao"], qual: teste}
    with pytest.raises(ValueError, match="teste"):
        _treinar(tmp_path, paths["treino"], paths["validacao"], treinado["modelo"])
    _nada_gravado(tmp_path)


@pytest.mark.parametrize("qual", ["treino", "validacao"])
def test_copia_do_teste_com_outro_nome_e_recusada_pelas_datas(treinado, tmp_path, qual):
    copia = tmp_path / "outro_nome.parquet"
    copia.write_bytes((Path(treinado["treino"]).parent / "teste.parquet").read_bytes())
    paths = {"treino": treinado["treino"], "validacao": treinado["validacao"], qual: copia}
    with pytest.raises(ValueError, match="teste"):
        _treinar(tmp_path, paths["treino"], paths["validacao"], treinado["modelo"])
    _nada_gravado(tmp_path)


def test_validacao_passada_como_treino_e_recusada_pelas_datas(treinado, tmp_path):
    with pytest.raises(ValueError, match="2025-10-01"):
        _treinar(tmp_path, treinado["validacao"], treinado["validacao"], treinado["modelo"])
    _nada_gravado(tmp_path)


# --- Critério 5: pontuações de todas as etiquetas em cada época; sugestões com τ = 0 e 0,25 ---

def test_pontuacoes_de_cada_epoca_uma_linha_por_contrato_e_uma_coluna_por_etiqueta(treinado):
    n_val, n_etiquetas = len(treinado["val_df"]), len(_classes(treinado))
    assert sorted(p.name for p in treinado["pontuacoes"].glob("epoca_*.npy")) == \
        [tb.scores_file(k) for k in range(1, EPOCAS + 1)]
    for k in range(1, EPOCAS + 1):
        p = np.load(treinado["pontuacoes"] / tb.scores_file(k))
        assert p.shape == (n_val, n_etiquetas) and p.dtype == np.float32
    # As épocas são mesmo diferentes (o modelo treinou entre elas)
    assert not np.allclose(np.load(treinado["pontuacoes"] / tb.scores_file(1)),
                           np.load(treinado["pontuacoes"] / tb.scores_file(EPOCAS)))
    assert [e["epoca"] for e in treinado["resumo"]["por_epoca"]] == list(range(1, EPOCAS + 1))


def test_sugestoes_de_cada_epoca_sao_o_top_5_das_pontuacoes_com_cada_tau(treinado):
    classes = np.array(_classes(treinado), dtype=object)
    # O log_prior conta-se outra vez aqui, só com o treino, e tem de ser o que ficou gravado
    prior = tfidf.log_prior(treinado["treino_df"]["codigo"], classes, n_min=N_MIN)
    np.testing.assert_allclose(np.load(treinado["pontuacoes"] / tb.LOG_PRIOR), prior)
    tau_muda_alguma = False
    for k in range(1, EPOCAS + 1):
        p = np.load(treinado["pontuacoes"] / tb.scores_file(k))
        sem = [list(s) for s in tfidf.top_k(p, classes)]
        com = [list(s) for s in tfidf.top_k(p - tfidf.TAU * prior, classes)]
        pasta = treinado["saida"] / tb.epoch_dir(k)
        assert pd.read_parquet(pasta / tt.suggestions_file(0.0))["sugestoes"].map(list).tolist() == sem
        assert pd.read_parquet(pasta / tt.suggestions_file(tfidf.TAU))["sugestoes"].map(list).tolist() == com
        tau_muda_alguma |= sem != com
    assert tau_muda_alguma  # pré-condição: com estes dados o τ muda a ordem nalguma época


def test_sugestoes_da_raiz_sao_as_da_ultima_epoca(treinado):
    ultima = treinado["saida"] / tb.epoch_dir(EPOCAS)
    for tau in (0.0, tfidf.TAU):
        assert pd.read_parquet(treinado["saida"] / tt.suggestions_file(tau)).equals(
            pd.read_parquet(ultima / tt.suggestions_file(tau)))


# --- Critério 6: um checkpoint por época, que se pode carregar e dá as mesmas pontuações ---

def test_um_checkpoint_por_epoca_com_a_traducao_numero_codigo(treinado):
    classes = _classes(treinado)
    por_epoca = treinado["resumo"]["por_epoca"]
    assert sorted(p.name for p in treinado["checkpoints"].glob("checkpoint-*")) == \
        sorted(e["checkpoint"] for e in por_epoca)
    for e in por_epoca:
        pasta = treinado["checkpoints"] / e["checkpoint"]
        # Com o estado do otimizador e da taxa, para se poder retomar desta época
        assert (pasta / "optimizer.pt").is_file() and (pasta / "scheduler.pt").is_file()
        model = AutoModelForSequenceClassification.from_pretrained(pasta)
        tok = AutoTokenizer.from_pretrained(pasta)
        assert [model.config.id2label[i] for i in range(len(classes))] == classes
        # O modelo de cada época reproduz as pontuações gravadas dessa época, pela ordem da validação
        p = bert.scores(model, tok, treinado["val_df"]["texto"], treinado["resumo"]["max_length"])
        np.testing.assert_allclose(p, np.load(treinado["pontuacoes"] / e["pontuacoes"]), rtol=1e-5, atol=1e-5)


# --- Retomar um treino interrompido do último checkpoint (decisão do E5-T2: gravar o otimizador) ---

def _treino_que_cai_na_epoca(base, treinado, monkeypatch, epoca, onde):
    """Treina como o `treinado`, mas o processo "cai" no fim de `epoca`, antes do seu checkpoint.

    `onde="pontuar"`: ao pontuar a validação, antes de gravar o que quer que seja dessa época.
    `onde="checkpoint"`: ao gravar o checkpoint, já com as pontuações e as sugestões dessa época
    gravadas (a retoma tem de as gravar outra vez por cima). É o que acontece se o pod cair.
    """
    if onde == "pontuar":
        alvo, nome = bert, "predict_scores"
    else:
        alvo, nome = Trainer, "_save_checkpoint"
    original, chamadas = getattr(alvo, nome), []

    def cai(*args, **kw):
        chamadas.append(1)
        if len(chamadas) == epoca:
            raise KeyboardInterrupt("o pod caiu")
        return original(*args, **kw)

    with monkeypatch.context() as m:
        m.setattr(alvo, nome, cai)
        with pytest.raises(KeyboardInterrupt):
            _treinar(base, treinado["treino"], treinado["validacao"], treinado["modelo"])


@pytest.mark.parametrize("onde", ["pontuar", "checkpoint"])
def test_treino_retomado_da_o_mesmo_que_sem_interrupcao(treinado, tmp_path, monkeypatch, onde):
    _treino_que_cai_na_epoca(tmp_path, treinado, monkeypatch, epoca=2, onde=onde)
    assert sorted(p.name for p in (tmp_path / "checkpoints").glob("checkpoint-*")) == \
        [treinado["resumo"]["por_epoca"][0]["checkpoint"]]
    assert (tmp_path / "saida" / tb.epoch_dir(2)).exists() == (onde == "checkpoint")

    # A retoma só treina e pontua as épocas que faltam (2 e 3), não volta ao início
    original, chamadas = bert.predict_scores, []
    monkeypatch.setattr(bert, "predict_scores", lambda *a, **kw: chamadas.append(1) or original(*a, **kw))
    _treinar(tmp_path, treinado["treino"], treinado["validacao"], treinado["modelo"], retomar=True)
    assert len(chamadas) == EPOCAS - 1

    # As pontuações e as sugestões de todas as épocas são as do treino que nunca parou
    for k in range(1, EPOCAS + 1):
        np.testing.assert_allclose(np.load(tmp_path / "pontuacoes" / tb.scores_file(k)),
                                   np.load(treinado["pontuacoes"] / tb.scores_file(k)), rtol=1e-5, atol=1e-5)
    nomes = [p.relative_to(treinado["saida"]) for p in _ficheiros_de_sugestoes(treinado["saida"])]
    assert nomes == [p.relative_to(tmp_path / "saida") for p in _ficheiros_de_sugestoes(tmp_path / "saida")]
    for nome in nomes:
        pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "saida" / nome), pd.read_parquet(treinado["saida"] / nome))
    r = json.loads((tmp_path / "saida" / tt.RESUMO).read_text(encoding="utf-8"))
    assert r["retomado_depois_da_epoca"] == 1
    # O por_epoca tem todas as épocas: a 1.ª do treino que caiu, as outras da retoma
    assert [(e["epoca"], e["checkpoint"]) for e in r["por_epoca"]] == \
        [(e["epoca"], e["checkpoint"]) for e in treinado["resumo"]["por_epoca"]]
    assert all(e["perda_treino"] is not None for e in r["por_epoca"])


@pytest.mark.parametrize("outro", [{"lote": 1}, {"epocas": EPOCAS + 1}, {"semente": 5}, {"taus": (0.0,)},
                                   {"pontuacoes_dir": "outras"}])
def test_retomar_com_outros_argumentos_e_recusado(treinado, tmp_path, monkeypatch, outro):
    # Com outro lote ou outras épocas o Trainer saltava os passos errados sem dar erro; com outra
    # pasta de pontuações, as épocas ficavam divididas por duas pastas
    _treino_que_cai_na_epoca(tmp_path, treinado, monkeypatch, epoca=2, onde="pontuar")
    antes = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    if "pontuacoes_dir" in outro:
        outro = {"pontuacoes_dir": tmp_path / outro["pontuacoes_dir"]}
    chave = next(iter(outro)).removesuffix("_dir")
    with pytest.raises(ValueError, match=f"diferentes: {chave}"):
        _treinar(tmp_path, treinado["treino"], treinado["validacao"], treinado["modelo"], retomar=True, **outro)
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == antes  # nada mudou


def test_retomar_com_o_treino_de_outra_pasta_e_recusado(treinado, tmp_path, monkeypatch):
    # Outra janela de treino tem o mesmo nome de ficheiro (treino.parquet)
    _treino_que_cai_na_epoca(tmp_path, treinado, monkeypatch, epoca=2, onde="pontuar")
    outra = tmp_path / "janela" / "treino.parquet"
    outra.parent.mkdir()
    outra.write_bytes(treinado["treino"].read_bytes())
    with pytest.raises(ValueError, match="diferentes: treino"):
        _treinar(tmp_path, outra, treinado["validacao"], treinado["modelo"], retomar=True)


def test_retomar_com_a_pasta_do_modelo_escrita_de_outra_forma_e_aceite(treinado, tmp_path, monkeypatch):
    # O mesmo modelo, uma vez com o caminho absoluto e outra com um relativo: é o mesmo treino
    _treino_que_cai_na_epoca(tmp_path, treinado, monkeypatch, epoca=2, onde="pontuar")
    monkeypatch.chdir(treinado["modelo"].parent)
    _treinar(tmp_path, treinado["treino"], treinado["validacao"], Path(treinado["modelo"].name), retomar=True)
    assert (tmp_path / "saida" / tt.RESUMO).is_file()


def test_retomar_sem_argumentos_gravados_da_erro_claro(treinado, tmp_path, monkeypatch):
    _treino_que_cai_na_epoca(tmp_path, treinado, monkeypatch, epoca=2, onde="pontuar")
    (tmp_path / "saida" / tb.ARGUMENTOS).unlink()
    with pytest.raises(ValueError, match="não existe"):
        _treinar(tmp_path, treinado["treino"], treinado["validacao"], treinado["modelo"], retomar=True)


def test_retomar_depois_da_ultima_epoca_so_escreve_o_que_faltava(treinado, tmp_path, monkeypatch):
    # O pod cai depois do último checkpoint, ao copiar as sugestões para a raiz: as épocas estão
    # todas treinadas, mas falta o treino.json. A retoma não treina nada e acaba o que faltava.
    original, chamadas = tb.shutil.copyfile, []

    def cai(*args, **kw):
        chamadas.append(1)
        raise KeyboardInterrupt("o pod caiu")

    with monkeypatch.context() as m:
        m.setattr(tb.shutil, "copyfile", cai)
        with pytest.raises(KeyboardInterrupt):
            _treinar(tmp_path, treinado["treino"], treinado["validacao"], treinado["modelo"])
    assert chamadas and not (tmp_path / "saida" / tt.RESUMO).exists()
    assert len(list((tmp_path / "checkpoints").glob("checkpoint-*"))) == EPOCAS

    pontuadas = []
    monkeypatch.setattr(bert, "predict_scores", lambda *a, **kw: pontuadas.append(1))
    _treinar(tmp_path, treinado["treino"], treinado["validacao"], treinado["modelo"], retomar=True)
    assert pontuadas == []  # nenhuma época voltou a correr
    r = json.loads((tmp_path / "saida" / tt.RESUMO).read_text(encoding="utf-8"))
    assert r["retomado_depois_da_epoca"] == EPOCAS
    assert [e["epoca"] for e in r["por_epoca"]] == list(range(1, EPOCAS + 1))
    assert all(e["perda_treino"] is not None for e in r["por_epoca"])
    for tau in (0.0, tfidf.TAU):
        nome = tt.suggestions_file(tau)
        pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "saida" / nome), pd.read_parquet(treinado["saida"] / nome))


def test_tau_negativo_e_recusado_antes_de_treinar(treinado, tmp_path):
    with pytest.raises(ValueError, match=">= 0"):
        _treinar(tmp_path, treinado["treino"], treinado["validacao"], treinado["modelo"], taus=(0.0, -0.25))
    assert not (tmp_path / "saida").exists() and not (tmp_path / "checkpoints").exists()


def test_retomar_sem_checkpoint_da_erro_antes_de_treinar(treinado, tmp_path):
    with pytest.raises(ValueError, match="nenhum checkpoint"):
        _treinar(tmp_path, treinado["treino"], treinado["validacao"], treinado["modelo"], retomar=True)
    assert not (tmp_path / "saida").exists()


# --- treino.json ---

def test_treino_json_tem_o_que_o_portao_e_a_comparacao_precisam(treinado):
    r = treinado["resumo"]
    assert set(r) == {
        "data", "modelo", "max_length", "n_min", "taus", "ficheiros", "ficheiro_do_modelo_por_defeito", "treino",
        "validacao", "exemplos_usados", "exemplos_saidos", "etiquetas", "taxa_aprendizagem", "lote", "epocas",
        "aquecimento", "decaimento_pesos", "semente", "dispositivo", "gpu", "n_gpus", "checkpoints", "pontuacoes", "por_epoca",
        "retomado_depois_da_epoca", "tempo_treinar_s", "pico_rss_mb", "nucleos", "versoes"}
    assert r["ficheiro_do_modelo_por_defeito"] == tt.suggestions_file(tfidf.TAU)
    assert r["ficheiros"] == {tt.SUGESTOES: 0.0, tt.suggestions_file(tfidf.TAU): tfidf.TAU}
    assert r["treino"] == {"ficheiro": "treino.parquet", "datas": ["2024-03-01", "2024-03-04"]}
    assert r["validacao"] == {"ficheiro": "validacao.parquet", "datas": ["2025-10-02", "2025-10-06"],
                              "contratos": len(VALIDACAO)}
    assert r["modelo"] == {"chave": None, "nome": str(treinado["modelo"]), "revisao": None}
    assert (r["n_min"], r["max_length"], r["epocas"], r["lote"], r["semente"]) == (N_MIN, 32, EPOCAS, 2, 0)
    assert (r["dispositivo"], r["gpu"], r["n_gpus"], r["etiquetas"]) == ("cpu", None, 0, 6)
    assert {"torch", "transformers", "accelerate", "tokenizers"} <= set(r["versoes"])
    assert all(e["perda_treino"] is not None for e in r["por_epoca"])
    assert r["retomado_depois_da_epoca"] is None
    # Os argumentos que a retoma vai confirmar
    a = json.loads((treinado["saida"] / tb.ARGUMENTOS).read_text(encoding="utf-8"))
    assert (a["lote"], a["epocas"], a["semente"], a["n_min"], a["taus"]) == (2, EPOCAS, 0, N_MIN, [0.0, tfidf.TAU])


# --- Critério 8: no pod, antes de treinar, confirma-se que o torch vê a GPU ---

def test_sem_gpu_o_treino_em_cuda_recusa_antes_de_ler_os_dados(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # Os ficheiros nem existem: o erro tem de ser o da GPU, antes de qualquer leitura
    with pytest.raises(RuntimeError, match="GPU"):
        tb.run(tmp_path / "nao_existe.parquet", tmp_path / "nao_existe.parquet", tmp_path / "saida",
               dispositivo="cuda", checkpoints_dir=tmp_path / "ck", pontuacoes_dir=tmp_path / "pt")
    _nada_gravado(tmp_path)


def test_comando_usa_cuda_por_defeito_e_recusa_sem_gpu(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="GPU"):
        tb.main(["--treino", str(tmp_path / "x.parquet"), "--validacao", str(tmp_path / "y.parquet"),
                 "--saida", str(tmp_path / "saida"), "--checkpoints", str(tmp_path / "ck"),
                 "--pontuacoes", str(tmp_path / "pt")])


def test_comando_no_processador_grava_tudo(tmp_path, run_pipeline, contratos_regua):
    # O comando só aceita N = 5, 10 ou 20: usa os contratos da régua, com etiquetas que chegam
    run_pipeline(tmp_path, contratos_regua)
    out = tmp_path / "out"
    modelo = make_tiny_model(tmp_path / "modelo", pd.read_parquet(out / "treino.parquet")["texto"])
    tb.main(["--treino", str(out / "treino.parquet"), "--validacao", str(out / "validacao.parquet"),
             "--modelo", str(modelo), "--saida", str(tmp_path / "saida"),
             "--checkpoints", str(tmp_path / "ck"), "--pontuacoes", str(tmp_path / "pt"),
             "--n-min", "5", "--max-length", "16", "--epocas", "1", "--lote", "4", "--taxa-aprendizagem", "1e-3",
             "--semente", "3", "--taus", "0", "0.5", "--dispositivo", "cpu"])
    r = json.loads((tmp_path / "saida" / tt.RESUMO).read_text(encoding="utf-8"))
    assert (r["n_min"], r["max_length"], r["epocas"], r["lote"], r["taxa_aprendizagem"], r["semente"]) == \
        (5, 16, 1, 4, 1e-3, 3)
    assert r["taus"] == [0.0, 0.5] and r["ficheiro_do_modelo_por_defeito"] is None
    assert (tmp_path / "saida" / "sugestoes_validacao_tau0.5.parquet").exists()
    assert (tmp_path / "pt" / tb.scores_file(1)).exists() and len(list((tmp_path / "ck").glob("checkpoint-*"))) == 1


def test_comando_sem_pastas_grava_em_metrics_bert_e_models_bert(tmp_path, run_pipeline, contratos_regua, monkeypatch):
    # As mesmas pastas que o `make treinar-bert` usa: metrics/bert/<modelo> (pequena, vai para o
    # git) e models/bert/<modelo> (grande, fica no pod). A raiz do projeto passa a ser o tmp_path.
    run_pipeline(tmp_path, contratos_regua)
    out = tmp_path / "out"
    modelo = make_tiny_model(tmp_path / "minusculo", pd.read_parquet(out / "treino.parquet")["texto"])
    monkeypatch.setattr(tb, "RAIZ", tmp_path)
    monkeypatch.setattr(tb, "MODELOS_DIR", tmp_path / "models" / "bert")
    tb.main(["--treino", str(out / "treino.parquet"), "--validacao", str(out / "validacao.parquet"),
             "--modelo", str(modelo), "--n-min", "5", "--max-length", "16", "--epocas", "1", "--lote", "4",
             "--dispositivo", "cpu"])
    assert (tmp_path / "metrics" / "bert" / "minusculo" / tt.RESUMO).is_file()
    assert (tmp_path / "models" / "bert" / "minusculo" / "pontuacoes" / tb.scores_file(1)).is_file()
    assert len(list((tmp_path / "models" / "bert" / "minusculo" / "checkpoints").glob("checkpoint-*"))) == 1


def test_comando_com_retomar_sem_checkpoint_da_erro(treinado, tmp_path):
    with pytest.raises(ValueError, match="nenhum checkpoint"):
        tb.main(["--treino", str(treinado["treino"]), "--validacao", str(treinado["validacao"]),
                 "--modelo", str(treinado["modelo"]), "--saida", str(tmp_path / "saida"),
                 "--checkpoints", str(tmp_path / "ck"), "--pontuacoes", str(tmp_path / "pt"),
                 "--max-length", "32", "--dispositivo", "cpu", "--retomar"])


# --- Pastas de outro treino não se misturam ---

@pytest.mark.parametrize("pasta", ["checkpoints", "pontuacoes", "saida/epoca_1"])
def test_pasta_de_outro_treino_e_recusada_antes_de_treinar(treinado, tmp_path, pasta):
    (tmp_path / pasta).mkdir(parents=True)
    (tmp_path / pasta / "antigo.txt").write_text("de outro treino")
    with pytest.raises(ValueError, match="outr"):
        _treinar(tmp_path, treinado["treino"], treinado["validacao"], treinado["modelo"])
    assert not (tmp_path / "saida" / tt.RESUMO).exists()


def test_pastas_por_defeito_ficam_em_models_fora_do_git():
    ck, pt = tb.default_dirs(Path("metrics") / "bert" / "bertimbau")
    assert ck == tb.RAIZ / "models" / "bert" / "bertimbau" / "checkpoints"
    assert pt == tb.RAIZ / "models" / "bert" / "bertimbau" / "pontuacoes"
