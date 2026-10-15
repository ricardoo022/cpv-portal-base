"""Testes unitários do `bert.py` (E5-T2): etiquetas, sugestões a partir das pontuações, tokenização,
argumentos do treino, carregamento do modelo e verificação da GPU.

Complementam os testes de integração (tests/integration/test_bert_pipeline.py), que treinam um
modelo minúsculo de ponta a ponta. Como esses, guardam a ligação entre textos, etiquetas e
sugestões, não a qualidade do modelo. Nenhum descarrega o modelo verdadeiro.
"""

import numpy as np
import pandas as pd
import pytest
import torch

import bert
import tfidf
from conftest import make_tiny_model

CLASSES = np.array(["15800000", "30192000", "33600000", "45233200", "90911200", "90919300"], dtype=object)


# --- Etiquetas ---

def test_etiquetas_por_ordem_crescente_com_a_regra_dos_codigos_raros():
    codigo = pd.Series(["90911200"] * 2 + ["33600000"] * 2 + ["45233200"] * 2 + ["30192000"] * 2
                       + ["15800000"] * 2 + ["33141100", "33141400"] + ["34110000"])
    etiqueta, classes = bert.training_labels(codigo, n_min=2)
    assert list(classes) == ["15800000", "30192000", "33140000", "33600000", "45233200", "90911200"]
    assert etiqueta.isna().sum() == 1  # o 34110000, raro e sozinho no grupo


def test_codigo_de_divisao_no_treino_da_erro():
    codigo = pd.Series(["90911200", "33600000", "45233200", "30192000", "15800000", "45000000"])
    with pytest.raises(ValueError, match="divisão"):
        bert.training_labels(codigo, n_min=1)


def test_menos_de_5_etiquetas_da_erro():
    with pytest.raises(ValueError, match="4 etiquetas"):
        bert.training_labels(pd.Series(["90911200", "33600000", "45233200", "30192000"]), n_min=1)


# --- Sugestões a partir das pontuações ---

def test_sugestoes_por_pontuacao_com_empate_para_o_menor_codigo():
    p = np.array([[0.0, 5.0, 1.0, 1.0, 3.0, 2.0]], dtype=np.float32)
    # 33600000 e 45233200 empatam (1.0): fica primeiro o menor código
    assert bert.suggest_from_scores(p, CLASSES) == [["30192000", "90911200", "90919300", "33600000", "45233200"]]


def test_tau_tira_vantagem_as_etiquetas_frequentes():
    p = np.array([[0.0, 0.0, 0.0, 0.0, 1.0, 1.1]], dtype=np.float32)
    prior = np.log(np.array([0.1, 0.1, 0.1, 0.1, 0.5, 0.1]))  # 90911200 é a mais frequente
    assert bert.suggest_from_scores(p, CLASSES, prior, tau=0.0)[0][:2] == ["90919300", "90911200"]
    # com τ = 1: 1.0 − log 0,5 = 1,69 contra 1,1 − log 0,1 = 3,40; 0 − log 0,1 = 2,30
    assert bert.suggest_from_scores(p, CLASSES, prior, tau=1.0)[0][:2] == ["90919300", "15800000"]
    assert bert.suggest_from_scores(p, CLASSES, prior, tau=1.0) == \
        [list(s) for s in tfidf.top_k(p - 1.0 * prior, CLASSES)]


@pytest.mark.parametrize("kw, msg", [
    ({"tau": 0.25}, "log_prior"),
    ({"tau": -0.1}, ">= 0"),
    ({"tau": 0.25, "log_prior": np.zeros(3)}, "uma entrada por etiqueta"),
])
def test_tau_invalido_ou_sem_prior_da_erro(kw, msg):
    with pytest.raises(ValueError, match=msg):
        bert.suggest_from_scores(np.zeros((1, len(CLASSES)), dtype=np.float32), CLASSES, **kw)


def test_pontuacoes_com_outro_numero_de_colunas_dao_erro():
    with pytest.raises(ValueError, match="colunas"):
        bert.suggest_from_scores(np.zeros((2, len(CLASSES) - 1), dtype=np.float32), CLASSES)


# --- Tokenização e lotes ---

@pytest.fixture(scope="module")
def modelo_minusculo(tmp_path_factory):
    textos = ["Limpeza das escolas", "Medicamentos hospitalares", "Papel e canetas", "Refeições na cantina",
              "Asfalto para estradas"]
    return make_tiny_model(tmp_path_factory.mktemp("bert") / "modelo", textos)


def test_textos_longos_cortam_se_no_max_length(modelo_minusculo):
    # O tokenizador minúsculo, como o do BERTimbau, não tem um model_max_length útil: só o
    # max_length passado corta o texto
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(modelo_minusculo)
    ids = bert.encode(tok, ["Limpeza das escolas " * 30, "Papel"], max_length=8)
    assert [len(i) for i in ids] == [8, 3]
    assert ids[0][0] == tok.cls_token_id and ids[0][-1] == tok.sep_token_id


def test_lote_preenche_ate_ao_maior_texto_e_marca_o_preenchimento():
    batch = bert.Collator(pad_token_id=0)([{"input_ids": np.array([2, 7, 3]), "labels": 4},
                                           {"input_ids": np.array([2, 3]), "labels": 1}])
    assert batch["input_ids"].tolist() == [[2, 7, 3], [2, 3, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert batch["labels"].tolist() == [4, 1]


# --- Carregamento do modelo ---

@pytest.mark.parametrize("arquitetura", ["bert", "deberta"])
def test_modelo_carrega_em_float32_com_uma_saida_por_etiqueta(tmp_path, arquitetura):
    # O Albertina está gravado em bfloat16 no Hugging Face; o transformers 5 carregava-o nesse
    # tipo se não se pedisse float32
    pasta = make_tiny_model(tmp_path / "m", ["Limpeza das escolas", "Papel e canetas"], arquitetura,
                            dtype=torch.bfloat16)
    model, tok, info = bert.load_model(pasta, CLASSES, max_length=16)
    assert {p.dtype for p in model.parameters()} == {torch.float32}
    assert model.config.num_labels == len(CLASSES)
    assert [model.config.id2label[i] for i in range(len(CLASSES))] == list(CLASSES)
    assert info == {"nome": str(pasta), "revisao": None, "max_length": 16}


def test_pasta_de_modelo_sem_max_length_da_erro(modelo_minusculo):
    with pytest.raises(ValueError, match="max_length"):
        bert.load_model(modelo_minusculo, CLASSES)


@pytest.mark.parametrize("max_length", [1, 0, 2.5])
def test_max_length_invalido_da_erro(modelo_minusculo, max_length):
    with pytest.raises(ValueError, match="max_length"):
        bert.load_model(modelo_minusculo, CLASSES, max_length=max_length)


def test_modelo_do_e5_usa_o_nome_a_revisao_e_o_comprimento_do_bert_models(monkeypatch):
    # Sem descarregar nada: regista o que se pede ao Hugging Face
    pedidos = []

    def regista(nome, **kw):
        pedidos.append((nome, kw.get("revision")))
        return object()

    monkeypatch.setattr(bert.AutoTokenizer, "from_pretrained", regista)
    monkeypatch.setattr(bert.AutoModelForSequenceClassification, "from_pretrained", regista)
    for chave in ("bertimbau", "albertina"):
        pedidos.clear()
        _, _, info = bert.load_model(chave, CLASSES)
        m = bert.bert_models.MODELOS[chave]
        assert pedidos == [(m["nome"], m["revisao"])] * 2
        assert info["max_length"] == bert.bert_models.COMPRIMENTO_MAXIMO[chave]


# --- Argumentos do treino ---

def test_argumentos_por_defeito_sao_os_do_artigo_do_bert_e_um_checkpoint_completo_por_epoca(tmp_path):
    a = bert.training_arguments(tmp_path, dispositivo="cpu")
    assert (a.learning_rate, a.per_device_train_batch_size, a.num_train_epochs) == (2e-5, 32, 3)
    assert (a.warmup_steps, a.weight_decay, str(a.lr_scheduler_type.value)) == (0.1, 0.01, "linear")
    assert (a.seed, a.data_seed) == (0, 0)
    assert str(a.save_strategy.value) == "epoch" and not a.save_only_model and a.save_total_limit is None
    assert a.report_to == []  # "none": nenhum registo automático (o MLflow entra no E5-T3)
    assert a.dataloader_num_workers == 0 and a.use_cpu


def test_semente_chega_ao_treino_e_a_ordem_dos_exemplos(tmp_path):
    a = bert.training_arguments(tmp_path, semente=7, dispositivo="cpu")
    assert (a.seed, a.data_seed) == (7, 7)


def test_em_cuda_nao_se_forca_o_processador(tmp_path):
    assert not bert.training_arguments(tmp_path, dispositivo="cuda").use_cpu


@pytest.mark.parametrize("kw, msg", [
    ({"aquecimento": 1.0}, "fração"),
    ({"epocas": 0}, "epocas"),
    ({"dispositivo": "xpu"}, "dispositivo"),
])
def test_argumentos_invalidos_dao_erro(tmp_path, kw, msg):
    with pytest.raises(ValueError, match=msg):
        bert.training_arguments(tmp_path, **{"dispositivo": "cpu", **kw})


# --- GPU ---

def test_sem_gpu_a_verificacao_da_erro(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="GPU"):
        bert.check_gpu()


def test_com_gpu_a_verificacao_devolve_o_nome(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: "NVIDIA RTX A5000")
    assert bert.check_gpu() == "NVIDIA RTX A5000"


# --- Recusas do treino, antes de treinar ---

TEXTO = pd.Series(["Limpeza das escolas", "Medicamentos hospitalares", "Papel e canetas", "Refeições na cantina",
                   "Asfalto para estradas"])
CODIGO = pd.Series(["90911200", "33600000", "30192000", "15800000", "45233200"])


def test_texto_e_codigo_desalinhados_dao_erro(modelo_minusculo, tmp_path):
    # Índices diferentes: o texto de um contrato ficaria com a etiqueta de outro
    with pytest.raises(ValueError, match="índices diferentes"):
        bert.train(TEXTO, CODIGO.set_axis(range(1, 6)), modelo_minusculo, tmp_path / "ck", n_min=1,
                   max_length=16, dispositivo="cpu")
    assert not (tmp_path / "ck").exists()


def test_validacao_sem_on_epoch_da_erro(modelo_minusculo, tmp_path):
    # Sem quem as receba, as pontuações da validação perdiam-se em silêncio
    with pytest.raises(ValueError, match="on_epoch"):
        bert.train(TEXTO, CODIGO, modelo_minusculo, tmp_path / "ck", textos_val=TEXTO, n_min=1, max_length=16,
                   dispositivo="cpu")
    assert not (tmp_path / "ck").exists()


@pytest.mark.parametrize("pasta", ["nao_existe", "vazia"])
def test_retomar_sem_checkpoint_no_train_da_erro(modelo_minusculo, tmp_path, pasta):
    (tmp_path / "vazia").mkdir()
    assert bert.last_checkpoint(tmp_path / pasta) is None
    with pytest.raises(ValueError, match="nenhum checkpoint"):
        bert.train(TEXTO, CODIGO, modelo_minusculo, tmp_path / pasta, n_min=1, max_length=16, dispositivo="cpu",
                   retomar=True)


def test_menos_etiquetas_do_que_sugestoes_da_erro():
    with pytest.raises(ValueError, match="só há 4 etiquetas"):
        bert.suggest_from_scores(np.zeros((2, 4), dtype=np.float32), CLASSES[:4])
