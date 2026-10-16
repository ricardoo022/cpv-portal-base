"""Testes de integração do comando de sugestões com um checkpoint do BERT (E5-T4, `src/suggest.py --bert`).

Critério do E5-T4: "dada uma descrição de compra, o comando de sugestões devolve 5 códigos deste
modelo, nenhum ao nível da divisão". Gera .xlsx inventados, corre a conversão e a preparação
(`run_pipeline`) e tuna o modelo minúsculo dos testes do E5-T2 (`make_tiny_model`) com
`train_bert.run` no processador; depois o comando lê um checkpoint gravado por esse treino. As
sugestões de um texto da validação têm de ser as que o treino gravou para esse texto, com o mesmo τ
e na época desse checkpoint: isso guarda a ligação checkpoint -> etiquetas -> frequências -> ordem.
Nunca se descarrega o BERTimbau nem o Albertina.
"""

import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import bert_models
import suggest as sg
import tfidf
import train_bert as tb  # no topo: o conftest substitui o git do train_bert (sem_git_real)
import train_tfidf as tt
from conftest import make_tiny_model

# Os mesmos temas do test_bert_pipeline.py: frequências diferentes por etiqueta, para o τ mudar a
# ordem. N = 2: os dois códigos da classe 3314 sobem juntos para 33140000; o 34110000 sai.
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
VALIDACAO = [
    (2025, "v-papel", datetime(2025, 10, 2), "Papel e canetas", None, None, "30192000-0 - Papel"),
    (2025, "v-novo", datetime(2025, 10, 3), "Viaturas elétricas", None, None, "34110000-0 - Viaturas"),
    (2025, "v-limpeza", datetime(2025, 10, 4), "Limpeza das escolas", None, None, "90911200-0 - Limpeza"),
    (2025, "v-seringas", datetime(2025, 10, 5), "Seringas descartáveis", None, None, "33141100-0 - Seringas"),
    (2025, "v-obras", datetime(2025, 10, 6), "Obras novas", None, None, "45000000-0 - Obras"),
]
TESTE = [(2025, "t-1", datetime(2025, 11, 5), "Papel novo", None, None, "30192000-0 - Papel")]
EPOCAS = 3
MAX_LENGTH = 32  # o do treino minúsculo; as posições do modelo minúsculo só vão até 64
SUGGEST = Path(sg.__file__)


@pytest.fixture(scope="module")
def treinado(tmp_path_factory, run_pipeline):
    base = tmp_path_factory.mktemp("bert_sugerir")
    run_pipeline(base, TREINO + VALIDACAO + TESTE)
    modelo = make_tiny_model(base / "modelo", [c[3] for c in TREINO])
    # As pastas como as grava o train_bert.py por defeito: <base>/checkpoints e <base>/pontuacoes lado a lado
    tb.run(base / "out" / "treino.parquet", base / "out" / "validacao.parquet", base / "saida", modelo=modelo,
           n_min=N_MIN, max_length=MAX_LENGTH, taxa_aprendizagem=3e-3, lote=2, epocas=EPOCAS, aquecimento=0.0,
           dispositivo="cpu", checkpoints_dir=base / "checkpoints", pontuacoes_dir=base / "pontuacoes")
    resumo = json.loads((base / "saida" / tt.RESUMO).read_text(encoding="utf-8"))
    return {"base": base, "saida": base / "saida", "pontuacoes": base / "pontuacoes",
            "checkpoints": {e["epoca"]: base / "checkpoints" / e["checkpoint"] for e in resumo["por_epoca"]},
            "val_df": pd.read_parquet(base / "out" / "validacao.parquet")}


def _blocos(texto):
    return [b for b in texto.strip().split("\n\n") if b]


def _codigos(stdout):
    return [[l.split()[1] for l in b.splitlines()[1:]] for b in _blocos(stdout)]


def _gravadas(treinado, epoca, tau):
    return pd.read_parquet(treinado["saida"] / tb.epoch_dir(epoca) / tt.suggestions_file(tau))


def _sugerir(*args):
    return subprocess.run([sys.executable, str(SUGGEST), *map(str, args)], capture_output=True, text=True)


def test_comando_com_checkpoint_da_5_codigos_do_modelo_iguais_aos_gravados(treinado):
    # O critério do E5-T4, pelo comando: uma descrição da validação e uma nova, com o τ por defeito
    ck = treinado["checkpoints"][EPOCAS]
    r = _sugerir("--bert", ck, "--max-length", MAX_LENGTH, "Papel e canetas", "aquisição de papel A4")
    assert r.returncode == 0, r.stderr
    codigos = _codigos(r.stdout)
    assert len(codigos) == 2
    classes = set(json.loads((treinado["pontuacoes"] / tb.ETIQUETAS).read_text(encoding="utf-8")))
    for s in codigos:
        assert len(s) == 5 and len(set(s)) == 5
        assert all(len(c) == 8 and c.isdigit() for c in s)
        assert not any(c.endswith("000000") for c in s)
        assert set(s) <= classes
    gravadas = _gravadas(treinado, EPOCAS, tfidf.TAU).set_index("idcontrato")["sugestoes"].map(list)
    assert codigos[0] == gravadas["v-papel"]


@pytest.mark.parametrize("epoca", range(1, EPOCAS + 1))
def test_sugestoes_de_cada_checkpoint_sao_as_gravadas_nessa_epoca_com_cada_tau(treinado, epoca):
    # Todos os textos da validação, com τ = 0 e τ = 0,25: o checkpoint, as etiquetas, o log_prior e
    # a ordem do comando são os do train_bert.py
    carregado = sg.load_bert(treinado["checkpoints"][epoca], max_length=MAX_LENGTH)
    textos = treinado["val_df"]["texto"].tolist()
    for tau in (0.0, tfidf.TAU):
        sug, _ = sg.bert_suggestions(carregado, textos, tau=tau)
        assert sug == _gravadas(treinado, epoca, tau)["sugestoes"].map(list).tolist(), (epoca, tau)


def test_tau_muda_a_ordem_nalgum_checkpoint(treinado):
    # Pré-condição do teste acima: sem ela, um comando que ignorasse o τ passava
    assert any(_gravadas(treinado, k, 0.0)["sugestoes"].map(list).tolist()
               != _gravadas(treinado, k, tfidf.TAU)["sugestoes"].map(list).tolist() for k in range(1, EPOCAS + 1))


def test_comando_passa_o_tau_e_a_pasta_das_pontuacoes(treinado, tmp_path):
    # Os checkpoints copiados sem a pasta das pontuações ao lado (a por defeito não existe), as
    # pontuações noutra pasta (--pontuacoes) e τ = 0: as sugestões gravadas com τ = 0
    outra = tmp_path / "outra"
    shutil.copytree(treinado["pontuacoes"], outra)
    textos = treinado["val_df"]["texto"].tolist()
    for k in range(1, EPOCAS + 1):
        ck = tmp_path / f"so_ck_{k}" / "checkpoints" / treinado["checkpoints"][k].name
        shutil.copytree(treinado["checkpoints"][k], ck)
        assert not (ck.parent.parent / "pontuacoes").exists()
        r = _sugerir("--bert", ck, "--pontuacoes", outra, "--max-length", MAX_LENGTH, "--tau", 0, *textos)
        assert r.returncode == 0, r.stderr
        assert _codigos(r.stdout) == _gravadas(treinado, k, 0.0)["sugestoes"].map(list).tolist()


def test_texto_longo_corta_se_ao_max_length(treinado):
    # O tokenizador do BERTimbau não corta sozinho (bert_models.py): sem o max_length um texto
    # longo passava das 64 posições do modelo minúsculo e dava erro. Com 32 tokens ficam
    # [CLS] + as 30 primeiras palavras + [SEP] (no modelo minúsculo cada palavra é um token).
    carregado = sg.load_bert(treinado["checkpoints"][EPOCAS], max_length=MAX_LENGTH)
    palavras = " ".join(t for textos in TEMAS.values() for t in textos).split()
    longo = " ".join((palavras * 10)[:200])
    curto = " ".join((palavras * 10)[:MAX_LENGTH - 2])
    sug, pont = sg.bert_suggestions(carregado, [longo, curto])
    assert sug[0] == sug[1]
    np.testing.assert_array_equal(pont[0], pont[1])


def test_base_da_o_max_length_do_e5_t1(treinado):
    for base in bert_models.MODELOS:
        carregado = sg.load_bert(treinado["checkpoints"][EPOCAS], base=base)
        assert carregado["max_length"] == bert_models.COMPRIMENTO_MAXIMO[base]
    # o --max-length dado ganha
    assert sg.load_bert(treinado["checkpoints"][EPOCAS], base="bertimbau", max_length=20)["max_length"] == 20


def test_sem_base_nem_max_length_da_erro(treinado):
    r = _sugerir("--bert", treinado["checkpoints"][EPOCAS], "Papel e canetas")
    assert r.returncode != 0
    assert "--base" in r.stderr


def _copia(treinado, tmp_path):
    """Uma cópia do checkpoint da última época e das pontuações, lado a lado como no treino."""
    ck = tmp_path / "checkpoints" / treinado["checkpoints"][EPOCAS].name
    shutil.copytree(treinado["checkpoints"][EPOCAS], ck)
    shutil.copytree(treinado["pontuacoes"], tmp_path / "pontuacoes")
    return ck, tmp_path / "pontuacoes"


def test_pontuacoes_de_outro_treino_sao_recusadas(treinado, tmp_path):
    # Outras etiquetas no etiquetas.json: as frequências seriam de outras etiquetas
    ck, pont = _copia(treinado, tmp_path)
    etiquetas = json.loads((pont / tb.ETIQUETAS).read_text(encoding="utf-8"))
    (pont / tb.ETIQUETAS).write_text(json.dumps(etiquetas[1:] + ["99999999"]), encoding="utf-8")
    with pytest.raises(ValueError, match="treinos diferentes"):
        sg.load_bert(ck, max_length=MAX_LENGTH)


def test_log_prior_com_outro_tamanho_e_recusado(treinado, tmp_path):
    ck, pont = _copia(treinado, tmp_path)
    np.save(pont / tb.LOG_PRIOR, np.load(pont / tb.LOG_PRIOR)[:-1])
    with pytest.raises(ValueError, match="entradas"):
        sg.load_bert(ck, max_length=MAX_LENGTH)


def test_checkpoint_com_etiqueta_de_divisao_e_recusado(treinado, tmp_path):
    # O treino recusa divisões; se um checkpoint as tivesse, o comando podia sugeri-las
    ck, pont = _copia(treinado, tmp_path)
    config = json.loads((ck / "config.json").read_text(encoding="utf-8"))
    etiquetas = json.loads((pont / tb.ETIQUETAS).read_text(encoding="utf-8"))
    etiquetas[0] = "45000000"
    config["id2label"]["0"] = "45000000"
    config["label2id"] = {c: i for i, c in enumerate(etiquetas)}
    (ck / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (pont / tb.ETIQUETAS).write_text(json.dumps(etiquetas), encoding="utf-8")
    with pytest.raises(ValueError, match="divisão"):
        sg.load_bert(ck, max_length=MAX_LENGTH)


def test_cuda_sem_gpu_e_recusado(treinado):
    # A verificação do treino (bert.check_gpu): sem ela o erro seria do torch, a meio de carregar
    import torch
    if torch.cuda.is_available():
        pytest.skip("há GPU: a verificação passa")
    with pytest.raises(RuntimeError, match="não vê nenhuma GPU"):
        sg.load_bert(treinado["checkpoints"][EPOCAS], max_length=MAX_LENGTH, dispositivo="cuda")
    # e o comando passa o --dispositivo
    r = _sugerir("--bert", treinado["checkpoints"][EPOCAS], "--max-length", MAX_LENGTH, "--dispositivo", "cuda", "x")
    assert r.returncode != 0 and "não vê nenhuma GPU" in r.stderr, r.stderr


def test_nomes_dos_ficheiros_sao_os_do_treino():
    # O suggest.py repete-os para não importar o torch no comando do TF-IDF
    assert (sg.ETIQUETAS, sg.LOG_PRIOR) == (tb.ETIQUETAS, tb.LOG_PRIOR)
