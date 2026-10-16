"""Testes do sinal de erro 1 do E5 (`scripts/sinal_decorar.py`): decorar poucos contratos do treino.

Gera .xlsx inventados, corre a conversão e a preparação (`run_pipeline`) com os contratos da
régua (7 etiquetas com N = 10, uma delas a classe 33140000 que junta dois códigos raros) e treina
um modelo minúsculo no processador com 4 desses contratos. O que se guarda: a camada de
classificação tem as etiquetas do treino completo, não só as dos contratos sorteados; o modelo
decora-os; um erro de ligação (etiquetas desalinhadas dos textos) faz o sinal falhar; e as
mesmas recusas do treino a sério (o teste, a GPU, o YAML).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import bert
import train_bert  # noqa: F401  (o conftest substitui o git_info só se o train_bert já estiver importado)
from conftest import GIT_FALSO, make_tiny_model
from labels import collapse_rare_codes

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "scripts"))
import sinal_decorar as sd  # noqa: E402

N = 4
RAPIDO = dict(n=N, max_length=16, taxa_aprendizagem=3e-3, lote=2, epocas=20, aquecimento=0.0, dispositivo="cpu")


@pytest.fixture(scope="module")
def dados(tmp_path_factory, run_pipeline, contratos_regua):
    base = tmp_path_factory.mktemp("sinal")
    run_pipeline(base, contratos_regua)
    treino = base / "out" / "treino.parquet"
    df = pd.read_parquet(treino)
    return {"treino": treino, "validacao": base / "out" / "validacao.parquet", "df": df,
            "modelo": make_tiny_model(base / "modelo", df["texto"])}


def _correr(dados, saida, **kw):
    return sd.run(dados["treino"], saida, modelo=dados["modelo"], **{**RAPIDO, **kw})


def test_modelo_tem_as_etiquetas_do_treino_completo_e_decora_os_contratos(dados, tmp_path, monkeypatch):
    # A camada de classificação do modelo treinado: uma saída por etiqueta do treino completo
    treinar, saidas = bert.train, []

    def guarda_o_modelo(*a, **k):
        out = treinar(*a, **k)
        saidas.append(out["model"])
        return out

    monkeypatch.setattr(sd.bert, "train", guarda_o_modelo)
    r = _correr(dados, tmp_path / "saida")

    esperadas = sorted(collapse_rare_codes(dados["df"]["codigo"], 10).dropna().unique())
    assert len(esperadas) == 7
    assert r["etiquetas"] == len(esperadas) == saidas[0].config.num_labels
    assert [saidas[0].config.id2label[i] for i in range(len(esperadas))] == esperadas
    # Os contratos sorteados não chegam para todas: só o treino completo dá as 7
    assert r["etiquetas_nos_contratos"] < len(esperadas)

    assert (r["acertos"], r["contratos"], r["criterio_acertos"], r["passou"]) == (N, N, 4, True)
    assert r["top1"] == 1.0
    assert [e["epoca"] for e in r["por_epoca"]] == list(range(1, 21))
    assert r["por_epoca"][-1]["acertos"] == N and all(e["perda_treino"] is not None for e in r["por_epoca"])
    # Cada contrato com a etiqueta do treino completo (a regra dos raros contada em todo o treino)
    etiqueta = collapse_rare_codes(dados["df"]["codigo"], 10)
    por_id = dict(zip(dados["df"]["idcontrato"].astype(str), etiqueta))
    assert len(r["contratos_sorteados"]) == N
    for c in r["contratos_sorteados"]:
        assert c["etiqueta"] == por_id[c["idcontrato"]] == c["top1"]
    # O ficheiro gravado é o resultado devolvido
    assert json.loads((tmp_path / "saida" / sd.RESULTADO).read_text(encoding="utf-8")) == r
    assert (r["taxa_aprendizagem"], r["epocas"], r["semente"], r["n_min"], r["max_length"]) == (3e-3, 20, 0, 10, 16)
    assert r["treino"]["sha256"] == sd.file_sha256(dados["treino"])
    assert r["git"] == GIT_FALSO and r["memoria_gpu_mb"] is None and r["gpu"] is None
    # Sem checkpoints: nada fica fora da pasta de saída
    assert sorted(p.name for p in (tmp_path / "saida").iterdir()) == [sd.RESULTADO]


def test_etiquetas_desalinhadas_dos_textos_fazem_o_sinal_falhar(dados, tmp_path, monkeypatch):
    # O erro que o sinal existe para apanhar: cada texto treinado com a etiqueta de outro
    treinar = bert.train

    def desalinhado(texto, codigo, *a, **k):
        return treinar(texto, pd.Series(np.roll(codigo.to_numpy(), 1), index=codigo.index), *a, **k)

    monkeypatch.setattr(sd.bert, "train", desalinhado)
    # Uma semente em que os 4 contratos sorteados têm etiquetas todas diferentes (rodar muda todas)
    r = _correr(dados, tmp_path / "saida", semente=_semente_com_etiquetas_diferentes(dados))
    assert not r["passou"] and r["acertos"] < r["criterio_acertos"]


def _semente_com_etiquetas_diferentes(dados):
    etiqueta = collapse_rare_codes(dados["df"]["codigo"], 10)
    return next(s for s in range(100) if etiqueta.loc[sd.sample(etiqueta, N, s)].nunique() == N)


def test_sorteio_repete_se_com_a_mesma_semente_e_muda_com_outra(dados):
    etiqueta = collapse_rare_codes(dados["df"]["codigo"], 10)
    assert list(sd.sample(etiqueta, N, 0)) == list(sd.sample(etiqueta, N, 0))
    assert list(sd.sample(etiqueta, N, 0)) != list(sd.sample(etiqueta, N, 1))


def test_sorteio_so_escolhe_contratos_com_etiqueta():
    # Os que a regra dos códigos raros tira (NaN) não entram em nenhum treino
    etiqueta = pd.Series(["a", None, "b", None, "c", None], index=[10, 11, 12, 13, 14, 15])
    for s in range(20):
        assert sorted(sd.sample(etiqueta, 3, s)) == [10, 12, 14]
    with pytest.raises(ValueError, match="só tem 3"):
        sd.sample(etiqueta, 4)


# --- Recusas, antes de treinar ---

def _nao_treina(monkeypatch):
    def falha(*a, **k):
        raise AssertionError("treinou")
    monkeypatch.setattr(sd.bert, "train", falha)


def test_ficheiro_chamado_teste_e_recusado(dados, tmp_path, monkeypatch):
    _nao_treina(monkeypatch)
    teste = tmp_path / "teste.parquet"
    teste.write_bytes(dados["treino"].read_bytes())
    with pytest.raises(ValueError, match="teste"):
        sd.run(teste, tmp_path / "saida", modelo=dados["modelo"], **{k: v for k, v in RAPIDO.items()})
    assert not (tmp_path / "saida").exists()


@pytest.mark.parametrize("qual", ["validacao", "teste"])
def test_contratos_da_validacao_ou_do_teste_sao_recusados_pelas_datas(dados, tmp_path, monkeypatch, qual):
    _nao_treina(monkeypatch)
    copia = tmp_path / "outro_nome.parquet"
    copia.write_bytes((dados["treino"].parent / f"{qual}.parquet").read_bytes())
    with pytest.raises(ValueError, match="2025-10-01"):
        sd.run(copia, tmp_path / "saida", modelo=dados["modelo"], **RAPIDO)
    assert not (tmp_path / "saida").exists()


def test_sem_gpu_recusa_antes_de_ler_os_dados(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="GPU"):
        sd.run(tmp_path / "nao_existe.parquet", tmp_path / "saida", taxa_aprendizagem=1e-4, dispositivo="cuda")


@pytest.mark.parametrize("sem", ["experiencia", "commit"])
def test_na_gpu_sem_yaml_ou_sem_commit_recusa(dados, tmp_path, monkeypatch, sem):
    monkeypatch.setattr(sd.bert, "check_gpu", lambda: "GPU fingida")
    _nao_treina(monkeypatch)
    yaml_ok = tmp_path / "e.yaml"
    yaml_ok.write_text("taxa_aprendizagem: 1.0e-4\n", encoding="utf-8")
    if sem == "commit":
        monkeypatch.setattr(sd.train_bert, "git_info", lambda: {"commit": None, "alteracoes_por_gravar": None})
    kw = {} if sem == "experiencia" else {"experiencia": yaml_ok}
    with pytest.raises(ValueError, match="obrigatório"):
        sd.run(dados["treino"], tmp_path / "saida", modelo=dados["modelo"], **{**RAPIDO, "dispositivo": "cuda", **kw})


@pytest.mark.parametrize("taxa", [None, 0.0])
def test_sem_taxa_de_aprendizagem_recusa(dados, tmp_path, monkeypatch, taxa):
    _nao_treina(monkeypatch)
    with pytest.raises(ValueError, match="taxa"):
        sd.run(dados["treino"], tmp_path / "saida", modelo=dados["modelo"], **{**RAPIDO, "taxa_aprendizagem": taxa})


def test_resultado_de_outro_sinal_na_pasta_e_recusado(dados, tmp_path, monkeypatch):
    _nao_treina(monkeypatch)
    (tmp_path / "saida").mkdir()
    (tmp_path / "saida" / sd.RESULTADO).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="já existe"):
        _correr(dados, tmp_path / "saida")


def test_yaml_da_experiencia_fica_no_resultado(dados, tmp_path):
    texto = "# E5-T4\nsinal_decorar:\n  taxa_aprendizagem: 3.0e-3\n"
    yaml_path = tmp_path / "e5_t4.yaml"
    yaml_path.write_text(texto, encoding="utf-8")
    r = _correr(dados, tmp_path / "saida", epocas=1, experiencia=yaml_path)
    assert r["experiencia"]["conteudo"] == texto and r["experiencia"]["sha256"] == sd.file_sha256(yaml_path)


# --- O comando ---

def test_comando_no_processador_grava_e_acaba_com_0_se_passou(dados, tmp_path, capsys):
    codigo = sd.main(["--treino", str(dados["treino"]), "--modelo", str(dados["modelo"]),
                      "--saida", str(tmp_path / "saida"), "--taxa-aprendizagem", "3e-3", "--n", str(N),
                      "--epocas", "20", "--lote", "2", "--aquecimento", "0", "--max-length", "16",
                      "--dispositivo", "cpu"])
    assert codigo == 0 and "passou" in capsys.readouterr().out
    r = json.loads((tmp_path / "saida" / sd.RESULTADO).read_text(encoding="utf-8"))
    assert (r["contratos"], r["epocas"], r["lote"], r["passou"]) == (N, 20, 2, True)


def test_comando_acaba_com_1_se_o_sinal_falhou(dados, tmp_path, capsys):
    # Uma época com uma taxa minúscula: não decora; o ficheiro fica gravado na mesma
    codigo = sd.main(["--treino", str(dados["treino"]), "--modelo", str(dados["modelo"]),
                      "--saida", str(tmp_path / "saida"), "--taxa-aprendizagem", "1e-9", "--n", str(N),
                      "--epocas", "1", "--max-length", "16", "--dispositivo", "cpu"])
    assert codigo == 1 and "FALHOU" in capsys.readouterr().out
    assert not json.loads((tmp_path / "saida" / sd.RESULTADO).read_text(encoding="utf-8"))["passou"]


def test_comando_sem_taxa_de_aprendizagem_da_erro(dados, tmp_path):
    with pytest.raises(SystemExit):
        sd.main(["--treino", str(dados["treino"]), "--saida", str(tmp_path / "saida"), "--dispositivo", "cpu"])
