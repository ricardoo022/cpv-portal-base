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
from test_pipeline import CONTRATOS_REGUA, run_pipeline

# N = 6: com estes contratos, 33141400 (5 exemplos no treino) é raro, a classe 3314 só com ele
# também, e o exemplo sai. Se a validação (mais 3 contratos 33141400) entrasse nas contagens,
# o 33141400 passava a etiqueta. Ficam 7 etiquetas, chega para 5 sugestões.
N_MIN = 6
SO_VALIDACAO = "zzpalavravalidacao"  # palavra que só aparece nos textos da validação

TREINO_REGUA = [c for c in CONTRATOS_REGUA if c[2] < datetime(2025, 10, 1)]
VALIDACAO = [
    (2025, "v-limpeza", datetime(2025, 10, 2), "Limpeza 0", None, None, "90919300-5 - Limpeza"),  # texto visto
    (2025, "v-papel", datetime(2025, 10, 3), f"Papel {SO_VALIDACAO} A4", None, None, "30197630-1 - Papel"),
    *[(2025, f"v-luvas-{i}", datetime(2025, 10, 4), f"Luvas {SO_VALIDACAO} {i}", None, None, "33141400-0 - Luvas")
      for i in range(3)],
    (2025, "v-obras", datetime(2025, 10, 5), f"Obras {SO_VALIDACAO}", None, None, "45000000-0 - Obras"),
]
TESTE = [(2025, "t-1", datetime(2025, 11, 5), "Papel novo", None, None, "30197630-1 - Papel")]
CONTRATOS = TREINO_REGUA + VALIDACAO + TESTE


@pytest.fixture(scope="module")
def treinado(tmp_path_factory):
    base = tmp_path_factory.mktemp("tfidf")
    run_pipeline(base, CONTRATOS)
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
        "data", "n_min", "min_df", "treino", "validacao", "exemplos_usados", "exemplos_saidos",
        "etiquetas", "vocabulario", "n_iter", "convergiu", "tempo_vetorizar_treinar_s",
        "tempo_sugerir_s", "pico_rss_mb", "nucleos", "versoes", "coef_bytes"}
    tr = treinado["treino_df"]
    assert resumo["n_min"] == N_MIN and resumo["min_df"] == 1
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
    assert tt.tfidf.suggest(gravado, textos).tolist() == treinado["sugestoes"]["sugestoes"].map(list).tolist()


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


def test_probabilidades_por_ordem_decrescente(treinado):
    out = sg.format_suggestions(treinado["model"], ["papel de impressão"])
    probs = [float(l.split("probabilidade ")[1].rstrip(")")) for l in out.splitlines()[1:] if l]
    assert len(probs) == 5 and probs == sorted(probs, reverse=True)
