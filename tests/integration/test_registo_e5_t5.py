"""Os resultados gravados do E5-T5 (`metrics/bert/e5_t5_*`) e os números que os `docs/` citam deles.

O treino pago já não se repete (o pod e os checkpoints foram-se), por isso estas pastas são registos
fixos, como as `e4_*` e as `e5_t4_*`. Os testes guardam três coisas:

- o YAML pré-registado (`experiencias/e5_t5.yaml`) não mudou depois de correr: o SHA-256 gravado no
  sinal, no ensaio e no treino tem de ser o do ficheiro de hoje;
- os ficheiros dizem o que a secção 23 do `qualidade_dados.md` e o `pod_info.json` dizem;
- (só com dados) as sugestões gravadas, avaliadas outra vez na validação, dão o top-5 escrito na
  secção 23. Salta no CI, que não tem `data/`. Nunca lê o teste.
"""

import json
from pathlib import Path

import pytest

import model_gate as mg
from cpv_data import file_sha256
from train_tfidf import VALIDACAO

RAIZ = Path(__file__).resolve().parents[2]
BERT = RAIZ / "metrics" / "bert"
YAML = RAIZ / "experiencias" / "e5_t5.yaml"
COMMIT = "98dcb338f567dc0524248813268151934c54c54a"
SHA_TREINO = "0769501ae2a306dbc1fe72e1aa64ccdd5cd86b15f269ab707110a6fc0457a8a7"
SHA_VALIDACAO = "f0cce1a06d532124feb25eaa963600b7ef2e8b24b0ae42c4955e6b438782b9c9"

TREINO = BERT / "e5_t5_albertina" / "treino.json"
ENSAIO = BERT / "e5_t5_ensaio" / "treino.json"
SINAL = BERT / "e5_t5_sinal_decorar" / "sinal_decorar.json"
POD = BERT / "e5_t5_albertina" / "pod_info.json"
TREINO_E5_T4 = BERT / "e5_t4_bertimbau" / "treino.json"

dados = pytest.mark.skipif(not VALIDACAO.exists(), reason="dados preparados ausentes (correr make preparar)")


def _ler(caminho):
    return json.loads(caminho.read_text(encoding="utf-8"))


@pytest.mark.parametrize("caminho", [SINAL, ENSAIO, TREINO], ids=lambda p: p.parent.name)
def test_correu_com_o_yaml_de_hoje_num_commit_limpo(caminho):
    d = _ler(caminho)
    assert d["experiencia"]["ficheiro"] == "experiencias/e5_t5.yaml"
    assert d["experiencia"]["sha256"] == file_sha256(YAML), "o YAML pré-registado mudou depois de correr"
    assert d["git"] == {"commit": COMMIT, "alteracoes_por_gravar": False}
    assert d["dispositivo"] == "cuda"
    assert d["modelo"]["nome"] == "PORTULAN/albertina-100m-portuguese-ptpt-encoder"
    sha_treino = d["sha256"]["treino"] if "sha256" in d else d["treino"]["sha256"]
    assert sha_treino == SHA_TREINO


@pytest.mark.parametrize("caminho", [ENSAIO, TREINO], ids=lambda p: p.parent.name)
def test_ensaio_e_treino_leram_a_validacao_do_portatil(caminho):
    # O sinal de decorar não lê a validação; o ensaio e o treino gravam o SHA-256 das duas
    d = _ler(caminho)
    assert d["sha256"] == {"treino": SHA_TREINO, "validacao": SHA_VALIDACAO}
    argumentos = _ler(caminho.parent / "argumentos.json")
    assert (argumentos["treino_sha256"], argumentos["validacao_sha256"]) == (SHA_TREINO, SHA_VALIDACAO)


def test_treino_completo_tem_a_receita_do_yaml_e_a_do_e5_t4():
    d = _ler(TREINO)
    receita = ("taxa_aprendizagem", "lote", "epocas", "aquecimento", "decaimento_pesos", "semente",
               "n_min", "max_length", "etiquetas")
    assert tuple(d[k] for k in receita) == (2e-05, 32, 3, 0.1, 0.01, 0, 10, 256, 3000)
    # Decisão do utilizador (7 de novembro de 2026): entre o E5-T4 e o E5-T5 só muda o modelo
    t4 = _ler(TREINO_E5_T4)
    assert tuple(t4[k] for k in receita) == tuple(d[k] for k in receita)
    assert d["ensaio"] is False
    assert d["treino"]["ficheiro"] == "treino.parquet" and d["validacao"]["ficheiro"] == "validacao.parquet"
    assert [e["epoca"] for e in d["por_epoca"]] == [1, 2, 3]
    assert d["ficheiro_do_modelo_por_defeito"] == "sugestoes_validacao_tau0.25.parquet"


def test_sinal_de_decorar_passou_na_epoca_18_com_100_epocas():
    """Secção 23: 100 de 100 na última época, 95 ou mais pela primeira vez na época 18."""
    d = _ler(SINAL)
    assert (d["epocas"], d["contratos"], d["etiquetas"], d["taxa_aprendizagem"]) == (100, 100, 3000, 1e-4)
    assert (d["acertos"], d["passou"], d["primeira_epoca_criterio"], d["criterio_acertos"]) == (100, True, 18, 95)
    acertos = {e["epoca"]: e["acertos"] for e in d["por_epoca"]}
    assert max(acertos[k] for k in range(1, 18)) < 95 <= acertos[18]
    assert (acertos[10], acertos[16], acertos[17], acertos[18]) == (44, 94, 92, 99)


def test_sinal_usou_a_mesma_amostra_do_bertimbau():
    """A mesma semente sorteia os mesmos 100 contratos com as mesmas etiquetas nos dois modelos."""
    def amostra(d):
        return [(c["idcontrato"], c["etiqueta"]) for c in d["contratos_sorteados"]]

    t4 = _ler(BERT / "e5_t4_sinal_decorar" / "sinal_decorar.json")
    assert amostra(_ler(SINAL)) == amostra(t4)


def test_custo_e_memoria_do_pod_info_saem_do_treino_json():
    treino, pod = _ler(TREINO), _ler(POD)
    tc = pod["treino_completo"]
    assert tc["tempo_treinar_s"] == treino["tempo_treinar_s"]
    assert tc["memoria_gpu_pico_alocada_mb"] == treino["memoria_gpu_mb"]["pico_alocada"]
    assert tc["memoria_gpu_pico_reservada_mb"] == treino["memoria_gpu_mb"]["pico_reservada"]
    preco = pod["pod"]["preco_por_hora_usd"]
    assert tc["custo_usd_calculado"] == round(treino["tempo_treinar_s"] / 3600 * preco, 2)
    assert pod["pod_completo"]["custo_usd_calculado"] == round(pod["pod_completo"]["segundos_de_vida"] / 3600 * preco, 2)
    assert pod["pod"]["gpu"] == treino["gpu"] == _ler(TREINO_E5_T4)["gpu"]  # a mesma placa do E5-T4


def test_ensaio_mediu_quase_o_dobro_do_tempo_por_passo_do_bertimbau():
    """Secção 23: 0,1418 s por passo contra 0,073 do BERTimbau, e 21 700 MB reservados no pico."""
    e5, e4 = _ler(ENSAIO), _ler(BERT / "e5_t4_ensaio" / "treino.json")
    assert (e5["segundos_por_passo"], e5["segundos_por_epoca_estimados"]) == (0.1418, 2118.5)
    assert e5["memoria_gpu_mb"] == {"pico_alocada": 16781, "pico_reservada": 21700}
    assert e5["passos_por_epoca"] == e4["passos_por_epoca"]
    assert round(e5["segundos_por_passo"] / e4["segundos_por_passo"], 2) == 1.94


def test_sugestoes_de_exemplo_sao_5_e_nenhuma_e_uma_divisao():
    linhas = (TREINO.parent / "sugestoes_exemplos.txt").read_text(encoding="utf-8").splitlines()
    descricoes = [l for l in linhas if l.startswith("Descrição:")]
    codigos = [l.split()[1] for l in linhas if l.startswith("  ")]
    assert len(descricoes) == 3 and len(codigos) == 15
    assert all(len(c) == 8 and c.isdigit() and not c.endswith("000000") for c in codigos)


@pytest.mark.data
@dados
@pytest.mark.parametrize("pasta, ficheiro, esperado", [
    ("epoca_1", "sugestoes_validacao_tau0.25.parquet",
     {"top5_codigo_todos": 71.10, "top5_classe_todos": 77.12, "top5_codigo_texto_novo": 68.44,
      "top5_classe_texto_novo": 75.66}),
    ("epoca_2", "sugestoes_validacao_tau0.25.parquet",
     {"top5_codigo_todos": 75.73, "top5_classe_todos": 81.50, "top5_codigo_texto_novo": 72.84,
      "top5_classe_texto_novo": 79.83}),
    (".", "sugestoes_validacao_tau0.25.parquet",
     {"top5_codigo_todos": 76.98, "top5_classe_todos": 82.38, "top5_codigo_texto_novo": 74.12,
      "top5_classe_texto_novo": 80.60}),
    (".", "sugestoes_validacao.parquet", {"top5_codigo_todos": 77.20}),
])
def test_curva_e_tau0_da_secao_23(pasta, ficheiro, esperado):
    """A tabela da secção 23 e o τ = 0 da última época, das sugestões gravadas."""
    import pandas as pd

    import evaluation as ev

    val = pd.read_parquet(VALIDACAO, columns=["idcontrato", "codigos", "visto_no_treino"])
    sug = pd.read_parquet(TREINO.parent / pasta / ficheiro, columns=["idcontrato", "sugestoes"])
    assert sug["idcontrato"].astype(str).tolist() == val["idcontrato"].astype(str).tolist()
    sugestoes = pd.Series([list(s) for s in sug["sugestoes"]], index=val.index)
    m = mg.gate_metrics(ev.report(sugestoes, val["codigos"], val["visto_no_treino"]))
    assert {k: round(100 * m[k], 2) for k in esperado} == esperado


@pytest.mark.data
@dados
def test_o_portao_le_a_ultima_epoca_com_tau_025():
    block, _ = mg.evaluate_run(TREINO.parent)
    assert block["contratos"] == 23945
    assert round(100 * block["metricas"]["top5_codigo_todos"], 2) == 76.98
