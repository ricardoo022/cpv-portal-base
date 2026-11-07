"""Os resultados gravados do E5-T4 (`metrics/bert/e5_t4_*`) e os números que os `docs/` citam deles.

O treino pago já não se repete (o pod e os checkpoints foram-se), por isso estas pastas são registos
fixos, como as `e4_*`. Os testes guardam três coisas:

- o YAML pré-registado (`experiencias/e5_t4.yaml`) não mudou depois de correr: o SHA-256 gravado no
  ensaio, nos dois sinais e no treino tem de ser o do ficheiro de hoje;
- os ficheiros dizem o que as secções 21 e 22 do `qualidade_dados.md` e o `pod_info.json` dizem;
- (só com dados) as sugestões gravadas, avaliadas outra vez na validação, dão o top-5 escrito na
  secção 22. Salta no CI, que não tem `data/`. Nunca lê o teste.
"""

import json
from pathlib import Path

import pytest

import model_gate as mg
from cpv_data import file_sha256
from train_tfidf import VALIDACAO

RAIZ = Path(__file__).resolve().parents[2]
BERT = RAIZ / "metrics" / "bert"
YAML = RAIZ / "experiencias" / "e5_t4.yaml"
COMMIT = "840360652ab13667ffd09fde92d28a29278fd0a4"
SHA_TREINO = "0769501ae2a306dbc1fe72e1aa64ccdd5cd86b15f269ab707110a6fc0457a8a7"

TREINO = BERT / "e5_t4_bertimbau" / "treino.json"
ENSAIO = BERT / "e5_t4_ensaio" / "treino.json"
SINAL = BERT / "e5_t4_sinal_decorar" / "sinal_decorar.json"
DIAGNOSTICO = BERT / "e5_t4_sinal_decorar_diag_100ep" / "sinal_decorar.json"
POD = BERT / "e5_t4_bertimbau" / "pod_info.json"

dados = pytest.mark.skipif(not VALIDACAO.exists(), reason="dados preparados ausentes (correr make preparar)")


def _ler(caminho):
    return json.loads(caminho.read_text(encoding="utf-8"))


@pytest.mark.parametrize("caminho", [ENSAIO, SINAL, DIAGNOSTICO, TREINO], ids=lambda p: p.parent.name)
def test_correu_com_o_yaml_de_hoje_num_commit_limpo(caminho):
    d = _ler(caminho)
    assert d["experiencia"]["ficheiro"] == "experiencias/e5_t4.yaml"
    assert d["experiencia"]["sha256"] == file_sha256(YAML), "o YAML pré-registado mudou depois de correr"
    assert d["git"] == {"commit": COMMIT, "alteracoes_por_gravar": False}
    assert d["dispositivo"] == "cuda"
    sha_treino = d["sha256"]["treino"] if "sha256" in d else d["treino"]["sha256"]
    assert sha_treino == SHA_TREINO


def test_treino_completo_tem_a_receita_do_yaml_e_nao_le_o_teste():
    d = _ler(TREINO)
    assert (d["taxa_aprendizagem"], d["lote"], d["epocas"], d["aquecimento"], d["decaimento_pesos"]) == (
        2e-05, 32, 3, 0.1, 0.01)
    assert (d["n_min"], d["max_length"], d["etiquetas"], d["ensaio"]) == (10, 256, 3000, False)
    assert d["treino"]["ficheiro"] == "treino.parquet" and d["validacao"]["ficheiro"] == "validacao.parquet"
    assert [e["epoca"] for e in d["por_epoca"]] == [1, 2, 3]
    assert d["ficheiro_do_modelo_por_defeito"] == "sugestoes_validacao_tau0.25.parquet"


def test_sinal_oficial_falhou_e_o_diagnostico_passou_na_epoca_27():
    """Secção 21: 86 de 100 em 30 épocas, nenhuma a chegar aos 95; com 100 épocas, 95 na época 27."""
    oficial, diag = _ler(SINAL), _ler(DIAGNOSTICO)
    assert (oficial["epocas"], oficial["acertos"], oficial["passou"]) == (30, 86, False)
    assert oficial["primeira_epoca_criterio"] is None
    assert max(e["acertos"] for e in oficial["por_epoca"]) < oficial["criterio_acertos"] == 95
    assert (diag["epocas"], diag["acertos"], diag["passou"], diag["primeira_epoca_criterio"]) == (100, 100, True, 27)
    # O diagnóstico só muda as épocas: a amostra e o resto da condição são os mesmos.
    for chave in ("taxa_aprendizagem", "lote", "aquecimento", "decaimento_pesos", "semente"):
        assert oficial[chave] == diag[chave], chave

    def amostra(d):  # sem o `top1`, que é a resposta do modelo e muda com as épocas
        return [(c["idcontrato"], c["etiqueta"]) for c in d["contratos_sorteados"]]

    assert amostra(oficial) == amostra(diag)


def test_custo_e_memoria_do_pod_info_saem_do_treino_json():
    treino, pod = _ler(TREINO), _ler(POD)
    tc = pod["treino_completo"]
    assert tc["tempo_treinar_s"] == treino["tempo_treinar_s"]
    assert tc["memoria_gpu_pico_alocada_mb"] == treino["memoria_gpu_mb"]["pico_alocada"]
    assert tc["memoria_gpu_pico_reservada_mb"] == treino["memoria_gpu_mb"]["pico_reservada"]
    preco = pod["pod"]["preco_por_hora_usd"]
    assert tc["custo_usd_calculado"] == round(treino["tempo_treinar_s"] / 3600 * preco, 2)
    assert pod["pod_completo"]["custo_usd_calculado"] == round(pod["pod_completo"]["segundos_de_vida"] / 3600 * preco, 2)
    assert pod["pod"]["gpu"] == treino["gpu"]


@pytest.mark.data
@dados
def test_top5_da_secao_22_volta_a_dar_se_com_a_validacao_atual():
    block, _ = mg.evaluate_run(TREINO.parent)
    assert block["contratos"] == 23945
    m = {k: round(100 * v, 2) for k, v in block["metricas"].items()}
    assert m == {"top5_codigo_todos": 73.39, "top5_classe_todos": 79.59,
                 "top5_codigo_texto_novo": 70.36, "top5_classe_texto_novo": 77.76}
