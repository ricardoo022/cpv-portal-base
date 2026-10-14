"""Testes do portão de qualidade (E4-T5), sem dados: correm no CI.

Dois grupos:
- o campeão guardado no git (`metrics/validacao.json`): vem da validação de outubro de 2025 e
  nenhum top-5 desceu face ao do `main` (o CI escreve o do `main` em `VALIDACAO_MAIN`);
- as regras do portão com blocos inventados: `check_block` (conjunto, período, métricas) e
  `check_not_worse` (qualquer descida falha, empate passa).
"""

import json
import os
from pathlib import Path

import pandas as pd
import pytest
from mlflow.models.evaluation.validation import ModelValidationFailedException

import model_gate as mg

CAMPEAO = {m: 0.8 for m in mg.METRICAS}


def bloco(**muda):
    b = {"conjunto": "validacao", "periodo": ["2025-10-01", "2025-10-31"], "metricas": dict(CAMPEAO)}
    b.update(muda)
    return b


# --- O campeão guardado no git ---

def _guardado():
    return json.loads(mg.CAMPEAO.read_text(encoding="utf-8"))


def test_campeao_guardado_vem_da_validacao():
    mg.check_block(_guardado())


def test_campeao_guardado_nao_desce_face_ao_main():
    main = os.environ.get("VALIDACAO_MAIN")
    if not main:
        pytest.skip("sem o campeão do main (só o CI o escreve, e o main pode ainda não o ter)")
    do_main = json.loads(Path(main).read_text(encoding="utf-8"))
    mg.check_not_worse(_guardado()["metricas"], do_main["metricas"])


# --- check_block ---

def test_bloco_da_validacao_passa():
    mg.check_block(bloco())


@pytest.mark.parametrize("conjunto", ["teste", "treino", None])
def test_bloco_de_outro_conjunto_falha(conjunto):
    with pytest.raises(mg.GateError, match="validação"):
        mg.check_block(bloco(conjunto=conjunto))


@pytest.mark.parametrize("periodo", [
    ["2025-10-01", "2025-11-01"],   # entra no teste
    ["2025-11-01", "2025-12-31"],   # é o teste
    ["2025-09-30", "2025-10-31"],   # entra no treino
])
def test_bloco_fora_de_outubro_de_2025_falha(periodo):
    with pytest.raises(mg.GateError, match="período"):
        mg.check_block(bloco(periodo=periodo))


def test_bloco_sem_uma_metrica_falha():
    metricas = dict(CAMPEAO)
    del metricas["top5_classe_texto_novo"]
    with pytest.raises(mg.GateError, match="top5_classe_texto_novo"):
        mg.check_block(bloco(metricas=metricas))


# --- check_not_worse ---

def test_portao_olha_para_o_codigo_e_a_classe_em_todos_e_no_texto_novo():
    assert set(mg.METRICAS) == {"top5_codigo_todos", "top5_classe_todos",
                                "top5_codigo_texto_novo", "top5_classe_texto_novo"}


def test_empate_passa():
    mg.check_not_worse(dict(CAMPEAO), CAMPEAO)


def test_subir_passa():
    mg.check_not_worse({m: v + 0.01 for m, v in CAMPEAO.items()}, CAMPEAO)


@pytest.mark.parametrize("metrica", mg.METRICAS)
def test_qualquer_descida_numa_metrica_falha(metrica):
    novas = {m: v + 0.05 for m, v in CAMPEAO.items()}  # as outras sobem
    novas[metrica] = CAMPEAO[metrica] - 1e-6
    with pytest.raises(ModelValidationFailedException, match=metrica):
        mg.check_not_worse(novas, CAMPEAO)


def test_metrica_em_falta_no_novo_falha():
    novas = dict(CAMPEAO)
    del novas["top5_codigo_todos"]
    with pytest.raises(ModelValidationFailedException, match="missing"):
        mg.check_not_worse(novas, CAMPEAO)


# --- check_same_validation ---

VALIDACAO = {"contratos": 2, "periodo": ["2025-10-01", "2025-10-31"],
             "contratos_sha256": mg.contracts_fingerprint(["a", "b"])}


def test_mesma_validacao_passa():
    mg.check_same_validation(dict(VALIDACAO), VALIDACAO)


@pytest.mark.parametrize("muda", [{"contratos": 3}, {"periodo": ["2025-10-02", "2025-10-31"]},
                                  {"contratos_sha256": mg.contracts_fingerprint(["a", "c"])}])
def test_validacao_diferente_do_campeao_falha(muda):
    campeao = VALIDACAO
    with pytest.raises(mg.GateError, match="avaliar outra vez o campeão"):
        mg.check_same_validation({**campeao, **muda}, campeao)


def test_impressao_digital_muda_com_os_contratos_e_com_a_ordem():
    base = mg.contracts_fingerprint(["a", "b"])
    assert mg.contracts_fingerprint(["a", "b"]) == base
    assert mg.contracts_fingerprint(["a", "c"]) != base
    assert mg.contracts_fingerprint(["b", "a"]) != base


def test_bloco_sem_periodo_da_erro_do_portao():
    b = bloco()
    del b["periodo"]
    with pytest.raises(mg.GateError, match="período"):
        mg.check_block(b)


def test_cada_grupo_e_medida_vai_para_o_seu_nome():
    # Relatório no formato do `ev.report`, com um top-5 diferente em cada linha
    linhas = [(g, m) for g in ("todos", "texto novo") for m in ("codigo", "classe", "divisao", "so_divisao")]
    rep = pd.DataFrame({"top5": [i / 10 for i in range(len(linhas))]},
                       index=pd.MultiIndex.from_tuples(linhas, names=["grupo", "medida"]))
    assert mg.gate_metrics(rep) == {"top5_codigo_todos": 0.0, "top5_classe_todos": 0.1,
                                    "top5_codigo_texto_novo": 0.4, "top5_classe_texto_novo": 0.5}
