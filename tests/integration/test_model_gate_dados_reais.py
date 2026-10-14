"""O campeão guardado (`metrics/validacao.json`) volta a dar-se com a validação real.

Avalia outra vez as sugestões que o ficheiro indica e confirma que as métricas são as guardadas:
se a preparação ou a avaliação mudarem, o teste falha e o portão tem de correr outra vez
(`make portao`). Salta se data/processed/ não existir (por exemplo, no CI). Nunca lê o teste.
"""

import json

import pytest

import model_gate as mg
from train_tfidf import VALIDACAO

dados = pytest.mark.skipif(not VALIDACAO.exists(), reason="dados preparados ausentes (correr make preparar)")


@pytest.mark.data
@dados
def test_metricas_do_campeao_sao_as_da_validacao_atual():
    campeao = json.loads(mg.CAMPEAO.read_text(encoding="utf-8"))
    block, _ = mg.evaluate_run((mg.RAIZ / campeao["sugestoes"]).parent)
    assert block["sugestoes"] == campeao["sugestoes"]
    assert block["periodo"] == campeao["periodo"]
    assert block["contratos"] == campeao["contratos"]
    assert block["contratos_sha256"] == campeao["contratos_sha256"]
    assert block["metricas"] == campeao["metricas"]
