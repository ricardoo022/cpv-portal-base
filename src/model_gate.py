"""Portão de qualidade (E4-T5): um modelo novo não pode ficar abaixo do campeão na validação.

Padrão campeão / desafiante (guia de MLOps da Google; passo *Evaluator* do TFX): o campeão é o
melhor modelo até agora, e as suas métricas da validação ficam no git em `metrics/validacao.json`.
Um modelo novo (o desafiante) avalia-se na mesma validação. Se algum top-5 do portão (código
completo ou classe, em todos os contratos ou só nos de texto novo) ficar abaixo do do campeão, o
portão falha e o ficheiro não muda. Se ficarem todos iguais ou acima, o novo passa a campeão e o
ficheiro é reescrito. Qualquer descida conta: retreinar a mesma configuração dá exatamente os
mesmos números (E4-S1), por isso uma descida não é acaso.

A comparação é a do MLflow (`validate_evaluation_results`, com o valor do campeão como limiar), e
cada avaliação fica registada como um run do MLflow, por defeito no ficheiro `mlflow.db` da raiz
(fora do git; ver com `mlflow ui --backend-store-uri sqlite:///mlflow.db`). Só se registam
parâmetros e métricas, nunca dados.

O CI não tem os dados e não mede o modelo: confirma que o `metrics/validacao.json` vem da
validação e que nenhum top-5 desceu face ao do `main` (`check_not_worse`, o mesmo critério).

Corre no portátil: só precisa da validação e das sugestões que o treino gravou no pod, não do
modelo. Lê a pasta de um treino (`treino.json`) e avalia o ficheiro de sugestões do modelo por
defeito (`ficheiro_do_modelo_por_defeito`, o τ escolhido), não o de τ = 0.

Exemplo (da raiz do projeto):
    .venv/bin/python src/model_gate.py --pasta metrics/tfidf
"""

import argparse
import json
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.models import EvaluationResult, MetricThreshold

import evaluation as ev
from cpv_data import TEST_START, VAL_START
from train_tfidf import RESUMO, VALIDACAO, check_not_test

RAIZ = Path(__file__).resolve().parents[1]
CAMPEAO = RAIZ / "metrics" / "validacao.json"
TRACKING = f"sqlite:///{RAIZ / 'mlflow.db'}"
CONJUNTO = "validacao"
MEDIDAS = ("codigo", "classe")
GRUPOS = {"todos": "todos", "texto novo": "texto_novo"}  # grupo do `ev.report` -> nome na métrica
METRICAS = tuple(f"top5_{m}_{g}" for g in GRUPOS.values() for m in MEDIDAS)


class GateError(Exception):
    """O bloco de métricas não serve para o portão (outro conjunto, outro período, métricas em falta)."""


def gate_metrics(report):
    """As métricas do portão a partir do `ev.report`: top-5 no código e na classe, por grupo."""
    return {f"top5_{m}_{nome}": float(report.loc[(grupo, m), "top5"])
            for grupo, nome in GRUPOS.items() for m in MEDIDAS}


def check_block(block):
    """Erro se o bloco não vier da validação (outubro de 2025) ou lhe faltar alguma métrica."""
    if block.get("conjunto") != CONJUNTO:
        raise GateError(f"as métricas vêm de {block.get('conjunto')!r}, não da validação")
    inicio, fim = (pd.Timestamp(d) for d in block["periodo"])
    if inicio < VAL_START or fim >= TEST_START:
        raise GateError(f"período {block['periodo']} fora da validação "
                        f"[{VAL_START.date()}, {TEST_START.date()})")
    em_falta = set(METRICAS) - set(block["metricas"])
    if em_falta:
        raise GateError(f"métricas em falta: {sorted(em_falta)}")


def check_not_worse(novas, campeao):
    """Erro (`ModelValidationFailedException` do MLflow) se alguma métrica descer face ao campeão.

    Cada métrica do campeão é o limiar mínimo (`MetricThreshold(threshold=...)`): igual passa,
    qualquer descida falha, e uma métrica em falta no novo também falha.
    """
    limiares = {m: MetricThreshold(threshold=campeao[m], greater_is_better=True) for m in METRICAS}
    mlflow.validate_evaluation_results(limiares, EvaluationResult(metrics=novas, artifacts={}))


def _relative(path):
    path = Path(path).resolve()
    return str(path.relative_to(RAIZ)) if path.is_relative_to(RAIZ) else str(path)


def evaluate_run(pasta, val_path=VALIDACAO):
    """Avalia na validação as sugestões do modelo por defeito da pasta de um treino.

    Devolve o bloco de métricas (o que vai para o `validacao.json`) e o resumo do treino.
    """
    check_not_test(val_path)
    pasta = Path(pasta)
    resumo = json.loads((pasta / RESUMO).read_text(encoding="utf-8"))
    nome = resumo.get("ficheiro_do_modelo_por_defeito")
    if nome is None:
        raise GateError(f"{pasta / RESUMO} não indica o ficheiro do modelo por defeito")

    val = pd.read_parquet(val_path, columns=["idcontrato", "data", "codigos", "visto_no_treino"])
    sug = pd.read_parquet(pasta / nome, columns=["idcontrato", "sugestoes"])
    if sug["idcontrato"].astype(str).tolist() != val["idcontrato"].astype(str).tolist():
        raise GateError(f"{pasta / nome} não tem os contratos da validação pela mesma ordem")
    sugestoes = pd.Series([list(s) for s in sug["sugestoes"]], index=val.index)
    report = ev.report(sugestoes, val["codigos"], val["visto_no_treino"])

    block = {
        "conjunto": CONJUNTO,
        "periodo": [str(val["data"].min().date()), str(val["data"].max().date())],
        "contratos": len(val),
        "sugestoes": _relative(pasta / nome),
        "metricas": gate_metrics(report),
    }
    return block, resumo


def training_params(resumo):
    """Parâmetros do treino que vão para o MLflow (só os que existem no `treino.json`)."""
    tau = resumo.get("ficheiros", {}).get(resumo.get("ficheiro_do_modelo_por_defeito"))
    params = {k: resumo.get(k) for k in ("n_min", "min_df", "alpha", "epocas", "com_entidade")}
    params["tau"] = tau
    params["treino_inicio"], params["treino_fim"] = resumo["treino"]["datas"]
    return {k: v for k, v in params.items() if v is not None}


# Números do `treino.json` que também vão para o MLflow, para ver o custo de cada treino ao lado
# das métricas (os que faltarem num treino antigo ficam de fora)
CUSTO = ("exemplos_usados", "exemplos_saidos", "etiquetas", "vocabulario", "tempo_vetorizar_treinar_s",
         "tempo_sugerir_s", "pico_rss_mb", "nucleos")


def training_cost(resumo):
    return {f"treino_{k}": float(resumo[k]) for k in CUSTO if resumo.get(k) is not None}


def run(pasta, val_path=VALIDACAO, campeao_path=CAMPEAO, tracking_uri=TRACKING, nome="tfidf"):
    """Avalia o treino da `pasta`, regista-o no MLflow e compara-o com o campeão.

    Se não houver campeão (`campeao_path` não existe), o novo passa a campeão. Se alguma métrica
    descer, levanta `ModelValidationFailedException` e não toca no `campeao_path`.
    Devolve o bloco gravado.
    """
    block, resumo = evaluate_run(pasta, val_path)
    check_block(block)
    campeao_path = Path(campeao_path)
    campeao = json.loads(campeao_path.read_text(encoding="utf-8")) if campeao_path.exists() else None
    if campeao is not None:
        check_block(campeao)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("portao_validacao")
    with mlflow.start_run(run_name=nome):
        mlflow.log_params({**training_params(resumo), "conjunto": block["conjunto"],
                           "periodo": " a ".join(block["periodo"]), "sugestoes": block["sugestoes"]})
        mlflow.log_metrics({**block["metricas"], **training_cost(resumo)})
        if campeao is not None:
            mlflow.set_tag("campeao_anterior", campeao["sugestoes"])
            try:
                check_not_worse(block["metricas"], campeao["metricas"])
            except Exception:
                mlflow.set_tag("portao", "recusado")
                raise
        mlflow.set_tag("portao", "aceite")

    block = {"modelo": nome, **block}
    campeao_path.parent.mkdir(parents=True, exist_ok=True)
    campeao_path.write_text(json.dumps(block, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return block


def main():
    p = argparse.ArgumentParser(description="Portão de qualidade: o modelo novo não pode ficar abaixo do campeão.")
    p.add_argument("--pasta", required=True, help="pasta do treino (com treino.json e as sugestões)")
    p.add_argument("--validacao", default=str(VALIDACAO), help="validacao.parquet")
    p.add_argument("--campeao", default=str(CAMPEAO), help="ficheiro das métricas do campeão")
    p.add_argument("--nome", default="tfidf", help="nome do modelo no MLflow e no ficheiro")
    p.add_argument("--tracking", default=TRACKING, help="registo do MLflow (por defeito mlflow.db na raiz)")
    a = p.parse_args()
    block = run(a.pasta, a.validacao, a.campeao, tracking_uri=a.tracking, nome=a.nome)
    print("Aceite (igual ou acima do campeão anterior): campeão gravado em", a.campeao)
    for m, v in block["metricas"].items():
        print(f"  {m}: {v:.4f}")


if __name__ == "__main__":
    main()
