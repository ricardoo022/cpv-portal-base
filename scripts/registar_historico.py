"""Regista no MLflow todos os treinos guardados em `metrics/`, para os ver lado a lado (E4-T5).

Uma vez, no portátil, depois de cada spike que grave sugestões em `metrics/`. Não passa pelo
portão de qualidade (`src/model_gate.py`) e não muda nenhum ficheiro: só lê e regista, numa
experiência à parte (`historico_e4`). Um run por ficheiro de sugestões (um por τ), com:
- parâmetros: o spike, a opção (a pasta), N, `alpha`, épocas (ou "regra por defeito"), τ,
  entidade, datas do treino e versões das bibliotecas;
- métricas: top-1, top-5 e F1 macro no código completo, na classe e na divisão, em todos os
  contratos e só no texto novo (`evaluation.report`), mais o custo do treino (`treino.json`);
- a régua "códigos mais comuns" (E4-T2) como mais um run;
- a curva por época do E4-S5 (`metrics/e4_s5/curva/`) como um só run, com as métricas por época
  (no MLflow, o passo `step` é a época), o que dá um gráfico com a curva.

O texto novo depende do treino: um treino com outra janela (E4-S2, E4-S6; o início do treino
não é o do `treino.parquet` preparado) avalia-se com a validação da sua janela (`data/janelas/<ano>/validacao.parquet`), cujo `visto_no_treino` é o dessa
janela. Se a validação da janela não existir, o treino fica de fora (com aviso).

Correr outra vez não duplica: um run cujo nome já existe na experiência salta-se.

Exemplo (da raiz do projeto):
    .venv/bin/python scripts/registar_historico.py
"""

import argparse
import json
import re
import sys
from pathlib import Path

import mlflow
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import baselines as bl  # noqa: E402
import evaluation as ev  # noqa: E402
import model_gate as mg  # noqa: E402
from labels import DEFAULT_N_MIN  # noqa: E402
from train_tfidf import RESUMO, SUGESTOES, check_not_test  # noqa: E402

EXPERIENCIA = "historico_e4"
DADOS = RAIZ / "data" / "processed"
JANELAS = RAIZ / "data" / "janelas"
CURVA = Path("e4_s5") / "curva"
COLUNAS_VAL = ["idcontrato", "codigos", "visto_no_treino"]
_TAU = re.compile(r"sugestoes_validacao_tau([0-9.]+)\.parquet$")
_EPOCA = re.compile(r"sugestoes_epoca_(\d+)\.parquet$")


def tau_of(nome):
    """τ de um ficheiro de sugestões pelo nome (o de τ = 0 não tem sufixo)."""
    if nome == SUGESTOES:
        return 0.0
    m = _TAU.search(nome)
    return float(m.group(1)) if m else None


def report_metrics(report):
    """Todas as métricas do `ev.report` com nomes do MLflow: `top5_codigo_todos`, `f1_macro_classe_texto_novo`..."""
    out = {}
    for (grupo, medida), linha in report.iterrows():
        if medida not in ev.MEASURES:
            continue
        g = mg.GRUPOS[grupo]
        for col in ("top1", "top5", "f1_macro"):
            out[f"{col}_{medida}_{g}"] = float(linha[col])
        out[f"contratos_{medida}_{g}"] = float(linha["contratos"])
    return out


def evaluate_file(path, val):
    """Métricas de um ficheiro de sugestões na validação `val` (mesmos contratos, mesma ordem)."""
    sug = pd.read_parquet(path, columns=["idcontrato", "sugestoes"])
    if sug["idcontrato"].astype(str).tolist() != val["idcontrato"].astype(str).tolist():
        raise mg.GateError(f"{path} não tem os contratos da validação pela mesma ordem")
    sugestoes = pd.Series([list(s) for s in sug["sugestoes"]], index=val.index)
    return report_metrics(ev.report(sugestoes, val["codigos"], val["visto_no_treino"]))


def validation_for(resumo, dados, janelas, inicio_por_defeito):
    """Caminho da validação que corresponde à janela do treino (None se não existir).

    Um treino que começa no mesmo dia que o `treino.parquet` de `dados` usou a janela por
    defeito; os outros usam a validação da sua janela, `janelas/<ano do início>/`.
    """
    inicio = resumo["treino"]["datas"][0]
    if inicio == inicio_por_defeito:
        path = dados / "validacao.parquet"
    else:
        path = janelas / inicio[:4] / "validacao.parquet"
    return path if path.exists() else None


def run_params(resumo, spike, opcao, tau, sugestoes):
    params = {k: v for k, v in mg.training_params(resumo).items() if k != "tau"}
    params.update({"spike": spike, "opcao": opcao, "tau": tau, "sugestoes": sugestoes,
                   "epocas": resumo.get("epocas") or "regra por defeito"})
    params.update({f"versao_{k}": v for k, v in resumo.get("versoes", {}).items()})
    return params


def training_runs(metrics_dir):
    """(pasta, resumo) de cada treino guardado, por ordem do caminho."""
    for path in sorted(Path(metrics_dir).rglob(RESUMO)):
        yield path.parent, json.loads(path.read_text(encoding="utf-8"))


def _existing(nome_experiencia):
    exp = mlflow.get_experiment_by_name(nome_experiencia)
    if exp is None:
        return set()
    runs = mlflow.search_runs([exp.experiment_id])
    if not len(runs):
        return set()
    # Só os que acabaram: um run que rebentou a meio volta a fazer-se na corrida seguinte
    return set(runs.loc[runs["status"] == "FINISHED", "tags.mlflow.runName"])


def run(metrics_dir=RAIZ / "metrics", dados=DADOS, janelas=JANELAS, tracking_uri=mg.TRACKING,
        experiencia=EXPERIENCIA, verbose=True):
    """Regista cada ficheiro de sugestões, a régua e a curva. Devolve os nomes dos runs novos."""
    metrics_dir, dados, janelas = Path(metrics_dir), Path(dados), Path(janelas)
    val_path = dados / "validacao.parquet"
    check_not_test(val_path)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiencia)
    ja = _existing(experiencia)
    inicio_por_defeito = str(pd.read_parquet(dados / "treino.parquet", columns=["data"])["data"].min().date())
    novos = []
    validacoes = {}

    def val_de(path):
        check_not_test(path)
        if path not in validacoes:
            validacoes[path] = pd.read_parquet(path, columns=COLUNAS_VAL)
        return validacoes[path]

    def registar(nome, params, metricas, tags=None):
        if nome in ja:
            return
        with mlflow.start_run(run_name=nome):
            mlflow.log_params(params)
            mlflow.log_metrics(metricas)
            mlflow.set_tags(tags or {})
        novos.append(nome)
        if verbose:
            print("registado:", nome)

    campeao = None
    if (metrics_dir / mg.CAMPEAO.name).exists():
        campeao = json.loads((metrics_dir / mg.CAMPEAO.name).read_text(encoding="utf-8"))["sugestoes"]

    # 1. Cada treino, um run por ficheiro de sugestões (um por τ)
    for pasta, resumo in training_runs(metrics_dir):
        rel = pasta.relative_to(metrics_dir)
        spike, opcao = rel.parts[0], "/".join(rel.parts[1:]) or rel.parts[0]
        vpath = validation_for(resumo, dados, janelas, inicio_por_defeito)
        if vpath is None:
            print(f"aviso: {rel} fica de fora (sem a validação da janela que começa em "
                  f"{resumo['treino']['datas'][0]})")
            continue
        for ficheiro in sorted(pasta.glob("sugestoes_validacao*.parquet")):
            tau = tau_of(ficheiro.name)
            if tau is None:
                print(f"aviso: {rel / ficheiro.name} fica de fora (não se sabe o τ pelo nome)")
                continue
            sugestoes = str(Path("metrics") / rel / ficheiro.name)
            nome = f"{rel}/tau{tau:g}"
            tags = {"spike": spike, "campeao": str(sugestoes == campeao).lower()}
            registar(nome, run_params(resumo, spike, opcao, tau, sugestoes),
                     {**evaluate_file(ficheiro, val_de(vpath)), **mg.training_cost(resumo)}, tags)

    # 2. A régua "códigos mais comuns" (E4-T2), com o N por defeito
    if "regua" not in ja:
        codigo = pd.read_parquet(dados / "treino.parquet", columns=["codigo"])["codigo"]
        top = bl.most_common_labels(codigo, n_min=DEFAULT_N_MIN)
        val = val_de(val_path)
        sug = bl.suggest_most_common(top, val.index)
        metricas = report_metrics(ev.report(sug, val["codigos"], val["visto_no_treino"]))
        registar("regua", {"spike": "e4_t2", "opcao": "regua", "n_min": DEFAULT_N_MIN,
                           "etiquetas": " ".join(top)}, metricas, {"spike": "e4_t2", "campeao": "false"})

    # 3. A curva por época do E4-S5: um run, com o passo = a época
    curva = metrics_dir / CURVA
    if (curva / "curva.json").exists() and str(CURVA) not in ja:
        resumo = json.loads((curva / "curva.json").read_text(encoding="utf-8"))
        val = val_de(val_path)
        with mlflow.start_run(run_name=str(CURVA)):
            mlflow.log_params({"spike": "e4_s5", "opcao": "curva", "n_min": resumo["n_min"],
                               "alpha": resumo["alpha"], "tau": resumo["tau"], "epocas": resumo["epocas"],
                               "nota": resumo.get("nota", "")[:500]})
            mlflow.set_tags({"spike": "e4_s5", "campeao": "false"})
            for ficheiro in sorted(curva.glob("sugestoes_epoca_*.parquet")):
                epoca = int(_EPOCA.search(ficheiro.name).group(1))
                mlflow.log_metrics(evaluate_file(ficheiro, val), step=epoca)
        novos.append(str(CURVA))
        if verbose:
            print("registado:", CURVA)
    return novos


def main():
    p = argparse.ArgumentParser(description="Regista no MLflow todos os treinos guardados em metrics/.")
    p.add_argument("--metrics", default=str(RAIZ / "metrics"), help="pasta dos resultados guardados")
    p.add_argument("--dados", default=str(DADOS), help="pasta com treino.parquet e validacao.parquet")
    p.add_argument("--janelas", default=str(JANELAS), help="pasta das outras janelas de treino (E4-T4)")
    p.add_argument("--tracking", default=mg.TRACKING, help="registo do MLflow (por defeito mlflow.db na raiz)")
    a = p.parse_args()
    novos = run(a.metrics, a.dados, a.janelas, a.tracking)
    print(f"{len(novos)} runs novos na experiência {EXPERIENCIA}")


if __name__ == "__main__":
    main()
