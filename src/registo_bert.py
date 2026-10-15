"""Regista no MLflow um treino do BERT (E5-T3): um run, com a curva por época.

O treino (`src/train_bert.py`) corre no pod e grava uma pasta `--saida` pequena: `treino.json`,
uma pasta `epoca_<k>/` por época com as sugestões da validação de cada τ, e na raiz as sugestões
da última época. Essa pasta copia-se para o portátil, e este módulo lê-a e regista o treino no
`mlflow.db` do portátil (o mesmo do portão e do histórico). Não se copia o `mlflow.db` do pod:
copiado por cima, apagava os runs que o portátil já tem; o registo refaz-se a partir dos ficheiros.

Antes de registar confirma que a validação do portátil é a mesma que o pod usou: o SHA-256 do
`validacao.parquet` tem de ser o do `treino.json` (`sha256.validacao`), e o do `treino.parquet` ao
lado o do treino (`sha256.treino`); um só byte diferente dá outra impressão digital. Confirma
também, como o portão, que nada vem do teste e que a validação é a da janela do treino.

O run tem:
- parâmetros: o modelo e a sua revisão, os hiperparâmetros, as datas e o SHA-256 do treino e da
  validação, o commit do git, o YAML da experiência, as versões das bibliotecas, ... (todo o
  `treino.json` que diz como repetir o treino; o que não existir regista-se como "nenhum");
- métricas por época, com o passo (`step`) = a época, o que dá no MLflow o gráfico da curva:
  top-1 e top-5 no código completo e na classe, em todos os contratos e só no texto novo
  (`evaluation.report`), e a perda no treino. As do τ do modelo por defeito têm os nomes do portão
  (`top5_codigo_todos`, ...); as dos outros τ levam o sufixo `_tau<τ>` (`top5_codigo_todos_tau0`);
- métricas sem passo: o custo do treino (tempo, memória, exemplos, etiquetas, segundos por época);
- etiquetas (tags): o SHA-256 do `treino.json`, o modelo e o texto do YAML da experiência.

Só parâmetros e métricas: nem dados, nem modelos, nem ficheiros (como no portão).

Correr outra vez a mesma pasta não duplica: se já houver na experiência um run acabado (FINISHED)
com o mesmo SHA-256 do `treino.json`, não regista. Um run que rebentou a meio não conta.

Exemplo (no portátil, da raiz do projeto, depois de copiar a pasta do pod):
    .venv/bin/python src/registo_bert.py --pasta metrics/bert/bertimbau
"""

import argparse
import json
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient

import evaluation as ev
import model_gate as mg
from cpv_data import MAX_YAML, TEST_START, file_sha256
from train_tfidf import RESUMO, VALIDACAO, check_dates, check_not_test

EXPERIENCIA = "treinos_bert"
NENHUM = "nenhum"  # o MLflow não aceita None; um campo que falta fica com este texto
MEDIDAS = ("codigo", "classe")
TOPS = ("top1", "top5")
COLUNAS_VAL = ["idcontrato", "codigos", "visto_no_treino"]
# Hiperparâmetros e o resto do `treino.json` que vão tal e qual para os parâmetros
DIRETOS = ("max_length", "n_min", "taxa_aprendizagem", "lote", "epocas", "aquecimento", "decaimento_pesos",
           "semente", "dispositivo", "gpu", "n_gpus", "retomado_depois_da_epoca")
# Custo do treino: métricas sem passo, `treino_<chave>`
CUSTO = ("tempo_treinar_s", "pico_rss_mb", "exemplos_usados", "exemplos_saidos", "etiquetas")


class RegistoError(Exception):
    """A pasta do treino ou a validação não servem para registar o treino."""


def _texto(v):
    return NENHUM if v is None else v


def metric_names(report, sufixo=""):
    """Top-1 e top-5 no código e na classe, por grupo: `top5_codigo_todos`, `top1_classe_texto_novo`..."""
    return {f"{top}_{m}_{nome}{sufixo}": float(report.loc[(grupo, m), top])
            for grupo, nome in mg.GRUPOS.items() for m in MEDIDAS for top in TOPS}


def _suffix(nome, resumo):
    """Sufixo das métricas de um ficheiro: nenhum no do modelo por defeito, `_tau<τ>` nos outros."""
    return "" if nome == resumo["ficheiro_do_modelo_por_defeito"] else f"_tau{resumo['ficheiros'][nome]:g}"


def epoch_metrics(pasta, resumo, val):
    """Métricas de cada época, {k: {métrica: valor}}, a partir das sugestões de `pasta/epoca_<k>/`.

    `val`: a validação (`idcontrato`, `codigos`, `visto_no_treino`); cada ficheiro de sugestões
    tem de ter os mesmos contratos pela mesma ordem.
    """
    pasta = Path(pasta)
    ids = val["idcontrato"].astype(str).tolist()
    out = {}
    for e in resumo["por_epoca"]:
        k = int(e["epoca"])
        metricas = {}
        for nome in resumo["ficheiros"]:
            path = pasta / f"epoca_{k}" / nome
            if not path.is_file():
                raise RegistoError(f"falta {path}")
            sug = pd.read_parquet(path, columns=["idcontrato", "sugestoes"])
            if sug["idcontrato"].astype(str).tolist() != ids:
                raise RegistoError(f"{path} não tem os contratos da validação pela mesma ordem")
            sugestoes = pd.Series([list(s) for s in sug["sugestoes"]], index=val.index)
            report = ev.report(sugestoes, val["codigos"], val["visto_no_treino"])
            metricas.update(metric_names(report, _suffix(nome, resumo)))
        if e.get("perda_treino") is not None:
            metricas["perda_treino"] = float(e["perda_treino"])
        out[k] = metricas
    return out


def training_params(resumo, pasta):
    """Parâmetros do run: tudo o que diz como repetir o treino (None -> "nenhum")."""
    modelo, git = resumo.get("modelo") or {}, resumo.get("git") or {}
    exp = resumo.get("experiencia") or {}
    params = {
        "modelo_chave": modelo.get("chave"), "modelo_nome": modelo.get("nome"),
        "modelo_revisao": modelo.get("revisao"),
        **{k: resumo.get(k) for k in DIRETOS},
        "taus": " ".join(f"{t:g}" for t in resumo["taus"]),
        "treino_inicio": resumo["treino"]["datas"][0], "treino_fim": resumo["treino"]["datas"][1],
        "treino_sha256": resumo["sha256"]["treino"], "validacao_sha256": resumo["sha256"]["validacao"],
        "git_commit": git.get("commit"), "git_alteracoes_por_gravar": git.get("alteracoes_por_gravar"),
        "experiencia_ficheiro": exp.get("ficheiro"), "experiencia_sha256": exp.get("sha256"),
        "tau_por_defeito": resumo["ficheiros"][resumo["ficheiro_do_modelo_por_defeito"]],
        "pasta": mg._relative(pasta),
    }
    params.update({f"versao_{k}": v for k, v in resumo.get("versoes", {}).items()})
    return {k: _texto(v) for k, v in params.items()}


def training_cost(resumo):
    """Custo do treino (métricas sem passo). Os segundos por época contam desde o início do treino,
    ou desde a retoma: com `--retomar`, os segundos das épocas seguintes começam do zero."""
    out = {f"treino_{k}": float(resumo[k]) for k in CUSTO if resumo.get(k) is not None}
    por_epoca = resumo["por_epoca"]
    feitas = len(por_epoca) - (resumo.get("retomado_depois_da_epoca") or 0)
    if feitas > 0 and por_epoca[-1].get("segundos_desde_o_inicio") is not None:
        out["treino_segundos_por_epoca"] = float(por_epoca[-1]["segundos_desde_o_inicio"]) / feitas
    return out


def _already_logged(client, experiencia, sha):
    """True se a experiência já tem um run acabado (FINISHED) deste `treino.json`."""
    exp = client.get_experiment_by_name(experiencia)
    if exp is None:
        return False
    runs = client.search_runs([exp.experiment_id],
                              filter_string=f"tags.treino_json_sha256 = '{sha}' and attributes.status = 'FINISHED'")
    return len(runs) > 0


def _check(pasta, resumo, val_path):
    """As verificações antes de registar (as do portão e a do SHA-256 da validação)."""
    origem = pasta / RESUMO
    check_not_test(val_path)
    check_dates(val_path, TEST_START)
    if not resumo.get("ficheiro_do_modelo_por_defeito"):
        raise RegistoError(f"{origem} não indica o ficheiro do modelo por defeito")
    mg.check_training_dates(resumo, origem)
    mg.check_window(resumo, val_path, origem)
    sha = (resumo.get("sha256") or {}).get("validacao")
    if not sha:
        raise RegistoError(f"{origem} não tem o SHA-256 da validação (chave sha256): sem ele não se confirma que a "
                           "validação do portátil é a que o pod usou; o treino é anterior ao E5-T3?")
    if file_sha256(val_path) != sha:
        raise RegistoError(f"{val_path} não é a validação que o pod usou (o SHA-256 é outro): use o "
                           "validacao.parquet que foi copiado para o pod")
    # O treino ao lado (o mesmo que o check_window leu) também tem de ser o do pod (critério do E5-T4)
    treino_path = Path(val_path).parent / "treino.parquet"
    if file_sha256(treino_path) != (resumo.get("sha256") or {}).get("treino"):
        raise RegistoError(f"{treino_path} não é o treino que o pod usou (o SHA-256 é outro): use o "
                           "treino.parquet que foi copiado para o pod")
    yaml = (resumo.get("experiencia") or {}).get("conteudo")
    if yaml is not None and len(yaml) > MAX_YAML:
        raise RegistoError(f"o YAML da experiência tem {len(yaml)} carateres; uma tag do MLflow leva até {MAX_YAML}")


def log_training(pasta, val_path=VALIDACAO, tracking_uri=mg.TRACKING, experiencia=EXPERIENCIA, nome=None):
    """Regista o treino da `pasta` como um run do MLflow. Devolve o `run_id`, ou None se já estava."""
    pasta = Path(pasta)
    path = pasta / RESUMO
    if not path.is_file():
        raise RegistoError(f"falta {path}: a pasta não é a saída de um treino")
    resumo = json.loads(path.read_text(encoding="utf-8"))
    _check(pasta, resumo, val_path)

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri)
    sha = file_sha256(path)
    if _already_logged(client, experiencia, sha):
        return None

    # Avaliar antes de abrir o run: um erro nas sugestões não deixa um run a meio
    val = pd.read_parquet(val_path, columns=COLUNAS_VAL)
    por_epoca = epoch_metrics(pasta, resumo, val)

    modelo = resumo.get("modelo") or {}
    yaml = (resumo.get("experiencia") or {}).get("conteudo")
    mlflow.set_experiment(experiencia)
    with mlflow.start_run(run_name=nome or pasta.resolve().name) as run:
        mlflow.log_params(training_params(resumo, pasta))
        for k, metricas in sorted(por_epoca.items()):
            mlflow.log_metrics(metricas, step=k)  # no MLflow o passo é a época: dá a curva
        mlflow.log_metrics(training_cost(resumo))
        mlflow.set_tags({"treino_json_sha256": sha, "modelo": _texto(modelo.get("chave") or modelo.get("nome")),
                         "experiencia_yaml": _texto(yaml)})
    return run.info.run_id


def main(argv=None):
    p = argparse.ArgumentParser(description="Regista no MLflow um treino do BERT (um run, com a curva por época).")
    p.add_argument("--pasta", required=True, help="pasta --saida do treino (com treino.json e as pastas epoca_<k>)")
    p.add_argument("--validacao", default=str(VALIDACAO), help="validacao.parquet (a mesma que o pod usou)")
    p.add_argument("--tracking", default=mg.TRACKING, help="registo do MLflow (por defeito mlflow.db na raiz)")
    p.add_argument("--experiencia", default=EXPERIENCIA, help="experiência do MLflow")
    p.add_argument("--nome", default=None, help="nome do run (por defeito o nome da pasta)")
    a = p.parse_args(argv)
    run_id = log_training(a.pasta, a.validacao, tracking_uri=a.tracking, experiencia=a.experiencia, nome=a.nome)
    if run_id is None:
        print(f"Já estava registado na experiência {a.experiencia}: {a.pasta}")
    else:
        print(f"Registado na experiência {a.experiencia}: {a.pasta} (run {run_id})")


if __name__ == "__main__":
    main()
