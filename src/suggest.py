"""Comando de sugestões (E4-H1, E5-T4): 5 códigos CPV para cada descrição escrita à mão.

Lê um modelo TF-IDF gravado pelo `train_tfidf.py` (`--modelo`) ou um checkpoint do BERT gravado
pelo `train_bert.py` (`--bert`, ver abaixo) e mostra, para cada descrição, os 5 códigos por
ordem, com a probabilidade que o modelo dá a cada um. A ordem usa a correção das etiquetas
frequentes com o τ escolhido no E4-S4 (`tfidf.TAU`), por isso não é sempre a das probabilidades:
uma etiqueta muito frequente pode ter mais probabilidade e ficar abaixo de uma rara.

O ficheiro do modelo só existe no pod da Runpod: tem cerca de 4,8 GB (E4-S0) e não cabe na
memória do portátil (5,8 GB partilhados com o editor), por isso este comando corre no pod, a
seguir ao treino.

BERT (E5-T4): `--bert` é a pasta de um checkpoint (`models/bert/<saída>/checkpoints/checkpoint-<passo>/`,
com o modelo, o tokenizador e a tradução número -> código no `config.json`). As frequências das
etiquetas (`log_prior.npy`) e a lista de etiquetas (`etiquetas.json`) vêm da pasta das pontuações
do mesmo treino, por defeito `models/bert/<saída>/pontuacoes/` (dois níveis acima do checkpoint,
como o `train_bert.py` as grava). A ordem é a do `train_bert.py` (`bert.suggest_from_scores`,
pontuação − τ · log frequência), por isso as sugestões de um texto da validação são as gravadas
para esse texto com o mesmo τ e a época desse checkpoint. A "probabilidade" é o softmax das
pontuações (sem a correção). Os textos cortam-se ao `max_length` do modelo (`--base`, que lê o
`COMPRIMENTO_MAXIMO` do E5-T1 em `bert_models.py`) ou ao dado com `--max-length`. Um modelo BERT
base (~0,4 GB) cabe no portátil: corre no processador (`--dispositivo cpu`, por defeito) ou na
GPU do pod (`--dispositivo cuda`).

Exemplos (da raiz do projeto):
    .venv/bin/python src/suggest.py --modelo models/tfidf_n10.joblib
    .venv/bin/python src/suggest.py --modelo models/tfidf_n10.joblib "aquisição de papel A4"
    .venv/bin/python src/suggest.py --bert models/bert/bertimbau/checkpoints/checkpoint-1000 --base bertimbau
Sem descrições, usa as 3 de `EXEMPLOS`.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

import tfidf

# O torch e o transformers só se importam com --bert (dentro das funções do BERT): o comando
# do TF-IDF e os seus testes não os carregam.
ETIQUETAS = "etiquetas.json"  # os mesmos nomes do train_bert.py, que importa o torch
LOG_PRIOR = "log_prior.npy"

EXEMPLOS = [
    "serviços de limpeza de três escolas básicas durante 12 meses",
    "fornecimento de refeições escolares para os alunos do 1.º ciclo do concelho",
    "empreitada de repavimentação de estradas municipais",
]


def _format(textos, sugestoes, proba, classes, casas):
    coluna = {c: i for i, c in enumerate(classes)}
    linhas = []
    for i, (texto, codigos) in enumerate(zip(textos, sugestoes)):
        linhas.append(f"Descrição: {texto}")
        for pos, codigo in enumerate(codigos, start=1):
            p = float(np.round(proba[i, coluna[codigo]], casas))
            linhas.append(f"  {pos}. {codigo}  (probabilidade {p:.{casas}f})")
        linhas.append("")
    return "\n".join(linhas)


def format_suggestions(model, textos, casas=3, tau=tfidf.TAU):
    """Texto com cada descrição e as suas 5 sugestões numeradas, com a probabilidade (TF-IDF)."""
    textos = list(textos)
    sugestoes = tfidf.suggest(model, textos, tau=tau)
    return _format(textos, sugestoes, model.predict_proba(textos), model.classes_, casas)


def load_bert(checkpoint, pontuacoes_dir=None, base=None, max_length=None, dispositivo="cpu"):
    """O checkpoint do BERT pronto a sugerir: {"model", "tokenizer", "classes", "log_prior", "max_length"}.

    `pontuacoes_dir`: onde estão o `etiquetas.json` e o `log_prior.npy` do mesmo treino; por
    defeito `<checkpoint>/../../pontuacoes`. `base` ("bertimbau" ou "albertina") dá o
    `max_length` do E5-T1; `max_length` dado ganha (os testes usam um modelo minúsculo).
    Erro se as etiquetas do checkpoint não forem as do `etiquetas.json` (outro treino), se o
    `log_prior` não tiver uma entrada por etiqueta, ou se alguma etiqueta for uma divisão.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    import bert

    bert.check_dispositivo(dispositivo)
    if base is None and max_length is None:
        raise ValueError("diga o modelo de base (--base bertimbau|albertina), para o max_length do E5-T1, "
                         "ou o --max-length")
    max_length = bert.model_max_length(base if base is not None else "", max_length)
    if dispositivo == "cuda":
        bert.check_gpu()
    checkpoint = Path(checkpoint)
    pontuacoes_dir = Path(pontuacoes_dir) if pontuacoes_dir is not None else checkpoint.parent.parent / "pontuacoes"
    classes = json.loads((pontuacoes_dir / ETIQUETAS).read_text(encoding="utf-8"))
    log_prior = np.load(pontuacoes_dir / LOG_PRIOR)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, dtype=torch.float32).to(dispositivo)
    id2label = [str(model.config.id2label[i]) for i in range(model.config.num_labels)]
    if id2label != [str(c) for c in classes]:
        raise ValueError(f"as etiquetas do checkpoint {checkpoint} não são as de {pontuacoes_dir / ETIQUETAS}: "
                         "o checkpoint e as pontuações são de treinos diferentes")
    if log_prior.shape != (len(classes),):
        raise ValueError(f"{pontuacoes_dir / LOG_PRIOR}: tem {log_prior.shape} entradas; são precisas {len(classes)}")
    divisoes = [c for c in classes if c.endswith("000000")]
    if divisoes:
        # O treino recusa-as (bert.training_labels); se aparecerem, o modelo sugeria divisões
        raise ValueError(f"o checkpoint tem etiquetas ao nível da divisão: {divisoes}")
    return {"model": model, "tokenizer": tokenizer, "classes": np.asarray(classes, dtype=object),
            "log_prior": log_prior, "max_length": max_length}


def bert_suggestions(carregado, textos, tau=tfidf.TAU):
    """As 5 sugestões de cada texto e as pontuações (logits), como no `train_bert.py`."""
    import bert

    pont = bert.scores(carregado["model"], carregado["tokenizer"], list(textos), carregado["max_length"])
    return bert.suggest_from_scores(pont, carregado["classes"], carregado["log_prior"], tau), pont


def format_bert_suggestions(carregado, textos, casas=3, tau=tfidf.TAU):
    """Como o `format_suggestions`, para o BERT; a probabilidade é o softmax das pontuações."""
    textos = list(textos)
    sugestoes, pont = bert_suggestions(carregado, textos, tau)
    pont = pont - pont.max(axis=1, keepdims=True)
    proba = np.exp(pont) / np.exp(pont).sum(axis=1, keepdims=True)
    return _format(textos, sugestoes, proba, carregado["classes"], casas)


def main(argv=None):
    p = argparse.ArgumentParser(description="Sugere 5 códigos CPV para cada descrição.")
    qual = p.add_mutually_exclusive_group(required=True)
    qual.add_argument("--modelo", help="modelo TF-IDF gravado pelo train_tfidf.py (.joblib)")
    qual.add_argument("--bert", help="pasta de um checkpoint do BERT gravado pelo train_bert.py "
                                     "(models/bert/<saída>/checkpoints/checkpoint-<passo>)")
    p.add_argument("--tau", type=float, default=tfidf.TAU, help="correção das etiquetas frequentes (0 = sem)")
    p.add_argument("--pontuacoes", default=None,
                   help="BERT: pasta com etiquetas.json e log_prior.npy (por defeito <checkpoint>/../../pontuacoes)")
    p.add_argument("--base", choices=["bertimbau", "albertina"], default=None,
                   help="BERT: o modelo de base do checkpoint, para o max_length do E5-T1 (bert_models.py)")
    p.add_argument("--max-length", type=int, default=None, help="BERT: comprimento máximo em tokens (ganha à --base)")
    p.add_argument("--dispositivo", choices=["cpu", "cuda"], default="cpu", help="BERT: cpu ou cuda (GPU)")
    p.add_argument("descricoes", nargs="*", help="descrições; sem nenhuma usa 3 exemplos")
    a = p.parse_args(argv)
    textos = a.descricoes or EXEMPLOS
    if a.modelo is not None:
        print(format_suggestions(joblib.load(a.modelo), textos, tau=a.tau))
    else:
        carregado = load_bert(a.bert, a.pontuacoes, a.base, a.max_length, a.dispositivo)
        print(format_bert_suggestions(carregado, textos, tau=a.tau))


if __name__ == "__main__":
    main()
