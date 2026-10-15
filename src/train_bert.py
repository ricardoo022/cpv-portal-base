"""Treino do BERTimbau ou do Albertina (E5-T2) e sugestões para a validação, época a época.

Lê só o treino e a validação, por caminhos explícitos, e passa-os primeiro pelas verificações do
E4-T5 (`train_tfidf.check_inputs`): recusa o teste pelo nome (`teste.parquet`) e pelas datas.
Com `--dispositivo cuda` (o que se usa no pod), antes disso confirma que o `torch` vê a GPU
(`bert.check_gpu`). Passos:

1. tuna o modelo (`bert.train`) com o `texto` e o `codigo` do treino; as etiquetas e as contagens
   do N (regra dos códigos raros) saem só do treino;
2. no fim de cada época k:
   - o `Trainer` grava o modelo (`<checkpoints>/checkpoint-<passo>`), com o tokenizador e a
     tradução número -> código, para se poder usar o modelo da época escolhida no E5-S1;
   - grava as pontuações de todas as etiquetas para cada contrato da validação
     (`<pontuacoes>/epoca_<k>.npy`, float32, linhas pela ordem do `validacao.parquet`), para se
     aplicar qualquer τ depois sem treinar outra vez. Com ~24 mil contratos e ~3 000 etiquetas
     são ~0,3 GB por época: ficam no pod, fora do git (por defeito em `models/bert/`);
   - grava as sugestões dessa época com cada τ (`<saida>/epoca_<k>/sugestoes_validacao*.parquet`);
3. copia as sugestões da última época para a `<saida>`, no mesmo formato e com os mesmos nomes
   do TF-IDF (`sugestoes_validacao.parquet` com τ = 0 e `sugestoes_validacao_tau0.25.parquet`),
   que a avaliação (E4-T1) e o portão (E4-T5) leem sem mudanças;
4. grava `treino.json` (as chaves que o portão lê, como no TF-IDF, mais os hiperparâmetros, o
   modelo e a sua revisão, os tempos e as versões, e uma entrada por época).

A `<saida>` é pequena (vai para o portátil e para o git); os checkpoints e as pontuações são
grandes e ficam no pod. Para não misturar treinos, o treino recusa pastas de checkpoints ou de
pontuações que não estejam vazias, e uma `<saida>` que já tenha pastas `epoca_*`.

Retomar (`--retomar`): se o treino parar a meio (o pod caiu), volta-se a correr o mesmo comando
com `--retomar`. O treino continua do último checkpoint (que leva o estado do otimizador) e só
correm as épocas que faltam; as épocas já feitas ficam como estavam, e as que tinham começado
depois do último checkpoint voltam a gravar-se por cima.

Exemplo (no pod, da raiz do projeto):
    .venv/bin/python src/train_bert.py --modelo bertimbau --saida metrics/bert/bertimbau
"""

import argparse
import datetime as dt
import json
import os
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

import bert
import bert_models
import tfidf
from labels import DEFAULT_N_MIN
from train_tfidf import RESUMO, TREINO, VALIDACAO, _date_range, check_inputs, peak_rss_mb, suggestions_file

RAIZ = Path(__file__).resolve().parents[1]
MODELOS_DIR = RAIZ / "models" / "bert"  # fora do git
ETIQUETAS = "etiquetas.json"
LOG_PRIOR = "log_prior.npy"
EPOCA_INFO = "epoca.json"  # dentro de cada epoca_<k>/: o que entra no `por_epoca` do treino.json
ARGUMENTOS = "argumentos.json"  # na <saida>: os argumentos do treino, para a retoma os confirmar


def epoch_dir(k):
    return f"epoca_{k}"


def scores_file(k):
    return f"epoca_{k}.npy"


def default_dirs(out_dir):
    """Pastas dos checkpoints e das pontuações por defeito: `models/bert/<nome da saída>/`."""
    base = MODELOS_DIR / Path(out_dir).name
    return base / "checkpoints", base / "pontuacoes"


def _check_empty(path, o_que, apagar):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"{path}: a pasta de {o_que} não está vazia; use outra pasta para não misturar treinos, "
                         f"ou --retomar se o treino caiu a meio, ou {apagar}")


def _versions():
    import accelerate
    import torch
    import transformers
    return {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "accelerate": accelerate.__version__,
            "tokenizers": __import__("tokenizers").__version__, "numpy": np.__version__,
            "pandas": pd.__version__, "cuda": torch.version.cuda}


def run(train_path, val_path, out_dir, modelo="bertimbau", n_min=DEFAULT_N_MIN, taus=(0.0, tfidf.TAU),
        max_length=None, taxa_aprendizagem=bert.TAXA_APRENDIZAGEM, lote=bert.LOTE, epocas=bert.EPOCAS,
        aquecimento=bert.AQUECIMENTO, decaimento_pesos=bert.DECAIMENTO_PESOS, semente=bert.SEMENTE,
        dispositivo="cuda", checkpoints_dir=None, pontuacoes_dir=None, retomar=False):
    """Tuna o `modelo`, grava as pontuações e as sugestões de cada época e o resumo.

    `modelo`: "bertimbau", "albertina" (`bert_models.MODELOS`) ou a pasta de um modelo gravado
    (os testes usam um modelo minúsculo; então `max_length` tem de ser dado). `taus`: um ficheiro
    de sugestões por τ, em cada época. Os hiperparâmetros por defeito são os do artigo do BERT
    (`bert.py`). `dispositivo`: "cuda" (no pod; verifica primeiro que há GPU) ou "cpu" (testes).
    `checkpoints_dir` e `pontuacoes_dir`: por defeito em `models/bert/<nome da saída>/`.
    `retomar`: continua um treino interrompido do seu último checkpoint, com as mesmas pastas e
    os mesmos argumentos (ver a descrição do módulo).
    Devolve o dicionário do `bert.train`.
    """
    bert.check_dispositivo(dispositivo)
    gpu = bert.check_gpu() if dispositivo == "cuda" else None  # antes de ler qualquer ficheiro
    check_inputs(train_path, val_path)
    if any(t < 0 for t in taus):
        raise ValueError(f"τ tem de ser >= 0: {list(taus)}")
    out_dir = Path(out_dir)
    padrao_ck, padrao_pt = default_dirs(out_dir)
    checkpoints_dir = Path(checkpoints_dir or padrao_ck)
    pontuacoes_dir = Path(pontuacoes_dir or padrao_pt)
    # Tudo o que muda o treino. Ao retomar tem de ser igual: com outro lote ou outro número de
    # épocas o Trainer saltava os passos errados sem dar erro, e as épocas misturavam configurações.
    # Os caminhos completos: o treino de outra janela também se chama treino.parquet. O max_length
    # já resolvido: dar 256 ou deixar o do E5-T1 (256) é o mesmo treino.
    argumentos = {"treino": str(Path(train_path).resolve()), "validacao": str(Path(val_path).resolve()),
                  "modelo": str(modelo), "n_min": n_min, "taus": [float(t) for t in taus],
                  "max_length": bert.model_max_length(modelo, max_length),
                  "taxa_aprendizagem": taxa_aprendizagem, "lote": lote, "epocas": epocas, "aquecimento": aquecimento,
                  "decaimento_pesos": decaimento_pesos, "semente": semente}
    apagar = f"apague {out_dir}, {checkpoints_dir} e {pontuacoes_dir} e treine de novo"
    if retomar:
        ultimo = bert.last_checkpoint(checkpoints_dir)
        if ultimo is None:
            raise ValueError(f"{checkpoints_dir}: não há nenhum checkpoint para retomar; o treino caiu antes da "
                             f"1.ª época acabar: {apagar}")
        if not (out_dir / ARGUMENTOS).is_file():
            raise ValueError(f"{out_dir / ARGUMENTOS}: não existe, por isso não se sabe com que argumentos o treino "
                             f"começou; confirme a pasta de saída (--saida), ou {apagar}")
        guardados = json.loads((out_dir / ARGUMENTOS).read_text(encoding="utf-8"))
        diferentes = sorted(k for k in argumentos if guardados.get(k) != argumentos[k])
        if diferentes:
            raise ValueError(f"para retomar, os argumentos têm de ser os do treino interrompido ({out_dir / ARGUMENTOS}); "
                             f"diferentes: {', '.join(f'{k} ({guardados.get(k)!r} -> {argumentos[k]!r})' for k in diferentes)}")
        with open(Path(ultimo) / "trainer_state.json", encoding="utf-8") as f:
            retomado = int(round(json.load(f)["epoch"]))  # a última época acabada
    else:
        retomado = None
        _check_empty(checkpoints_dir, "checkpoints", apagar)
        _check_empty(pontuacoes_dir, "pontuações", apagar)
        if out_dir.exists() and any(out_dir.glob("epoca_*")):
            raise ValueError(f"{out_dir}: já tem pastas epoca_* de outro treino; use outra pasta de saída, ou "
                             f"--retomar se o treino caiu a meio, ou {apagar}")
    out_dir.mkdir(parents=True, exist_ok=True)
    pontuacoes_dir.mkdir(parents=True, exist_ok=True)
    if not retomar:
        (out_dir / ARGUMENTOS).write_text(json.dumps(argumentos, ensure_ascii=False, indent=1), encoding="utf-8")

    treino = pd.read_parquet(train_path, columns=["data", "texto", "codigo"])
    datas_treino = _date_range(treino["data"])
    val = pd.read_parquet(val_path, columns=["idcontrato", "data", "texto"])
    datas_val = _date_range(val["data"])
    ids = val["idcontrato"].astype(str).to_numpy()

    # As etiquetas e as frequências, pelas mesmas regras do `bert.train` (só com o treino), para as
    # sugestões de cada época; no fim confirma-se que são as do modelo treinado.
    _, classes = bert.training_labels(treino["codigo"], n_min)
    prior = tfidf.log_prior(treino["codigo"], classes, n_min=n_min)
    # As épocas já feitas antes da interrupção (com `retomar`) vêm dos seus epoca.json
    epocas_info = [json.loads((out_dir / epoch_dir(k) / EPOCA_INFO).read_text(encoding="utf-8"))
                   for k in range(1, (retomado or 0) + 1)]

    def on_epoch(k, checkpoint, pontuacoes):
        np.save(pontuacoes_dir / scores_file(k), pontuacoes)
        pasta = out_dir / epoch_dir(k)
        pasta.mkdir(exist_ok=True)  # existe só ao retomar, se a época tinha começado
        for tau in taus:
            sug = bert.suggest_from_scores(pontuacoes, classes, prior, tau)
            bert.suggestions_frame(ids, sug).to_parquet(pasta / suggestions_file(tau), index=False)
        info = {"epoca": k, "checkpoint": checkpoint, "pontuacoes": scores_file(k),
                "segundos_desde_o_inicio": round(time.perf_counter() - t0, 2)}
        (pasta / EPOCA_INFO).write_text(json.dumps(info), encoding="utf-8")
        epocas_info.append(info)

    t0 = time.perf_counter()
    out = bert.train(treino["texto"], treino["codigo"], modelo, checkpoints_dir, textos_val=val["texto"],
                     on_epoch=on_epoch, n_min=n_min, max_length=max_length, taxa_aprendizagem=taxa_aprendizagem,
                     lote=lote, epocas=epocas, aquecimento=aquecimento, decaimento_pesos=decaimento_pesos,
                     semente=semente, dispositivo=dispositivo, retomar=retomar)
    t_treinar = time.perf_counter() - t0
    if list(out["classes"]) != list(classes) or not np.array_equal(out["log_prior"], prior):
        raise RuntimeError("as etiquetas das pontuações não são as do modelo treinado")
    np.save(pontuacoes_dir / LOG_PRIOR, out["log_prior"])
    with open(pontuacoes_dir / ETIQUETAS, "w", encoding="utf-8") as f:
        json.dump([str(c) for c in classes], f)

    # A perda média de cada época, do registo do Trainer (logging_strategy="epoch")
    perdas = {int(round(h["epoch"])): h["loss"] for h in out["trainer"].state.log_history if "loss" in h}
    for e in epocas_info:
        e["perda_treino"] = perdas.get(e["epoca"])

    # As sugestões da última época ficam também na raiz da saída, com os nomes do TF-IDF
    ultima = out_dir / epoch_dir(epocas_info[-1]["epoca"])
    for tau in taus:
        shutil.copyfile(ultima / suggestions_file(tau), out_dir / suggestions_file(tau))

    info = out["info"]
    resumo = {
        "data": dt.datetime.now().isoformat(timespec="seconds"),
        "modelo": {"chave": modelo if modelo in bert_models.MODELOS else None, "nome": info["nome"],
                   "revisao": info["revisao"]},
        "max_length": info["max_length"],
        "n_min": n_min,
        "taus": list(taus),
        "ficheiros": {suggestions_file(t): t for t in taus},
        "ficheiro_do_modelo_por_defeito": suggestions_file(tfidf.TAU) if tfidf.TAU in taus else None,
        "treino": {"ficheiro": Path(train_path).name, "datas": datas_treino},
        "validacao": {"ficheiro": Path(val_path).name, "datas": datas_val, "contratos": len(ids)},
        "exemplos_usados": out["exemplos_usados"],
        "exemplos_saidos": out["exemplos_saidos"],
        "etiquetas": int(len(classes)),
        "taxa_aprendizagem": taxa_aprendizagem,
        "lote": lote,
        "epocas": epocas,
        "aquecimento": aquecimento,
        "decaimento_pesos": decaimento_pesos,
        "semente": semente,
        "dispositivo": dispositivo,
        "gpu": gpu,
        "checkpoints": str(checkpoints_dir),
        "pontuacoes": str(pontuacoes_dir),
        "por_epoca": epocas_info,
        # Com `retomar`, a última época acabada antes da interrupção; o tempo e os segundos das
        # épocas seguintes contam só a partir da retoma
        "retomado_depois_da_epoca": retomado,
        "tempo_treinar_s": round(t_treinar, 2),
        "pico_rss_mb": round(peak_rss_mb()),
        "nucleos": os.cpu_count(),
        "versoes": _versions(),
    }
    with open(out_dir / RESUMO, "w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=1)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Tuna o BERTimbau ou o Albertina e sugere 5 códigos para a "
                                            "validação, em cada época.")
    p.add_argument("--treino", default=str(TREINO), help="treino.parquet")
    p.add_argument("--validacao", default=str(VALIDACAO), help="validacao.parquet")
    p.add_argument("--modelo", default="bertimbau",
                   help=f"{' ou '.join(bert_models.MODELOS)}, ou a pasta de um modelo gravado")
    p.add_argument("--saida", default=None,
                   help="pasta das sugestões e do treino.json (por defeito metrics/bert/<modelo>)")
    p.add_argument("--checkpoints", default=None, help="pasta dos checkpoints (por defeito models/bert/<saída>/checkpoints)")
    p.add_argument("--pontuacoes", default=None, help="pasta das pontuações (por defeito models/bert/<saída>/pontuacoes)")
    p.add_argument("--n-min", type=int, default=DEFAULT_N_MIN, choices=[5, 10, 20],
                   help="N da regra dos códigos raros")
    p.add_argument("--taus", type=float, nargs="+", default=[0.0, tfidf.TAU],
                   help="τ da correção das etiquetas frequentes; um ficheiro de sugestões por τ e por época")
    p.add_argument("--max-length", type=int, default=None,
                   help="comprimento máximo em tokens (por defeito o do E5-T1, bert_models.py)")
    p.add_argument("--taxa-aprendizagem", type=float, default=bert.TAXA_APRENDIZAGEM, help="taxa de aprendizagem")
    p.add_argument("--lote", type=int, default=bert.LOTE, help="exemplos por lote")
    p.add_argument("--epocas", type=int, default=bert.EPOCAS, help="número de épocas")
    p.add_argument("--aquecimento", type=float, default=bert.AQUECIMENTO,
                   help="fração dos passos com a taxa a subir de 0 até à taxa pedida")
    p.add_argument("--decaimento-pesos", type=float, default=bert.DECAIMENTO_PESOS, help="decaimento dos pesos")
    p.add_argument("--semente", type=int, default=bert.SEMENTE, help="semente aleatória")
    p.add_argument("--retomar", action="store_true",
                   help="continua um treino interrompido do último checkpoint (mesmas pastas e argumentos)")
    p.add_argument("--dispositivo", choices=bert.DISPOSITIVOS, default="cuda",
                   help="cuda (no pod; confirma primeiro que o torch vê a GPU) ou cpu")
    a = p.parse_args(argv)
    saida = a.saida or str(RAIZ / "metrics" / "bert" / Path(a.modelo).name)
    run(a.treino, a.validacao, saida, modelo=a.modelo, n_min=a.n_min, taus=a.taus, max_length=a.max_length,
        taxa_aprendizagem=a.taxa_aprendizagem, lote=a.lote, epocas=a.epocas, aquecimento=a.aquecimento,
        decaimento_pesos=a.decaimento_pesos, semente=a.semente, dispositivo=a.dispositivo,
        checkpoints_dir=a.checkpoints, pontuacoes_dir=a.pontuacoes, retomar=a.retomar)
    print("Gravado:", saida)


if __name__ == "__main__":
    main()
