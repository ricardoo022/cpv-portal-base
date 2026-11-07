"""Sinal de erro 1 do E5 (E5-T4 e E5-T5): o modelo consegue decorar 100 contratos do treino?

Condição escrita no épico E5 (docs/backlog.md, "Decorar poucos exemplos"): treinado só com 100
contratos do treino, sorteados com uma semente fixa, com as etiquetas do treino completo (cerca de
3 000) e até 30 épocas, o modelo acerta no top-1 em pelo menos 95 deles. Se não acertar, há um
erro no código de treino (por exemplo, etiquetas desalinhadas dos textos).

Passos:
1. lê o treino completo e calcula as etiquetas como o treino a sério (`bert.training_labels`, a
   regra dos códigos raros com o N, contada em todo o treino): a camada de classificação tem uma
   saída por cada uma dessas ~3 000 etiquetas, e não só pelas dos 100 contratos, para se testar a
   ligação real entre textos, etiquetas e saídas;
2. sorteia os 100 contratos (com a semente) entre os que ficam com etiqueta (os que a regra dos
   códigos raros tira não entram em nenhum treino);
3. tuna o modelo só com eles (`bert.train`, com as `classes` do passo 1), sem gravar checkpoints,
   e no fim de cada época pontua os mesmos 100 textos (sem *dropout*) e conta os acertos no top-1
   (τ = 0: só a pontuação, sem a correção das etiquetas frequentes, que usa as frequências do
   treino completo e não é o que se testa aqui);
4. grava `<saida>/sinal_decorar.json`: os acertos na última época, o critério, a primeira época que
   o cumpriu e se passou (passa se alguma época o cumprir, como está no YAML, mesmo que os acertos
   desçam depois), os acertos de cada época, os 100 contratos (`idcontrato`, etiqueta e top-1 do
   modelo na última época), a taxa de
   aprendizagem, as épocas, a semente, o número de etiquetas, o tempo e a memória da GPU, o
   SHA-256 do treino, o commit e o YAML da experiência.

Não se usa o `train_bert.py` porque ali a validação são contratos de outubro de 2025 (o
`check_inputs` recusa outros); aqui os textos pontuados são os do próprio treino. Recusa o teste
como o treino a sério: pelo nome (`teste.parquet`) e pelas datas (o treino tem de acabar antes da
validação). Na GPU (`--dispositivo cuda`, por defeito) confirma primeiro que o `torch` vê a GPU, e
recusa-se sem `--experiencia` ou sem commit do git, como o `train_bert.py`. A taxa de aprendizagem
não tem valor por defeito: o épico diz que fica escrita no YAML antes de correr.

O comando acaba com código 1 se o sinal falhar (o ficheiro fica gravado na mesma).

Exemplo (no pod, da raiz do projeto):
    .venv/bin/python scripts/sinal_decorar.py --modelo bertimbau --taxa-aprendizagem 1e-4 \
        --saida metrics/bert/sinal_decorar_bertimbau --experiencia experiencias/e5_t4.yaml
"""

import argparse
import datetime as dt
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
import bert  # noqa: E402
import bert_models  # noqa: E402
import train_bert  # noqa: E402
from cpv_data import VAL_START, file_sha256  # noqa: E402
from labels import DEFAULT_N_MIN  # noqa: E402
from train_tfidf import TREINO, check_dates, check_not_test  # noqa: E402

N_CONTRATOS = 100
EPOCAS = 30
CRITERIO = 0.95  # 95 de 100
SEMENTE = 0
RESULTADO = "sinal_decorar.json"


def sample(etiqueta, n, semente=SEMENTE):
    """Os índices de `n` linhas com etiqueta, sorteados com a `semente`, pela ordem do ficheiro."""
    com = etiqueta.index[etiqueta.notna()]
    if n > len(com):
        raise ValueError(f"o treino só tem {len(com)} contratos com etiqueta; pediram-se {n}")
    escolhidos = np.random.default_rng(semente).choice(len(com), size=n, replace=False)
    return com[np.sort(escolhidos)]


def first_epoch_meeting(por_epoca, criterio):
    """A primeira época com pelo menos `criterio` acertos, ou None se nenhuma chegar lá."""
    return next((e["epoca"] for e in por_epoca if e["acertos"] >= criterio), None)


def run(train_path, out_dir, modelo="bertimbau", taxa_aprendizagem=None, n=N_CONTRATOS, epocas=EPOCAS,
        semente=SEMENTE, n_min=DEFAULT_N_MIN, max_length=None, lote=bert.LOTE, aquecimento=bert.AQUECIMENTO,
        decaimento_pesos=bert.DECAIMENTO_PESOS, dispositivo="cuda", experiencia=None):
    """Treina com `n` contratos do treino e conta quantos acerta no top-1. Devolve o resultado gravado."""
    bert.check_dispositivo(dispositivo)
    gpu = bert.check_gpu() if dispositivo == "cuda" else None  # antes de ler qualquer ficheiro
    if taxa_aprendizagem is None or taxa_aprendizagem <= 0:
        raise ValueError(f"a taxa de aprendizagem é obrigatória e > 0 (a do YAML): {taxa_aprendizagem!r}")
    if dispositivo == "cuda" and experiencia is None:
        raise ValueError("na GPU o YAML da experiência é obrigatório (--experiencia experiencias/<item>.yaml), "
                         "onde a taxa de aprendizagem do sinal fica escrita antes de correr")
    check_not_test(train_path)
    check_dates(train_path, VAL_START)
    experiencia = train_bert.read_experiment(experiencia) if experiencia is not None else None
    git = train_bert.git_info()
    if git["commit"] is None and dispositivo == "cuda":
        raise ValueError("não se leu o commit do git (ver o docs/maquina_runpod.md, secção 7); na GPU o commit é "
                         "obrigatório")
    out_dir = Path(out_dir)
    if (out_dir / RESULTADO).exists():
        raise ValueError(f"{out_dir / RESULTADO} já existe; use outra pasta de saída ou apague-o")

    treino = pd.read_parquet(train_path, columns=["idcontrato", "texto", "codigo"])
    etiqueta, classes = bert.training_labels(treino["codigo"], n_min)  # contadas em todo o treino
    idx = sample(etiqueta, n, semente)
    textos, alvo = treino.loc[idx, "texto"], etiqueta.loc[idx]

    por_epoca, top1 = [], {}

    def on_epoch(k, checkpoint, pontuacoes):
        top1["ultima"] = [s[0] for s in bert.suggest_from_scores(pontuacoes, classes, k=1)]
        por_epoca.append({"epoca": k, "acertos": int(sum(a == b for a, b in zip(top1["ultima"], alvo)))})

    t0 = time.perf_counter()
    # Sem checkpoints (30 épocas x ~1,3 GB cada): a pasta é só a de trabalho do Trainer
    with tempfile.TemporaryDirectory() as pasta_trainer:
        out = bert.train(textos, alvo, modelo, pasta_trainer, textos_val=textos, on_epoch=on_epoch, classes=classes,
                         max_length=max_length, taxa_aprendizagem=taxa_aprendizagem, lote=lote, epocas=epocas,
                         aquecimento=aquecimento, decaimento_pesos=decaimento_pesos, semente=semente,
                         dispositivo=dispositivo, gravar_checkpoints=False)
    tempo = time.perf_counter() - t0
    perdas = {int(round(h["epoch"])): h["loss"] for h in out["trainer"].state.log_history if "loss" in h}
    for e in por_epoca:
        e["perda_treino"] = perdas.get(e["epoca"])

    acertos = por_epoca[-1]["acertos"]
    criterio = math.ceil(CRITERIO * n)
    primeira_epoca = first_epoch_meeting(por_epoca, criterio)
    info = out["info"]
    resultado = {
        "data": dt.datetime.now().isoformat(timespec="seconds"),
        "acertos": acertos,
        "contratos": n,
        "top1": acertos / n,
        "criterio_acertos": criterio,
        "primeira_epoca_criterio": primeira_epoca,
        "passou": primeira_epoca is not None,
        "modelo": {"chave": modelo if modelo in bert_models.MODELOS else None, "nome": info["nome"],
                   "revisao": info["revisao"]},
        "max_length": info["max_length"],
        "n_min": n_min,
        "etiquetas": int(len(classes)),  # as do treino completo: o número de saídas do modelo
        "etiquetas_nos_contratos": int(alvo.nunique()),
        "taxa_aprendizagem": taxa_aprendizagem,
        "epocas": epocas,
        "lote": lote,
        "aquecimento": aquecimento,
        "decaimento_pesos": decaimento_pesos,
        "semente": semente,
        "dispositivo": dispositivo,
        "gpu": gpu,
        "por_epoca": por_epoca,
        "contratos_sorteados": [{"idcontrato": str(i), "etiqueta": str(e), "top1": t}
                                for i, e, t in zip(treino.loc[idx, "idcontrato"], alvo, top1["ultima"])],
        "tempo_s": round(tempo, 2),
        "memoria_gpu_mb": train_bert.gpu_memory_mb(dispositivo),
        "treino": {"ficheiro": Path(train_path).name, "sha256": file_sha256(train_path)},
        "git": git,
        "experiencia": experiencia,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / RESULTADO).write_text(json.dumps(resultado, ensure_ascii=False, indent=1), encoding="utf-8")
    return resultado


def main(argv=None):
    p = argparse.ArgumentParser(description="Sinal de erro 1 do E5: treina com 100 contratos do treino (com as "
                                            "etiquetas do treino completo) e conta os acertos no top-1.")
    p.add_argument("--treino", default=str(TREINO), help="treino.parquet (o completo: dá as etiquetas)")
    p.add_argument("--modelo", default="bertimbau",
                   help=f"{' ou '.join(bert_models.MODELOS)}, ou a pasta de um modelo gravado")
    p.add_argument("--saida", default=None,
                   help=f"pasta do {RESULTADO} (por defeito metrics/bert/sinal_decorar_<modelo>)")
    p.add_argument("--taxa-aprendizagem", type=float, required=True,
                   help="taxa de aprendizagem (a escrita no YAML antes de correr)")
    p.add_argument("--n", type=int, default=N_CONTRATOS, help="contratos sorteados")
    p.add_argument("--epocas", type=int, default=EPOCAS, help="número de épocas")
    p.add_argument("--semente", type=int, default=SEMENTE, help="semente do sorteio e do treino")
    p.add_argument("--n-min", type=int, default=DEFAULT_N_MIN, choices=[5, 10, 20],
                   help="N da regra dos códigos raros (contada no treino completo)")
    p.add_argument("--max-length", type=int, default=None,
                   help="comprimento máximo em tokens (por defeito o do E5-T1, bert_models.py)")
    p.add_argument("--lote", type=int, default=bert.LOTE, help="exemplos por lote")
    p.add_argument("--aquecimento", type=float, default=bert.AQUECIMENTO,
                   help="fração dos passos com a taxa a subir de 0 até à taxa pedida")
    p.add_argument("--decaimento-pesos", type=float, default=bert.DECAIMENTO_PESOS, help="decaimento dos pesos")
    p.add_argument("--experiencia", default=None, help="YAML da experiência (obrigatório na GPU)")
    p.add_argument("--dispositivo", choices=bert.DISPOSITIVOS, default="cuda",
                   help="cuda (no pod; confirma primeiro que o torch vê a GPU) ou cpu")
    a = p.parse_args(argv)
    saida = a.saida or str(RAIZ / "metrics" / "bert" / f"sinal_decorar_{Path(a.modelo).name}")
    r = run(a.treino, saida, modelo=a.modelo, taxa_aprendizagem=a.taxa_aprendizagem, n=a.n, epocas=a.epocas,
            semente=a.semente, n_min=a.n_min, max_length=a.max_length, lote=a.lote, aquecimento=a.aquecimento,
            decaimento_pesos=a.decaimento_pesos, dispositivo=a.dispositivo, experiencia=a.experiencia)
    print(f"Última época: {r['acertos']} de {r['contratos']} no top-1 (critério: >= {r['criterio_acertos']}); "
          f"primeira época a cumprir: {r['primeira_epoca_criterio']}: "
          f"{'passou' if r['passou'] else 'FALHOU'}. Gravado: {Path(saida) / RESULTADO}")
    return 0 if r["passou"] else 1


if __name__ == "__main__":
    sys.exit(main())
