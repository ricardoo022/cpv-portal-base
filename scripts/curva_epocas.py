"""Spike E4-S5: top-5 na validação depois de cada época do TF-IDF (curva só para mostrar).

Treina o mesmo modelo de `src/tfidf.py` (TF-IDF com `min_df=2` e `tfidf.VECTORIZER`, e
`SGDClassifier(loss="log_loss", alpha=..., random_state=tfidf.SEED)`, etiquetas de
`labels.collapse_rare_codes` sem os exemplos que saem), mas uma época de cada vez, e grava as
sugestões da validação depois de cada época. A curva não entra na decisão do E4-S5; os pontos
exatos (3, 6, 12, 18 e 24 épocas) vêm de treinos separados com `fit`.

Como se faz: o scikit-learn não deixa guardar o modelo a meio de um `fit`, por isso usa-se
`SGDClassifier.partial_fit`, que faz uma época por chamada. Confirmado no código do
scikit-learn 1.7.2 (`sklearn/linear_model/_stochastic_gradient.py` e `_sgd_fast.pyx.tp`):

- O passo da aprendizagem continua de época para época como num `fit`. O `fit` põe `t_ = 1`
  (`_fit`, linha 705) e o `partial_fit` só o cria se ainda não existir (`_partial_fit`,
  linhas 636-637); cada chamada passa `est.t_` ao `_plain_sgd` (`fit_binary`, linha 492) e no
  fim soma-lhe `n_iter_ * n_exemplos` (`_fit_multiclass`, linha 813). Com
  `learning_rate="optimal"` o passo é `1 / (alpha * (optimal_init + t - 1))`
  (`_sgd_fast.pyx.tp`, linha 473), por isso a época k do `partial_fit` usa os mesmos passos que
  a época k de um `fit`.
- Mas a ordem dos exemplos NÃO é a de `fit(max_iter=k, tol=None)`. No `fit`, as k épocas correm
  dentro do mesmo `_plain_sgd`, e no início de cada uma `dataset.shuffle(seed)` (linha 461)
  baralha o array de índices que já vinha baralhado da época anterior (`_seq_dataset.pyx.tp`,
  linhas 137-145, Fisher-Yates sobre `index_data_ptr`): a ordem muda de época para época. Cada
  `partial_fit` cria um dataset novo (`fit_binary`, linha 444), com os índices pela ordem original
  (`_seq_dataset.pyx.tp`, linha 331, `CSRDataset`, o do texto vetorizado), e a mesma semente (`_fit_multiclass`, linhas 784-785,
  tira as sementes de `check_random_state(self.random_state)`, que com uma semente inteira
  recomeça igual em cada chamada): todas as épocas veem os exemplos pela mesma ordem.
  Na 1.ª época as duas coisas coincidem (o teste verifica-o); daí em diante a curva é uma
  aproximação de `fit(max_iter=k, tol=None)`. (Há ainda uma diferença de arredondamento, muito
  menor: cada chamada acaba com `w.reset_wscale()`, `_sgd_fast.pyx.tp` linha 595, que passa a
  escala dos pesos para os coeficientes em float32; sem baralhar, `fit` e `partial_fit` diferem
  em ~1e-7 relativo, nos dados minúsculos dos testes.)

Em cada época grava `sugestoes_epoca_{e:02d}.parquet` (`idcontrato`, `sugestoes`), no formato
de `sugestoes_validacao.parquet` do `src/train_tfidf.py`, pela ordem do `validacao.parquet`; no
fim grava `curva.json`. Não avalia (a avaliação faz-se no portátil, num notebook, com
`src/evaluation.py`) e não grava o modelo. Recusa `teste.parquet`.

Corre no pod da Runpod (docs/maquina_runpod.md), não no portátil: o treino completo não cabe nos
5,8 GB. Escreve uma linha por época, para se seguir com `tail -f` num `tmux`.

Exemplo (da raiz do projeto):
    .venv/bin/python scripts/curva_epocas.py --saida metrics/e4_s5/curva 2>&1 | tee curva.log
"""

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
import tfidf  # noqa: E402
import train_tfidf as tt  # noqa: E402
from evaluation import TOP_K  # noqa: E402
from labels import collapse_rare_codes  # noqa: E402

TREINO = RAIZ / "data" / "processed" / "treino.parquet"
VALIDACAO = RAIZ / "data" / "processed" / "validacao.parquet"
RESUMO = "curva.json"
N_MIN = 10
MIN_DF = 2
LOTE = 2000  # linhas da validação pontuadas de cada vez, como em `tfidf.suggest_many`
NOTA = "partial_fit: mesma ordem dos exemplos em todas as épocas; aproximação de fit(max_iter=k, tol=None)"


def suggestions_file(epoca):
    """Nome do ficheiro das sugestões da validação depois da época `epoca` (1, 2, ...)."""
    return f"sugestoes_epoca_{epoca:02d}.parquet"


def _suggest(clf, X_val, prior, tau, lote=LOTE):
    """5 sugestões por linha de `X_val`, com a correção das etiquetas frequentes de força `tau`."""
    classes = np.asarray(clf.classes_)
    out = []
    for start in range(0, X_val.shape[0], lote):
        scores = np.asarray(clf.decision_function(X_val[start:start + lote]))
        adjusted = scores if tau == 0 else scores - tau * prior
        out.extend(tfidf.top_k(adjusted, classes, TOP_K))
    return out


def run(train_path, val_path, out_dir, epocas=24, alpha=tfidf.ALPHA, tau=tfidf.TAU, n_min=N_MIN,
        n_jobs=-1):
    """Treina `epocas` épocas com `partial_fit` e grava as sugestões da validação de cada uma.

    Devolve o `SGDClassifier` depois da última época (os testes comparam-no com um `fit`).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import SGDClassifier

    tt.check_not_test(train_path)
    tt.check_not_test(val_path)
    if epocas < 1:
        raise ValueError(f"--epocas tem de ser >= 1: {epocas}")
    if tau < 0:
        raise ValueError(f"τ tem de ser >= 0: {tau}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Treino: etiquetas como no `tfidf.train` (os exemplos que saem não entram)
    treino = pd.read_parquet(train_path, columns=["data", "texto", "codigo"])
    if treino["codigo"].str.endswith("000000").any():
        raise ValueError("o treino tem códigos ao nível da divisão (XX000000)")
    datas_treino = tt._date_range(treino["data"])
    label = collapse_rare_codes(treino["codigo"], n_min=n_min)
    keep = label.notna().to_numpy()
    y = label[keep].to_numpy()
    classes = np.unique(y)  # ordenadas, as mesmas que um `fit` tiraria de `y`
    if len(classes) < TOP_K:
        raise ValueError(f"o treino só tem {len(classes)} etiquetas depois da regra dos códigos raros; "
                         f"são precisas {TOP_K}")
    prior = tfidf.log_prior(treino["codigo"], classes, n_min=n_min)

    # 2. Vetorizar uma só vez: o vocabulário e os pesos IDF saem só do treino
    t0 = time.perf_counter()
    vec = TfidfVectorizer(min_df=MIN_DF, **tfidf.VECTORIZER)
    X = vec.fit_transform(treino["texto"][keep].tolist())
    n_exemplos, n_saidos = int(keep.sum()), int((~keep).sum())
    del treino, label
    val = pd.read_parquet(val_path, columns=["idcontrato", "data", "texto"])
    datas_val = tt._date_range(val["data"])
    ids = val["idcontrato"].astype(str).to_numpy()
    X_val = vec.transform(val["texto"].tolist())
    del val
    t_vetorizar = time.perf_counter() - t0
    print(f"vetorizado em {t_vetorizar:.1f} s: {n_exemplos} exemplos, {len(classes)} etiquetas, "
          f"{len(vec.vocabulary_)} termos; validação {len(ids)} contratos", flush=True)

    # 3. Uma época de cada vez
    clf = SGDClassifier(loss="log_loss", alpha=alpha, random_state=tfidf.SEED, n_jobs=n_jobs)
    tempos = []
    for e in range(1, epocas + 1):
        t0 = time.perf_counter()
        if e == 1:
            clf.partial_fit(X, y, classes=classes)
        else:
            clf.partial_fit(X, y)
        t_treinar = time.perf_counter() - t0

        t0 = time.perf_counter()
        sugestoes = _suggest(clf, X_val, prior, tau)
        out = pd.DataFrame({"idcontrato": ids, "sugestoes": [[str(c) for c in s] for s in sugestoes]})
        out.to_parquet(out_dir / suggestions_file(e), index=False)
        t_sugerir = time.perf_counter() - t0

        tempos.append({"epoca": e, "treino_s": round(t_treinar, 2), "sugestoes_s": round(t_sugerir, 2)})
        print(f"época {e}/{epocas}: treino {t_treinar:.1f} s, sugestões {t_sugerir:.1f} s, "
              f"pico RSS {tt.peak_rss_mb():.0f} MB", flush=True)

    # 4. Resumo
    resumo = {
        "data": dt.datetime.now().isoformat(timespec="seconds"),
        "alpha": clf.alpha,
        "tau": tau,
        "n_min": n_min,
        "min_df": MIN_DF,
        "epocas": epocas,
        "ficheiros": [suggestions_file(e) for e in range(1, epocas + 1)],
        "treino": {"ficheiro": Path(train_path).name, "datas": datas_treino,
                   "exemplos_usados": n_exemplos, "exemplos_saidos": n_saidos},
        "validacao": {"ficheiro": Path(val_path).name, "datas": datas_val, "contratos": len(ids)},
        "etiquetas": int(len(clf.classes_)),
        "vocabulario": int(len(vec.vocabulary_)),
        # t_ = 1 + épocas × exemplos: confirma que o passo continuou de época para época
        "t_final": float(clf.t_),
        "tempo_vetorizar_s": round(t_vetorizar, 2),
        "tempos": tempos,
        "pico_rss_mb": round(tt.peak_rss_mb()),
        "nucleos": os.cpu_count(),
        "versoes": tt._versions(),
        "nota": NOTA,
    }
    with open(out_dir / RESUMO, "w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=1)
    return clf


def main(argv=None):
    p = argparse.ArgumentParser(description="Sugestões da validação depois de cada época do TF-IDF (E4-S5).")
    p.add_argument("--treino", default=str(TREINO), help="treino.parquet")
    p.add_argument("--validacao", default=str(VALIDACAO), help="validacao.parquet")
    p.add_argument("--saida", required=True, help="pasta das sugestões por época e do curva.json")
    p.add_argument("--epocas", type=int, default=24, help="número de épocas")
    p.add_argument("--alpha", type=float, default=tfidf.ALPHA, help="regularização (por defeito a do E4-S4)")
    p.add_argument("--tau", type=float, default=tfidf.TAU, help="τ da correção das etiquetas frequentes")
    p.add_argument("--n-min", type=int, default=N_MIN, help="N da regra dos códigos raros")
    p.add_argument("--n-jobs", type=int, default=-1, help="núcleos do SGDClassifier (-1 = todos)")
    a = p.parse_args(argv)
    run(a.treino, a.validacao, a.saida, epocas=a.epocas, alpha=a.alpha, tau=a.tau, n_min=a.n_min,
        n_jobs=a.n_jobs)
    print("Gravado:", a.saida, flush=True)


if __name__ == "__main__":
    main()
