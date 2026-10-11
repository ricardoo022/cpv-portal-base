"""Ensaio do spike E4-S0: quanto custa treinar TF-IDF + classificador linear?

Mede o tempo e o pico de memória de um treino (numa amostra ou no treino completo) e acrescenta
uma linha JSON ao ficheiro de saída. Os números do E4-S0 vêm daqui (`metrics/e4_s0/`).
Não é código do pipeline: não avalia qualidade, não grava modelos, não lê validação nem teste.
O treino a sério nasce no E4-H1.

Exemplos (da raiz do projeto):
    .venv/bin/python scripts/ensaio_treino.py --n-min 5 --amostra 10000 --modelo sgd \
        --saida metrics/e4_s0/portatil.jsonl
    .venv/bin/python scripts/ensaio_treino.py --n-min 10 --amostra 0 --modelo sgd \
        --saida metrics/e4_s0/pod.jsonl          # --amostra 0 = treino completo
"""

import argparse
import datetime as dt
import json
import os
import platform
import resource
import sys
import time
import warnings

import numpy as np
import pandas as pd

from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from labels import collapse_rare_codes  # noqa: E402

TREINO = RAIZ / "data" / "processed" / "treino.parquet"
SEMENTE = 0


def pico_rss_mb():
    """Pico de memória residente do processo até agora (ru_maxrss vem em KB no Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def max_features_arg(valor):
    return None if valor in ("None", "none", "0") else int(valor)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-min", type=int, required=True, choices=[5, 10, 20])
    p.add_argument("--amostra", type=int, required=True, help="linhas; 0 = treino completo")
    p.add_argument("--max-features", type=max_features_arg, default=None)
    p.add_argument("--modelo", choices=["logreg", "sgd"], required=True)
    p.add_argument("--max-iter", type=int, default=100, help="só para logreg")
    p.add_argument("--saida", required=True)
    p.add_argument("--so-vetorizar", action="store_true",
                   help="mede só o vocabulário (n_features) sem treinar; poupa memória")
    a = p.parse_args()

    import scipy
    import sklearn
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression, SGDClassifier

    rss_inicio = pico_rss_mb()

    # 1. Etiquetas: a regra aplica-se à coluna COMPLETA (as contagens dependem de todo o treino);
    #    só depois se tira a amostra, para as linhas terem a mesma etiqueta que no treino real.
    codigo = pd.read_parquet(TREINO, columns=["codigo"])["codigo"]
    etiqueta = collapse_rare_codes(codigo, a.n_min)
    n_linhas_total = len(etiqueta)
    n_classes_total = int(etiqueta.nunique())
    validas = np.flatnonzero(etiqueta.notna().to_numpy())
    rng = np.random.default_rng(SEMENTE)
    if a.amostra == 0:
        idx = validas
    else:
        idx = np.sort(rng.choice(validas, size=min(a.amostra, len(validas)), replace=False))
    y = etiqueta.to_numpy()[idx]
    del codigo, etiqueta

    texto = pd.read_parquet(TREINO, columns=["texto"])["texto"]
    X_txt = texto.to_numpy()[idx].tolist()
    del texto
    rss_apos_carregar = pico_rss_mb()

    # 2. Vetorizar.
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                          max_features=a.max_features, dtype=np.float32)
    t0 = time.perf_counter()
    X = vec.fit_transform(X_txt)
    t_vetorizar = time.perf_counter() - t0
    rss_apos_vetorizar = pico_rss_mb()
    # O vocabulário "stop_words_" guarda os termos cortados; apaga-se para não pesar na memória.
    vec.stop_words_ = None

    if a.so_vetorizar:
        resultado = {"data": dt.datetime.now().isoformat(timespec="seconds"),
                     "argumentos": vars(a), "linhas_amostra": len(y),
                     "n_classes_amostra": int(len(np.unique(y))),
                     "n_features": int(X.shape[1]), "nnz_X": int(X.nnz),
                     "tempo_vetorizar_s": round(t_vetorizar, 2),
                     "pico_rss_mb": {"apos_carregar": round(rss_apos_carregar),
                                     "apos_vetorizar": round(rss_apos_vetorizar)}}
        with open(a.saida, "a", encoding="utf-8") as f:
            f.write(json.dumps(resultado, ensure_ascii=False) + "\n")
        print(json.dumps(resultado, ensure_ascii=False, indent=1))
        return

    # 3. Treinar.
    if a.modelo == "logreg":
        clf = LogisticRegression(solver="saga", max_iter=a.max_iter)
    else:
        clf = SGDClassifier(loss="log_loss", n_jobs=-1, random_state=0)

    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always", ConvergenceWarning)
        t0 = time.perf_counter()
        clf.fit(X, y)
        t_treinar = time.perf_counter() - t0
    convergiu = not any(issubclass(w.category, ConvergenceWarning) for w in avisos)
    rss_final = pico_rss_mb()

    resultado = {
        "data": dt.datetime.now().isoformat(timespec="seconds"),
        "maquina": {"nucleos": os.cpu_count(), "hostname": platform.node()},
        "argumentos": vars(a),
        "versoes": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                    "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__},
        "linhas_treino_total": n_linhas_total,
        "classes_treino_total": n_classes_total,
        "linhas_amostra": len(y),
        "n_classes_amostra": int(len(clf.classes_)),
        "n_features": int(X.shape[1]),
        "nnz_X": int(X.nnz),
        "tempo_vetorizar_s": round(t_vetorizar, 2),
        "tempo_treinar_s": round(t_treinar, 2),
        "n_iter": np.atleast_1d(clf.n_iter_).tolist() if hasattr(clf, "n_iter_") else None,
        "convergiu": convergiu,
        "avisos": sorted({str(w.message)[:200] for w in avisos}),
        "coef_dtype": str(clf.coef_.dtype),
        "coef_shape": list(clf.coef_.shape),
        "coef_bytes": int(clf.coef_.nbytes),
        "pico_rss_mb": {"inicio": round(rss_inicio), "apos_carregar": round(rss_apos_carregar),
                        "apos_vetorizar": round(rss_apos_vetorizar), "final": round(rss_final)},
    }
    with open(a.saida, "a", encoding="utf-8") as f:
        f.write(json.dumps(resultado, ensure_ascii=False) + "\n")
    print(json.dumps(resultado, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
