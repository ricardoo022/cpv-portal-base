"""Treino do TF-IDF + regressão logística (E4-H1) e sugestões para a validação.

Lê só o treino e a validação, por caminhos explícitos; recusa um ficheiro chamado
`teste.parquet`, porque o teste não se olha antes do fim (CLAUDE.md). Passos:

1. treina o modelo (`tfidf.train`) com o `texto` e o `codigo` do treino; as etiquetas e as
   contagens do N (regra dos códigos raros) saem só do treino;
2. grava o modelo (joblib), se for pedido;
3. sugere 5 códigos para cada contrato da validação e grava `sugestoes_validacao.parquet`
   (`idcontrato`, `sugestoes`), pela ordem do `validacao.parquet`. Este ficheiro é pequeno (sem
   texto) para ir para o git e ser lido no portátil, onde o modelo não cabe;
   Com vários τ (`--taus`, spike E4-S4), grava também um ficheiro por τ diferente de 0
   (`sugestoes_validacao_tau0.5.parquet`, ...), com a correção das etiquetas frequentes;
4. grava `treino.json` com o que é preciso para repetir e comparar o treino (datas, N, tamanho
   do vocabulário, tempos, memória, versões).

Corre no pod da Runpod (docs/maquina_runpod.md), não no portátil: o treino completo pede cerca
de 6 GB e o portátil tem 5,8 GB partilhados com o editor (E4-S0).

Exemplo (da raiz do projeto):
    .venv/bin/python src/train_tfidf.py --n-min 10
    .venv/bin/python src/train_tfidf.py --alpha 1e-5 --taus 0 0.25 0.5 1 --sem-modelo \
        --saida metrics/e4_s4/alpha_1e-05
"""

import argparse
import datetime as dt
import json
import os
import platform
import resource
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import tfidf
from labels import DEFAULT_N_MIN, collapse_rare_codes

RAIZ = Path(__file__).resolve().parents[1]
TREINO = RAIZ / "data" / "processed" / "treino.parquet"
VALIDACAO = RAIZ / "data" / "processed" / "validacao.parquet"
SAIDA = RAIZ / "metrics" / "e4_h1"
SUGESTOES = "sugestoes_validacao.parquet"
RESUMO = "treino.json"
PROIBIDO = "teste.parquet"


def check_not_test(path):
    """Erro se `path` for o ficheiro de teste: o teste só se usa no fim do projeto."""
    if Path(path).name == PROIBIDO:
        raise ValueError(f"{path}: o conjunto de teste não se lê no treino nem nas decisões")


def peak_rss_mb():
    """Pico de memória residente do processo até agora (ru_maxrss vem em KB no Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _date_range(data):
    return [str(pd.Timestamp(data.min()).date()), str(pd.Timestamp(data.max()).date())]


def _versions():
    import scipy
    import sklearn
    return {"python": platform.python_version(), "sklearn": sklearn.__version__,
            "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__}


def suggestions_file(tau):
    """Nome do ficheiro das sugestões da validação com o τ dado (τ = 0: o nome do E4-H1)."""
    return SUGESTOES if tau == 0 else f"sugestoes_validacao_tau{tau:g}.parquet"


def run(train_path, val_path, out_dir, model_path=None, n_min=DEFAULT_N_MIN, min_df=2,
        alpha=tfidf.ALPHA, taus=(0.0, tfidf.TAU)):
    """Treina, grava o modelo (se `model_path`), as sugestões da validação e o resumo.

    `alpha` é a regularização do `SGDClassifier`; `taus` são os τ da correção das etiquetas
    frequentes, um ficheiro de sugestões por τ. Por defeito, os valores escolhidos no E4-S4, e
    também τ = 0 (sem correção), para comparar.

    Devolve o modelo treinado (um Pipeline do sklearn).
    """
    from sklearn.exceptions import ConvergenceWarning

    check_not_test(train_path)
    check_not_test(val_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Treino: só as colunas precisas
    treino = pd.read_parquet(train_path, columns=["data", "texto", "codigo"])
    datas_treino = _date_range(treino["data"])
    etiqueta = collapse_rare_codes(treino["codigo"], n_min)  # só para contar; o train refaz-o
    n_usados, n_saidos = int(etiqueta.notna().sum()), int(etiqueta.isna().sum())
    del etiqueta

    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always", ConvergenceWarning)
        t0 = time.perf_counter()
        model = tfidf.train(treino["texto"], treino["codigo"], n_min=n_min, min_df=min_df, alpha=alpha)
        t_treinar = time.perf_counter() - t0
    convergiu = not any(issubclass(w.category, ConvergenceWarning) for w in avisos)
    del treino  # o texto do treino já não é preciso; liberta memória antes da validação

    # 2. Modelo
    if model_path is not None:
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)

    # 3. Sugestões para a validação, pela ordem do ficheiro
    val = pd.read_parquet(val_path, columns=["idcontrato", "data", "texto"])
    datas_val = _date_range(val["data"])
    t0 = time.perf_counter()
    por_tau = tfidf.suggest_many(model, val["texto"], taus)
    t_sugerir = time.perf_counter() - t0
    ids = val["idcontrato"].astype(str).to_numpy()
    for tau, sugestoes in por_tau.items():
        out = pd.DataFrame({"idcontrato": ids, "sugestoes": [[str(c) for c in s] for s in sugestoes]})
        out.to_parquet(out_dir / suggestions_file(tau), index=False)

    # 4. Resumo
    vec, clf = model.named_steps["tfidf"], model.named_steps["clf"]
    resumo = {
        "data": dt.datetime.now().isoformat(timespec="seconds"),
        "n_min": n_min,
        "min_df": min_df,
        "alpha": clf.alpha,
        "taus": list(taus),
        "treino": {"ficheiro": Path(train_path).name, "datas": datas_treino},
        "validacao": {"ficheiro": Path(val_path).name, "datas": datas_val, "contratos": len(out)},
        "exemplos_usados": n_usados,
        "exemplos_saidos": n_saidos,
        "etiquetas": int(len(clf.classes_)),
        "vocabulario": int(len(vec.vocabulary_)),
        "n_iter": np.atleast_1d(clf.n_iter_).tolist(),
        "convergiu": convergiu,
        "tempo_vetorizar_treinar_s": round(t_treinar, 2),
        "tempo_sugerir_s": round(t_sugerir, 2),
        "pico_rss_mb": round(peak_rss_mb()),
        "nucleos": os.cpu_count(),
        "versoes": _versions(),
        "coef_bytes": int(clf.coef_.nbytes),
    }
    with open(out_dir / RESUMO, "w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=1)
    return model


def main():
    p = argparse.ArgumentParser(description="Treina o TF-IDF e sugere 5 códigos para a validação.")
    p.add_argument("--treino", default=str(TREINO), help="treino.parquet")
    p.add_argument("--validacao", default=str(VALIDACAO), help="validacao.parquet")
    p.add_argument("--n-min", type=int, default=DEFAULT_N_MIN, choices=[5, 10, 20],
                   help="N da regra dos códigos raros")
    p.add_argument("--saida", default=str(SAIDA), help="pasta das sugestões e do treino.json")
    p.add_argument("--modelo", default=None, help="ficheiro do modelo (por defeito models/tfidf_n{N}.joblib)")
    p.add_argument("--sem-modelo", action="store_true", help="não grava o modelo (poupa ~5 GB de disco)")
    p.add_argument("--alpha", type=float, default=tfidf.ALPHA, help="regularização (por defeito a do E4-S4)")
    p.add_argument("--taus", type=float, nargs="+", default=[0.0, tfidf.TAU],
                   help="τ da correção das etiquetas frequentes; um ficheiro de sugestões por τ")
    a = p.parse_args()
    modelo = None if a.sem_modelo else (a.modelo or str(RAIZ / "models" / f"tfidf_n{a.n_min}.joblib"))
    run(a.treino, a.validacao, a.saida, model_path=modelo, n_min=a.n_min, alpha=a.alpha, taus=a.taus)
    print("Gravado:", modelo or "(sem modelo)", "e", a.saida)


if __name__ == "__main__":
    main()
