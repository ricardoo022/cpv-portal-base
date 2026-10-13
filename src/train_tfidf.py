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
   do vocabulário, épocas, tempos, memória, versões).

Épocas (spike E4-S5): por defeito, todas as etiquetas treinam exatamente `tfidf.EPOCAS` épocas
(24, escolhidas no E4-S5), sem paragem automática (`max_iter`, `tol=None`); `--epocas K` muda o
número. Com `--regra-por-defeito`, cada classificador um-contra-todos para pela regra do
scikit-learn (a perda no treino não desce `tol` × n.º de exemplos em 5 épocas seguidas): é o que
usam os spikes que mudam o N ou a janela do treino, onde as 24 épocas não foram medidas.

Entidade (spike E4-S3): `--com-entidade` junta a `adjudicante` ao início do texto de entrada, no
treino e na validação (`with_entity`). Por defeito o texto é o da decisão 5, sem a entidade.

Corre no pod da Runpod (docs/maquina_runpod.md), não no portátil: o treino completo pede cerca
de 6 GB e o portátil tem 5,8 GB partilhados com o editor (E4-S0).

Exemplo (da raiz do projeto):
    .venv/bin/python src/train_tfidf.py --n-min 10
    .venv/bin/python src/train_tfidf.py --alpha 1e-5 --taus 0 0.25 0.5 1 --regra-por-defeito \
        --sem-modelo --saida metrics/e4_s4/alpha_1e-5
    .venv/bin/python src/train_tfidf.py --epocas 12 --sem-modelo --n-jobs 8 \
        --saida metrics/e4_s5/epocas_12
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
SAIDA = RAIZ / "metrics" / "tfidf"  # metrics/e4_h1/ e metrics/e4_s4/ ficam como registo fixo
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


def with_entity(texto, adjudicante, sep=" | "):
    """Junta a entidade (`adjudicante`) ao início do texto de entrada: `entidade | texto`.

    Spike E4-S3: testa se a entidade compradora ajuda o modelo (decisão 7). A `adjudicante` é uma
    coluna `category`; converte-se para texto e limpa-se como o `build_text` do prepare.py (só os
    espaços). O `sep` é o mesmo que separa as partes do texto na decisão 5. Se a entidade estiver
    vazia (ou em falta), fica só o texto. O índice mantém-se, para `tfidf.train` o poder alinhar
    com o `codigo`.
    """
    ent = adjudicante.astype("object").fillna("").str.split().str.join(" ")
    return texto.where(ent == "", ent + sep + texto)


def run(train_path, val_path, out_dir, model_path=None, n_min=DEFAULT_N_MIN, min_df=2,
        alpha=tfidf.ALPHA, taus=(0.0, tfidf.TAU), epocas=tfidf.EPOCAS, n_jobs=-1, com_entidade=False):
    """Treina, grava o modelo (se `model_path`), as sugestões da validação e o resumo.

    `alpha` é a regularização do `SGDClassifier`; `taus` são os τ da correção das etiquetas
    frequentes, um ficheiro de sugestões por τ. Por defeito, os valores escolhidos no E4-S4, e
    também τ = 0 (sem correção), para comparar. `epocas`: k épocas fixas para todas as etiquetas
    (por defeito `tfidf.EPOCAS`, escolhido no E4-S5), ou None para a regra de paragem por defeito
    do scikit-learn (`tfidf.sgd_epochs`).
    `n_jobs`: núcleos do classificador; não muda o modelo, só o tempo (e a memória).
    `com_entidade`: junta a `adjudicante` ao início do texto (`with_entity`), no treino e na
    validação (spike E4-S3). Por defeito False: o texto é o da decisão 5, sem a entidade. Tem de
    ser igual no treino e na validação, senão o modelo treinava com uma entrada e sugeria com outra.

    Devolve o modelo treinado (um Pipeline do sklearn).
    """
    from sklearn.exceptions import ConvergenceWarning

    check_not_test(train_path)
    check_not_test(val_path)
    tfidf.sgd_epochs(epocas)  # épocas inválidas dão erro aqui, antes de ler os ficheiros
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Treino: só as colunas precisas
    treino = pd.read_parquet(train_path, columns=["data", "texto", "codigo", "adjudicante"]
                             if com_entidade else ["data", "texto", "codigo"])
    datas_treino = _date_range(treino["data"])
    etiqueta = collapse_rare_codes(treino["codigo"], n_min)  # só para contar; o train refaz-o
    n_usados, n_saidos = int(etiqueta.notna().sum()), int(etiqueta.isna().sum())
    del etiqueta
    texto_treino = with_entity(treino["texto"], treino["adjudicante"]) if com_entidade else treino["texto"]

    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always", ConvergenceWarning)
        t0 = time.perf_counter()
        model = tfidf.train(texto_treino, treino["codigo"], n_min=n_min, min_df=min_df, alpha=alpha,
                            n_jobs=n_jobs, epocas=epocas)
        t_treinar = time.perf_counter() - t0
    convergiu = not any(issubclass(w.category, ConvergenceWarning) for w in avisos)
    del treino, texto_treino  # o texto do treino já não é preciso; liberta memória antes da validação

    # 2. Modelo
    if model_path is not None:
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)

    # 3. Sugestões para a validação, pela ordem do ficheiro
    val = pd.read_parquet(val_path, columns=["idcontrato", "data", "texto", "adjudicante"]
                          if com_entidade else ["idcontrato", "data", "texto"])
    datas_val = _date_range(val["data"])
    texto_val = with_entity(val["texto"], val["adjudicante"]) if com_entidade else val["texto"]
    t0 = time.perf_counter()
    por_tau = tfidf.suggest_many(model, texto_val, taus)
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
        "com_entidade": bool(com_entidade),  # spike E4-S3: texto com a adjudicante junta?
        # Que τ tem cada ficheiro de sugestões, e qual é o do modelo por defeito (`tfidf.TAU`).
        # O nome principal é o de τ = 0, por isso quem avalia o modelo escolhido lê este campo.
        "ficheiros": {suggestions_file(t): t for t in taus},
        "ficheiro_do_modelo_por_defeito": suggestions_file(tfidf.TAU) if tfidf.TAU in taus else None,
        "treino": {"ficheiro": Path(train_path).name, "datas": datas_treino},
        "validacao": {"ficheiro": Path(val_path).name, "datas": datas_val, "contratos": len(ids)},
        "exemplos_usados": n_usados,
        "exemplos_saidos": n_saidos,
        "etiquetas": int(len(clf.classes_)),
        "vocabulario": int(len(vec.vocabulary_)),
        # Épocas pedidas (None: regra por defeito) e o que o classificador usou de facto
        "epocas": None if epocas is None else int(epocas),
        "tol": clf.tol,
        "max_iter": clf.max_iter,
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
    epocas = p.add_mutually_exclusive_group()
    epocas.add_argument("--epocas", type=int, default=tfidf.EPOCAS,
                        help="épocas fixas para todas as etiquetas, sem paragem automática "
                             f"(por defeito {tfidf.EPOCAS}, as do E4-S5)")
    epocas.add_argument("--regra-por-defeito", action="store_true",
                        help="em vez de épocas fixas, a regra de paragem do scikit-learn (tol=1e-3)")
    p.add_argument("--n-jobs", type=int, default=-1, help="núcleos do classificador (-1: todos)")
    p.add_argument("--com-entidade", action="store_true",
                   help="junta a adjudicante ao início do texto, no treino e na validação (spike E4-S3)")
    a = p.parse_args()
    modelo = None if a.sem_modelo else (a.modelo or str(RAIZ / "models" / f"tfidf_n{a.n_min}.joblib"))
    run(a.treino, a.validacao, a.saida, model_path=modelo, n_min=a.n_min, alpha=a.alpha, taus=a.taus,
        epocas=None if a.regra_por_defeito else a.epocas, n_jobs=a.n_jobs, com_entidade=a.com_entidade)
    print("Gravado:", modelo or "(sem modelo)", "e", a.saida)


if __name__ == "__main__":
    main()
