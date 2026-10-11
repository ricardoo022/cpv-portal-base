"""Modelo TF-IDF: texto -> 5 etiquetas de treino ordenadas (história E4-H1).

O TF-IDF transforma cada texto num vetor de pesos de palavras e pares de palavras; o classificador
é uma regressão logística um-contra-todos treinada com `SGDClassifier(loss="log_loss")` (decisão 5
do E4: a `LogisticRegression(solver="saga")` não acaba em tempo útil com ~3 000 etiquetas).

As etiquetas vêm de `labels.collapse_rare_codes` com o mesmo N que a régua (`baselines.py`), para
o modelo e a régua escolherem entre as mesmas etiquetas. Como nenhuma etiqueta de treino está ao
nível da divisão, o modelo nunca sugere divisões.

Correção das etiquetas frequentes (spike E4-S4, Menon et al., 2021, "logit adjustment"): em vez
de ordenar pela pontuação f(x), ordena-se por f(x) − τ · log(frequência da etiqueta no treino).
Com τ = 0 não muda nada; com τ > 0 as etiquetas frequentes perdem vantagem. O artigo usa as
pontuações de um modelo softmax; aqui aplica-se às pontuações do um-contra-todos, uma adaptação.

Repetível: a semente é fixa (`SEED`) e os empates na pontuação resolvem-se pelo menor código, por
isso o mesmo texto dá sempre as mesmas 5 sugestões pela mesma ordem, e dois treinos com os mesmos
dados dão o mesmo modelo (o `SGDClassifier` tira uma semente por etiqueta antes de dividir o
trabalho pelos núcleos, por isso o `n_jobs` não muda o resultado).
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

from evaluation import TOP_K
from labels import DEFAULT_N_MIN, collapse_rare_codes

# Os mesmos parâmetros do ensaio do E4-S0 (`scripts/ensaio_treino.py`), cujos tempos e memória
# medidos valem para este treino. float32 gasta metade da memória do float64.
VECTORIZER = dict(ngram_range=(1, 2), sublinear_tf=True, dtype=np.float32)
SEED = 0
# Escolhidos na validação no spike E4-S4 (notebooks/05_tfidf_afinacao.ipynb): a regularização do
# SGDClassifier e a força da correção das etiquetas frequentes. O alpha por defeito do sklearn
# (0,0001) travava tanto os pesos das palavras que o modelo sugeria quase sempre as etiquetas
# mais frequentes.
ALPHA = 1e-7
TAU = 0.25


def train(texto, codigo, n_min=DEFAULT_N_MIN, min_df=2, n_jobs=-1, alpha=ALPHA, **sgd_params):
    """Treina o TF-IDF + classificador com o treino (`texto` e `codigo`, com o mesmo índice).

    A etiqueta de cada exemplo é `collapse_rare_codes(codigo, n_min)`; os exemplos que saem (NaN)
    não entram no treino. `alpha` é a regularização; `sgd_params` vai para o `SGDClassifier`.
    Devolve um `Pipeline` com os passos "tfidf" e "clf". O modelo leva consigo o `log_prior_`
    das suas etiquetas (contado só com este treino), para a correção das etiquetas frequentes.
    """
    if not texto.index.equals(codigo.index):
        raise ValueError("`texto` e `codigo` têm índices diferentes")
    if codigo.str.endswith("000000").any():
        # O prepare.py tira-os do treino (decisão 8); se aparecerem, o modelo podia sugeri-los.
        raise ValueError("o treino tem códigos ao nível da divisão (XX000000)")
    label = collapse_rare_codes(codigo, n_min=n_min)
    keep = label.notna()
    n_labels = label[keep].nunique()
    if n_labels < TOP_K:
        raise ValueError(
            f"o treino só tem {n_labels} etiquetas depois da regra dos códigos raros; são precisas {TOP_K}"
        )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(min_df=min_df, **VECTORIZER)),
        ("clf", SGDClassifier(loss="log_loss", alpha=alpha, random_state=SEED, n_jobs=n_jobs, **sgd_params)),
    ])
    model.fit(texto[keep].tolist(), label[keep].to_numpy())
    model.log_prior_ = log_prior(codigo, model.classes_, n_min=n_min)
    # No scikit-learn 1.7 o TfidfVectorizer já não guarda `stop_words_` (a lista dos termos
    # cortados pelo min_df, que no ensaio do E4-S0 ocupava memória), por isso não há nada a apagar.
    return model


def top_k(scores, classes, k=TOP_K):
    """As k etiquetas de maior pontuação de cada linha de `scores`, da maior para a menor.

    `classes` tem de estar por ordem crescente (o `classes_` do scikit-learn está): a ordenação
    estável de -pontuação deixa os empates pela ordem das colunas, ou seja, o menor código primeiro.
    """
    order = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    return [list(row) for row in classes[order]]


def log_prior(codigo, classes, n_min=DEFAULT_N_MIN):
    """log(frequência) de cada etiqueta de `classes` no treino, pela ordem de `classes`.

    As etiquetas contam-se como no `train` (`collapse_rare_codes` com o mesmo N), só com o treino.
    """
    label = collapse_rare_codes(codigo, n_min=n_min).dropna()
    counts = label.value_counts()
    missing = [c for c in classes if c not in counts.index]
    if missing:
        raise ValueError(f"etiquetas do modelo que não estão no treino: {missing[:3]}")
    return np.log(counts.reindex(classes).to_numpy(dtype=float) / len(label))


def suggest_many(model, textos, taus=(0.0,), prior=None, k=TOP_K, batch_size=2000):
    """`suggest` para vários τ de uma vez: {τ: Series}. As pontuações calculam-se uma só vez.

    `prior` é o `log_prior` das etiquetas do modelo; sem ele usa-se o `log_prior_` que o `train`
    guarda no modelo. Só é preciso se algum τ for diferente de 0.
    """
    classes = np.asarray(model.classes_)
    if len(classes) < k:
        raise ValueError(f"o modelo só tem {len(classes)} etiquetas; são precisas {k}")
    if any(t != 0 for t in taus):
        if prior is None:
            prior = getattr(model, "log_prior_", None)
        if prior is None:
            raise ValueError("com τ diferente de 0 é preciso o `prior` (log_prior)")
        prior = np.asarray(prior, dtype=float)
        if prior.shape != classes.shape:
            raise ValueError("o `prior` não tem uma entrada por etiqueta do modelo")
    if not isinstance(textos, pd.Series):
        textos = pd.Series(list(textos), dtype=object)
    values = textos.tolist()
    out = {t: [] for t in taus}
    for start in range(0, len(values), batch_size):
        scores = np.asarray(model.decision_function(values[start:start + batch_size]))
        for t in taus:
            adjusted = scores if t == 0 else scores - t * prior
            out[t].extend(top_k(adjusted, classes, k))
    return {t: pd.Series(v, index=textos.index, dtype=object) for t, v in out.items()}


def suggest(model, textos, k=TOP_K, batch_size=2000, tau=0.0, prior=None):
    """k sugestões distintas por texto, da mais para a menos provável (formato da `evaluation`).

    `textos` é uma Series (o índice mantém-se) ou uma lista (índice 0, 1, 2, ...). Os textos
    pontuam-se em lotes de `batch_size` para a matriz textos x etiquetas não ocupar memória
    demais (a validação tem ~24 mil textos e o modelo ~3 000 etiquetas). Com `tau` > 0 aplica-se
    a correção das etiquetas frequentes, com o `prior` de `log_prior` (por defeito o do modelo).
    Por defeito τ = 0 (só a pontuação); o valor escolhido para usar é `TAU`, que o treino e o
    comando de sugestões passam.
    """
    return suggest_many(model, textos, (tau,), prior, k, batch_size)[tau]
