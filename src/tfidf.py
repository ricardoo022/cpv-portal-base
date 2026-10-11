"""Modelo TF-IDF: texto -> 5 etiquetas de treino ordenadas (história E4-H1).

O TF-IDF transforma cada texto num vetor de pesos de palavras e pares de palavras; o classificador
é uma regressão logística um-contra-todos treinada com `SGDClassifier(loss="log_loss")` (decisão 5
do E4: a `LogisticRegression(solver="saga")` não acaba em tempo útil com ~3 000 etiquetas).

As etiquetas vêm de `labels.collapse_rare_codes` com o mesmo N que a régua (`baselines.py`), para
o modelo e a régua escolherem entre as mesmas etiquetas. Como nenhuma etiqueta de treino está ao
nível da divisão, o modelo nunca sugere divisões.

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


def train(texto, codigo, n_min=DEFAULT_N_MIN, min_df=2, n_jobs=-1, **sgd_params):
    """Treina o TF-IDF + classificador com o treino (`texto` e `codigo`, com o mesmo índice).

    A etiqueta de cada exemplo é `collapse_rare_codes(codigo, n_min)`; os exemplos que saem (NaN)
    não entram no treino. `sgd_params` vai para o `SGDClassifier` (por exemplo, `alpha`).
    Devolve um `Pipeline` com os passos "tfidf" e "clf".
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
        ("clf", SGDClassifier(loss="log_loss", random_state=SEED, n_jobs=n_jobs, **sgd_params)),
    ])
    model.fit(texto[keep].tolist(), label[keep].to_numpy())
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


def suggest(model, textos, k=TOP_K, batch_size=2000):
    """k sugestões distintas por texto, da mais para a menos provável (formato da `evaluation`).

    `textos` é uma Series (o índice mantém-se) ou uma lista (índice 0, 1, 2, ...). Os textos
    pontuam-se em lotes de `batch_size` para a matriz textos x etiquetas não ocupar memória
    demais (a validação tem ~24 mil textos e o modelo ~3 000 etiquetas).
    """
    classes = np.asarray(model.classes_)
    if len(classes) < k:
        raise ValueError(f"o modelo só tem {len(classes)} etiquetas; são precisas {k}")
    if not isinstance(textos, pd.Series):
        textos = pd.Series(list(textos), dtype=object)
    values = textos.tolist()
    out = []
    for start in range(0, len(values), batch_size):
        scores = model.decision_function(values[start:start + batch_size])
        out.extend(top_k(np.asarray(scores), classes, k))
    return pd.Series(out, index=textos.index, dtype=object)
