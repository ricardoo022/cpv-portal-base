"""Testes unitários a src/tfidf.py (TF-IDF + SGDClassifier, história E4-H1)."""

import numpy as np
import pandas as pd
import pytest

import evaluation as ev
import labels as lb
import tfidf

# Um tema por código, com vocabulário próprio, para o modelo os separar com poucos exemplos.
# Os dois códigos de limpeza têm 1 exemplo cada: com N = 2 são raros e sobem os dois para a
# classe 90910000 (2 exemplos); com N = 3 a classe e o grupo 909 também são raros e saem.
N = 2
TEMAS = {
    "90911200": ["limpeza de escolas"],
    "90919300": ["limpeza dos escritórios municipais"],
    "33600000": ["medicamentos para o hospital", "fornecimento de medicamentos", "medicamentos genéricos"],
    "45233200": ["pavimentação da estrada", "repavimentação de estrada municipal", "obras na estrada"],
    "30192000": ["material de escritório", "papel e canetas", "tinteiros e papel"],
    "15800000": ["refeições escolares", "fornecimento de refeições", "refeições para idosos"],
    "34110000": ["aquisição de automóveis", "compra de automóveis ligeiros", "automóveis elétricos"],
}


def treino(temas=TEMAS):
    pares = [(t, c) for c, textos in temas.items() for t in textos]
    texto = pd.Series([t for t, _ in pares], index=range(100, 100 + len(pares)))
    codigo = pd.Series([c for _, c in pares], index=texto.index)
    return texto, codigo


@pytest.fixture(scope="module")
def modelo():
    texto, codigo = treino()
    return tfidf.train(texto, codigo, n_min=N, min_df=1)


def test_sugere_5_codigos_distintos_de_8_digitos_por_ordem_da_pontuacao(modelo):
    textos = pd.Series(["limpeza de escolas", "medicamentos", "estrada"])
    sug = tfidf.suggest(modelo, textos)
    scores = modelo.decision_function(textos.tolist())
    classes = list(modelo.classes_)
    for i, s in enumerate(sug):
        assert len(s) == ev.TOP_K == len(set(s))
        assert all(len(c) == 8 and c.isdigit() for c in s)
        pontos = [scores[i, classes.index(c)] for c in s]
        assert pontos == sorted(pontos, reverse=True)
        # nenhuma etiqueta de fora pontua mais do que a 5.ª sugestão
        assert max(scores[i, j] for j, c in enumerate(classes) if c not in s) <= pontos[-1]


def test_o_texto_de_treino_da_o_seu_codigo_em_primeiro(modelo):
    sug = tfidf.suggest(modelo, ["medicamentos para o hospital", "pavimentação da estrada"])
    assert sug[0][0] == "33600000"
    assert sug[1][0] == "45233200"


def test_codigo_raro_so_aparece_como_a_sua_classe(modelo):
    texto, codigo = treino()
    assert "90910000" in modelo.classes_
    assert "90911200" not in modelo.classes_ and "90919300" not in modelo.classes_
    assert tfidf.suggest(modelo, ["limpeza de escolas"])[0][0] == "90910000"
    assert set(modelo.classes_) == set(lb.collapse_rare_codes(codigo, N).dropna())


def test_as_sugestoes_sao_sempre_etiquetas_de_treino(modelo):
    texto, codigo = treino()
    sug = tfidf.suggest(modelo, pd.Series(["xyz", "limpeza", "refeições e automóveis"]))
    permitidas = set(lb.collapse_rare_codes(codigo, N).dropna())
    assert all(set(s) <= permitidas for s in sug)


def test_exemplos_que_saem_nao_entram_no_treino():
    # com N = 3 o grupo 909 (2 exemplos) é raro e sai; os outros 5 códigos têm 3 e ficam
    texto, codigo = treino()
    m = tfidf.train(texto, codigo, n_min=3, min_df=1)
    assert not any(c.startswith("909") for c in m.classes_)
    assert len(m.classes_) == 5
    assert "limpeza" not in m.named_steps["tfidf"].vocabulary_


def test_nunca_sugere_divisao(modelo):
    sug = tfidf.suggest(modelo, ["limpeza", "qualquer coisa", ""])
    assert not any(c.endswith("000000") for s in sug for c in s)


def test_treino_com_codigo_de_divisao_da_erro():
    texto, codigo = treino()
    codigo.iloc[0] = "45000000"
    with pytest.raises(ValueError, match="divisão"):
        tfidf.train(texto, codigo, n_min=1, min_df=1)


def test_mesmo_texto_duas_vezes_da_as_mesmas_sugestoes(modelo):
    sug = tfidf.suggest(modelo, ["limpeza de escolas", "limpeza de escolas"])
    assert sug[0] == sug[1]
    assert tfidf.suggest(modelo, ["limpeza de escolas"])[0] == sug[0]


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_dois_treinos_dao_o_mesmo_modelo(modelo, n_jobs):
    texto, codigo = treino()
    m = tfidf.train(texto, codigo, n_min=N, min_df=1, n_jobs=n_jobs)
    clf, ref = m.named_steps["clf"], modelo.named_steps["clf"]
    np.testing.assert_array_equal(clf.coef_, ref.coef_)
    textos = ["limpeza", "estrada e automóveis", "papel"]
    assert tfidf.suggest(m, textos).tolist() == tfidf.suggest(modelo, textos).tolist()


class Empatado:
    """Modelo falso: todas as etiquetas com a mesma pontuação, exceto a do índice `melhor`."""

    def __init__(self, classes, melhor=None):
        self.classes_ = np.array(classes)
        self.melhor = melhor

    def decision_function(self, textos):
        scores = np.zeros((len(textos), len(self.classes_)))
        if self.melhor is not None:
            scores[:, self.melhor] = 1.0
        return scores


def test_empate_fica_o_menor_codigo():
    classes = [f"{i}1111111" for i in range(1, 8)]  # por ordem crescente, como o classes_
    sug = tfidf.suggest(Empatado(classes, melhor=6), ["a", "b"])
    assert sug[0] == ["71111111", "11111111", "21111111", "31111111", "41111111"]
    assert sug[1] == sug[0]


def test_o_indice_mantem_se(modelo):
    textos = pd.Series(["limpeza", "estrada", "papel"], index=[7, 3, 42])
    sug = tfidf.suggest(modelo, textos)
    assert sug.index.equals(textos.index)
    assert sug.dtype == object
    assert sug.loc[3] == tfidf.suggest(modelo, ["estrada"])[0]


def test_lista_da_indice_0_1_2(modelo):
    sug = tfidf.suggest(modelo, ["limpeza", "estrada"])
    assert sug.index.equals(pd.RangeIndex(2))


def test_lotes_dao_o_mesmo_que_um_lote(modelo):
    textos = pd.Series(["limpeza", "estrada", "papel", "medicamentos", "automóveis"], index=list("abcde"))
    um = tfidf.suggest(modelo, textos)
    lotes = tfidf.suggest(modelo, textos, batch_size=2)
    assert lotes.index.equals(um.index)
    assert lotes.tolist() == um.tolist()


def test_menos_de_5_etiquetas_no_treino_da_erro():
    temas = dict(list(TEMAS.items())[:4])
    texto, codigo = treino(temas)
    with pytest.raises(ValueError, match="só tem 4 etiquetas"):
        tfidf.train(texto, codigo, n_min=1, min_df=1)


def test_etiquetas_que_saem_nao_contam_para_as_5():
    # 7 códigos distintos, mas com N = 4 todos saem: não sobra nenhuma etiqueta
    texto, codigo = treino()
    with pytest.raises(ValueError, match="só tem 0 etiquetas"):
        tfidf.train(texto, codigo, n_min=4, min_df=1)


def test_modelo_com_menos_de_k_etiquetas_da_erro():
    with pytest.raises(ValueError, match="só tem 4 etiquetas"):
        tfidf.suggest(Empatado([f"{i}1111111" for i in range(1, 5)]), ["a"])


def test_indices_diferentes_da_erro():
    texto, codigo = treino()
    with pytest.raises(ValueError, match="índices diferentes"):
        tfidf.train(texto, codigo.reset_index(drop=True), n_min=1, min_df=1)
