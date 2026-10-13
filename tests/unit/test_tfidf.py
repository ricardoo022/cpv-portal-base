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


@pytest.mark.parametrize("n_min", [N, 3])
def test_as_sugestoes_sao_sempre_etiquetas_de_treino(modelo, n_min):
    # E4-T3: pedindo tantas sugestões quantas as etiquetas do modelo, aparecem todas; o conjunto
    # tem de ser o das etiquetas de treino (depois da regra dos códigos raros). O τ só muda a
    # ordem, não o conjunto, por isso não entra aqui.
    # Compara-se com as etiquetas calculadas aqui, não com o `classes_`, que viria do mesmo erro.
    # Com N = 3 há exemplos que saem (o grupo 909): também não podem aparecer nas sugestões.
    texto, codigo = treino()
    m = modelo if n_min == N else tfidf.train(texto, codigo, n_min=n_min, min_df=1)
    permitidas = set(lb.collapse_rare_codes(codigo, n_min).dropna())
    sug = tfidf.suggest(m, pd.concat([texto, pd.Series(["xyz", ""])]), k=len(m.classes_))
    assert all(set(s) == permitidas for s in sug)


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
    sug = tfidf.suggest(Empatado(classes, melhor=6), ["a", "b"], tau=0.0)  # só a pontuação
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


# --- Spike E4-S4: correção das etiquetas frequentes (Menon et al., 2021) ---

class _Fixo:
    """Modelo falso com pontuações fixas, para controlar a ordenação."""

    def __init__(self, scores, classes):
        self.scores = np.asarray(scores, dtype=float)
        self.classes_ = np.asarray(classes)

    def decision_function(self, textos):
        return np.tile(self.scores, (len(textos), 1))


CLASSES = ["11000000", "22100000", "33600000", "44100000", "55100000", "66100000"]


def test_log_prior_conta_etiquetas_do_treino_pela_ordem_das_classes():
    codigo = pd.Series(["33600000"] * 6 + ["45233200"] * 2 + ["90911200"] * 2)
    prior = tfidf.log_prior(codigo, np.array(["33600000", "45233200", "90911200"]), n_min=1)
    np.testing.assert_allclose(prior, np.log([0.6, 0.2, 0.2]))


def test_log_prior_usa_as_etiquetas_depois_dos_codigos_raros():
    # Com N = 2, os dois códigos de limpeza sobem para a classe 90910000 (2 exemplos de 10)
    codigo = pd.Series(["33600000"] * 8 + ["90911200", "90919300"])
    prior = tfidf.log_prior(codigo, np.array(["33600000", "90910000"]), n_min=2)
    np.testing.assert_allclose(prior, np.log([0.8, 0.2]))


def test_log_prior_com_etiqueta_que_nao_esta_no_treino_da_erro():
    with pytest.raises(ValueError, match="não estão no treino"):
        tfidf.log_prior(pd.Series(["33600000"] * 3), np.array(["33600000", "45233200"]), n_min=1)


def test_tau_zero_ignora_as_frequencias_e_ordena_so_pela_pontuacao(modelo):
    textos = ["limpeza de escolas", "estrada", "papel"]
    prior = np.log(np.linspace(0.01, 0.9, len(modelo.classes_)))  # frequências bem diferentes
    so_pontuacao = tfidf.top_k(modelo.decision_function(textos), np.asarray(modelo.classes_))
    assert tfidf.suggest(modelo, textos, tau=0.0, prior=prior).tolist() == so_pontuacao


def test_tau_tira_vantagem_a_etiqueta_frequente():
    # 33600000 tem a maior pontuação, mas é muito frequente; com τ = 1 desce para o fim
    scores = [1.0, 0.9, 1.5, 0.8, 0.7, 0.6]
    prior = np.log([0.1, 0.1, 0.5, 0.1, 0.1, 0.1])
    m = _Fixo(scores, CLASSES)
    assert tfidf.suggest(m, ["x"], tau=0.0)[0][0] == "33600000"
    com = tfidf.suggest(m, ["x"], tau=1.0, prior=prior)[0]
    # pontuações corrigidas: 1.0+2.30, 0.9+2.30, 1.5+0.69, 0.8+2.30, 0.7+2.30, 0.6+2.30
    assert com == ["11000000", "22100000", "44100000", "55100000", "66100000"]


def test_suggest_many_da_o_mesmo_que_suggest_para_cada_tau(modelo):
    textos = pd.Series(["limpeza de escolas", "medicamentos", "automóveis"], index=[7, 3, 5])
    prior = np.log(np.linspace(0.05, 0.3, len(modelo.classes_)))
    varios = tfidf.suggest_many(modelo, textos, (0.0, 0.5, 2.0), prior, batch_size=2)
    for tau, s in varios.items():
        assert s.index.tolist() == [7, 3, 5]
        assert s.tolist() == tfidf.suggest(modelo, textos, tau=tau, prior=prior).tolist()


def test_tau_sem_prior_num_modelo_sem_frequencias_da_erro():
    with pytest.raises(ValueError, match="prior"):
        tfidf.suggest(_Fixo([1, 2, 3, 4, 5, 6], CLASSES), ["estrada"], tau=0.5)


def test_train_guarda_as_frequencias_do_treino_no_modelo(modelo):
    texto, codigo = treino()
    np.testing.assert_allclose(modelo.log_prior_, tfidf.log_prior(codigo, modelo.classes_, n_min=N))


def test_sem_prior_usa_as_frequencias_do_modelo(modelo):
    textos = ["limpeza de escolas", "estrada", "papel"]
    assert (tfidf.suggest(modelo, textos, tau=1.0).tolist()
            == tfidf.suggest(modelo, textos, tau=1.0, prior=modelo.log_prior_).tolist())


def test_train_usa_o_alpha_escolhido_por_defeito(modelo):
    assert tfidf.ALPHA == 1e-7 and tfidf.TAU == 0.25
    assert modelo.named_steps["clf"].alpha == tfidf.ALPHA
    texto, codigo = treino()
    assert tfidf.train(texto, codigo, n_min=N, min_df=1, alpha=1e-3).named_steps["clf"].alpha == 1e-3


def test_train_usa_as_epocas_escolhidas_por_defeito(modelo):
    # E4-S5: 24 épocas fixas para todas as etiquetas, sem paragem automática
    assert tfidf.EPOCAS == 24
    clf = modelo.named_steps["clf"]
    assert clf.n_iter_ == tfidf.EPOCAS and clf.max_iter == tfidf.EPOCAS and clf.tol is None


def test_train_sem_epocas_volta_a_regra_de_paragem_do_sklearn():
    # epocas=None: a regra de antes do E4-S5, que os spikes que mudam o N ou a janela usam
    texto, codigo = treino()
    clf = tfidf.train(texto, codigo, n_min=N, min_df=1, epocas=None).named_steps["clf"]
    assert clf.tol == 1e-3 and clf.max_iter == 1000
    assert clf.n_iter_ < tfidf.EPOCAS  # pré-condição: aqui a regra para antes das 24 épocas


@pytest.mark.parametrize("extra", [{"max_iter": 5}, {"tol": None}, {"max_iter": 5, "tol": None}])
def test_train_com_epocas_e_max_iter_ou_tol_da_erro(extra):
    # Dois sítios a dizer quantas épocas: um deles seria ignorado em silêncio
    texto, codigo = treino()
    with pytest.raises(ValueError, match="epocas"):
        tfidf.train(texto, codigo, n_min=N, min_df=1, **extra)
    with pytest.raises(ValueError, match="epocas"):
        tfidf.train(texto, codigo, n_min=N, min_df=1, epocas=3, **extra)


def test_train_sem_epocas_aceita_max_iter_e_tol():
    texto, codigo = treino()
    clf = tfidf.train(texto, codigo, n_min=N, min_df=1, epocas=None, max_iter=2, tol=None).named_steps["clf"]
    assert clf.n_iter_ == 2


def test_prior_com_tamanho_errado_da_erro(modelo):
    with pytest.raises(ValueError, match="uma entrada por etiqueta"):
        tfidf.suggest(modelo, ["estrada"], tau=0.5, prior=np.zeros(2))


def test_suggest_usa_o_tau_escolhido_por_defeito():
    # 33600000 tem a maior pontuação mas é muito frequente; com τ = 0,25 sai das 5 sugestões:
    # pontuação − 0,25·log(freq): 1,0+0,98 / 0,9+0,98 / 1,5+0,03 / 0,8+0,98 / 0,7+0,98 / 0,6+0,98
    m = _Fixo([1.0, 0.9, 1.5, 0.8, 0.7, 0.6], CLASSES)
    m.log_prior_ = np.log([0.02, 0.02, 0.9, 0.02, 0.02, 0.02])
    assert tfidf.suggest(m, ["x"], tau=0.0)[0][0] == "33600000"  # pré-condição: sem correção ganha
    assert tfidf.suggest(m, ["x"])[0] == ["11000000", "22100000", "44100000", "55100000", "66100000"]


@pytest.mark.parametrize("taus", [(-0.25,), (0.0, -1.0)])
def test_tau_negativo_da_erro(modelo, taus):
    with pytest.raises(ValueError, match=">= 0"):
        tfidf.suggest_many(modelo, ["estrada"], taus)


# --- E4-T3: o modelo decora um treino minúsculo ("overfit a tiny dataset") ---

def test_treino_minusculo_sem_codigos_raros_acerta_todos_os_textos_no_top_1(temas_decorar):
    # Com regularização fraca e N = 1, o modelo tem de acertar no top-1 de cada texto com que
    # treinou. Se não acertar, há um erro no treino (etiquetas trocadas, exemplos perdidos,
    # colunas fora de ordem...), mesmo que o treino corra sem dar erro.
    texto, codigo = treino(temas_decorar)
    # pré-condições: N = 1 não muda nada; N = 2 mudava (o tema com 1 texto saía)
    assert lb.collapse_rare_codes(codigo, n_min=1).equals(codigo)
    assert lb.collapse_rare_codes(codigo, n_min=2).isna().any()
    m = tfidf.train(texto, codigo, n_min=1, min_df=1, alpha=tfidf.ALPHA)
    # τ = 0: ordem só pela pontuação do modelo. Com τ > 0 a correção das etiquetas frequentes
    # mexia na ordem, e o teste deixava de medir só o que o classificador aprendeu.
    top1 = tfidf.suggest(m, texto, tau=0.0).str[0]
    errados = top1[top1 != codigo]
    assert errados.empty, f"top-1 errado em {len(errados)} de {len(codigo)} textos: {errados.to_dict()}"
