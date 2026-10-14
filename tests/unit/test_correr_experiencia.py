"""Testes unitários a scripts/correr_experiencia.py (spikes E4-S5, E4-S1, E4-S3 e E4-S2): ler o YAML e montar os comandos.

Os primeiros testes escrevem à mão as opções dos `experiencias/e4_s5.yaml` (modo `fases`),
`experiencias/e4_s1.yaml` (modo `grelha`), `experiencias/e4_s3.yaml` (modo `entidade`) e
`experiencias/e4_s2.yaml` (modo `janela`), as do backlog: se alguém mudar uma lista depois de ver resultados, o teste falha e a mudança tem de ficar
explícita.
"""

import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "scripts"))
import correr_experiencia as ce  # noqa: E402

E4_S5 = RAIZ / "experiencias" / "e4_s5.yaml"
E4_S1 = RAIZ / "experiencias" / "e4_s1.yaml"
E4_S3 = RAIZ / "experiencias" / "e4_s3.yaml"
E4_S2 = RAIZ / "experiencias" / "e4_s2.yaml"

BASE = """spike: X
n_min: 10
saida: metrics/x
alpha: {em_vigor: 1.0e-7, candidatos: [1.0e-8]}
tau: {em_vigor: 0.0, candidatos: [0.25, 0.5, 1.0]}
epocas: {em_vigor: null, candidatos: [3, 6, 12, 18, 24]}
curva_epocas: 24
"""


def escrever(tmp_path, texto):
    path = tmp_path / "exp.yaml"
    path.write_text(texto, encoding="utf-8")
    return path


def test_o_yaml_do_e4_s5_tem_as_opcoes_do_backlog():
    exp = ce.ler(E4_S5)
    assert exp["n_min"] == 10
    assert exp["alpha"] == {"em_vigor": 1e-7, "candidatos": [1e-8]}
    assert exp["tau"] == {"em_vigor": 0.0, "candidatos": [0.25, 0.5, 1.0]}
    assert exp["epocas"] == {"em_vigor": None, "candidatos": [3, 6, 12, 18, 24]}
    assert exp["curva_epocas"] == 24
    assert exp["saida"] == "metrics/e4_s5"


def test_alpha_escrito_sem_ponto_e_texto_no_yaml_e_da_erro(tmp_path):
    # O PyYAML lê "1e-8" como a string '1e-8'; sem a verificação, o treino recebia texto.
    path = escrever(tmp_path, BASE.replace("[1.0e-8]", "[1e-8]"))
    with pytest.raises(ValueError, match="1.0e-7"):
        ce.ler(path)


@pytest.mark.parametrize("velho, novo", [
    ("[3, 6, 12, 18, 24]", "[3, 6, 6]"),          # épocas repetidas
    ("[0.25, 0.5, 1.0]", "[0.0, 0.5]"),           # τ candidato igual ao em vigor
    ("[3, 6, 12, 18, 24]", "[0, 6]"),             # 0 épocas
    ("[3, 6, 12, 18, 24]", "[2.5]"),              # épocas não inteiras
    ("[1.0e-8]", "[-1.0e-8]"),                    # alpha negativo
    ("curva_epocas: 24", "curva_epocas: 24\nextra: 1"),  # chave a mais
    ("curva_epocas: 24", ""),                     # chave em falta
    ("n_min: 10", "n_min: 0"),                    # n_min inválido
    ("n_min: 10", "n_min: -5"),
])
def test_yaml_invalido_da_erro(tmp_path, velho, novo):
    with pytest.raises(ValueError):
        ce.ler(escrever(tmp_path, BASE.replace(velho, novo)))


@pytest.mark.parametrize("invalido", [0, -5, "dez", True])
def test_n_min_invalido_da_erro_com_mensagem_clara(tmp_path, invalido):
    path = escrever(tmp_path, BASE.replace("n_min: 10", f"n_min: {invalido}"))
    with pytest.raises(ValueError, match="n_min.*não é um inteiro >= 1"):
        ce.ler(path)



def test_fase_alpha_um_treino_por_alpha_com_todos_os_taus_e_a_regra_por_defeito():
    exp = ce.ler(E4_S5)
    lista = ce.comandos(exp, "alpha", n_jobs=16)
    assert [n for n, _, _ in lista] == ["alpha_1e-07", "alpha_1e-08"]
    for nome, args, _ in lista:
        assert args[1] == "src/train_tfidf.py"
        i = args.index("--taus")
        assert args[i + 1:i + 5] == ["0", "0.25", "0.5", "1"]
        # epocas.em_vigor: null -> regra por defeito do sklearn, dita explicitamente; sem
        # --regra-por-defeito, o train_tfidf.py treinava as 24 épocas do E4-S5 em silêncio
        assert "--regra-por-defeito" in args
        assert "--epocas" not in args
        assert args[args.index("--n-jobs") + 1] == "16"
        assert args[args.index("--n-min") + 1] == "10"
        assert args[args.index("--saida") + 1] == f"metrics/e4_s5/{nome}"
        assert "--sem-modelo" in args
    assert lista[1][1][lista[1][1].index("--alpha") + 1] == "1e-08"


def test_fase_epocas_um_treino_por_numero_de_epocas_e_a_curva_mais_demorados_primeiro():
    exp = ce.ler(E4_S5)
    lista = ce.comandos(exp, "epocas", alpha=1e-7, tau=0.25)
    assert [n for n, _, _ in lista] == ["curva", "epocas_24", "epocas_18", "epocas_12", "epocas_06", "epocas_03"]
    for nome, args, _ in lista[1:]:
        assert args[args.index("--epocas") + 1] == str(int(nome[-2:]))
        assert "--regra-por-defeito" not in args
        assert args[args.index("--alpha") + 1] == "1e-07"
        i = args.index("--taus")
        assert args[i + 1:i + 3] == ["0", "0.25"]
        assert args[args.index("--saida") + 1] == f"metrics/e4_s5/{nome}"
    curva = lista[0][1]
    assert curva[1] == "scripts/curva_epocas.py"
    assert curva[curva.index("--epocas") + 1] == "24"
    assert curva[curva.index("--tau") + 1] == "0.25"


def test_fase_alpha_com_epocas_em_vigor_fixas_passa_essas_epocas():
    exp = ce.ler(E4_S5)
    exp["epocas"]["em_vigor"] = 24
    for _, args, previstas in ce.comandos(exp, "alpha"):
        assert args[args.index("--epocas") + 1] == "24" and previstas == 24
        assert "--regra-por-defeito" not in args


def test_fase_epocas_sem_alpha_ou_tau_da_erro():
    exp = ce.ler(E4_S5)
    with pytest.raises(ValueError, match="alpha"):
        ce.comandos(exp, "epocas", alpha=1e-7)


def test_fase_epocas_com_tau_zero_so_grava_o_ficheiro_sem_correcao():
    exp = ce.ler(E4_S5)
    args = ce.comandos(exp, "epocas", alpha=1e-7, tau=0.0)[1][1]
    i = args.index("--taus")
    assert args[i + 1] == "0" and args[i + 2].startswith("--")


@pytest.mark.parametrize("fase, extra", [("alpha", {}), ("epocas", {"alpha": 1e-7, "tau": 0.25})])
def test_treino_e_validacao_passam_a_todos_os_comandos(fase, extra):
    # Sem isto, cada script usava os ficheiros de data/processed/ em vez dos pedidos.
    exp = ce.ler(E4_S5)
    for _, args, _ in ce.comandos(exp, fase, treino="t.parquet", validacao="v.parquet", **extra):
        assert args[args.index("--treino") + 1] == "t.parquet"
        assert args[args.index("--validacao") + 1] == "v.parquet"
    for _, args, _ in ce.comandos(exp, fase, **extra):
        assert "--treino" not in args and "--validacao" not in args


# --- Grelha (spike E4-S1): o produto N × épocas, com o α fixo e um ficheiro de sugestões por τ ---

BASE_GRELHA = """spike: X
modo: grelha
saida: metrics/x
alpha: 1.0e-7
n_min: [5, 10, 20]
epocas: [null, 3, 6, 12, 18, 24]
tau: [0.0, 0.25, 0.5]
"""


def test_o_yaml_do_e4_s1_tem_as_opcoes_do_backlog():
    # As opções do E4-S1, escritas à mão a partir do backlog: se alguém as mudar depois de ver
    # resultados, este teste falha e a mudança fica explícita.
    exp = ce.ler(E4_S1)
    assert exp["modo"] == "grelha"
    assert exp["alpha"] == 1e-7
    assert exp["n_min"] == [5, 10, 20]
    assert exp["epocas"] == [None, 3, 6, 12, 18, 24]
    assert exp["tau"] == [0.0, 0.25, 0.5]
    assert exp["saida"] == "metrics/e4_s1"


def test_fase_grelha_um_treino_por_n_e_epocas_com_os_taus():
    exp = ce.ler(E4_S1)
    lista = ce.comandos(exp, "grelha", n_jobs=8)
    assert len(lista) == len(exp["n_min"]) * len(exp["epocas"])  # 18 treinos
    nomes = [n for n, _, _ in lista]
    # Da mais demorada à menos; a regra por defeito conta como 12 épocas.
    assert nomes[:3] == ["n_min_05_ep_24", "n_min_10_ep_24", "n_min_20_ep_24"]
    for nome, args, _ in lista:
        assert args[1] == "src/train_tfidf.py"
        assert args[args.index("--alpha") + 1] == "1e-07"
        i = args.index("--taus")
        assert args[i + 1:i + 4] == ["0", "0.25", "0.5"]
        assert args[args.index("--n-jobs") + 1] == "8"
        assert "--sem-modelo" in args
        assert args[args.index("--saida") + 1] == f"metrics/e4_s1/{nome}"


def test_fase_grelha_usa_a_regra_por_defeito_quando_as_epocas_sao_null():
    exp = ce.ler(E4_S1)
    args = {n: a for n, a, _ in ce.comandos(exp, "grelha")}["n_min_10_ep_defeito"]
    assert "--regra-por-defeito" in args and "--epocas" not in args


def test_fase_grelha_tambem_passa_treino_e_validacao():
    exp = ce.ler(E4_S1)
    for _, args, _ in ce.comandos(exp, "grelha", treino="t.parquet", validacao="v.parquet"):
        assert args[args.index("--treino") + 1] == "t.parquet"
        assert args[args.index("--validacao") + 1] == "v.parquet"
    for _, args, _ in ce.comandos(exp, "grelha"):
        assert "--treino" not in args and "--validacao" not in args


def test_uma_grelha_so_corre_a_fase_grelha():
    with pytest.raises(ValueError, match="grelha"):
        ce.comandos(ce.ler(E4_S1), "alpha")


def test_uma_experiencia_em_fases_recusa_a_fase_grelha():
    with pytest.raises(ValueError, match="grelha"):
        ce.comandos(ce.ler(E4_S5), "grelha")


@pytest.mark.parametrize("velho, novo", [
    ("n_min: [5, 10, 20]", "n_min: [5, 5, 20]"),               # N repetido
    ("n_min: [5, 10, 20]", "n_min: []"),                       # lista vazia
    ("n_min: [5, 10, 20]", "n_min: 10"),                       # escalar onde se espera lista
    ("tau: [0.0, 0.25, 0.5]", "tau: [-0.25, 0.25]"),           # τ negativo
    ("alpha: 1.0e-7", "alpha: [1.0e-7]"),                      # alpha tem de ser número, não lista
    ("epocas: [null, 3, 6, 12, 18, 24]", "epocas: [0, 3]"),    # 0 épocas
    ("tau: [0.0, 0.25, 0.5]", "tau: [0.0, 0.25, 0.5]\ncurva_epocas: 24"),  # chave a mais numa grelha
])
def test_yaml_de_grelha_invalido_da_erro(tmp_path, velho, novo):
    with pytest.raises(ValueError):
        ce.ler(escrever(tmp_path, BASE_GRELHA.replace(velho, novo)))


# --- Spike E4-S3: a experiência da entidade (modo `entidade`) ---

BASE_ENTIDADE = """spike: X
saida: metrics/x
modo: entidade
n_min: 10
alpha: 1.0e-7
epocas: 24
tau: [0.25]
com_entidade: {sem_entidade: a, com_entidade: b}
grupos: [todos, entidades_no_treino, entidades_novas]
"""


def test_o_yaml_do_e4_s3_tem_as_opcoes_do_backlog():
    # As opções do E4-S3, escritas à mão a partir do backlog e do YAML pré-registado: se alguém as
    # mudar depois de ver resultados, este teste falha e a mudança tem de ficar explícita.
    exp = ce.ler(E4_S3)
    assert exp["modo"] == "entidade"
    assert exp["n_min"] == 10
    assert exp["alpha"] == 1e-7
    assert exp["epocas"] == 24
    assert exp["tau"] == [0.25]
    assert exp["saida"] == "metrics/e4_s3"
    assert exp["com_entidade"] == {"sem_entidade": "metrics/e4_s1/n_min_10_ep_24",
                                   "com_entidade": "metrics/e4_s3/com_entidade"}
    assert exp["grupos"] == ["todos", "entidades_no_treino", "entidades_novas"]


def test_a_experiencia_da_entidade_nao_corre_pelo_script():
    # É um só comando (src/train_tfidf.py --com-entidade): o script de experiências não a corre
    with pytest.raises(ValueError, match="com-entidade"):
        ce.comandos(ce.ler(E4_S3), "alpha")


@pytest.mark.parametrize("velho, novo", [
    ("tau: [0.25]", "tau: [-0.25]"),                          # τ negativo
    ("tau: [0.25]", "tau: [0.25, 0.25]"),                     # τ repetido
    ("tau: [0.25]", "tau: []"),                               # lista vazia
    ("alpha: 1.0e-7", "alpha: 1e-7"),                         # texto em vez de número
    ("alpha: 1.0e-7", "alpha: 0.0"),                          # alpha tem de ser > 0
    ("epocas: 24", "epocas: 0"),                              # 0 épocas
    ("epocas: 24", "epocas: 2.5"),                            # épocas não inteiras
    ("n_min: 10", "n_min: 0"),                                # n_min inválido
    ("com_entidade: {sem_entidade: a, com_entidade: b}", "com_entidade: {sem_entidade: a}"),  # só uma opção
    ("grupos: [todos, entidades_no_treino, entidades_novas]", "grupos: []"),                  # sem grupos
    ("grupos: [todos, entidades_no_treino, entidades_novas]", "grupos: [todos, todos]"),      # repetidos
    ("grupos: [todos, entidades_no_treino, entidades_novas]",
     "grupos: [todos, entidades_no_treino, entidades_novas]\nextra: 1"),                      # chave a mais
])
def test_yaml_da_entidade_invalido_da_erro(tmp_path, velho, novo):
    with pytest.raises(ValueError):
        ce.ler(escrever(tmp_path, BASE_ENTIDADE.replace(velho, novo)))


def test_yaml_da_entidade_sem_modo_continua_a_ser_fases(tmp_path):
    # Sem `modo`, o leitor assume `fases` (comportamento antigo): este YAML passa a faltar chaves
    with pytest.raises(ValueError, match="curva_epocas"):
        ce.ler(escrever(tmp_path, BASE_ENTIDADE.replace("modo: entidade\n", "")))


# --- Spike E4-S2: a janela de treino (modo `janela`) ---

BASE_JANELA = """spike: X
saida: metrics/x
modo: janela
n_min: 10
alpha: 1.0e-7
epocas: null
tau: [0.25]
janelas:
  - {nome: curta, inicio: "2025-01-01", pasta: a}
  - {nome: media, inicio: "2022-01-01", pasta: b}
grupo: g
"""


def test_o_yaml_do_e4_s2_tem_as_opcoes_do_backlog():
    # As opções do E4-S2, escritas à mão a partir do backlog e do YAML pré-registado: se alguém as
    # mudar depois de ver resultados, este teste falha e a mudança tem de ficar explícita.
    exp = ce.ler(E4_S2)
    assert exp["modo"] == "janela"
    assert exp["n_min"] == 10
    assert exp["alpha"] == 1e-7
    assert exp["epocas"] is None
    assert exp["tau"] == [0.25]
    assert exp["saida"] == "metrics/e4_s2"
    # Por esta ordem: a 1.ª (a mais curta) é a opção em vigor de partida e fica em caso de empate
    assert exp["janelas"] == [
        {"nome": "desde_2025", "inicio": "2025-01-01", "pasta": "metrics/e4_s2/desde_2025"},
        {"nome": "desde_2022", "inicio": "2022-01-01", "pasta": "metrics/e4_s1/n_min_10_ep_defeito"},
        {"nome": "desde_2019", "inicio": "2019-01-01", "pasta": "metrics/e4_s2/desde_2019"},
    ]
    assert exp["grupo"] == "texto_novo_na_janela_mais_longa"


def test_a_experiencia_das_janelas_nao_corre_pelo_script():
    # Cada janela prepara-se e treina-se à mão no pod: o script de experiências não a corre
    with pytest.raises(ValueError, match="janelas"):
        ce.comandos(ce.ler(E4_S2), "alpha")


def test_a_base_das_janelas_e_valida(tmp_path):
    # Sem isto, os casos inválidos abaixo podiam falhar por outra razão qualquer
    exp = ce.ler(escrever(tmp_path, BASE_JANELA))
    assert [j["nome"] for j in exp["janelas"]] == ["curta", "media"]


def test_data_da_janela_sem_aspas_da_erro_a_pedir_aspas(tmp_path):
    # Sem aspas, o PyYAML lê 2025-01-01 como datetime.date, não como texto
    path = escrever(tmp_path, BASE_JANELA.replace('"2025-01-01"', "2025-01-01"))
    with pytest.raises(ValueError, match="aspas"):
        ce.ler(path)


@pytest.mark.parametrize("velho, novo", [
    ('"2025-01-01"', "2025-01-01"),                                         # data sem aspas
    ('"2025-01-01"', '"2019-01-01"'),                                       # ordem crescente
    ('"2025-01-01"', '"2022-01-01"'),                                       # datas repetidas
    ('"2025-01-01"', '"2025-1-1"'),                                         # formato errado
    ('"2025-01-01"', '"2025-02-30"'),                                       # data que não existe
    ("nome: media", "nome: curta"),                                         # janela repetida
    ("pasta: b}", "pasta: b, extra: 1}"),                                   # chave a mais numa janela
    ("pasta: b}", "}"),                                                     # chave em falta numa janela
    ("pasta: a}", 'pasta: ""}'),                                            # pasta vazia
    ("alpha: 1.0e-7", "alpha: -1.0e-7"),                                    # alpha negativo
    ("tau: [0.25]", "tau: []"),                                             # τ vazio
    ('  - {nome: media, inicio: "2022-01-01", pasta: b}\n', ""),            # uma só janela
    ("grupo: g", 'grupo: ""'),                                              # grupo vazio
    ("epocas: null", "epocas: 0"),                                          # 0 épocas
    ("grupo: g", "grupo: g\nextra: 1"),                                     # chave a mais
    ("grupo: g", "grupo: g\ninformacao_epocas: 24"),                        # chave que saiu do YAML
])
def test_yaml_das_janelas_invalido_da_erro(tmp_path, velho, novo):
    assert velho in BASE_JANELA
    with pytest.raises(ValueError):
        ce.ler(escrever(tmp_path, BASE_JANELA.replace(velho, novo, 1)))


E4_S6 = RAIZ / "experiencias" / "e4_s6.yaml"


def test_o_yaml_do_e4_s6_tem_as_opcoes_do_backlog():
    exp = ce.ler(E4_S6)
    assert exp["modo"] == "janela"
    assert exp["n_min"] == 10
    assert exp["alpha"] == 1e-7
    assert exp["epocas"] == 24
    assert exp["tau"] == [0.25]
    assert exp["janelas"] == [
        {"nome": "desde_2022", "inicio": "2022-01-01", "pasta": "metrics/e4_s1/n_min_10_ep_24"},
        {"nome": "desde_2019", "inicio": "2019-01-01", "pasta": "metrics/e4_s6/desde_2019"},
    ]
    assert exp["grupo"] == "texto_novo_na_janela_mais_longa"
    assert exp["saida"] == "metrics/e4_s6"
