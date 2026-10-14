"""Testes unitários aos passos de src/prepare.py, com dados pequenos inventados."""

import pandas as pd
import pytest

import prepare as pr


def contratos(rows):
    """Contratos no formato que sai de load_one: (idcontrato, data, texto, codigos[, ficheiro])."""
    df = pd.DataFrame({
        "idcontrato": [r[0] for r in rows],
        "data": pd.to_datetime([r[1] for r in rows]),
        "ficheiro": [r[4] if len(r) > 4 else 2025 for r in rows],
        "texto": [r[2] for r in rows],
        "codigo": [r[3].split()[0] for r in rows],
        "codigos": [r[3] for r in rows],
        "adjudicante": "E", "tipoContrato": "T", "tipoprocedimento": "P",
    })
    return pr.add_columns(df)


# --- build_text (decisão 5) ---

def raw(obj, desc, aq=None):
    return pd.DataFrame({"objectoContrato": [obj], "descContrato": [desc], "DescrAcordoQuadro": [aq]})


def test_build_text_descricao_igual_ao_objeto_nao_se_repete():
    assert pr.build_text(raw("Limpeza de escolas", "  LIMPEZA de   escolas")).iloc[0] == "Limpeza de escolas"


def test_build_text_descricao_diferente_junta_se():
    assert pr.build_text(raw("Limpeza", "Limpeza de 3 escolas")).iloc[0] == "Limpeza | Limpeza de 3 escolas"


def test_build_text_junta_descricao_do_acordo_quadro():
    texto = pr.build_text(raw("Convite ao abrigo do AQ", "Convite ao abrigo do AQ", "AQ de medicamentos")).iloc[0]
    assert texto == "Convite ao abrigo do AQ | AQ de medicamentos"


def test_build_text_colunas_vazias_nao_escrevem_none():
    texto = pr.build_text(raw("Limpeza", None, None)).iloc[0]
    assert texto == "Limpeza"


def test_build_text_normaliza_espacos_mas_mantem_maiusculas():
    assert pr.build_text(raw("  Limpeza\n de  Escolas ", None)).iloc[0] == "Limpeza de Escolas"


# --- text_key ---

def test_text_key_textos_iguais_ignorando_maiusculas_e_espacos():
    k = pr.text_key(pd.Series(["Medicamentos", "  MEDICAMENTOS ", "Medicamentos vários"]))
    assert k[0] == k[1]
    assert k[0] != k[2]


# --- dedupe_ids (decisão 3) ---

def test_dedupe_ids_fica_a_copia_do_ficheiro_mais_recente():
    df = contratos([
        (1, "2024-03-01", "antigo", "90611000", 2023),
        (1, "2024-03-01", "novo", "90611000", 2024),
        (2, "2024-03-01", "outro", "90611000", 2024),
    ])
    out = pr.dedupe_ids(df)
    assert sorted(out["idcontrato"]) == [1, 2]
    assert out.loc[out["idcontrato"] == 1, "texto"].item() == "novo"


def test_dedupe_ids_no_mesmo_ficheiro_fica_a_data_mais_recente():
    # A cópia mais recente vem primeiro, para a ordem das linhas não decidir sozinha
    df = contratos([
        (1, "2024-05-01", "nova", "90611000", 2024),
        (1, "2024-03-01", "antiga", "90611000", 2024),
    ])
    assert pr.dedupe_ids(df)["texto"].tolist() == ["nova"]


def test_dedupe_ids_o_ficheiro_conta_antes_da_data():
    # A cópia do ficheiro antigo tem a data mais recente e vem em último: fica a do ficheiro novo
    df = contratos([
        (1, "2024-03-01", "ficheiro novo", "90611000", 2024),
        (1, "2024-06-01", "ficheiro antigo", "90611000", 2023),
    ])
    assert pr.dedupe_ids(df)["texto"].tolist() == ["ficheiro novo"]


# --- build_train (decisões 8 e 6) ---

def test_build_train_tira_codigos_ao_nivel_da_divisao():
    df = contratos([
        (1, "2024-01-10", "medicamentos", "33000000"),
        (2, "2024-01-11", "limpeza", "90911200"),
    ])
    tr = pr.build_train(df, verbose=False)
    assert tr["codigo"].tolist() == ["90911200"]


def test_build_train_um_exemplo_por_texto_com_o_codigo_mais_comum():
    df = contratos([
        (1, "2024-01-10", "Medicamentos", "33600000"),
        (2, "2024-01-11", "medicamentos", "33600000"),
        (3, "2024-01-12", " MEDICAMENTOS", "33690000"),
    ])
    tr = pr.build_train(df, verbose=False)
    assert len(tr) == 1
    assert tr["codigo"].item() == "33600000"
    assert tr["n_contratos"].item() == 3  # conta todos os contratos do texto, também o minoritário


def test_build_train_empate_fica_o_menor_codigo():
    df = contratos([
        (1, "2024-01-10", "luvas", "33141420"),
        (2, "2024-01-11", "luvas", "18424300"),
    ])
    tr = pr.build_train(df, verbose=False)
    assert tr["codigo"].item() == "18424300"


def test_build_train_ignora_validacao_e_teste():
    df = contratos([
        (1, "2024-01-10", "limpeza", "90911200"),
        (2, "2025-10-05", "jardins", "77311000"),
        (3, "2025-11-05", "papel", "30197630"),
    ])
    tr = pr.build_train(df, verbose=False)
    assert tr["idcontrato"].tolist() == [1]


# --- build_eval (passo 9) ---

def test_build_eval_fica_com_todos_e_marca_textos_vistos():
    df = contratos([
        (1, "2024-01-10", "limpeza", "90911200"),
        (2, "2025-10-05", "Limpeza", "90911200"),   # texto igual ao do treino
        (3, "2025-10-06", "jardins", "77000000"),   # código ao nível da divisão: fica na validação
        (4, "2025-10-07", "jardins", "77311000"),   # texto repetido: fica na validação
    ])
    tr = pr.build_train(df, verbose=False)
    val = pr.build_eval(df, "validação", set(tr["chave"]), verbose=False)
    assert sorted(val["idcontrato"]) == [2, 3, 4]
    assert val.set_index("idcontrato")["visto_no_treino"].to_dict() == {2: True, 3: False, 4: False}


# --- check_sets (passo 10) ---

def conjuntos_validos():
    df = contratos([
        (1, "2024-01-10", "limpeza", "90911200"),
        (2, "2025-10-05", "jardins", "77311000"),
        (3, "2025-11-05", "papel", "30197630"),
    ])
    tr = pr.build_train(df, verbose=False)
    sets = {"treino": tr.drop(columns="chave")}
    for nome in ["validação", "teste"]:
        sets[nome] = pr.build_eval(df, nome, set(tr["chave"]), verbose=False)
    return sets


def test_check_sets_aceita_conjuntos_validos():
    pr.check_sets(conjuntos_validos())


def test_check_sets_apanha_contrato_em_dois_conjuntos():
    sets = conjuntos_validos()
    sets["teste"].loc[0, "idcontrato"] = sets["treino"].loc[0, "idcontrato"]
    with pytest.raises(pr.DataCheckError, match="mais do que um conjunto"):
        pr.check_sets(sets)


def test_check_sets_apanha_teste_fora_do_periodo():
    sets = conjuntos_validos()
    sets["teste"].loc[0, "data"] = pd.Timestamp("2025-10-15")
    with pytest.raises(pr.DataCheckError, match="teste fora do período"):
        pr.check_sets(sets)


def test_check_sets_apanha_codigo_ao_nivel_da_divisao_no_treino():
    sets = conjuntos_validos()
    sets["treino"].loc[0, "nivel"] = 2
    with pytest.raises(pr.DataCheckError, match="nível da divisão"):
        pr.check_sets(sets)


def test_check_sets_apanha_texto_repetido_no_treino():
    sets = conjuntos_validos()
    extra = sets["treino"].iloc[[0]].assign(idcontrato=99, texto="LIMPEZA")
    sets["treino"] = pd.concat([sets["treino"], extra], ignore_index=True)
    with pytest.raises(pr.DataCheckError, match="texto repetido"):
        pr.check_sets(sets)


@pytest.mark.parametrize("texto", ["jardins | None", "None", "jardins | nan | AQ", "jardins |  nan "])
def test_check_sets_apanha_none_escrito_no_texto(texto):
    sets = conjuntos_validos()
    sets["validação"].loc[0, "texto"] = texto
    with pytest.raises(pr.DataCheckError, match="'None' ou 'nan'"):
        pr.check_sets(sets)


@pytest.mark.parametrize("texto", ["águas domésticas nan Quinta dos Farias", "jardins | None of the above"])
def test_check_sets_aceita_none_ou_nan_dentro_de_uma_frase(texto):
    # Só um pedaço inteiro entre separadores é um campo vazio escrito como texto.
    sets = conjuntos_validos()
    sets["validação"].loc[0, "texto"] = texto
    pr.check_sets(sets)


def test_check_sets_apanha_codigo_mal_formado():
    sets = conjuntos_validos()
    sets["treino"].loc[0, "codigo"] = "9091120"
    with pytest.raises(pr.DataCheckError, match="mal formado"):
        pr.check_sets(sets)


# --- Início do treino indicado (E4-T4) ---

INICIO_2019 = pd.Timestamp("2019-01-01")


def escrever_raw(pasta, ano, linhas):
    """Um contratos<ano>.parquet como os de convert_to_parquet.py (todas as colunas em texto)."""
    n = len(linhas)
    pd.DataFrame({
        "idcontrato": [i for i, _ in linhas],
        "dataPublicacao": [d for _, d in linhas],
        "objectoContrato": [f"Contrato {i}" for i, _ in linhas],
        "descContrato": [None] * n,
        "DescrAcordoQuadro": [None] * n,
        "CPV": ["90911200-8 - Limpeza de edifícios"] * n,
        "adjudicante": ["500000000 - Entidade"] * n,
        "tipoContrato": ["Aquisição de serviços"] * n,
        "tipoprocedimento": ["Ajuste Direto Regime Geral"] * n,
    }).to_parquet(pasta / f"contratos{ano}.parquet", index=False)


def test_add_columns_com_inicio_anterior_poe_contratos_antigos_no_treino():
    df = contratos([
        (1, "2020-06-01", "limpeza", "90911200", 2020),
        (2, "2025-10-05", "jardins", "77311000"),
    ])
    assert pd.isna(df.set_index("idcontrato").loc[1, "periodo"])  # por defeito (2022) fica de fora
    com_inicio = pr.add_columns(df.drop(columns=["nivel", "periodo", "chave"]), train_start=INICIO_2019)
    assert com_inicio["periodo"].tolist() == ["treino", "validação"]


def test_build_train_com_inicio_anterior_marca_texto_antigo_como_visto():
    df = contratos([
        (1, "2020-06-01", "limpeza", "90911200", 2020),
        (2, "2024-01-10", "papel", "30197630", 2024),
        (3, "2025-10-05", "Limpeza", "90911200"),
    ])
    base = df.drop(columns=["nivel", "periodo", "chave"])
    for inicio, visto in [(None, False), (INICIO_2019, True)]:
        df = pr.add_columns(base, train_start=inicio)
        tr = pr.build_train(df, verbose=False)
        val = pr.build_eval(df, "validação", set(tr["chave"]), verbose=False)
        assert val["visto_no_treino"].tolist() == [visto]


def test_load_all_le_contratos_desde_o_inicio_indicado(tmp_path):
    escrever_raw(tmp_path, 2020, [("1", "2018-12-31"), ("2", "2019-01-01"), ("3", "2021-12-31")])
    escrever_raw(tmp_path, 2025, [("4", "2022-01-01"), ("5", "2025-10-05")])
    por_defeito, n_defeito = pr.load_all(tmp_path)
    assert sorted(por_defeito["idcontrato"]) == ["4", "5"]
    assert n_defeito == 2
    com_inicio, n_inicio = pr.load_all(tmp_path, train_start=INICIO_2019)
    assert sorted(com_inicio["idcontrato"]) == ["2", "3", "4", "5"]
    assert n_inicio == 4


def test_check_sets_usa_o_inicio_do_treino_indicado():
    # O treino de conjuntos_validos() tem um contrato de 2024-01-10: com início em junho de 2024
    # fica antes do início, e a verificação tem de o apanhar.
    sets = conjuntos_validos()
    pr.check_sets(sets, train_start=pd.Timestamp("2024-01-10"))
    with pytest.raises(pr.DataCheckError, match="treino fora do período"):
        pr.check_sets(sets, train_start=pd.Timestamp("2024-06-01"))


def test_check_sets_por_defeito_apanha_treino_antes_de_2022():
    sets = conjuntos_validos()
    sets["treino"].loc[0, "data"] = pd.Timestamp("2021-06-01")
    with pytest.raises(pr.DataCheckError, match="treino fora do período"):
        pr.check_sets(sets)
    with pytest.raises(pr.DataCheckError, match="treino fora do período"):
        pr.check_sets(sets, train_start=None)


def test_check_sets_aceita_treino_antigo_com_inicio_anterior():
    sets = conjuntos_validos()
    sets["treino"].loc[0, "data"] = pd.Timestamp("2021-06-01")
    pr.check_sets(sets, train_start=INICIO_2019)


@pytest.mark.parametrize("inicio", ["2025-10-01", pd.Timestamp("2025-10-01"), "2025-12-01", "2026-03-01"])
def test_run_inicio_do_treino_nao_anterior_a_validacao_da_erro(tmp_path, inicio):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    escrever_raw(raw_dir, 2025, [("1", "2024-01-10"), ("2", "2025-10-05"), ("3", "2025-11-05")])
    with pytest.raises(ValueError):
        pr.run(raw_dir=raw_dir, out_dir=tmp_path / "out", verbose=False, train_start=inicio)
    assert not list(tmp_path.glob("out/*.parquet"))


def test_linha_de_comando_outro_inicio_sem_saida_da_erro_sem_escrever(tmp_path, monkeypatch, capsys):
    # Corre src/prepare.py como `python src/prepare.py --inicio-treino 2019-01-01`, mas dentro do
    # pytest e com as pastas de cpv_data trocadas por pastas temporárias: mesmo que o argparse não
    # parasse, nunca leria nem reescreveria os dados reais de data/processed/.
    import runpy
    import sys

    import cpv_data as cd

    raiz = tmp_path / "raiz"
    raw_dir = raiz / "data" / "processed" / "raw"
    raw_dir.mkdir(parents=True)
    escrever_raw(raw_dir, 2025, [("1", "2024-01-10"), ("2", "2025-10-05"), ("3", "2025-11-05")])
    monkeypatch.setattr(cd, "ROOT", raiz)
    monkeypatch.setattr(cd, "RAW_PARQUET", raw_dir)
    monkeypatch.setattr(sys, "argv", ["prepare.py", "--inicio-treino", "2019-01-01"])
    with pytest.raises(SystemExit) as erro:
        runpy.run_path(pr.__file__, run_name="__main__")
    assert erro.value.code == 2
    assert "--saida" in capsys.readouterr().err
    assert list(raiz.rglob("*.parquet")) == [raw_dir / "contratos2025.parquet"]  # só o de entrada


def test_linha_de_comando_outro_inicio_com_saida_em_data_processed_da_erro(tmp_path, monkeypatch):
    import runpy
    import sys

    import cpv_data as cd

    raiz = tmp_path / "raiz"
    raw_dir = raiz / "data" / "processed" / "raw"
    raw_dir.mkdir(parents=True)
    escrever_raw(raw_dir, 2025, [("1", "2024-01-10"), ("2", "2025-10-05"), ("3", "2025-11-05")])
    monkeypatch.setattr(cd, "ROOT", raiz)
    monkeypatch.setattr(cd, "RAW_PARQUET", raw_dir)
    saida = str(raiz / "data" / "processed")
    monkeypatch.setattr(sys, "argv", ["prepare.py", "--inicio-treino", "2019-01-01", "--saida", saida])
    with pytest.raises(SystemExit) as erro:
        runpy.run_path(pr.__file__, run_name="__main__")
    assert erro.value.code == 2
    assert list(raiz.rglob("*.parquet")) == [raw_dir / "contratos2025.parquet"]


@pytest.mark.parametrize("conjunto, contrato", [("validação", "2"), ("teste", "3")])
def test_run_copia_antiga_num_ficheiro_mais_recente_nao_tira_contrato_da_avaliacao(tmp_path, conjunto, contrato):
    # O contrato tem uma cópia de out/nov 2025 (ficheiro 2025) e outra de 2020 num ficheiro mais
    # recente (2026). Com o início por defeito a de 2020 nem é lida; com 2019 ganharia (decisão 3)
    # e o contrato sairia da avaliação sem aviso. A preparação tem de parar.
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    escrever_raw(raw_dir, 2025, [("1", "2024-01-10"), ("2", "2025-10-05"), ("3", "2025-11-05")])
    escrever_raw(raw_dir, 2026, [(contrato, "2020-06-01")])
    sets = pr.run(raw_dir=raw_dir, out_dir=tmp_path / "defeito", verbose=False)
    assert sets[conjunto]["idcontrato"].tolist() == [contrato]
    with pytest.raises(pr.DataCheckError, match="cópia mais antiga"):
        pr.run(raw_dir=raw_dir, out_dir=tmp_path / "2019", verbose=False, train_start=INICIO_2019)
    assert not list(tmp_path.glob("2019/*.parquet"))


@pytest.mark.parametrize("argumentos", [[], ["--inicio-treino", "2022-01-01"]])
def test_linha_de_comando_com_inicio_por_defeito_grava_em_data_processed(tmp_path, monkeypatch, argumentos):
    # `make preparar` corre o comando sem argumentos: tem de continuar a gravar na pasta de sempre,
    # sem pedir --saida. Dizer a data por defeito explicitamente também não precisa de --saida.
    import runpy
    import sys

    import cpv_data as cd

    raiz = tmp_path / "raiz"
    raw_dir = raiz / "data" / "processed" / "raw"
    raw_dir.mkdir(parents=True)
    escrever_raw(raw_dir, 2025, [("1", "2024-01-10"), ("2", "2025-10-05"), ("3", "2025-11-05")])
    monkeypatch.setattr(cd, "ROOT", raiz)
    monkeypatch.setattr(cd, "RAW_PARQUET", raw_dir)
    monkeypatch.setattr(sys, "argv", ["prepare.py", *argumentos])
    runpy.run_path(pr.__file__, run_name="__main__")
    gravados = sorted(p.name for p in (raiz / "data" / "processed").glob("*.parquet"))
    assert gravados == sorted(f"{f}.parquet" for f in pr.FILES.values())
