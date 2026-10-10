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


def test_check_sets_apanha_none_escrito_no_texto():
    sets = conjuntos_validos()
    sets["validação"].loc[0, "texto"] = "jardins | None"
    with pytest.raises(pr.DataCheckError, match="None"):
        pr.check_sets(sets)


def test_check_sets_apanha_codigo_mal_formado():
    sets = conjuntos_validos()
    sets["treino"].loc[0, "codigo"] = "9091120"
    with pytest.raises(pr.DataCheckError, match="mal formado"):
        pr.check_sets(sets)
