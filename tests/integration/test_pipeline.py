"""Teste de integração: o pipeline inteiro, de .xlsx a treino/validação/teste.

Cria ficheiros .xlsx inventados no formato do Portal BASE, corre a conversão para Parquet
(src/convert_to_parquet.py) e a preparação (src/prepare.py), e verifica os ficheiros que saem,
contrato a contrato. Cada contrato inventado põe à prova uma regra de docs/qualidade_dados.md.
No fim, a régua (src/baselines.py) conta as etiquetas do treino que a preparação escreveu, e a
avaliação (src/evaluation.py) corre sobre a validação que a preparação escreveu.
"""

from datetime import datetime

import pandas as pd
import pytest

import baselines as bl
import evaluation as ev
import prepare as pr

LIMPEZA_ESCOLAS = "90919300-5 - Serviços de limpeza de escolas"

# (ficheiro, idcontrato, dataPublicacao, objectoContrato, descContrato, DescrAcordoQuadro, CPV)
CONTRATOS = [
    # 1, 2, 3: o mesmo texto com códigos diferentes -> um exemplo, código mais comum, n_contratos=3
    (2023, "1", datetime(2023, 5, 10), "Limpeza de escolas", "Limpeza de escolas", None, LIMPEZA_ESCOLAS),
    (2024, "2", datetime(2024, 2, 1), "limpeza de ESCOLAS", "limpeza de ESCOLAS", None, LIMPEZA_ESCOLAS),
    (2024, "3", datetime(2024, 3, 1), "Limpeza  de escolas", None, None, "90911200-8 - Limpeza de edifícios"),
    # 4: código ao nível da divisão -> sai do treino
    (2024, "4", datetime(2024, 6, 1), "Medicamentos", "Medicamentos", None, "33000000-0 - Equipamento médico"),
    # 5: publicado em 2025 mas no ficheiro de 2024; com descrição do acordo-quadro
    (2024, "5", datetime(2025, 1, 15), "Convite ao abrigo do acordo-quadro", "Convite ao abrigo do acordo-quadro",
     "AQ de medicamentos oncológicos", "33652000-5 - Agentes antineoplásicos"),
    # 6: o mesmo idcontrato em dois ficheiros -> fica só a cópia do mais recente (decisão 3).
    # Os textos diferentes são inventados (nos dados as cópias têm o mesmo texto): sem eles,
    # o passo dos textos repetidos juntava as cópias e o teste passava sem o dedupe_ids.
    (2024, "6", datetime(2025, 2, 1), "Papel para impressora", None, None, "30197630-1 - Papel de impressão"),
    (2025, "6", datetime(2025, 2, 1), "Papel de impressão", None, None, "30197630-1 - Papel de impressão"),
    # 7: dois códigos; descrição diferente do objeto
    (2025, "7", datetime(2025, 3, 1), "Seguro", "Seguro de acidentes de trabalho", None,
     "66512100-3 - Seguro de acidentes\n66510000-8 - Seguros"),
    # 8: sem CPV -> sai
    (2025, "8", datetime(2025, 4, 1), "Sem código", None, None, None),
    # 9 e 14: fora do período -> saem
    (2021, "9", datetime(2021, 12, 31), "Antigo", None, None, LIMPEZA_ESCOLAS),
    (2025, "14", datetime(2026, 1, 5), "Futuro", None, None, LIMPEZA_ESCOLAS),
    # 10, 11: validação (outubro de 2025)
    (2025, "10", datetime(2025, 10, 10), "LIMPEZA DE ESCOLAS", None, None, LIMPEZA_ESCOLAS),
    (2025, "11", datetime(2025, 10, 20), "Medicamentos", None, None, "33000000-0 - Equipamento médico"),
    # 12, 13: teste (novembro e dezembro de 2025). O 13 só fica visto_no_treino se o contrato 6
    # guardar a cópia de 2025 ("Papel de impressão"), por isso também falha se a decisão 3 falhar.
    (2025, "12", datetime(2025, 11, 5), "Jardinagem do parque", None, None, "77311000-3 - Manutenção de jardins"),
    (2025, "13", datetime(2025, 12, 20), "papel de impressão", None, None, "30197630-1 - Papel de impressão"),
]


@pytest.fixture(scope="module")
def sets(tmp_path_factory, run_pipeline):
    return run_pipeline(tmp_path_factory.mktemp("pipeline"), CONTRATOS)


def by_id(df):
    return df.set_index("idcontrato")


# --- Onde acaba cada contrato ---

def test_cada_contrato_acaba_no_conjunto_certo(sets):
    assert sorted(sets["treino"]["idcontrato"], key=int) == ["2", "5", "6", "7"]
    assert sorted(sets["validação"]["idcontrato"], key=int) == ["10", "11"]
    assert sorted(sets["teste"]["idcontrato"], key=int) == ["12", "13"]


def test_contratos_sem_cpv_ou_fora_do_periodo_saem(sets):
    todos = set().union(*(set(df["idcontrato"]) for df in sets.values()))
    assert {"8", "9", "14"}.isdisjoint(todos)


# --- Regras do treino ---

def test_texto_repetido_vira_um_exemplo_com_o_codigo_mais_comum(sets):
    limpeza = by_id(sets["treino"]).loc["2"]
    assert limpeza["codigo"] == "90919300"
    assert limpeza["n_contratos"] == 3


def test_codigo_ao_nivel_da_divisao_sai_do_treino(sets):
    assert "4" not in set(sets["treino"]["idcontrato"])
    assert (sets["treino"]["nivel"] > 2).all()


def test_contrato_publicado_em_2025_num_ficheiro_de_2024_entra(sets):
    aq = by_id(sets["treino"]).loc["5"]
    assert aq["texto"] == "Convite ao abrigo do acordo-quadro | AQ de medicamentos oncológicos"


def test_idcontrato_repetido_entre_ficheiros_fica_uma_vez(sets):
    # As duas cópias têm textos diferentes, por isso só o dedupe_ids as junta (não o passo dos textos)
    assert (sets["treino"]["idcontrato"] == "6").sum() == 1
    assert by_id(sets["treino"]).loc["6", "texto"] == "Papel de impressão"


def test_varios_codigos_etiqueta_e_o_primeiro_e_guardam_se_todos(sets):
    seguro = by_id(sets["treino"]).loc["7"]
    assert seguro["codigo"] == "66512100"
    assert seguro["codigos"] == "66512100 66510000"
    assert seguro["texto"] == "Seguro | Seguro de acidentes de trabalho"


# --- Regras da validação e do teste ---

def test_validacao_mantem_codigo_ao_nivel_da_divisao(sets):
    med = by_id(sets["validação"]).loc["11"]
    assert med["nivel"] == 2


def test_textos_vistos_no_treino_ficam_marcados(sets):
    assert by_id(sets["validação"])["visto_no_treino"].to_dict() == {"10": True, "11": False}
    assert by_id(sets["teste"])["visto_no_treino"].to_dict() == {"12": False, "13": True}


def test_nenhum_contrato_em_dois_conjuntos(sets):
    ids = [set(df["idcontrato"]) for df in sets.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])


def test_datas_sao_lidas_do_excel_como_datas(sets):
    assert pd.api.types.is_datetime64_any_dtype(sets["teste"]["data"])
    assert by_id(sets["teste"]).loc["12", "data"] == pd.Timestamp("2025-11-05")


# --- A avaliação lê o que a preparação escreve ---

PREENCHER = ["99999991", "99999992", "99999993", "99999994", "99999995"]  # não acertam em nada


def oraculo(df):
    """Sugestões que são os próprios códigos do contrato: tudo o que se pode acertar, acerta."""
    return df["codigos"].str.split().map(lambda c: (c + PREENCHER)[:5])


def test_avaliacao_corre_sobre_a_validacao_da_preparacao(sets):
    # Validação: o 10 (90919300, texto visto no treino) e o 11 (só 33000000, texto novo).
    # O 10 conta no código completo, na classe e na divisão; o 11 só na divisão, e aparece
    # em so_divisao. Com o oráculo, tudo o que conta acerta.
    val = sets["validação"]
    out = ev.report(oraculo(val), val["codigos"], val["visto_no_treino"])
    todos = out.loc["todos"]
    assert todos["contratos"].tolist() == [1, 1, 2, 1]
    assert (todos[["top1", "top5", "f1_macro"]] == 1.0).all().all()
    # Texto novo: só o 11 -> fora do código completo e da classe, dentro de so_divisao
    assert out.loc["texto novo", "contratos"].tolist() == [0, 0, 1, 1]
    assert out.loc[("texto novo", "so_divisao"), "top5"] == 1.0


def test_medidas_hierarquicas_correm_sobre_a_validacao_da_preparacao(sets):
    val = sets["validação"]
    out = ev.hierarchical_report(oraculo(val), val["codigos"], val["visto_no_treino"])
    assert (out == 1.0).all().all()


# --- A régua conta o treino que a preparação escreve ---

def regua(treino, k):
    # N = 1: com tão poucos contratos, o N do modelo (10) tirava todos os exemplos
    return bl.most_common_labels(treino["codigo"], n_min=1, k=k)


def test_regua_conta_exemplos_do_treino_da_preparacao(sets):
    # Treino: 90919300 (1 exemplo que junta os contratos 1, 2 e 3), 30197630, 33652000 e 66512100.
    # Contando exemplos, os 4 empatam com 1 e ficam por ordem de código: o 90919300 fica em último,
    # embora tenha 3 contratos. Contando contratos ficava em 1.º.
    assert regua(sets["treino"], 4) == ["30197630", "33652000", "66512100", "90919300"]
    assert bl.most_common_labels(sets["treino"]["codigo"], sets["treino"]["n_contratos"], n_min=1, k=1) == ["90919300"]


def test_regua_com_o_n_do_modelo_sem_etiquetas_suficientes_da_erro(sets):
    # Com N = 10, os 4 exemplos deste treino são todos raros (até ao grupo) e saem: sobram 0
    # etiquetas, e a régua tem de o dizer em vez de sugerir códigos que o modelo não aprende.
    with pytest.raises(ValueError, match="só tem 0 etiquetas"):
        bl.most_common_labels(sets["treino"]["codigo"])


def test_regua_nunca_sugere_divisoes(sets):
    # O contrato 4 (33000000) saiu do treino (decisão 8) e nenhuma etiqueta sobe até à divisão
    assert not any(c.endswith("000000") for c in regua(sets["treino"], 4))


def test_regua_avaliada_sobre_a_validacao_da_preparacao(sets):
    # Sugestões: 30197630, 33652000, 66512100, 90919300 e um código que não acerta em nada.
    # O 10 (90919300) acerta o código só à 4.ª: top-1 0, top-5 1. O 11 (só 33000000) acerta a
    # divisão 33 à 2.ª (33652000): top-1 0, top-5 1 na divisão e em so_divisao.
    val = sets["validação"]
    sug = bl.suggest_most_common(regua(sets["treino"], 4) + PREENCHER[:1], val.index)
    assert (sug.map(lambda s: s[:4]) == pd.Series([regua(sets["treino"], 4)] * len(val), index=val.index)).all()
    out = ev.report(sug, val["codigos"], val["visto_no_treino"]).loc["todos"]
    assert out.loc["codigo", ["contratos", "top1", "top5"]].tolist() == [1, 0.0, 1.0]
    assert out.loc["divisao", ["contratos", "top1", "top5"]].tolist() == [2, 0.0, 1.0]
    assert out.loc["so_divisao", ["contratos", "top1", "top5"]].tolist() == [1, 0.0, 1.0]


# --- A régua com o N do modelo (10), num treino com códigos suficientes ---
#
# Os contratos de `contratos_regua` (tests/conftest.py, partilhados com o teste do TF-IDF) dão a
# régua com o N por defeito e 5 sugestões. A explicação de cada código está lá.

TOP_REGUA = ["90919300", "30197630", "66512100", "77311000", "33140000"]


@pytest.fixture(scope="module")
def sets_regua(tmp_path_factory, run_pipeline, contratos_regua):
    return run_pipeline(tmp_path_factory.mktemp("regua"), contratos_regua)


def test_regua_com_o_n_do_modelo_sobre_o_treino_da_preparacao(sets_regua):
    assert bl.most_common_labels(sets_regua["treino"]["codigo"]) == TOP_REGUA


def test_regua_com_o_n_do_modelo_sobe_os_raros_e_ignora_divisoes(sets_regua):
    top = bl.most_common_labels(sets_regua["treino"]["codigo"])
    assert "33140000" in top and not any(c.startswith("331411") for c in top)
    assert "45000000" not in top
    # Com N = 5 ou sem a regra dos raros (N = 1), a classe não existe e entra 45233200
    for n in (1, 5):
        assert bl.most_common_labels(sets_regua["treino"]["codigo"], n_min=n)[4] == "45233200"


def test_regua_conta_exemplos_e_nao_contratos_no_pipeline(sets_regua):
    tr = sets_regua["treino"]
    assert tr.loc[tr["codigo"] == "98300000", "n_contratos"].tolist() == [6] * 10
    assert "98300000" not in bl.most_common_labels(tr["codigo"])
    assert bl.most_common_labels(tr["codigo"], tr["n_contratos"])[0] == "98300000"


def test_regua_com_o_n_do_modelo_avaliada_na_validacao(sets_regua):
    # Sugestões: 90919300, 30197630, 66512100, 77311000, 33140000.
    #   v-limpeza  (90919300, texto visto) -> acerta código, classe e divisão à 1.ª
    #   v-material (33141100)              -> código não; classe 3314 e divisão 33 à 5.ª
    #   v-estradas (45233200)              -> nada
    #   v-obras    (só 45000000)           -> só divisão (so_divisao); nada
    val = sets_regua["validação"]
    sug = bl.suggest_most_common(bl.most_common_labels(sets_regua["treino"]["codigo"]), val.index)
    out = ev.report(sug, val["codigos"], val["visto_no_treino"])
    todos = out.loc["todos"]
    assert todos.loc["codigo", ["contratos", "top1", "top5"]].tolist() == [3, 1 / 3, 1 / 3]
    assert todos.loc["classe", ["contratos", "top1", "top5"]].tolist() == [3, 1 / 3, 2 / 3]
    assert todos.loc["divisao", ["contratos", "top1", "top5"]].tolist() == [4, 1 / 4, 2 / 4]
    assert todos.loc["so_divisao", ["contratos", "top5"]].tolist() == [1, 0.0]
    # Texto novo: sem o v-limpeza, o único acerto que sobra é o v-material na classe
    novo = out.loc["texto novo"]
    assert novo.loc["codigo", ["contratos", "top5"]].tolist() == [2, 0.0]
    assert novo.loc["classe", ["contratos", "top5"]].tolist() == [2, 0.5]


# --- Propriedades do pipeline ---

def test_correr_duas_vezes_da_exatamente_o_mesmo(sets, tmp_path, run_pipeline):
    outra_vez = run_pipeline(tmp_path, CONTRATOS)
    for nome in sets:
        pd.testing.assert_frame_equal(sets[nome], outra_vez[nome])


def test_pasta_sem_ficheiros_da_erro_claro(tmp_path):
    with pytest.raises(pr.DataCheckError, match="nenhum contratos"):
        pr.run(raw_dir=tmp_path, out_dir=tmp_path / "out", verbose=False)
