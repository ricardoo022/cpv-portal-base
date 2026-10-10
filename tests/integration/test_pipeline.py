"""Teste de integração: o pipeline inteiro, de .xlsx a treino/validação/teste.

Cria ficheiros .xlsx inventados no formato do Portal BASE, corre a conversão para Parquet
(src/convert_to_parquet.py) e a preparação (src/prepare.py), e verifica os ficheiros que saem,
contrato a contrato. Cada contrato inventado põe à prova uma regra de docs/qualidade_dados.md.
"""

from datetime import datetime

import openpyxl
import pandas as pd
import pytest

import convert_to_parquet as conv
import prepare as pr

COLS = pr.RAW_COLS + ["Ano"]

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
    # 6: o mesmo idcontrato em dois ficheiros, com o texto corrigido no mais recente -> fica só esse
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
    # 12, 13: teste (novembro e dezembro de 2025)
    (2025, "12", datetime(2025, 11, 5), "Jardinagem do parque", None, None, "77311000-3 - Manutenção de jardins"),
    (2025, "13", datetime(2025, 12, 20), "papel de impressão", None, None, "30197630-1 - Papel de impressão"),
]


def write_xlsx(path, rows):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(COLS)
    for _, idc, data, obj, desc, aq, cpv in rows:
        values = {"idcontrato": idc, "dataPublicacao": data, "objectoContrato": obj, "descContrato": desc,
                  "DescrAcordoQuadro": aq, "CPV": cpv, "adjudicante": "500000000 - Entidade",
                  "tipoContrato": "Aquisição de serviços", "tipoprocedimento": "Ajuste Direto Regime Geral",
                  "Ano": data.year}
        ws.append([values[c] for c in COLS])
    wb.save(path)


def run_pipeline(base):
    xlsx_dir, raw_dir, out_dir = base / "xlsx", base / "raw", base / "out"
    xlsx_dir.mkdir()
    raw_dir.mkdir()
    for ano in sorted({c[0] for c in CONTRATOS}):
        xlsx = xlsx_dir / f"contratos{ano}.xlsx"
        write_xlsx(xlsx, [c for c in CONTRATOS if c[0] == ano])
        conv.convert(xlsx, raw_dir / f"contratos{ano}.parquet")
    pr.run(raw_dir=raw_dir, out_dir=out_dir, verbose=False)
    return {nome: pd.read_parquet(out_dir / f"{f}.parquet") for nome, f in pr.FILES.items()}


@pytest.fixture(scope="module")
def sets(tmp_path_factory):
    return run_pipeline(tmp_path_factory.mktemp("pipeline"))


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


# --- Propriedades do pipeline ---

def test_correr_duas_vezes_da_exatamente_o_mesmo(sets, tmp_path):
    outra_vez = run_pipeline(tmp_path)
    for nome in sets:
        pd.testing.assert_frame_equal(sets[nome], outra_vez[nome])


def test_pasta_sem_ficheiros_da_erro_claro(tmp_path):
    with pytest.raises(pr.DataCheckError, match="nenhum contratos"):
        pr.run(raw_dir=tmp_path, out_dir=tmp_path / "out", verbose=False)
