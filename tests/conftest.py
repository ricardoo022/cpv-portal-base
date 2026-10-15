"""Ajudantes partilhados pelos testes de integração.

`run_pipeline` corre o pipeline real (conversão para Parquet e preparação) sobre .xlsx inventados
no formato do Portal BASE; `contratos_regua` são os contratos inventados com códigos suficientes
para o N por defeito (10) e 5 sugestões. Usam-nos o teste do pipeline e o do TF-IDF.
"""

from datetime import datetime

import openpyxl
import pandas as pd
import pytest

import convert_to_parquet as conv
import prepare as pr

COLS = pr.RAW_COLS + ["Ano"]


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


def _run_pipeline(base, contratos, train_start=None):
    xlsx_dir, raw_dir, out_dir = base / "xlsx", base / "raw", base / "out"
    xlsx_dir.mkdir()
    raw_dir.mkdir()
    for ano in sorted({c[0] for c in contratos}):
        xlsx = xlsx_dir / f"contratos{ano}.xlsx"
        write_xlsx(xlsx, [c for c in contratos if c[0] == ano])
        conv.convert(xlsx, raw_dir / f"contratos{ano}.parquet")
    # Sem train_start não se passa o argumento: assim corre-se o valor por defeito do próprio run
    extra = {} if train_start is None else {"train_start": train_start}
    pr.run(raw_dir=raw_dir, out_dir=out_dir, verbose=False, **extra)
    return {nome: pd.read_parquet(out_dir / f"{f}.parquet") for nome, f in pr.FILES.items()}


@pytest.fixture(scope="session")
def run_pipeline():
    """run_pipeline(base, contratos, train_start=None) -> {"treino", "validação", "teste": DataFrame}.

    Cada contrato é (ficheiro, idcontrato, dataPublicacao, objectoContrato, descContrato,
    DescrAcordoQuadro, CPV). Escreve em `base` (uma pasta de `tmp_path`): os .parquet de entrada
    em `base / "raw"` e os três conjuntos em `base / "out"`. `train_start` vai para `prepare.run`.
    """
    return _run_pipeline


# A régua com o N do modelo (10): textos todos diferentes ("<nome> <i>"), para a régua e o modelo
# correrem como no projeto. Por código, no treino (publicados em 2024):

def _gerar(nome, codigo, n_textos, repete=1, data=datetime(2024, 3, 1), ano=2024):
    cpv = f"{codigo}-0 - {nome}"
    return [(ano, f"{nome}-{i}-{j}", data, f"{nome} {i}", None, None, cpv)
            for i in range(n_textos) for j in range(repete)]


CONTRATOS_REGUA = [
    *_gerar("Limpeza", "90919300", 30),       # 1.º: 30 exemplos
    *_gerar("Papel", "30197630", 25),         # 2.º: 25
    *_gerar("Seguro", "66512100", 20),        # 3.º: 20
    *_gerar("Jardins", "77311000", 15),       # 4.º: 15
    # Dois códigos da classe 3314, com 7 e 5 exemplos: raros com N = 10, sobem juntos para 33140000
    # (12). Empata com 45233200 (12) e fica à frente por ser o menor código: 5.º. Com N = 5 não
    # são raros, a classe não existe e o 5.º passa a ser 45233200: o teste distingue N = 5 de 10.
    *_gerar("Material", "33141100", 7),
    *_gerar("Luvas", "33141400", 5),
    *_gerar("Estradas", "45233200", 12),      # 6.º: fica de fora
    # 10 exemplos, cada um repetido em 6 contratos (n_contratos = 6): 60 contratos, mas só 10
    # exemplos. Contando contratos seria o 1.º; contando exemplos fica de fora.
    *_gerar("Diversos", "98300000", 10, repete=6),
    # Código de divisão: sai do treino (decisão 8), embora sejam 50 exemplos
    *_gerar("Obras", "45000000", 50),
    # Validação (outubro de 2025)
    (2025, "v-limpeza", datetime(2025, 10, 2), "Limpeza 0", None, None, "90919300-5 - Limpeza"),   # texto visto
    (2025, "v-material", datetime(2025, 10, 3), "Material novo", None, None, "33141100-0 - Material"),
    (2025, "v-estradas", datetime(2025, 10, 4), "Estradas novas", None, None, "45233200-0 - Estradas"),
    (2025, "v-obras", datetime(2025, 10, 5), "Obras novas", None, None, "45000000-0 - Obras"),
    # Teste (novembro de 2025): a preparação exige-o; a régua nunca o lê
    (2025, "t-1", datetime(2025, 11, 5), "Papel novo", None, None, "30197630-1 - Papel"),
]


@pytest.fixture(scope="session")
def contratos_regua():
    return CONTRATOS_REGUA


# E4-T3: um tema por código, sem palavras de conteúdo em comum. 2 textos por código, menos o
# 66512100, que tem 1: com N = 1 fica como etiqueta, com N = 2 sairia. Assim os testes de
# sobreajuste distinguem N = 1 de N = 2, e cada texto de treino fica com o seu código.
TEMAS_DECORAR = {
    "90911200": ["Limpeza das escolas", "Lavagem dos pavilhões escolares"],
    "33600000": ["Medicamentos hospitalares", "Antibióticos e analgésicos"],
    "45233200": ["Pavimentação rodoviária", "Asfalto para estradas"],
    "30192000": ["Papel e canetas", "Tinteiros para impressoras"],
    "15800000": ["Refeições na cantina", "Almoços para idosos"],
    "34110000": ["Automóveis ligeiros", "Viaturas elétricas"],
    "66512100": ["Seguro de acidentes pessoais"],
}


@pytest.fixture(scope="session")
def temas_decorar():
    return TEMAS_DECORAR


# E5-T2: modelos minúsculos de pesos aleatórios, construídos localmente (nunca se descarrega o
# modelo verdadeiro: o CI não tem acesso fiável ao Hugging Face). Mesmas arquiteturas e mesmas
# classes de tokenizador dos modelos do E5: BERT com `BertTokenizer` (WordPiece, como o BERTimbau)
# e DeBERTa (v1) com `DebertaTokenizer` (BPE ao nível do byte, como o Albertina).
ESPECIAIS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
# Pesos iniciais maiores do que os 0,02 do BERT e sem dropout: com 32 dimensões e pesos de 0,02 o
# modelo dava quase a mesma saída a todos os textos e não decorava 5 exemplos em 20 épocas.
CONFIG_MINUSCULA = dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                        max_position_embeddings=64, initializer_range=0.2, hidden_dropout_prob=0.0,
                        attention_probs_dropout_prob=0.0)


def make_tiny_model(pasta, textos, arquitetura="bert", dtype=None):
    """Grava em `pasta` um tokenizador feito com as palavras de `textos` e um modelo minúsculo.

    "bert": vocabulário de palavras inteiras (uma palavra = um token; as outras dão [UNK]).
    "deberta": BPE treinado com a biblioteca `tokenizers` sobre os `textos`. `dtype` (por exemplo
    torch.bfloat16) grava os pesos nesse tipo, como o Albertina está gravado no Hugging Face.
    Devolve a pasta, para passar como `modelo` ao treino.
    """
    import torch
    from transformers import (BertConfig, BertModel, BertTokenizer, DebertaConfig, DebertaModel,
                              DebertaTokenizer)

    pasta.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    if arquitetura == "bert":
        palavras = sorted({p for t in textos for p in t.split()})
        tok = BertTokenizer(vocab={p: i for i, p in enumerate(ESPECIAIS + palavras)}, do_lower_case=False)
        model = BertModel(BertConfig(vocab_size=len(tok), pad_token_id=tok.pad_token_id, **CONFIG_MINUSCULA))
    elif arquitetura == "deberta":
        from tokenizers import ByteLevelBPETokenizer
        bpe = ByteLevelBPETokenizer()
        bpe.train_from_iterator(list(textos), vocab_size=400, min_frequency=1, special_tokens=ESPECIAIS,
                                show_progress=False)
        bpe.save_model(str(pasta))
        tok = DebertaTokenizer(vocab=str(pasta / "vocab.json"), merges=str(pasta / "merges.txt"))
        # A atenção relativa do Albertina (config.json do modelo verdadeiro)
        model = DebertaModel(DebertaConfig(vocab_size=len(tok), pad_token_id=tok.pad_token_id, relative_attention=True,
                                           pos_att_type=["c2p", "p2c"], position_biased_input=False,
                                           type_vocab_size=0, **CONFIG_MINUSCULA))
    else:
        raise ValueError(arquitetura)
    if dtype is not None:
        model = model.to(dtype)
    tok.save_pretrained(pasta)
    model.save_pretrained(pasta)
    return pasta
