"""Leitura dos treinos da grelha do BERT (spike E5-S1) para o notebook 12.

Cada treino da grelha (um modelo, uma taxa de aprendizagem, uma semente) é uma pasta `--saida` do
`train_bert.py` copiada do pod. Dentro dela, `epoca_<k>/` tem as sugestões da época k com cada τ
(`sugestoes_validacao.parquet` para τ = 0, `sugestoes_validacao_tau0.25.parquet`,
`sugestoes_validacao_tau0.5.parquet`, ...) e o `epoca.json` (com a perda na validação desde o
E5-T7); na raiz fica o `treino.json`. Uma **opção** da grelha é (taxa, época, τ): 3 × 3 × 3 = 27
por modelo, de só 3 treinos, porque a época e o τ não pedem treino novo.

Aqui ficam só as regras que o notebook usa e que têm teste: onde está cada pasta, as verificações
do `treino.json` (janela, teste, validação, hiperparâmetros), a leitura das sugestões alinhada pela
validação, as horas e o custo de cada treino, o controlo e os extremos. A apresentação (tabelas,
texto) fica no notebook. Não lê nada sozinho a não ser os ficheiros que recebe.
"""

import json
import math
import re
from pathlib import Path

import pandas as pd

from cpv_data import TEST_START, TRAIN_START, VAL_START
from train_tfidf import suggestions_file

TESTE = "teste.parquet"
TREINO_JSON = "treino.json"
EPOCA_JSON = "epoca.json"
POD_INFO = "pod_info.json"
TOP_K = 5
_CODIGO = re.compile(r"\d{8}")


def taxa_txt(taxa):
    """A taxa como se escreve no nome da pasta: 2e-05 -> "2e-5", 0.0001 -> "1e-4", 2.5e-5 -> "2.5e-5".

    O `f"{taxa:g}"` do Python dá "2e-05" (expoente com 2 dígitos) e "0.0001" (sem expoente); as
    pastas do E5-S1 usam sempre o expoente sem zeros à esquerda.
    """
    mantissa, expoente = f"{taxa:e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(expoente)}"


def pasta_treino(saida, modelo, taxa, semente):
    """Pasta de um treino da grelha: `<saida>/<modelo>_taxa_<taxa>_semente_<semente>`.

    Exemplo: ("metrics/e5_s1", "albertina", 3e-5, 0) -> metrics/e5_s1/albertina_taxa_3e-5_semente_0.
    """
    return Path(saida) / f"{modelo}_taxa_{taxa_txt(taxa)}_semente_{semente}"


def pasta_epoca(pasta, epoca):
    """`epoca_<k>/` dentro da pasta de um treino (o nome do `train_bert.epoch_dir`)."""
    return Path(pasta) / f"epoca_{epoca}"


def ficheiro_sugestoes(pasta, epoca, tau):
    """As sugestões de uma opção: época k e τ de um treino (τ = 0.5 -> ..._tau0.5.parquet)."""
    return pasta_epoca(pasta, epoca) / suggestions_file(tau)


def verificar_treino(resumo, origem, *, modelo, taxa, semente, epocas, n_min, taus, validacao_sha256=None):
    """Erro (ValueError) se o `treino.json` de uma pasta não for o treino que a grelha pediu.

    Confirma, sem abrir os dados: o modelo, a taxa, a semente, o número de épocas e o N pedidos;
    que não é um ensaio de custo; que gravou as sugestões de cada τ; que o treino é da janela de
    2022 (`TRAIN_START`) e acaba antes da validação; que nem o treino nem a validação foram o
    `teste.parquet`; que a validação vai até ao fim de outubro (antes do teste); e, se o
    `treino.json` tiver o SHA-256 da validação, que é igual ao do ficheiro do portátil (senão
    o "texto novo" e os contratos podiam ser outros). Devolve a lista de avisos (o que não se
    pôde confirmar), vazia se tudo foi confirmado.
    """
    erros, avisos = [], []
    if resumo.get("ensaio"):
        erros.append("é um ensaio de custo (sem sugestões)")
    chave = (resumo.get("modelo") or {}).get("chave")
    if chave != modelo:
        erros.append(f"modelo {chave!r}, a grelha pediu {modelo!r}")
    if not math.isclose(float(resumo.get("taxa_aprendizagem", math.nan)), taxa, rel_tol=1e-9):
        erros.append(f"taxa {resumo.get('taxa_aprendizagem')}, a grelha pediu {taxa:g}")
    if resumo.get("semente") != semente:
        erros.append(f"semente {resumo.get('semente')}, a grelha pediu {semente}")
    if resumo.get("epocas") != epocas:
        erros.append(f"{resumo.get('epocas')} épocas, a grelha pediu {epocas}")
    if resumo.get("n_min") != n_min:
        erros.append(f"N = {resumo.get('n_min')}, a grelha pediu {n_min}")
    feitos = [float(t) for t in resumo.get("taus", [])]
    falta = [t for t in taus if not any(math.isclose(t, f) for f in feitos)]
    if falta:
        erros.append(f"não gravou as sugestões com τ = {falta} (gravou {feitos})")

    treino, val = resumo.get("treino") or {}, resumo.get("validacao") or {}
    if TESTE in (treino.get("ficheiro"), val.get("ficheiro")):
        erros.append("o treino ou a validação foi o ficheiro de teste")
    datas = treino.get("datas") or [None, None]
    if datas[0] != str(TRAIN_START.date()):
        erros.append(f"o treino começa em {datas[0]}, não em {TRAIN_START.date()} (janela de 2022)")
    if datas[1] is None or pd.Timestamp(datas[1]) >= VAL_START:
        erros.append(f"o treino vai até {datas[1]}, não acaba antes da validação ({VAL_START.date()})")
    dval = val.get("datas") or [None, None]
    if dval[1] is None or pd.Timestamp(dval[1]) >= TEST_START:
        erros.append(f"a validação vai até {dval[1]}, entra no período do teste ({TEST_START.date()})")

    sha = (resumo.get("sha256") or {}).get("validacao")
    if sha is None:
        avisos.append("o treino.json não tem o SHA-256 da validação: não se confirmou que é o mesmo ficheiro")
    elif validacao_sha256 is not None and sha != validacao_sha256:
        erros.append("o SHA-256 da validação do pod é diferente do do portátil")

    if erros:
        raise ValueError(f"{origem}: " + "; ".join(erros))
    return avisos


def ler_sugestoes(path, idcontrato):
    """Sugestões de um ficheiro, alinhadas pela validação (pelo `idcontrato`, nunca pela posição).

    `idcontrato` é a coluna da validação (com o índice da validação). Erro se o ficheiro não tiver
    exatamente esses contratos, um por linha, ou se algum não tiver 5 códigos distintos de 8
    dígitos. Devolve uma Series de listas com o índice da validação.
    """
    sug = pd.read_parquet(path, columns=["idcontrato", "sugestoes"])
    sug["idcontrato"] = sug["idcontrato"].astype(str)
    ids = idcontrato.astype(str)
    if sug["idcontrato"].duplicated().any() or set(sug["idcontrato"]) != set(ids):
        raise ValueError(f"{path}: os contratos não são os da validação, um por linha")
    s = sug.set_index("idcontrato")["sugestoes"].reindex(ids.to_numpy()).map(list)
    s.index = idcontrato.index
    if not s.map(lambda l: len(set(l)) == TOP_K and all(isinstance(c, str) and _CODIGO.fullmatch(c) for c in l)).all():
        raise ValueError(f"{path}: há contratos sem {TOP_K} códigos distintos de 8 dígitos")
    return s


def ler_pod_info(pasta):
    """O `pod_info.json` do treino (na pasta dele) ou da grelha (na pasta acima); None se não houver.

    O preço por hora não vem no `treino.json`: escreve-se à mão no `pod_info.json`, a partir das
    leituras da Runpod. Na grelha vários treinos podem partilhar um pod, por isso procura-se
    primeiro na pasta do treino e depois na de cima.
    """
    for p in (Path(pasta) / POD_INFO, Path(pasta).parent / POD_INFO):
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


def custo(resumo, pod_info):
    """Horas de GPU e dólares de um treino.

    Horas = `tempo_treinar_s` / 3600 (só o treino, sem instalação nem cópias, como no E5-T4).
    Dólares = horas × `preco_por_hora_usd` do `pod_info.json` (em `pod` ou na raiz). Sem preço,
    os dólares ficam NaN. Exemplo: 6458,35 s a 0,89 $/h -> 1,79 h -> 1,60 $.
    """
    segundos = resumo.get("tempo_treinar_s")
    horas = segundos / 3600 if segundos is not None else math.nan
    preco = None
    if pod_info:
        preco = (pod_info.get("pod") or {}).get("preco_por_hora_usd", pod_info.get("preco_por_hora_usd"))
    preco = float(preco) if preco is not None else math.nan
    return {"horas_gpu": horas, "preco_por_hora_usd": preco, "dolares": horas * preco, "gpu": resumo.get("gpu")}


def perda_validacao(pasta, epoca):
    """`perda_validacao` do `epoca.json` da época (E5-T7); NaN se o ficheiro ou o campo faltar.

    Os treinos anteriores ao E5-T7 (E5-T4, E5-T5) não a têm. É só informação: quem decide é o top-5.
    """
    p = pasta_epoca(pasta, epoca) / EPOCA_JSON
    if not p.exists():
        return math.nan
    valor = json.loads(p.read_text(encoding="utf-8")).get("perda_validacao")
    return float(valor) if valor is not None else math.nan


def controlo(valor, referencia, tolerancia):
    """O controlo do E5-S1: a repetição do primeiro treino fica a menos da tolerância dele?

    Tudo em pontos percentuais (76,98, não 0,7698). Passa se |valor − referência| <= tolerância.
    Devolve (passa, diferença = valor − referência).
    """
    diferenca = valor - referencia
    return abs(diferenca) <= tolerancia + 1e-9, diferenca


def nos_extremos(opcao, taxas, epocas, taus):
    """Os eixos em que a opção (taxa, época, τ) está num extremo da lista, como texto.

    Regra do E5-S1: a taxa ou o τ mais baixo ou mais alto, ou a última época (a primeira não conta:
    o treino continuou depois dela). Exemplo: (5e-5, 3, 0.25) com taxas [2e-5, 3e-5, 5e-5] e
    épocas [1, 2, 3] -> ["taxa 5e-5 é a mais alta", "época 3 é a última"].
    """
    taxa, epoca, tau = opcao
    out = []
    if math.isclose(taxa, min(taxas)):
        out.append(f"taxa {taxa_txt(taxa)} é a mais baixa")
    elif math.isclose(taxa, max(taxas)):
        out.append(f"taxa {taxa_txt(taxa)} é a mais alta")
    if epoca == max(epocas):
        out.append(f"época {epoca} é a última")
    if math.isclose(tau, min(taus)):
        out.append(f"τ {tau:g} é o mais baixo")
    elif math.isclose(tau, max(taus)):
        out.append(f"τ {tau:g} é o mais alto")
    return out
