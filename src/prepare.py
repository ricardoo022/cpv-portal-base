"""Fase 3: aplica as decisões de docs/qualidade_dados.md e cria treino, validação e teste.

Lê data/processed/raw/ (todos os ficheiros) e grava em data/processed/:
    treino.parquet, validacao.parquet, teste.parquet

Para com erro se alguma das verificações no fim falhar.

Uso:
    .venv/bin/python src/prepare.py
"""

from pathlib import Path

import pandas as pd

import cpv_data as cd

OUT_DIR = cd.ROOT / "data" / "processed"
SEP = " | "  # separador entre as partes do texto de entrada
RAW_COLS = ["idcontrato", "dataPublicacao", "objectoContrato", "descContrato", "DescrAcordoQuadro",
            "CPV", "adjudicante", "tipoContrato", "tipoprocedimento"]
KEEP = ["idcontrato", "data", "texto", "codigo", "codigos", "nivel",
        "adjudicante", "tipoContrato", "tipoprocedimento"]
FILES = {"treino": "treino", "validação": "validacao", "teste": "teste"}


class DataCheckError(Exception):
    """Uma verificação dos dados preparados falhou."""


def check(condition, message):
    # Ao contrário de `assert`, não pode ser desligado com `python -O`.
    if not condition:
        raise DataCheckError(message)


def log(msg, n, verbose=True):
    if verbose:
        print(f"  {msg:<64}{n:>10,}".replace(",", " "))


# --- Passos 1 a 5: ler e construir as colunas, ficheiro a ficheiro ---

def build_text(df):
    """Decisão 5: objectoContrato + descContrato (se diferente) + DescrAcordoQuadro (se existir)."""
    clean = lambda s: s.fillna("").str.split().str.join(" ")
    obj, desc, aq = clean(df["objectoContrato"]), clean(df["descContrato"]), clean(df["DescrAcordoQuadro"])
    junta_desc = (desc != "") & (cd.normalize_text(desc) != cd.normalize_text(obj))
    texto = obj.where(~junta_desc, obj + SEP + desc)
    return texto.where(aq == "", texto + SEP + aq)


def text_key(texto):
    """Impressão digital (hash) do texto normalizado: textos iguais dão o mesmo número."""
    return pd.util.hash_pandas_object(cd.normalize_text(texto), index=False).to_numpy()


def load_one(path, since=cd.TRAIN_START, until=cd.YEAR_END):
    """Lê um ficheiro, filtra pela data e constrói as colunas. Devolve (contratos, nº lidos no período)."""
    raw = pd.read_parquet(path, columns=RAW_COLS)
    raw["data"] = pd.to_datetime(raw["dataPublicacao"], errors="coerce")
    raw = raw[(raw["data"] >= since) & (raw["data"] < until)]
    n_lidos = len(raw)
    codigos = cd.all_codes(raw["CPV"])
    tem_codigo = (codigos.map(len) > 0).to_numpy()
    raw, codigos = raw[tem_codigo], codigos[tem_codigo]
    out = pd.DataFrame({
        "idcontrato": raw["idcontrato"],
        "data": raw["data"],
        "ficheiro": int(Path(path).stem.removeprefix("contratos")),
        "texto": build_text(raw),
        "codigo": codigos.str[0],
        "codigos": codigos.str.join(" "),  # todos os códigos, separados por espaço
        "adjudicante": raw["adjudicante"],
        "tipoContrato": raw["tipoContrato"],
        "tipoprocedimento": raw["tipoprocedimento"],
    })
    return out, n_lidos


def load_all(raw_dir):
    """Lê todos os contratos*.parquet da pasta. Devolve (contratos, nº lidos no período)."""
    paths = sorted(Path(raw_dir).glob("contratos*.parquet"))
    check(len(paths) > 0, f"nenhum contratos*.parquet em {raw_dir}")
    parts, n_lidos = [], 0
    for p in paths:
        part, n = load_one(p)
        parts.append(part)
        n_lidos += n
    df = pd.concat(parts, ignore_index=True)
    for c in ["adjudicante", "tipoContrato", "tipoprocedimento"]:
        df[c] = df[c].astype("category")
    return df, n_lidos


def dedupe_ids(df):
    """Decisão 3: uma cópia por idcontrato (a do ficheiro mais recente; no mesmo ficheiro, a de data mais recente)."""
    return df.sort_values(["ficheiro", "data"]).drop_duplicates("idcontrato", keep="last")


def add_columns(df):
    df = df.copy()
    df["nivel"] = cd.code_level(df["codigo"])
    df["periodo"] = cd.period(df["data"])
    df["chave"] = text_key(df["texto"])
    return df


# --- Passos 7 e 8: só no treino ---

def build_train(df, verbose=True):
    """Decisões 8 e 6: sem códigos ao nível da divisão; um exemplo por texto, com o código mais comum."""
    tr = df[df["periodo"] == "treino"]
    log("contratos", len(tr), verbose)
    tr = tr[tr["nivel"] > 2]
    n_especificos = len(tr)
    log("7. sem códigos ao nível da divisão", n_especificos, verbose)

    pares = tr.groupby(["chave", "codigo"]).size().rename("n").reset_index()
    # Empate entre códigos com o mesmo número de contratos: fica o menor código (para ser determinístico)
    pares = pares.sort_values(["chave", "n", "codigo"], ascending=[True, False, True])
    melhor = pares.drop_duplicates("chave").set_index("chave")["codigo"]
    total = pares.groupby("chave")["n"].sum()

    tr = tr[tr["codigo"].to_numpy() == melhor.reindex(tr["chave"]).to_numpy()]
    log("8a. contratos que seguem o código mais comum do seu texto", len(tr), verbose)
    tr = tr.sort_values("data").drop_duplicates("chave", keep="last").copy()
    tr["n_contratos"] = total.reindex(tr["chave"]).to_numpy()
    log("8b. exemplos de treino (um por texto)", len(tr), verbose)

    check(int(tr["n_contratos"].sum()) == n_especificos, "n_contratos não soma os contratos do treino")
    return tr[KEEP + ["n_contratos", "chave"]].reset_index(drop=True)


# --- Passo 9: validação e teste ficam todos ---

def build_eval(df, nome, chaves_treino, verbose=True):
    part = df[df["periodo"] == nome].copy()
    part["visto_no_treino"] = part["chave"].isin(chaves_treino)
    log("contratos", len(part), verbose)
    log("9. com texto que já aparece no treino", int(part["visto_no_treino"].sum()), verbose)
    log("   com código ao nível da divisão (avaliados só na divisão)", int((part["nivel"] == 2).sum()), verbose)
    return part[KEEP + ["visto_no_treino"]].reset_index(drop=True)


# --- Passo 10: verificações ---

def check_sets(sets):
    """Regras que têm de ser sempre verdade. Levanta DataCheckError se alguma falhar."""
    t, v, te = sets["treino"], sets["validação"], sets["teste"]
    for nome, x in sets.items():
        check(len(x) > 0, f"{nome} vazio")
        check(x["idcontrato"].is_unique, f"idcontrato repetido em {nome}")
        check(x["codigo"].str.fullmatch(r"\d{8}").all(), f"código mal formado em {nome}")
        check((x["texto"].str.strip() != "").all(), f"texto vazio em {nome}")
        check(not x["texto"].str.contains(r"\b(?:None|nan)\b").any(), f"'None' ou 'nan' escrito no texto em {nome}")
        check((x["codigo"] == x["codigos"].str.split().str[0]).all(), f"etiqueta não é o primeiro código em {nome}")
    check(t["data"].min() >= cd.TRAIN_START and t["data"].max() < cd.VAL_START, "treino fora do período")
    check(v["data"].min() >= cd.VAL_START and v["data"].max() < cd.TEST_START, "validação fora do período")
    check(te["data"].min() >= cd.TEST_START and te["data"].max() < cd.YEAR_END, "teste fora do período")
    ids = [set(x["idcontrato"]) for x in (t, v, te)]
    check(not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]), "contrato em mais do que um conjunto")
    check((t["nivel"] > 2).all(), "código ao nível da divisão no treino")
    norm = cd.normalize_text(t["texto"])
    check(norm.is_unique, "texto repetido no treino")
    check(pd.Series(text_key(t["texto"])).is_unique, "duas impressões digitais iguais no treino")


def run(raw_dir=cd.RAW_PARQUET, out_dir=OUT_DIR, verbose=True):
    """Corre o pipeline todo e grava os três ficheiros. Devolve os três conjuntos."""
    df, n_lidos = load_all(raw_dir)
    if verbose:
        print("Funil (contratos que ficam em cada passo):")
    log("1. publicados de jan 2022 a dez 2025 (todos os ficheiros)", n_lidos, verbose)
    log("3. com pelo menos um código CPV", len(df), verbose)
    df = dedupe_ids(df)
    log("2. uma cópia por idcontrato", len(df), verbose)
    df = add_columns(df)

    if verbose:
        print("\nTreino (jan 2022 a set 2025):")
    treino = build_train(df, verbose)
    chaves_treino = set(treino["chave"])
    sets = {"treino": treino.drop(columns="chave")}
    for nome in ["validação", "teste"]:
        if verbose:
            print(f"\n{nome.capitalize()}:")
        sets[nome] = build_eval(df, nome, chaves_treino, verbose)

    check_sets(sets)
    if verbose:
        print("\n10. Verificações: todas passaram.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for nome, ficheiro in FILES.items():
        sets[nome].to_parquet(out_dir / f"{ficheiro}.parquet", index=False)
    if verbose:
        print("11. Gravado em", out_dir, "(treino, validacao, teste).parquet")
    return sets


if __name__ == "__main__":
    run()
