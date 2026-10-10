"""Fase 3: aplica as decisões de docs/qualidade_dados.md e cria treino, validação e teste.

Lê data/processed/raw/ (todos os ficheiros) e grava em data/processed/:
    treino.parquet, validacao.parquet, teste.parquet

Para com erro se alguma das verificações no fim falhar.

Uso:
    .venv/bin/python src/prepare.py
"""

import pandas as pd

import cpv_data as cd

OUT_DIR = cd.ROOT / "data" / "processed"
SEP = " | "  # separador entre as partes do texto de entrada
RAW_COLS = ["idcontrato", "dataPublicacao", "objectoContrato", "descContrato", "DescrAcordoQuadro",
            "CPV", "adjudicante", "tipoContrato", "tipoprocedimento"]
KEEP = ["idcontrato", "data", "texto", "codigo", "codigos", "nivel",
        "adjudicante", "tipoContrato", "tipoprocedimento"]


def log(msg, n):
    print(f"  {msg:<64}{n:>10,}".replace(",", " "))


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


def load_one(year, counts):
    """Passos 1 a 5 para um ficheiro, deitando fora as colunas originais no fim."""
    raw = pd.read_parquet(cd.RAW_PARQUET / f"contratos{year}.parquet", columns=RAW_COLS)
    raw["data"] = pd.to_datetime(raw["dataPublicacao"], errors="coerce")
    raw = raw[(raw["data"] >= cd.TRAIN_START) & (raw["data"] < cd.YEAR_END)]
    counts["lidos"] += len(raw)
    codigos = cd.all_codes(raw["CPV"])
    raw = raw[codigos.map(len) > 0].copy()
    codigos = codigos[raw.index]
    out = pd.DataFrame({
        "idcontrato": raw["idcontrato"],
        "data": raw["data"],
        "ficheiro": year,
        "texto": build_text(raw),
        "codigo": codigos.str[0],
        "codigos": codigos.str.join(" "),  # todos os códigos, separados por espaço
        "adjudicante": raw["adjudicante"],
        "tipoContrato": raw["tipoContrato"],
        "tipoprocedimento": raw["tipoprocedimento"],
    })
    return out


def main():
    counts = {"lidos": 0}
    df = pd.concat([load_one(y, counts) for y in cd.ALL_YEARS], ignore_index=True)
    for c in ["adjudicante", "tipoContrato", "tipoprocedimento"]:
        df[c] = df[c].astype("category")

    print("Funil (contratos que ficam em cada passo):")
    log("1. publicados de jan 2022 a dez 2025 (todos os ficheiros)", counts["lidos"])
    log("3. com pelo menos um código CPV", len(df))
    df = df.sort_values(["ficheiro", "data"]).drop_duplicates("idcontrato", keep="last")
    log("2. uma cópia por idcontrato", len(df))

    df["nivel"] = cd.code_level(df["codigo"])
    df["periodo"] = cd.period(df["data"])
    df["chave"] = text_key(df["texto"])

    # --- Treino ---
    tr = df[df["periodo"] == "treino"]
    print("\nTreino (jan 2022 a set 2025):")
    log("contratos", len(tr))
    tr = tr[tr["nivel"] > 2]
    log("7. sem códigos ao nível da divisão", len(tr))
    pares = tr.groupby(["chave", "codigo"]).size().rename("n").reset_index()
    pares = pares.sort_values(["chave", "n", "codigo"], ascending=[True, False, True])
    melhor = pares.drop_duplicates("chave").set_index("chave")["codigo"]
    total = pares.groupby("chave")["n"].sum()
    tr = tr[tr["codigo"].to_numpy() == melhor.reindex(tr["chave"]).to_numpy()]
    log("8a. contratos que seguem o código mais comum do seu texto", len(tr))
    tr = tr.sort_values("data").drop_duplicates("chave", keep="last").copy()
    tr["n_contratos"] = total.reindex(tr["chave"]).to_numpy()
    log("8b. exemplos de treino (um por texto)", len(tr))
    treino = tr[KEEP + ["n_contratos"]].reset_index(drop=True)

    # --- Validação e teste: ficam todos ---
    chaves_treino = set(tr["chave"])
    sets = {"treino": treino}
    for nome in ["validação", "teste"]:
        part = df[df["periodo"] == nome].copy()
        part["visto_no_treino"] = part["chave"].isin(chaves_treino)
        sets[nome] = part[KEEP + ["visto_no_treino"]].reset_index(drop=True)
        print(f"\n{nome.capitalize()}:")
        log("contratos", len(part))
        log("9. com texto que já aparece no treino", int(part["visto_no_treino"].sum()))
        log("   com código ao nível da divisão (avaliados só na divisão)", int((part["nivel"] == 2).sum()))

    # --- 10. Verificações: se alguma falhar, o script para ---
    t, v, te = sets["treino"], sets["validação"], sets["teste"]
    assert t["data"].min() >= cd.TRAIN_START and t["data"].max() < cd.VAL_START, "treino fora do período"
    assert v["data"].min() >= cd.VAL_START and v["data"].max() < cd.TEST_START, "validação fora do período"
    assert te["data"].min() >= cd.TEST_START and te["data"].max() < cd.YEAR_END, "teste fora do período"
    ids = [set(x["idcontrato"]) for x in (t, v, te)]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]), "contrato em mais do que um conjunto"
    assert all(x["idcontrato"].is_unique for x in (t, v, te)), "idcontrato repetido"
    assert (t["nivel"] > 2).all(), "código ao nível da divisão no treino"
    assert pd.Series(text_key(t["texto"])).is_unique, "texto repetido no treino"
    assert all(x["codigo"].str.fullmatch(r"\d{8}").all() for x in (t, v, te)), "código mal formado"
    assert all((x["texto"].str.strip() != "").all() for x in (t, v, te)), "texto vazio"
    print("\n10. Verificações: todas passaram.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for nome, ficheiro in [("treino", "treino"), ("validação", "validacao"), ("teste", "teste")]:
        sets[nome].to_parquet(OUT_DIR / f"{ficheiro}.parquet", index=False)
    print("11. Gravado em", OUT_DIR.relative_to(cd.ROOT), "(treino, validacao, teste).parquet")


if __name__ == "__main__":
    main()
