"""Corre no pod os treinos de um spike descrito num YAML (`experiencias/*.yaml`), vários de cada vez.

O YAML diz que opções se testam e por que ordem se comparam, escrito antes de correr. Este script
só lança os treinos (`src/train_tfidf.py` e `scripts/curva_epocas.py`); a decisão faz-se no
portátil, num notebook, que lê o mesmo YAML com `ler`.

Três modos, conforme o YAML:

- **entidade** (`modo: entidade`; spike E4-S3): uma só variante do texto de entrada (com ou sem a
  `adjudicante`) e a forma de comparar. Não se corre por este script: é um único comando
  (`src/train_tfidf.py --com-entidade`) e a decisão lê o YAML com `ler`, como os outros.
- **fases** (por defeito; spike E4-S5), com duas fases, porque a 2.ª depende da decisão da 1.ª:
  - `alpha`: um treino por `alpha` (o em vigor e os candidatos), com as épocas em vigor no YAML e
    um ficheiro de sugestões por τ (o em vigor e os candidatos), para o passo 2 não pedir outro treino.
    `epocas.em_vigor: null` é a regra de paragem por defeito do scikit-learn e passa-se
    explicitamente (`--regra-por-defeito`): sem isso, o `train_tfidf.py` treinava as `tfidf.EPOCAS`
    épocas fixas escolhidas no E4-S5, sem ninguém dar por isso;
  - `epocas`: com o `alpha` e o τ escolhidos (`--alpha`, `--tau`), um treino por número fixo de
    épocas e a curva época a época.
- **grelha** (`modo: grelha`; spike E4-S1), uma só fase `grelha`: um treino por combinação de N e
  épocas (o produto cartesiano), cada um com `--taus` a lista de τ do YAML. O τ não retreina o
  modelo — só muda as sugestões ao sugerir — por isso as combinações com τ saem todas do mesmo treino.

Cada treino usa `--n-jobs` processadores e correm `--paralelo` ao mesmo tempo (num pod de 32
processadores: 4 × 8). Os mais demorados começam primeiro, para o último a acabar não ficar sozinho.

Exemplos (da raiz do projeto, no pod):
    .venv/bin/python scripts/correr_experiencia.py experiencias/e4_s5.yaml alpha --paralelo 2 --n-jobs 16
    .venv/bin/python scripts/correr_experiencia.py experiencias/e4_s5.yaml epocas --alpha 1e-7 --tau 0.25
    .venv/bin/python scripts/correr_experiencia.py experiencias/e4_s1.yaml grelha --paralelo 4 --n-jobs 8
    .venv/bin/python scripts/correr_experiencia.py experiencias/e4_s1.yaml grelha --mostrar
"""

import argparse
import datetime as dt
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parents[1]
PY = sys.executable
OBRIGATORIAS_FASES = {"spike", "saida", "n_min", "alpha", "tau", "epocas", "curva_epocas"}
OBRIGATORIAS_GRELHA = {"spike", "saida", "n_min", "alpha", "tau", "epocas"}
OBRIGATORIAS_ENTIDADE = {"spike", "saida", "n_min", "alpha", "epocas", "tau", "com_entidade", "grupos"}
MODOS = ("fases", "grelha", "entidade")
OBRIGATORIAS = {"fases": OBRIGATORIAS_FASES, "grelha": OBRIGATORIAS_GRELHA,
                "entidade": OBRIGATORIAS_ENTIDADE}


def _numero(valor, onde):
    # No YAML 1.1 "1e-7" é texto; um número tem de vir como número.
    if isinstance(valor, bool) or not isinstance(valor, (int, float)):
        raise ValueError(f"{onde}: {valor!r} não é um número (no YAML escreve 1.0e-7, não 1e-7)")
    return float(valor)


def _inteiro_positivo(valor, onde):
    if isinstance(valor, bool) or not isinstance(valor, int) or valor < 1:
        raise ValueError(f"{onde}: {valor!r} não é um inteiro >= 1")
    return valor


def _lista(valor, onde):
    """Uma lista não vazia de valores distintos (o eixo de uma grelha)."""
    if isinstance(valor, (str, bytes)) or not isinstance(valor, (list, tuple)) or not valor:
        raise ValueError(f"{onde}: tem de ser uma lista não vazia de valores: {valor!r}")
    if len(set(valor)) != len(valor):
        raise ValueError(f"{onde}: há valores repetidos: {list(valor)}")
    return list(valor)


def ler(path):
    """Lê e valida o YAML de uma experiência (modo `fases`, `grelha` ou `entidade`).

    `fases` (por defeito, ver `e4_s5.yaml`): dois passos, cada eixo testado na forma
    {"em_vigor": valor, "candidatos": [...]}, com `curva_epocas` obrigatório.
    `grelha` (ver `e4_s1.yaml`): cada eixo é uma lista de valores e testa-se o produto cartesiano;
    `alpha` é um número fixo (não uma lista) e `curva_epocas` não existe. O `modo` é opcional e,
    sem ele, é `fases`.
    `entidade` (ver `e4_s3.yaml`): uma só variante do texto, com o `n_min`, o `alpha`, as `epocas` e
    os `tau` fixos, o mapa `com_entidade` (nome -> pasta de cada opção) e os `grupos` da comparação.
    """
    with open(path, encoding="utf-8") as f:
        exp = yaml.safe_load(f)
    if not isinstance(exp, dict):
        raise ValueError(f"{path}: o YAML tem de ser um mapa de chaves")
    modo = exp.get("modo", "fases")
    if modo not in MODOS:
        raise ValueError(f"{path}: `modo` tem de ser um de {MODOS}, não {modo!r}")
    obrigatorias = OBRIGATORIAS[modo]
    faltam = obrigatorias - set(exp)
    a_mais = set(exp) - (obrigatorias | {"modo"})
    if faltam or a_mais:
        raise ValueError(f"{path}: chaves em falta {sorted(faltam)}, a mais {sorted(a_mais)}")
    out = {"spike": str(exp["spike"]), "saida": str(exp["saida"]), "modo": modo}
    if modo == "entidade":
        out["n_min"] = _inteiro_positivo(exp["n_min"], "n_min")
        out["alpha"] = _numero(exp["alpha"], "alpha")
        if out["alpha"] <= 0:
            raise ValueError("alpha tem de ser > 0")
        ep = exp["epocas"]
        out["epocas"] = None if ep is None else _inteiro_positivo(ep, "epocas")
        out["tau"] = [_numero(v, "tau") for v in _lista(exp["tau"], "tau")]
        if any(t < 0 for t in out["tau"]):
            raise ValueError("τ tem de ser >= 0")
        opcoes = exp["com_entidade"]
        if not isinstance(opcoes, dict) or len(opcoes) < 2:
            raise ValueError(f"com_entidade: mapa de duas opções (nome -> pasta): {opcoes!r}")
        out["com_entidade"] = {}
        for nome, pasta in opcoes.items():
            if not isinstance(nome, str) or not nome or not isinstance(pasta, str) or not pasta:
                raise ValueError(f"com_entidade: nome e pasta têm de ser texto não vazio: {nome!r}: {pasta!r}")
            out["com_entidade"][nome] = pasta
        grupos = _lista(exp["grupos"], "grupos")
        if not all(isinstance(g, str) and g for g in grupos):
            raise ValueError(f"grupos: nomes de grupo (texto não vazio): {grupos!r}")
        out["grupos"] = grupos
        return out
    if modo == "grelha":
        out["alpha"] = _numero(exp["alpha"], "alpha")
        if out["alpha"] <= 0:
            raise ValueError("alpha tem de ser > 0")
        out["n_min"] = [_inteiro_positivo(v, "n_min") for v in _lista(exp["n_min"], "n_min")]
        out["tau"] = [_numero(v, "tau") for v in _lista(exp["tau"], "tau")]
        if any(t < 0 for t in out["tau"]):
            raise ValueError("τ tem de ser >= 0")
        out["epocas"] = [None if v is None else _inteiro_positivo(v, "epocas")
                         for v in _lista(exp["epocas"], "epocas")]
        return out
    out["n_min"] = _inteiro_positivo(exp["n_min"], "n_min")
    out["curva_epocas"] = _inteiro_positivo(exp["curva_epocas"], "curva_epocas")
    for nome in ("alpha", "tau"):
        bloco = exp[nome]
        out[nome] = {"em_vigor": _numero(bloco["em_vigor"], f"{nome}.em_vigor"),
                     "candidatos": [_numero(v, f"{nome}.candidatos") for v in bloco["candidatos"]]}
    em_vigor = exp["epocas"]["em_vigor"]
    out["epocas"] = {"em_vigor": None if em_vigor is None else _inteiro_positivo(em_vigor, "epocas.em_vigor"),
                     "candidatos": [_inteiro_positivo(v, "epocas.candidatos") for v in exp["epocas"]["candidatos"]]}
    for nome in ("alpha", "tau", "epocas"):
        todos = [out[nome]["em_vigor"], *out[nome]["candidatos"]]
        if len(set(todos)) != len(todos):
            raise ValueError(f"{nome}: há opções repetidas: {todos}")
    if any(a <= 0 for a in [out["alpha"]["em_vigor"], *out["alpha"]["candidatos"]]):
        raise ValueError("alpha tem de ser > 0")
    if any(t < 0 for t in [out["tau"]["em_vigor"], *out["tau"]["candidatos"]]):
        raise ValueError("τ tem de ser >= 0")
    return out


def args_epocas(epocas):
    """Argumentos do `train_tfidf.py` para as épocas: None é a regra por defeito do scikit-learn."""
    return ["--regra-por-defeito"] if epocas is None else ["--epocas", str(epocas)]


def pasta_alpha(exp, alpha):
    """Pasta dos resultados do treino com este `alpha` e as épocas em vigor."""
    return Path(exp["saida"]) / f"alpha_{alpha:g}"


def pasta_epocas(exp, epocas):
    """Pasta dos resultados do treino com este número fixo de épocas."""
    return Path(exp["saida"]) / f"epocas_{epocas:02d}"


def pasta_curva(exp):
    return Path(exp["saida"]) / "curva"


def pasta_grelha(exp, n_min, epocas):
    """Pasta de um treino da grelha: N e épocas (`defeito` para a regra por defeito)."""
    ep = "defeito" if epocas is None else f"{epocas:02d}"
    return Path(exp["saida"]) / f"n_min_{n_min:02d}_ep_{ep}"


def comandos(exp, fase, alpha=None, tau=None, n_jobs=-1, treino=None, validacao=None):
    """Os treinos de uma fase: lista de (nome, argumentos, épocas previstas), da mais demorada à menos.

    As épocas previstas só servem para ordenar; a regra por defeito conta como 12 (o máximo no E4-S4).
    Na fase `alpha` as épocas são as `em_vigor` do YAML (`args_epocas`).
    Uma experiência em `grelha` só tem a fase `grelha`: o produto cartesiano de N × épocas, com
    `--taus` a lista de τ do YAML (o τ não retreina, só muda as sugestões ao sugerir).
    `treino` e `validacao` são os parquet a usar; sem eles, cada script usa os de `data/processed/`.
    """
    if exp.get("modo") == "entidade":
        raise ValueError("a experiência da entidade treina-se com um só comando "
                         "(src/train_tfidf.py --com-entidade); este script não a corre")
    comum = ["--n-jobs", str(n_jobs)]
    comum += [] if treino is None else ["--treino", str(treino)]
    comum += [] if validacao is None else ["--validacao", str(validacao)]
    if exp.get("modo") == "grelha":
        if fase != "grelha":
            raise ValueError(f"experiência em grelha: a fase tem de ser 'grelha', não {fase!r}")
        taus = [f"{t:g}" for t in exp["tau"]]
        out = [(f"n_min_{n:02d}_ep_{'defeito' if k is None else f'{k:02d}'}",
                [PY, "src/train_tfidf.py", "--n-min", str(n), "--alpha", f"{exp['alpha']:g}",
                 "--taus", *taus, *args_epocas(k), "--sem-modelo", *comum,
                 "--saida", str(pasta_grelha(exp, n, k))], 12 if k is None else k)
               for n in exp["n_min"] for k in exp["epocas"]]
        return sorted(out, key=lambda c: -c[2])
    comum = ["--n-min", str(exp["n_min"]), *comum]
    if fase == "alpha":
        taus = [exp["tau"]["em_vigor"], *exp["tau"]["candidatos"]]
        epocas = exp["epocas"]["em_vigor"]
        out = [(f"alpha_{a:g}",
                [PY, "src/train_tfidf.py", "--alpha", f"{a:g}", "--taus", *[f"{t:g}" for t in taus],
                 *args_epocas(epocas), "--sem-modelo", *comum, "--saida", str(pasta_alpha(exp, a))],
                12 if epocas is None else epocas)
               for a in [exp["alpha"]["em_vigor"], *exp["alpha"]["candidatos"]]]
    elif fase == "epocas":
        if alpha is None or tau is None:
            raise ValueError("a fase das épocas precisa do alpha e do τ escolhidos (--alpha, --tau)")
        taus = ["0"] if tau == 0 else ["0", f"{tau:g}"]
        out = [(f"epocas_{k:02d}",
                [PY, "src/train_tfidf.py", "--alpha", f"{alpha:g}", "--taus", *taus, *args_epocas(k),
                 "--sem-modelo", *comum, "--saida", str(pasta_epocas(exp, k))], k)
               for k in exp["epocas"]["candidatos"]]
        # A curva também sugere para a validação em cada época: conta como o dobro.
        out.append(("curva",
                    [PY, "scripts/curva_epocas.py", "--alpha", f"{alpha:g}", "--tau", f"{tau:g}",
                     "--epocas", str(exp["curva_epocas"]), *comum, "--saida", str(pasta_curva(exp))],
                    2 * exp["curva_epocas"]))
    else:
        raise ValueError(f"fase desconhecida: {fase!r} (alpha, epocas ou grelha)")
    return sorted(out, key=lambda c: -c[2])


def _correr(nome, args, log_dir):
    log = Path(log_dir) / f"pod_{nome}.log"
    with open(log, "w", encoding="utf-8") as f:
        f.write(f"INICIO {nome} {dt.datetime.now(dt.timezone.utc):%H:%M:%S}\n{' '.join(args)}\n")
        f.flush()
        codigo = subprocess.run(args, cwd=RAIZ, stdout=f, stderr=subprocess.STDOUT).returncode
        f.write(f"FIM {nome} {dt.datetime.now(dt.timezone.utc):%H:%M:%S} codigo={codigo}\n")
    return nome, codigo


def main(argv=None):
    p = argparse.ArgumentParser(description="Corre os treinos de uma fase de um spike descrito num YAML.")
    p.add_argument("experiencia", help="o YAML da experiência (experiencias/*.yaml)")
    p.add_argument("fase", choices=["alpha", "epocas", "grelha"])
    p.add_argument("--alpha", type=float, help="fase epocas: o alpha escolhido")
    p.add_argument("--tau", type=float, help="fase epocas: o τ escolhido")
    p.add_argument("--paralelo", type=int, default=4, help="treinos ao mesmo tempo")
    p.add_argument("--n-jobs", type=int, default=8, help="processadores de cada treino")
    p.add_argument("--treino", help="treino.parquet (por defeito o de data/processed/)")
    p.add_argument("--validacao", help="validacao.parquet (por defeito o de data/processed/)")
    p.add_argument("--mostrar", action="store_true", help="só mostra os comandos, não corre nada")
    a = p.parse_args(argv)
    exp = ler(a.experiencia)
    lista = comandos(exp, a.fase, a.alpha, a.tau, a.n_jobs, a.treino, a.validacao)
    for nome, args, _ in lista:
        print(nome, ":", " ".join(args), flush=True)
    if a.mostrar:
        return
    log_dir = RAIZ / exp["saida"]
    log_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=a.paralelo) as pool:
        resultados = list(pool.map(lambda c: _correr(c[0], c[1], log_dir), lista))
    falhou = [n for n, c in resultados if c != 0]
    print("falharam:", falhou if falhou else "nenhum", flush=True)
    sys.exit(1 if falhou else 0)


if __name__ == "__main__":
    main()
