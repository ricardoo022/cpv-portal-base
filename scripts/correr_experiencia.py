"""Corre no pod os treinos de um spike descrito num YAML (`experiencias/*.yaml`), vários de cada vez.

O YAML diz que opções se testam e por que ordem se comparam, escrito antes de correr. Este script
só lança os treinos (`src/train_tfidf.py` e `scripts/curva_epocas.py`); a decisão faz-se no
portátil, num notebook, que lê o mesmo YAML com `ler`.

Duas fases, porque a 2.ª depende da decisão da 1.ª (spike E4-S5):

- `alpha`: um treino por `alpha` (o em vigor e os candidatos), com as épocas em vigor no YAML e
  um ficheiro de sugestões por τ (o em vigor e os candidatos), para o passo 2 não pedir outro treino.
  `epocas.em_vigor: null` é a regra de paragem por defeito do scikit-learn e passa-se
  explicitamente (`--regra-por-defeito`): sem isso, o `train_tfidf.py` treinava as `tfidf.EPOCAS`
  épocas fixas escolhidas no E4-S5, sem ninguém dar por isso;
- `epocas`: com o `alpha` e o τ escolhidos (`--alpha`, `--tau`), um treino por número fixo de
  épocas e a curva época a época.

Cada treino usa `--n-jobs` processadores e correm `--paralelo` ao mesmo tempo (num pod de 32
processadores: 4 × 8). Os mais demorados começam primeiro, para o último a acabar não ficar sozinho.

Exemplos (da raiz do projeto, no pod):
    .venv/bin/python scripts/correr_experiencia.py experiencias/e4_s5.yaml alpha --paralelo 2 --n-jobs 16
    .venv/bin/python scripts/correr_experiencia.py experiencias/e4_s5.yaml epocas --alpha 1e-7 --tau 0.25
    .venv/bin/python scripts/correr_experiencia.py experiencias/e4_s5.yaml epocas --alpha 1e-7 --tau 0.25 --mostrar
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
CHAVES = {"spike", "n_min", "saida", "alpha", "tau", "epocas", "curva_epocas"}


def _numero(valor, onde):
    # No YAML 1.1 "1e-7" é texto; um número tem de vir como número.
    if isinstance(valor, bool) or not isinstance(valor, (int, float)):
        raise ValueError(f"{onde}: {valor!r} não é um número (no YAML escreve 1.0e-7, não 1e-7)")
    return float(valor)


def _inteiro_positivo(valor, onde):
    if isinstance(valor, bool) or not isinstance(valor, int) or valor < 1:
        raise ValueError(f"{onde}: {valor!r} não é um inteiro >= 1")
    return valor


def ler(path):
    """Lê e valida o YAML de uma experiência. Devolve um dict com os valores já convertidos.

    `alpha` e `tau` ficam {"em_vigor": float, "candidatos": [float]}; `epocas` fica
    {"em_vigor": None (regra por defeito) ou int, "candidatos": [int]}.
    """
    with open(path, encoding="utf-8") as f:
        exp = yaml.safe_load(f)
    if not isinstance(exp, dict) or set(exp) != CHAVES:
        faltam = CHAVES - set(exp or {})
        a_mais = set(exp or {}) - CHAVES
        raise ValueError(f"{path}: chaves em falta {sorted(faltam)}, a mais {sorted(a_mais)}")
    out = {"spike": str(exp["spike"]), "saida": str(exp["saida"]),
           "n_min": _inteiro_positivo(exp["n_min"], "n_min"),
           "curva_epocas": _inteiro_positivo(exp["curva_epocas"], "curva_epocas")}
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


def comandos(exp, fase, alpha=None, tau=None, n_jobs=-1, treino=None, validacao=None):
    """Os treinos de uma fase: lista de (nome, argumentos, épocas previstas), da mais demorada à menos.

    As épocas previstas só servem para ordenar; a regra por defeito conta como 12 (o máximo no E4-S4).
    Na fase `alpha` as épocas são as `em_vigor` do YAML (`args_epocas`).
    `treino` e `validacao` são os parquet a usar; sem eles, cada script usa os de `data/processed/`.
    """
    comum = ["--n-min", str(exp["n_min"]), "--n-jobs", str(n_jobs)]
    comum += [] if treino is None else ["--treino", str(treino)]
    comum += [] if validacao is None else ["--validacao", str(validacao)]
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
        raise ValueError(f"fase desconhecida: {fase!r} (alpha ou epocas)")
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
    p.add_argument("fase", choices=["alpha", "epocas"])
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
