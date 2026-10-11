"""Comando de sugestões (E4-H1): 5 códigos CPV para cada descrição escrita à mão.

Lê um modelo gravado pelo `train_tfidf.py` e mostra, para cada descrição, os 5 códigos por
ordem, com a probabilidade que o modelo dá a cada um. A ordem usa a correção das etiquetas
frequentes com o τ escolhido no E4-S4 (`tfidf.TAU`), por isso não é sempre a das probabilidades:
uma etiqueta muito frequente pode ter mais probabilidade e ficar abaixo de uma rara.

O ficheiro do modelo só existe no pod da Runpod: tem cerca de 4,8 GB (E4-S0) e não cabe na
memória do portátil (5,8 GB partilhados com o editor), por isso este comando corre no pod, a
seguir ao treino.

Exemplos (da raiz do projeto):
    .venv/bin/python src/suggest.py --modelo models/tfidf_n10.joblib
    .venv/bin/python src/suggest.py --modelo models/tfidf_n10.joblib "aquisição de papel A4"
Sem descrições, usa as 3 de `EXEMPLOS`.
"""

import argparse

import joblib
import numpy as np

import tfidf

EXEMPLOS = [
    "serviços de limpeza de três escolas básicas durante 12 meses",
    "fornecimento de refeições escolares para os alunos do 1.º ciclo do concelho",
    "empreitada de repavimentação de estradas municipais",
]


def format_suggestions(model, textos, casas=3, tau=tfidf.TAU):
    """Texto com cada descrição e as suas 5 sugestões numeradas, com a probabilidade."""
    textos = list(textos)
    sugestoes = tfidf.suggest(model, textos, tau=tau)
    proba = model.predict_proba(textos)
    coluna = {c: i for i, c in enumerate(model.classes_)}
    linhas = []
    for i, (texto, codigos) in enumerate(zip(textos, sugestoes)):
        linhas.append(f"Descrição: {texto}")
        for pos, codigo in enumerate(codigos, start=1):
            p = float(np.round(proba[i, coluna[codigo]], casas))
            linhas.append(f"  {pos}. {codigo}  (probabilidade {p:.{casas}f})")
        linhas.append("")
    return "\n".join(linhas)


def main(argv=None):
    p = argparse.ArgumentParser(description="Sugere 5 códigos CPV para cada descrição.")
    p.add_argument("--modelo", required=True, help="modelo gravado pelo train_tfidf.py (.joblib)")
    p.add_argument("--tau", type=float, default=tfidf.TAU, help="correção das etiquetas frequentes (0 = sem)")
    p.add_argument("descricoes", nargs="*", help="descrições; sem nenhuma usa 3 exemplos")
    a = p.parse_args(argv)
    model = joblib.load(a.modelo)
    print(format_suggestions(model, a.descricoes or EXEMPLOS, tau=a.tau))


if __name__ == "__main__":
    main()
