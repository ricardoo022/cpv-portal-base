# notebooks/ · exploração dos dados

Medições e análises que levaram às decisões do [`docs/qualidade_dados.md`](../docs/qualidade_dados.md). Cada número escrito nos docs sai de uma célula daqui.

Correm com `notebooks/` como pasta de trabalho (acrescentam `../src` ao caminho do Python). Nunca usam o teste (contratos de novembro e dezembro de 2025) para tomar decisões.

| Notebook | Pergunta | Resultados em |
|---|---|---|
| [`00_qualidade.ipynb`](00_qualidade.ipynb) | Os dados são usáveis? Campos vazios, contratos repetidos, formatos estranhos. | secções 1 a 6 do `qualidade_dados.md` |
| [`01_exploracao.ipynb`](01_exploracao.ipynb) | Que colunas ajudam, se o mesmo texto tem sempre o mesmo código, a que nível se escolhem os códigos, quantos exemplos tem cada código. Lê o `treino.parquet`, por isso precisa de `make preparar` antes. | secções 7 a 10 |
| [`02_anos_treino.ipynb`](02_anos_treino.ipynb) | Quantos anos usar no treino. | decisão 1 |

Voltar a correr o 01: `make notebook`.
