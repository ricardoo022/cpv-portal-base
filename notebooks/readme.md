# notebooks/ · exploração dos dados

Medições e análises que levaram às decisões do [`docs/qualidade_dados.md`](../docs/qualidade_dados.md). Cada número escrito nos docs sai de uma célula daqui.

Correm com `notebooks/` como pasta de trabalho (acrescentam `../src` ao caminho do Python). Nunca usam o teste (contratos de novembro e dezembro de 2025) para tomar decisões.

| Notebook | Pergunta | Resultados em |
|---|---|---|
| [`00_qualidade.ipynb`](00_qualidade.ipynb) | Os dados são usáveis? Campos vazios, contratos repetidos, formatos estranhos. | secções 1 a 6 do `qualidade_dados.md` |
| [`01_exploracao.ipynb`](01_exploracao.ipynb) | Que colunas ajudam, se o mesmo texto tem sempre o mesmo código, a que nível se escolhem os códigos, quantos exemplos tem cada código. Lê o `treino.parquet`, por isso precisa de `make preparar` antes. | secções 7 a 10 |
| [`02_anos_treino.ipynb`](02_anos_treino.ipynb) | Quantos anos usar no treino. | decisão 1 |
| [`03_regua.ipynb`](03_regua.ipynb) | Quanto acerta a régua "códigos mais comuns" na validação (E4-T2). Lê o treino e a validação, por isso precisa de `make preparar` antes. | secção 12 |
| [`04_tfidf.ipynb`](04_tfidf.ipynb) | Quanto acerta o TF-IDF na validação, comparado com a régua (E4-H1). Lê as sugestões gravadas pelo treino no pod (`metrics/e4_h1/`), o treino e a validação. | secção 13 |
| [`05_tfidf_afinacao.ipynb`](05_tfidf_afinacao.ipynb) | Que `alpha` (regularização) e que τ (correção das etiquetas frequentes) usar no TF-IDF (E4-S4). Lê as sugestões dos treinos no pod (`metrics/e4_s4/`), o treino e a validação. | secção 14 |

Voltar a correr o 01: `make notebook`. O 03: `make regua`. O 04: `make tfidf` (cerca de 5 minutos, quase tudo no bootstrap). O 05: `make afinacao`.
