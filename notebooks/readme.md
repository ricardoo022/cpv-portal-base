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
| [`06_tfidf_epocas.ipynb`](06_tfidf_epocas.ipynb) | Um `alpha` mais pequeno e quantas épocas treinar o TF-IDF (E4-S5), com a curva do top-5 época a época. Lê as opções de `experiencias/e4_s5.yaml`, as sugestões dos treinos no pod (`metrics/e4_s5/`), o treino e a validação. | secção 15 |
| [`07_tfidf_grelha.ipynb`](07_tfidf_grelha.ipynb) | Que combinação de N, épocas e τ usar no TF-IDF (E4-S1), numa grelha. Confirma que a célula N = 10/24 épocas reproduz o E4-S5. Lê as opções de `experiencias/e4_s1.yaml`, as sugestões dos treinos no pod (`metrics/e4_s1/`), o treino e a validação. | secção 16 |
| [`08_entidade.ipynb`](08_entidade.ipynb) | Se juntar a entidade compradora (`adjudicante`) ao texto ajuda o TF-IDF (E4-S3), em três grupos: todos, entidades que estão no treino e entidades novas. Lê as opções e a regra de `experiencias/e4_s3.yaml`, as sugestões sem entidade (`metrics/e4_s1/`) e com entidade (`metrics/e4_s3/`), o treino e a validação. | secção 17, decisão 7 |
| [`09_janela.ipynb`](09_janela.ipynb) | Que janela de treino usar no TF-IDF (E4-S2): desde 2025, 2022 ou 2019, decidido nos contratos com texto novo para o treino desde 2019, mais a análise completa com todas as medidas dos spikes anteriores. Lê as opções de `experiencias/e4_s2.yaml`, as sugestões (`metrics/e4_s1/` e `metrics/e4_s2/`), a validação, o treino desde 2022 e, de `data/janelas/`, o treino desde 2019 e as validações das janelas (copiados do pod). | secção 18, decisão 1 |
| [`10_janela_24_epocas.ipynb`](10_janela_24_epocas.ipynb) | Se a janela desde 2019 também ganha com as 24 épocas do modelo em vigor (E4-S6). É o 09 com outro YAML: lê `experiencias/e4_s6.yaml`, as sugestões (`metrics/e4_s1/` e `metrics/e4_s6/`), a validação, o treino desde 2022 e, de `data/janelas/2019/`, o treino e a validação desde 2019. | secção 19, decisão 1 |
| [`11_comprimento_tokens.ipynb`](11_comprimento_tokens.ipynb) | Quantos *tokens* tem cada texto no BERTimbau e no Albertina, e que comprimento máximo usar no treino (E5-T1). Lê o texto do treino e da validação e descarrega só os tokenizadores (os nomes e a revisão vêm de `src/bert_models.py`). | secção 20 |

Voltar a correr o 01: `make notebook`. O 03: `make regua`. O 04: `make tfidf` (cerca de 5 minutos, quase tudo no bootstrap). O 05: `make afinacao`. O 06: `make epocas`. O 07: `make grelha`. O 08: `make entidade` (cerca de 3 minutos, quase tudo no bootstrap). O 09: `make janela`. O 10: `make janela-24`. O 11: `make tokens` (cerca de 3 minutos).
