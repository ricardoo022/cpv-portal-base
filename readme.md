# cpv-portal-base

Modelo que sugere o código CPV de um contrato público a partir da descrição da compra, treinado com os contratos reais do Portal BASE.

Exemplo: entra "serviços de limpeza de três escolas básicas durante 12 meses", sai uma lista ordenada de códigos CPV (por exemplo `90919300-5` Serviços de limpeza de escolas, `90911200-8` Serviços de limpeza de edifícios, ...), para o técnico escolher em vez de procurar na tabela.

## O problema

O problema é pequeno, e convém dizê-lo com clareza.

- O CPV (Vocabulário Comum para os Contratos Públicos) é a tabela da UE que classifica o que se compra. Tem cerca de 9 500 códigos em árvore: 45 divisões (2 primeiros dígitos), depois grupos, classes e categorias.
- Quem prepara um procedimento escolhe o código à mão. Com tantos códigos parecidos, erra-se ou escolhe-se um genérico.
- O que resolve: poupa ao técnico uns minutos a procurar o código certo e reduz os códigos mal escolhidos.
- Porque um CPV errado tem custo:
  - Menos concorrência. As empresas procuram concursos por CPV. Com o código errado, quem podia concorrer não vê o concurso.
  - Regras mal aplicadas. Alguns limites e regimes legais dependem do tipo de compra.
  - Estatísticas erradas sobre o que o Estado compra.
- Ninguém nos disse que isto dói. Chegámos aqui a partir dos dados disponíveis, não a partir de uma queixa de um técnico.

Neste momento só queremos resolver isto: sugerir o CPV certo a partir do texto.

## Casos que já existem noutros países

Há evidência de que escolher o CPV é mesmo um problema, mas nenhuma fonte tem números de tempo perdido.

1. **Zwolle, Países Baixos.** A câmara encomendou à Haskoning, em 2022 e 2023, um modelo que prevê o CPV a partir do texto do concurso.
   - A ferramenta mostra os 5 códigos mais prováveis, para o técnico escolher entre 5 em vez de milhares.
   - Chegou a cerca de 76% de accuracy nas 45 divisões.
   - A conclusão foi que ainda não era boa o suficiente para uso real, por causa dos níveis mais detalhados.
   - É uma entidade pública que pagou para resolver isto.
2. **Estudo italiano, 5 milhões de concursos.** Diz que é difícil até para especialistas identificar o CPV certo e que o número de códigos leva a erros frequentes. Testou SVM com TF-IDF, BERT afinado e IT5 afinado nas 45 divisões (resultados em [Réguas](#réguas)).
3. **Artigo sobre qualidade de dados na contratação pública europeia.** Diz que os concursos são classificados com erro ou confusão com frequência. Exemplos: um arrendamento de escritórios marcado como obra de construção, e um anúncio com 83 códigos CPV.

O que isto nos diz:

- O problema existe e há quem pague para o resolver.
- Já foi feito em italiano e neerlandês. Em português não encontrámos nada.
- Já há régua: cerca de 75% a 79% de accuracy nas divisões.
- O modelo clássico é forte. No estudo italiano o SVM bateu o BERT no F1 macro. Tem de entrar como referência, e é possível que os modelos tunados não o batam.
- A accuracy engana. Com 78% de accuracy, o F1 macro ficou em 0,40: os modelos acertam nas divisões grandes e falham nas pequenas.
- Em Zwolle o que falhou foram os níveis detalhados, que é onde o problema está. É aí que este projeto tem de medir, com top-5 como na ferramenta deles.

## Objetivo deste projeto

1. Aprender o workflow completo de treino de um modelo: dados, split, baseline, fine-tuning, validação, teste e análise de erros.
2. Ter um caso com dados reais portugueses para o portefólio.

## A tarefa, em termos de código

- Entrada: `objectoContrato`, mais a `descContrato` quando é diferente, mais a `DescrAcordoQuadro` quando existe. É informação que o técnico tem quando escolhe o código. A entidade (`adjudicante`) entra só como experiência.
- Saída: lista ordenada de códigos CPV (top-5).
- Etiqueta: o primeiro código do `CPV` que cada entidade escolheu. No treino tiram-se os códigos ao nível da divisão, e os códigos raros sobem para a classe ou o grupo (decisões em [`docs/qualidade_dados.md`](docs/qualidade_dados.md)).
- Avaliação ao nível em que o problema existe (`src/evaluation.py`):
  - Principal: top-1 e top-5 no código completo (o código exato) e na classe (4 dígitos). Os contratos que só têm um código de divisão não contam no código completo (o modelo não aprende divisões) e mostram-se à parte.
  - Secundária: divisão (2 dígitos). É fácil e não poupa trabalho a ninguém, mas permite comparar com as réguas abaixo.
  - Sempre F1 macro, porque as classes são muito desequilibradas.
  - Só para mostrar: precisão, recall e F1 hierárquicos, que dão meio acerto a uma sugestão perto do código certo.
  - Para decidir entre dois modelos: bootstrap emparelhado na validação (`src/comparison.py`).

## Dados

- Fonte: "Contratos Públicos - Portal Base - IMPIC - Contratos de 2012 a 2026", no dados.gov.pt.
  https://dados.gov.pt/pt/datasets/contratos-publicos-portal-base-impic-contratos-de-2012-a-2026/
- Licença indicada na página: domínio público. Atualizado de 15 em 15 dias.
- Um ficheiro .xlsx por ano, de 2012 a 2026, entre 16 MB e 67 MB cada (542 MB no total). Uma folha (`Sheet1`) e as mesmas 39 colunas em todos.
- Ficam em `data/raw/`, que não vai para o git.
- Para descarregar ou atualizar: `.venv/bin/python scripts/download_data.py` (ou só alguns anos: `.venv/bin/python scripts/download_data.py 2025`).

Contratos por ficheiro (versão de 4 de outubro de 2026):

| Ano | Contratos | Ano | Contratos |
|---|---|---|---|
| 2012 | 64 783 | 2020 | 176 510 |
| 2013 | 83 217 | 2021 | 187 417 |
| 2014 | 102 516 | 2022 | 176 598 |
| 2015 | 107 636 | 2023 | 193 999 |
| 2016 | 114 897 | 2024 | 227 348 |
| 2017 | 128 583 | 2025 | 247 720 |
| 2018 | 127 891 | 2026 | 186 118 (até 4 de outubro) |
| 2019 | 145 513 | | |

As 39 colunas, o que é cada uma e se existe quando o técnico escolhe o código: [`docs/dados.md`](docs/dados.md).

Avisos sobre os dados:

- O `CPV` vem com a descrição colada (`90611000-3 - Serviços de limpeza de ruas`). É preciso extrair o código.
- Uma célula de `CPV` pode ter vários códigos separados por quebra de linha. Treinar com o primeiro; na avaliação, contar como acerto se a sugestão estiver entre os códigos do contrato.
- O ficheiro de um ano (e a coluna `Ano`) não é o ano de publicação: parece ser o ano de celebração do contrato. Dos contratos publicados em 2025, 225 106 estão no ficheiro de 2025, 25 877 no de 2024 e 2 930 no de 2023. Fazer o split pela `dataPublicacao`, lendo todos os ficheiros.
- Dentro do .xlsx as datas são números de série do Excel (45510 = 6 de agosto de 2024). O `src/convert_to_parquet.py` já as converte para datas normais (`2025-01-02T00:00:00`).
- Há textos repetidos entre anos e entidades (por exemplo a mesma limpeza de bermas em 2016, 2022, 2023 e 2024). No treino juntam-se num só exemplo; na validação e no teste ficam todos, marcados, para medir à parte se o modelo aprendeu ou decorou.
- Quase não há campos vazios no texto e no CPV, mas cerca de 18% dos contratos têm um código genérico, ao nível da divisão (por exemplo `33000000`). Resultados das verificações em [`docs/qualidade_dados.md`](docs/qualidade_dados.md).
- As etiquetas são o que cada entidade escolheu, e algumas escolheram mal. O modelo aprende também esses erros, e parte dos "erros" do modelo serão acertos. Por isso a análise de erros é obrigatória.
- Ler um .xlsx de 60 MB demora cerca de 1 minuto. Converte-se uma vez para Parquet com `.venv/bin/python src/convert_to_parquet.py` (todos os anos; fica em `data/processed/raw/`), que depois lê em segundos.

## Plano

Tunar modelos que já sabem português, em vez de criar um de raiz.

Um único modelo para todos os códigos. Modelos a tunar, os dois como classificador (uma camada final que escolhe entre os códigos CPV), com o mesmo código de treino:

| Modelo | Tipo | Tamanho | Português |
|---|---|---|---|
| BERTimbau base | BERT | 110M | Brasil |
| Albertina PT-PT 100M | DeBERTa | 100M | Portugal |

Comparação sem treino: o BERT multilingue já tunado com concursos europeus (https://huggingface.co/MKaan/multilingual-cpv-sector-classifier), só nas 45 divisões.

Dados usados (separação pela data de publicação, decisão em [`notebooks/02_anos_treino.ipynb`](notebooks/02_anos_treino.ipynb)):

| Parte | Período | Contratos |
|---|---|---|
| Treino | janeiro de 2022 a setembro de 2025 | 478 384 exemplos (de 794 883 contratos, depois de juntar textos repetidos e tirar códigos ao nível da divisão) |
| Validação | outubro de 2025 | 23 945 |
| Teste | novembro e dezembro de 2025 | 37 361 |
| Reserva | 2026 (até 4 de outubro) | não se usa por agora |

Os passos, pela ordem em que os fazemos, estão em [`docs/workflow.md`](docs/workflow.md).

## Réguas

Só existem para a divisão (2 dígitos):

| Fonte | Língua | Accuracy nas divisões | F1 macro |
|---|---|---|---|
| Estudo italiano, SVM com TF-IDF | italiano | 77,7% | 0,40 |
| Estudo italiano, BERT afinado | italiano | 74,1% | 0,30 |
| Estudo italiano, IT5-large afinado | italiano | 78,7% | 0,37 |
| Zwolle (Haskoning) | neerlandês e outras | cerca de 76% | n/d |

Dois avisos que saem daqui: o modelo clássico é forte e pode bater os modelos tunados, e a accuracy engana porque as divisões são muito desequilibradas. Reportar sempre o F1 macro.

Referências:
- Automatic Generation of Common Procurement Vocabulary Codes: https://ceur-ws.org/Vol-3596/paper45.pdf
- Haskoning, município de Zwolle: https://www.haskoning.com/en/projects/using-machine-learning-to-predicting-cpv-codes-from-tender-text
- Data Quality Barriers for Transparency in Public Procurement: https://www.mdpi.com/2078-2489/13/2/99

## Fora do âmbito, por agora

- Sugestão de preço base. Fundamentar o preço obriga o técnico a procurar contratos parecidos no Portal BASE, um a um: são horas, não minutos. Parece uma dor maior, mas também não está validada.
- Matching concurso-empresa com `adjudicatarios` e `concorrentes`.
- API e demo.

## Estrutura

```
cpv-portal-base/
  data/raw/         ficheiros .xlsx originais (fora do git)
  data/processed/   Parquet limpo e splits (fora do git)
  notebooks/        exploração: 00_qualidade, 01_exploracao, 02_anos_treino
  src/              pipeline: conversão, preparação, treino e avaliação
  tests/            testes unitários e de integração (pytest)
  scripts/          download_data.py
  docs/             workflow.md (os passos e o estado), backlog.md (épicos e user stories), dados.md (as colunas), qualidade_dados.md (verificações e decisões) e prs/ (a descrição de cada PR)
  requirements.txt  versões exatas das bibliotecas
```

Ambiente: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt` (Python 3.10).

Atalhos: `make help` mostra os comandos do projeto, cada um com uma frase a explicar o que faz (por exemplo `make preparar`, `make test`, `make test-avaliacao`).

Pipeline e testes:

```
.venv/bin/python scripts/download_data.py      # descarregar os .xlsx
.venv/bin/python src/convert_to_parquet.py     # .xlsx -> Parquet, todos os anos (data/processed/raw/)
.venv/bin/python src/prepare.py                # Parquet -> treino, validação, teste (data/processed/)
.venv/bin/pytest                               # testes unitários e de integração
```

- `tests/unit/`: cada função sozinha, com dados pequenos inventados.
- `tests/integration/`: o pipeline inteiro (xlsx -> treino/validação/teste) com 15 contratos inventados, cada um a pôr à prova uma regra de `docs/qualidade_dados.md`.
- Os testes foram verificados com erros introduzidos de propósito (mutation testing): apanharam os 9 que se tentaram (5 no pipeline, 4 nos códigos raros).
- CI: o GitHub Actions corre todos os testes em cada envio e em cada PR (`.github/workflows/testes.yml`). Não usa os dados reais.

## Estado

Ver as caixas de verificação em [`docs/workflow.md`](docs/workflow.md).
