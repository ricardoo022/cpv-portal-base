# Qualidade dos dados

Resultados das verificações de qualidade (fase 2 do [`workflow.md`](workflow.md)) e as decisões que saem delas.

- Dados: Portal BASE, versão de 4 de outubro de 2026.
- **Âmbito das medições (importante):**
  - As secções 1 a 6 foram medidas só com os ficheiros de 2025 e 2026 (433 838 contratos). Depois descobriu-se que o ficheiro de um ano é o ano de celebração (secção 2) e decidiu-se treinar desde 2022 (decisão 1). Servem para as decisões (são problemas técnicos que se repetem em todos os anos), mas os números não são os do treino final.
  - As secções 7 a 10 estão medidas com o **treino final**: contratos publicados de janeiro de 2022 a setembro de 2025, lidos de todos os ficheiros. A primeira versão destas secções usava só janeiro a setembro de 2025; onde a conclusão mudou, está dito.
  - A secção 11 é a saída do `src/prepare.py`.
  - A secção 12 já é da fase 4: a régua "códigos mais comuns" medida na validação.
- De onde vêm os números:
  - secções 1 a 6: [`notebooks/00_qualidade.ipynb`](../notebooks/00_qualidade.ipynb) (só conta);
  - secções 7 a 10: [`notebooks/01_exploracao.ipynb`](../notebooks/01_exploracao.ipynb) (só o treino; a validação só na secção 10). Dois números deste notebook têm de bater com o funil da secção 11, e batem: 794 883 contratos de treino e 14 348 fora do código mais comum do seu texto;
  - secção 12: [`notebooks/03_regua.ipynb`](../notebooks/03_regua.ipynb) (treino e validação; lê o `treino.parquet` e o `validacao.parquet`, por isso precisa do `prepare.py` antes).
- Os notebooks só contam. Os exemplos que mostram vêm apenas de contratos publicados antes de novembro de 2025, para não espreitar o período de teste.

## O que está bem

### 1. Quase tudo preenchido

| Campo | Preenchido |
|---|---|
| `objectoContrato` | 433 838 (100%) |
| `descContrato` | 433 838 (100%) |
| `CPV` | 433 835 (100%, faltam 3) |

Quase não há linhas a deitar fora.

### 2. Datas

Todos os contratos têm data de publicação válida.

| Publicados em | Contratos | Uso |
|---|---|---|
| 2024 | 6 | ignorar |
| 2025 | 225 113 | treino (janeiro a setembro), validação e teste |
| 2026 | 208 719 | por agora não se usa |

O ficheiro de um ano não coincide com o ano de publicação. **Correção:** uma versão anterior deste ficheiro dizia que os contratos publicados em 2025 estavam nos ficheiros de 2025 e 2026. Está errado. Lendo todos os ficheiros: 225 106 estão no de 2025, 25 877 no de 2024, 2 930 no de 2023 e 7 no de 2026. O ficheiro (e a coluna `Ano`) parece ser o ano de celebração: dos 25 877 do ficheiro de 2024, 25 777 foram celebrados em 2024. As secções 1 a 10 usaram só os ficheiros de 2025 e 2026 e deixaram de fora cerca de 30 mil contratos publicados em 2025 (12%); têm de ser medidas outra vez.

### 3. Vários códigos na mesma célula: raro

| Códigos na célula | Contratos |
|---|---|
| 1 | 428 593 (98,8%) |
| 2 | 4 282 (1,0%) |
| 3 | 514 (0,1%) |
| 4 | 196 (0,0%) |
| 5 ou mais | 250 (0,1%) |

Dos contratos com vários códigos, 1 317 (0,3% do total) têm todos os códigos na mesma divisão e 3 925 (0,9%) têm códigos de divisões diferentes.

### 4. Contratos repetidos com o mesmo `idcontrato`

1 199 linhas (0,3%) repetem um `idcontrato`. As cópias estão sempre no mesmo ficheiro e têm sempre o mesmo texto e o mesmo CPV. Só mudam duas colunas que não usamos: `TipoCriterioAdjudicacao` e `ContratEcologico`.

## O que é preciso ter em conta

### 5. A `descContrato` traz pouco

Em 313 691 contratos (72%) é igual ao `objectoContrato`, palavra por palavra (ignorando maiúsculas e espaços). Só em 28% acrescenta alguma coisa. Os dois campos têm um comprimento mediano de 12 palavras.

### 6. Textos duplicados: 32%

Contando só o `objectoContrato` (ignorando maiúsculas e espaços), 139 064 contratos (32%) têm um texto igual ao de outro contrato. Há 294 771 textos distintos.

Os mais repetidos, antes de novembro de 2025:

| Vezes | Texto |
|---|---|
| 1 870 | medicamentos |
| 722 | aquisição de serviço de viagens e alojamento – plataforma siav |
| 429 | convite à apresentação de propostas ao abrigo do(s) acordo(s)-quadro 2023/304 - ... |
| 421 | 1396/2025-convite à apresentação de propostas ao abrigo do(s) acordo(s)-quadro 2... |
| 402 | 1411/2025 - convite à apresentação de propostas ao abrigo dos acordos-quadro: 20... |

Isto traz dois problemas:

- **Textos que não dizem o que se compra.** "Medicamentos" pode ser qualquer um de dezenas de códigos de medicamentos. "Convite à apresentação de propostas ao abrigo do acordo-quadro..." não diz nada sobre o objeto. Nenhum modelo acerta no código completo só com estes textos.
- **O mesmo texto com códigos diferentes.** Há 5 220 textos que aparecem várias vezes com códigos diferentes, e cobrem 39 972 contratos (9%). Para o modelo é como ver a mesma pergunta com respostas diferentes. Isto marca um limite para o acerto, seja qual for o modelo.

### Fuga de dados (data leakage)

Acontece quando o modelo vê no treino informação que não devia, e por isso parece melhor no teste do que vai ser na realidade. No nosso caso: se "medicamentos" estiver no treino e no teste, o modelo "acerta" no teste porque decorou a resposta, não porque aprendeu.

Mas os textos repetidos também fazem parte do problema real: os técnicos vão voltar a escrever "medicamentos". Tirá-los todos do teste também falsearia o resultado.

### 7. Colunas que ajudam os textos vagos

Medido nos contratos de treino (janeiro de 2022 a setembro de 2025, 794 883 contratos).

| Coluna | Preenchida | O que traz |
|---|---|---|
| `DescrAcordoQuadro` | 14,5% de todos os contratos, mas 96,4% dos que falam de acordo-quadro | A descrição do acordo-quadro, por exemplo "Concurso público internacional para formação do acordo quadro de medicamentos..." |
| `adjudicante` | 100% | A entidade que compra |

Os textos que falam de acordo-quadro são 36 381 (4,6%). Muitos não dizem o que se compra ("convite à apresentação de propostas ao abrigo do acordo-quadro..."), mas a `DescrAcordoQuadro` diz.

As duas colunas são informação que o técnico já tem quando escolhe o código. Por isso não é fuga de dados.

### 8. O mesmo texto com códigos diferentes: quanto se perde ao escolher o mais comum

Medido nos contratos de treino (janeiro de 2022 a setembro de 2025), com o texto de entrada completo da decisão 5, construído com a mesma função do `prepare.py`.

Exemplo: o texto de entrada "medicamentos" (6 100 contratos, 7 entidades):

| Código | Vezes |
|---|---|
| `33000000` Equipamento médico, medicamentos e produtos para cuidados pessoais | 6 095 |
| `33690000` Medicamentos vários | 2 |
| `24000000` Produtos químicos | 1 |
| `32000000` Equipamento de rádio, televisão, comunicação... | 1 |
| `33600000` Produtos farmacêuticos | 1 |

Quase todos têm o código ao nível da divisão, por isso saem do treino com a decisão 8 e "medicamentos" fica com 5 contratos. Na primeira medição (só 2025, só o `objectoContrato`) havia 105 contratos com `33600000`. Hipótese, não verificada: com o texto completo esses contratos têm outro texto de entrada (por exemplo com a `descContrato` ou a descrição do acordo-quadro junta), e por isso já não contam como "medicamentos".

Pela ordem do `prepare.py` (primeiro tirar os códigos ao nível da divisão, decisão 8, e só depois juntar os textos repetidos):
- ficam 645 625 contratos;
- 6 849 textos têm mais do que um código;
- só 14 348 contratos (2,2%) têm um código diferente do mais comum do seu texto. Igual à primeira medição (2,2%) e igual ao funil da secção 11.

Primeira ideia, errada: um exemplo por par (texto, código). Os códigos de um texto repetido ficavam com um exemplo cada, e o modelo achava-os igualmente prováveis, apesar de um aparecer milhares de vezes e outro 2.

### 9. Códigos genéricos: a que nível as entidades escolhem o código

Medido nos contratos de treino (janeiro de 2022 a setembro de 2025, 794 883 contratos).

O nível de um código conta-se pelos dígitos antes dos zeros finais: `33600000` tem 3 dígitos significativos (`336`), por isso é um código ao nível do grupo. É uma aproximação: não foi verificado, código a código, que a tabela oficial do CPV segue sempre esta regra.

| Nível do código escolhido | Exemplo | Contratos | % |
|---|---|---|---|
| Divisão (2 dígitos) | `33000000` | 149 258 | 18,8% |
| Grupo (3) | `33600000` Produtos farmacêuticos | 196 885 | 24,8% |
| Classe (4) | `33690000` Medicamentos vários | 192 388 | 24,2% |
| Categoria (5) | `90911000` | 124 099 | 15,6% |
| Mais detalhado (6 a 8) | `90911200` Limpeza de edifícios | 132 253 | 16,6% |

Só 32% dos contratos têm código ao nível da categoria ou mais detalhado. A distribuição é quase igual à da primeira medição (só 2025: 17,6% ao nível da divisão).

Por divisão:
- A divisão `33` (equipamento médico e medicamentos) tem 257 682 contratos, 32% do treino. Desses, 24% estão ao nível da divisão e 65% ao nível do grupo ou acima.
- Tirar os códigos ao nível da divisão não apaga nenhuma divisão. A que fica pior é a `41` (água captada): de 113 contratos para 39.
- Tirar também os do nível do grupo custaria 43,5% do treino e 65% da divisão `33`.

Genérico não é o mesmo que errado. Um contrato que compra muitos medicamentos diferentes pode estar bem classificado como `33600000` Produtos farmacêuticos. Com estes dados não se distingue um genérico legítimo de um preguiçoso.

### 10. Quantos exemplos tem cada código (desequilíbrio de classes)

Medido nos **exemplos de treino** que o `prepare.py` grava (`treino.parquet`): sem códigos ao nível da divisão e com um só exemplo por texto (decisões 6 e 8). São 478 384 exemplos, que vêm de 645 625 contratos. É com estes que se contam os exemplos da decisão 9. A validação (outubro de 2025, 19 894 contratos sem código ao nível da divisão, do `validacao.parquet`) só entra na última coluna, para ver a cobertura.

Códigos distintos no treino:

| Nível | Códigos distintos |
|---|---|
| Divisão (2 dígitos) | 45 |
| Grupo (3) | 270 |
| Classe (4) | 1 152 |
| Código como foi escolhido (8) | 6 071 |

Exemplos por código (código como foi escolhido):
- Mediana: 8 exemplos. Metade dos códigos tem 8 ou menos.
- 1 026 códigos têm um só exemplo.
- O mais comum, `33600000` Produtos farmacêuticos, tem 37 151 (7,8% do treino).
- Os 10 códigos mais comuns cobrem 21,3% do treino.

Quantos códigos têm pelo menos N exemplos, e que parte dos dados cobrem:

| N | Códigos com pelo menos N | % dos exemplos de treino | % dos contratos de validação |
|---|---|---|---|
| 1 | 6 071 | 100,0% | 99,8% |
| 2 | 5 045 | 99,8% | 99,6% |
| 5 | 3 778 | 99,1% | 98,9% |
| 10 | 2 896 | 97,8% | 97,9% |
| 20 | 2 063 | 95,5% | 95,9% |
| 50 | 1 220 | 89,8% | 90,3% |
| 100 | 748 | 82,9% | 84,3% |

Como ler:
- Com N = 10, 97,9% dos contratos de validação têm um código que o modelo aprende tal como está. Os outros não ficam perdidos: com a decisão 9 o código sobe para a classe ou o grupo.
- As colunas do treino e da validação quase coincidem: o treino representa bem o mês seguinte.
- Comparado com a primeira medição (só 2025), a mediana passou de 4 para 8 exemplos e a cobertura com N = 10 de 92,2% para 97,9%. É o ganho de treinar desde 2022.
- O `02_anos_treino.ipynb` dava 98,2% para a mesma janela com N = 10. A diferença é que lá se contavam contratos, e aqui exemplos já com um só exemplo por texto: um código usado em muitos contratos com o mesmo texto fica com menos exemplos.

### 11. Resultado da preparação (funil)

Saída do `src/prepare.py` com a separação final (decisão 1), lendo todos os ficheiros.

| Passo | Contratos |
|---|---|
| Publicados de janeiro de 2022 a dezembro de 2025 | 859 649 |
| Com pelo menos um código CPV | 859 636 |
| Uma cópia por `idcontrato` | 856 189 |

Treino (janeiro de 2022 a setembro de 2025):

| Passo | Contratos |
|---|---|
| Contratos de treino | 794 883 |
| Sem códigos ao nível da divisão | 645 625 (saem 149 258, 18,8%) |
| Que seguem o código mais comum do seu texto | 631 277 (saem 14 348, 2,2%) |
| **Exemplos de treino (um por texto)** | **478 384** |

Validação e teste (ficam todos os contratos):

| | Validação (outubro de 2025) | Teste (novembro e dezembro de 2025) |
|---|---|---|
| Contratos | 23 945 | 37 361 |
| Com texto que já aparece no treino | 3 706 (15,5%) | 4 484 (12,0%) |
| Com código ao nível da divisão (avaliados só na divisão) | 4 051 (16,9%) | 5 642 (15,1%) |

### 12. Régua "códigos mais comuns" na validação (E4-T2)

De [`notebooks/03_regua.ipynb`](../notebooks/03_regua.ipynb), com `src/baselines.py` e a avaliação comum (`src/evaluation.py`).

**O que é:** sugerir sempre os mesmos 5 códigos, sem ler o texto. São as 5 etiquetas mais frequentes no treino **depois** da regra dos códigos raros (decisão 9), com o mesmo N que o modelo (por defeito 10), contadas uma vez por exemplo de treino (um por texto, decisão 10). Empate: o menor código. É o mínimo que o TF-IDF tem de bater (E4-H1).

| Posição | Etiqueta | Nome | Exemplos de treino |
|---|---|---|---|
| 1 | `33600000` | Produtos farmacêuticos | 37 151 (7,8%) |
| 2 | `33140000` | Material médico de consumo | 22 540 (4,7%) |
| 3 | `33690000` | Medicamentos vários | 10 848 (2,3%) |
| 4 | `33100000` | Equipamento médico | 9 347 (2,0%) |
| 5 | `98300000` | Serviços diversos | 4 277 (0,9%) |

As 5 juntas são 17,6% dos exemplos de treino, entre 3 000 etiquetas distintas. Com N = 5 e N = 20 as 5 etiquetas são as mesmas, pela mesma ordem: se o E4-S1 mudar o N, a régua não muda.

Resultados na validação (outubro de 2025):

| Grupo | Medida | Contratos | Top-1 | Top-5 | F1 macro |
|---|---|---|---|---|---|
| Todos | código completo | 19 906 | 13,7% | **26,3%** | 0,0001 |
| Todos | classe | 14 047 | 0,0% | **15,6%** | 0,0000 |
| Todos | divisão | 23 945 | 31,0% | 33,4% | 0,0105 |
| Todos | só divisão | 4 039 | 37,6% | 39,4% | 0,0121 |
| Texto novo | código completo | 16 565 | 11,4% | **23,7%** | 0,0001 |
| Texto novo | classe | 11 959 | 0,0% | **14,8%** | 0,0000 |
| Texto novo | divisão | 20 239 | 28,3% | 30,6% | 0,0098 |
| Texto novo | só divisão | 3 674 | 33,3% | 35,3% | 0,0111 |

Medidas hierárquicas da 1.ª sugestão (só se mostram): F1 micro 0,179 em todos e 0,160 no texto novo; F1 macro 0,025 nos dois.

**Contar exemplos ou contratos?** O treino tem um exemplo por texto (decisão 6), e o `n_contratos` diz quantos contratos tinham esse texto. Contar contratos (cada exemplo pesa o seu `n_contratos`) troca a 3.ª e a 4.ª etiqueta e põe `15800000` em 5.º lugar. Na validação, essa régua dá top-5 no código completo de 26,0% em todos os contratos e 23,3% no texto novo, menos 0,35 e 0,46 pontos. Com o bootstrap emparelhado do E4-T1, a régua que conta exemplos ganha em 100% das amostras nos dois grupos: a diferença conta, não é acaso. Na classe as duas dão exatamente o mesmo. Fica a que conta exemplos, a mais difícil de bater (decisão 10).

Como ler:
- **A régua a bater** (E4-H1) são os números a negrito: top-5 de 26,3% no código completo e 15,6% na classe em todos os contratos; 23,7% e 14,8% nos contratos com texto novo.
- Um modelo que não lê o texto já acerta o código completo em 1 de cada 4 contratos, porque 4 das 5 etiquetas mais comuns são de saúde (divisão 33). Por isso um número sozinho ("o modelo acerta 30%") não diz se o modelo é bom: compara-se sempre com esta régua.
- **Top-1 na classe é 0,0%.** A 1.ª sugestão, `33600000`, é um código de grupo (3 dígitos). Na classe comparam-se os 4 primeiros dígitos, `3360`, e nenhum contrato da validação tem um código da classe para baixo que comece por `3360`. O mesmo acontece com `33100000` e `98300000`: as três acertam 0 contratos na classe. Os 15,6% do top-5 vêm só de `33140000` (1 753 contratos, 12,5% dos que contam na classe) e `33690000` (438, 3,1%).
- O F1 macro é quase zero: a régua só acerta em 5 etiquetas, e o F1 macro faz a média sobre todas as etiquetas da validação. É o que se espera, e mostra porque o F1 macro se reporta sempre (decisão do E1-T1).
- Em **texto novo** a régua fica um pouco abaixo de **todos** (23,7% contra 26,3%), embora não leia o texto. Não é decorar: a etiqueta é uma das 5 da régua em 35,4% dos contratos com texto já visto e só em 19,4% dos de texto novo.
- **Só divisão:** são 4 039 contratos, e não os 4 051 da secção 11: lá conta-se a etiqueta (1.º código) ao nível da divisão; aqui, os contratos com **todos** os códigos ao nível da divisão. O notebook confirma as duas contagens.

### 13. TF-IDF contra a régua na validação (E4-H1)

**Substituída pela secção 14** quanto ao modelo por defeito: o `alpha` e o τ mudaram no spike E4-S4. Os números daqui ficam como registo do primeiro modelo.

De [`notebooks/04_tfidf.ipynb`](../notebooks/04_tfidf.ipynb), que lê as sugestões gravadas pelo treino no pod (`metrics/e4_h1/sugestoes_validacao.parquet`) e o resumo do treino (`metrics/e4_h1/treino.json`).

**O modelo** (`src/tfidf.py`, treinado com `src/train_tfidf.py` num pod da Runpod):
- **TF-IDF:** transforma cada texto num vetor com um peso por palavra e por par de palavras. Uma palavra pesa mais quando aparece no texto e é rara no resto do treino: "limpeza" pesa mais do que "serviços".
- **Classificador:** regressão logística um-contra-todos, treinada com o `SGDClassifier(loss="log_loss")` (decisão 5 do E4). "Um-contra-todos" quer dizer um modelo "sim/não" por etiqueta. As 5 sugestões são as 5 etiquetas com a pontuação mais alta.
- **Treino:** N = 10, janela desde 2022, 478 234 exemplos (150 saíram pela regra dos códigos raros), 3 000 etiquetas, 402 084 palavras e pares de palavras no vocabulário. Convergiu em 6 épocas.
- **Custo:** 454 s a vetorizar e treinar, 28 s a sugerir para os 23 945 contratos da validação. Os pesos do modelo ocupam 4,8 GB (`coef_bytes` do `treino.json`); o ficheiro gravado ocupa um pouco mais, porque leva também o vocabulário.

Resultados na validação (outubro de 2025), lado a lado com a régua da secção 12:

| Grupo | Medida | Contratos | Top-1 TF-IDF | Top-1 régua | Top-5 TF-IDF | Top-5 régua | F1 macro TF-IDF | F1 macro régua |
|---|---|---|---|---|---|---|---|---|
| Todos | código completo | 19 906 | 30,8% | 13,7% | **43,2%** | 26,3% | 0,0107 | 0,0001 |
| Todos | classe | 14 047 | 25,1% | 0,0% | **37,9%** | 15,6% | 0,0279 | 0,0000 |
| Todos | divisão | 23 945 | 43,3% | 31,0% | 62,2% | 33,4% | 0,1843 | 0,0105 |
| Todos | só divisão | 4 039 | 44,5% | 37,6% | 61,3% | 39,4% | 0,1313 | 0,0121 |
| Texto novo | código completo | 16 565 | 26,2% | 11,4% | **38,5%** | 23,7% | 0,0101 | 0,0001 |
| Texto novo | classe | 11 959 | 21,4% | 0,0% | **34,2%** | 14,8% | 0,0246 | 0,0000 |
| Texto novo | divisão | 20 239 | 38,7% | 28,3% | 59,0% | 30,6% | 0,1597 | 0,0098 |
| Texto novo | só divisão | 3 674 | 40,3% | 33,3% | 58,5% | 35,3% | 0,1242 | 0,0111 |

**A diferença conta?** Com o bootstrap emparelhado do E4-T1 (1 000 amostras), o TF-IDF ganha à régua em 100% das amostras no top-5 do código completo e da classe, tanto em todos os contratos como só no texto novo. O critério do E4-H1 está cumprido.

**Quanto vem de decorar textos?** Os contratos da validação com texto que já aparece no treino, comparados com os de texto novo:

| Grupo | Medida | Contratos | Top-1 TF-IDF | Top-5 TF-IDF | Top-5 régua |
|---|---|---|---|---|---|
| Texto já visto | código completo | 3 341 | 53,9% | 66,4% | 39,2% |
| Texto já visto | classe | 2 088 | 46,4% | 59,1% | 19,9% |
| Texto novo | código completo | 16 565 | 26,2% | 38,5% | 23,7% |
| Texto novo | classe | 11 959 | 21,4% | 34,2% | 14,8% |

Medidas hierárquicas da 1.ª sugestão (só se mostram): F1 micro 0,334 em todos e 0,289 no texto novo (régua: 0,179 e 0,160); F1 macro 0,073 e 0,068 (régua: 0,025 e 0,025).

Como ler:
- **O TF-IDF bate a régua com folga:** mais 16,9 pontos no top-5 do código completo e mais 22,3 na classe, em todos os contratos. No texto novo a vantagem é parecida (mais 14,8 e mais 19,4 pontos), por isso não vem só de decorar.
- **Decorar ajuda muito:** com texto já visto, o top-5 no código completo é 66,4%; com texto novo, 38,5%. O número que interessa para o técnico é o do texto novo, porque é o caso normal.
- **O F1 macro continua muito baixo** (0,0107 no código completo). O F1 macro faz a média sobre todas as etiquetas da validação, e o modelo acerta bem poucas etiquetas, as frequentes. As raras quase nunca saem em 1.º lugar. É o ponto fraco a seguir nos próximos modelos.
- **Sugestões das 3 descrições escritas à mão** (os nomes dos códigos são os da nomenclatura CPV oficial, Regulamento (CE) n.º 213/2008; o repositório não tem a tabela CPV) (`metrics/e4_h1/sugestoes_exemplos.txt`, do comando `src/suggest.py`): para "serviços de limpeza de três escolas básicas durante 12 meses" a 1.ª sugestão é `90910000` (serviços de limpeza). Para as refeições escolares, o código esperado aparece só em 4.º lugar (`55524000`, refeições para escolas). Para a repavimentação de estradas, a única sugestão de obras é `45453100` (em inglês *refurbishment work*, renovação), em 3.º lugar, e nenhuma é um código de estradas. Os outros lugares são códigos de saúde da régua (`33140000`, `33600000`...). Quando o texto se parece pouco com os do treino, o modelo puxa para as etiquetas mais frequentes.

### 14. Afinar o TF-IDF: regularização e etiquetas frequentes (E4-S4)

De [`notebooks/05_tfidf_afinacao.ipynb`](../notebooks/05_tfidf_afinacao.ipynb), que lê as sugestões gravadas pelos treinos no pod (`metrics/e4_s4/alpha_<alpha>/`). **Substitui os números da secção 13**: o modelo por defeito passa a ser o escolhido aqui.

**Porquê:** na secção 13, a 1.ª sugestão era uma das 5 etiquetas da régua em 84,7% dos contratos e o F1 macro no código completo era 0,0107. Duas correções testadas, com o treino completo (o melhor `alpha` depende do número de exemplos, por isso uma amostra no portátil não servia):

- **Regularização mais fraca.** A *regularização* é um travão que impede o modelo de dar pesos grandes às palavras. No `SGDClassifier` chama-se `alpha`, e no E4-H1 ficou no valor por defeito do scikit-learn, 0,0001. Uma etiqueta com 10 exemplos (o N) aparece em 10 dos 478 384 textos de treino, uma fração mais pequena do que esse travão. A hipótese (não medida diretamente: nenhuma célula mede os pesos das etiquetas raras) é que o modelo dessa etiqueta fica quase sem pesos de palavras, e ganham as etiquetas frequentes. O que se mediu é o efeito: a parte das 1.as sugestões que são uma das 5 etiquetas da régua desce de 84,7% para 30,7% ao baixar o `alpha`. A documentação do scikit-learn recomenda procurar o `alpha` entre 0,1 e 0,000001.
- **Correção das etiquetas frequentes** (Menon et al., 2021, *logit adjustment*): ordenar por pontuação − τ · log(frequência da etiqueta no treino). Com τ = 0 nada muda. O artigo usa pontuações de um modelo softmax; aqui aplica-se às do um-contra-todos, uma adaptação.

Resultados na validação (outubro de 2025), todos os contratos salvo onde diz "novo":

| `alpha` | τ | Top-5 código | Top-5 classe | Top-5 código (novo) | Top-5 classe (novo) | Top-1 código | F1 macro código | 1.ª sugestão na régua |
|---|---|---|---|---|---|---|---|---|
| 0,0001 (E4-H1) | 0 | 43,2% | 37,9% | 38,5% | 34,2% | 30,8% | 0,0107 | 84,7% |
| 0,0001 | 0,25 | 48,0% | 44,5% | 43,6% | 41,3% | 33,5% | 0,0148 | 75,9% |
| 0,0001 | 0,5 | 33,5% | 30,8% | 28,2% | 26,4% | 31,1% | 0,0190 | 28,1% |
| 0,0001 | 1 | 5,6% | 7,8% | 3,3% | 6,3% | 5,5% | 0,0051 | 3,9% |
| 0,00001 | 0 | 61,6% | 60,8% | 57,9% | 58,1% | 43,8% | 0,0516 | 49,9% |
| 0,00001 | 0,25 | 64,6% | 65,6% | 61,1% | 63,1% | 45,1% | 0,0619 | 44,1% |
| 0,00001 | 0,5 | 68,6% | 71,1% | 65,3% | 68,8% | 46,4% | 0,0776 | 38,1% |
| 0,00001 | 1 | 49,2% | 55,3% | 44,4% | 51,2% | 38,6% | 0,1079 | 19,4% |
| 0,000001 | 0 | 75,3% | 78,4% | 71,9% | 76,0% | 54,7% | 0,1617 | 35,2% |
| 0,000001 | 0,25 | 76,8% | 80,7% | 73,7% | 78,5% | 55,7% | 0,1899 | 31,8% |
| 0,000001 | 0,5 | 77,8% | 82,0% | 74,7% | 79,9% | 55,2% | 0,2089 | 28,7% |
| 0,000001 | 1 | 73,8% | 80,1% | 70,3% | 78,0% | 49,8% | 0,2288 | 18,7% |
| 0,0000001 | 0 | 76,9% | 82,2% | 73,4% | 79,9% | 57,0% | 0,2417 | 30,7% |
| **0,0000001** | **0,25** | **76,9%** | **82,9%** | **73,5%** | **80,8%** | **57,0%** | **0,2565** | **28,7%** |
| 0,0000001 | 0,5 | 76,4% | 82,9% | 73,0% | 80,7% | 55,9% | 0,2580 | 26,6% |
| 0,0000001 | 1 | 71,7% | 81,0% | 67,9% | 78,7% | 52,1% | 0,2582 | 19,1% |
| régua | - | 26,3% | 15,6% | 23,7% | 14,8% | 13,7% | 0,0001 | 100,0% |

**Escolha, com a regra escrita no backlog antes de correr** (primeiro o `alpha` com τ = 0, depois o τ; cada candidato contra a opção em vigor, com a regra do E4-T1):

| Comparação | Top-5 código | Top-5 classe | Fica |
|---|---|---|---|
| `alpha` 0,0001 → 0,00001 | +18,4 pontos (100% das amostras) | +22,9 (100%) | 0,00001 |
| `alpha` 0,00001 → 0,000001 | +13,7 (100%) | +17,6 (100%) | 0,000001 |
| `alpha` 0,000001 → 0,0000001 | +1,6 (100%) | +3,8 (100%) | 0,0000001 |
| τ 0 → 0,25 | +0,04 (62%: não conta) | +0,75 (100%) | 0,25 (desempata a classe) |
| τ 0,25 → 0,5 | −0,52 (0%) | −0,06 (30%) | 0,25 |
| τ 0,25 → 1 | −5,22 (0%) | −1,98 (0%) | 0,25 |

Escolhido: **`alpha` = 0,0000001 e τ = 0,25** (decisão 6 do E4, no `backlog.md`). Na avaliação completa: top-1 na divisão 81,0% e top-5 93,5%; no texto novo, top-1 na divisão 78,2%. Continua a bater a régua nos quatro casos do E4-H1, com o bootstrap.

Como ler:
- **O problema era a regularização.** Só baixar o `alpha` levou o top-5 no código completo de 43,2% para 76,9%, e o F1 macro de 0,0107 para 0,2417. A correção das etiquetas frequentes ajuda pouco quando o `alpha` já está certo (+0,75 pontos na classe); com o `alpha` forte do E4-H1, um τ grande deita tudo abaixo (τ = 1: 5,6%), porque tira a vantagem às etiquetas frequentes sem o modelo saber mais nada do texto.
- **O 0,0000001 entrou depois de ver os resultados.** O 0,000001 ganhou e era o extremo da lista; quando o melhor valor fica no extremo, não se sabe se um valor mais pequeno seria melhor. O 0,0000001 ganhou outra vez, com uma diferença que conta (+1,6 pontos no código completo, contra +13,7 no passo anterior), e **voltou a ser o extremo da lista**. Parou-se aí porque o ganho caiu muito, mas esse critério não estava escrito antes de correr, e o 0,00000001 não foi testado. **Limitação:** não se sabe se um `alpha` ainda mais pequeno seria melhor; o ganho por passo caiu de +13,7 para +1,6 pontos, mas não se pode garantir que continue a cair. Testá-lo custa um treino de cerca de 8 minutos no pod.
- **A regra em dois passos não mudou o resultado.** `alpha` = 0,000001 com τ = 0,5 tem mais 0,87 pontos no top-5 do código completo, mas menos 0,96 na classe (as duas diferenças contam). A guarda da classe elimina-a, também numa comparação direta (última célula do notebook).
- **Comparação com o publicado:** o top-1 na divisão (81,0%) fica agora na zona dos 75% a 79% dos estudos citados no `readme.md`. Não é uma comparação exata: eles treinaram só com as divisões, com outros dados.
- **Otimista, mas pouco:** o `alpha` e o τ foram escolhidos na validação, por isso estes números ficam um pouco acima do que o teste vai mostrar. O teste continua fechado.
- **Custo:** os 4 treinos levaram 6 a 8 minutos a vetorizar e treinar, com 6 a 12 épocas (`treino.json`, célula 3 do notebook); com a sugestão, cada um levou 7 a 9 minutos (`metrics/e4_s4/pod_treino*.log`). Com a preparação dos pods, cerca de 40 minutos de pod, perto de 0,21 $ (`metrics/e4_s4/pod_info.json`, escrito à mão a partir das leituras da Runpod).
- **Ainda não medido com o modelo novo:** as sugestões para as 3 descrições escritas à mão (secção 13) são do modelo do E4-H1. Os treinos deste spike não gravaram o modelo (`--sem-modelo`), por isso o comando de sugestões ainda não correu com o `alpha` e o τ novos.

### 15. Épocas e um `alpha` mais pequeno no TF-IDF (E4-S5)

De [`notebooks/06_tfidf_epocas.ipynb`](../notebooks/06_tfidf_epocas.ipynb), que lê as sugestões gravadas pelos treinos no pod (`metrics/e4_s5/`). **Substitui os números da secção 14**: o modelo por defeito passa a ser o escolhido aqui.

**Porquê:** duas coisas ficaram por medir no E4-S4. O `alpha` 0,0000001 ganhou mas era o extremo da lista testada. E o treino parava sempre pela regra por defeito do scikit-learn: cada um dos ~3 000 classificadores "sim/não" para quando a perda (o erro) **no treino** não desce pelo menos 0,001 × o número de exemplos em 5 épocas seguidas (*paragem automática*). Uma **época** é uma passagem completa por todos os exemplos de treino. Nunca se tinha medido se outro número de épocas dava melhores sugestões na **validação**.

**Como:** as opções e a ordem das comparações foram escritas antes de correr em [`experiencias/e4_s5.yaml`](../experiencias/e4_s5.yaml), que o treino no pod e o notebook leem. Três passos, cada candidato contra a opção em vigor com a regra do E4-T1: (1) `alpha` 0,00000001 contra 0,0000001, com τ = 0; (2) o τ, só se o `alpha` mudasse; (3) um número fixo de épocas, igual para todas as etiquetas e sem paragem automática (3, 6, 12, 18 e 24), contra a regra por defeito.

**Repetível:** o treino com `alpha` = 0,0000001 e a regra por defeito deu sugestões 100% iguais às do E4-S4, nos 4 τ (célula 7 do notebook). Os números da secção 14 confirmam-se.

Resultados na validação (outubro de 2025), N = 10, todos os contratos salvo onde diz "novo":

| `alpha` | τ | Épocas | Top-5 código | Top-5 classe | Top-5 código (novo) | Top-5 classe (novo) | Top-1 código | F1 macro código | 1.ª sugestão na régua |
|---|---|---|---|---|---|---|---|---|---|
| 0,0000001 (E4-S4) | 0 | regra por defeito | 76,9% | 82,2% | 73,4% | 79,9% | 57,0% | 0,2417 | 30,7% |
| 0,0000001 (E4-S4) | 0,25 | regra por defeito | 76,9% | 82,9% | 73,5% | 80,8% | 57,0% | 0,2565 | 28,7% |
| 0,00000001 | 0 | regra por defeito | 69,4% | 75,8% | 64,9% | 72,7% | 53,1% | 0,2221 | 30,4% |
| 0,00000001 | 0,25 | regra por defeito | 69,4% | 76,2% | 64,8% | 73,2% | 53,0% | 0,2224 | 28,8% |
| 0,0000001 | 0,25 | 3 | 74,5% | 81,8% | 70,8% | 79,6% | 55,4% | 0,2446 | 28,0% |
| 0,0000001 | 0,25 | 6 | 76,8% | 82,7% | 73,3% | 80,5% | 56,5% | 0,2532 | 28,6% |
| 0,0000001 | 0,25 | 12 | 77,5% | 83,1% | 74,2% | 81,0% | 57,2% | 0,2552 | 28,6% |
| 0,0000001 | 0,25 | 18 | 77,5% | 83,1% | 74,1% | 81,0% | 57,6% | 0,2552 | 29,0% |
| **0,0000001** | **0,25** | **24** | **78,3%** | **84,1%** | **75,1%** | **82,2%** | **57,7%** | **0,2568** | **28,9%** |
| régua | - | - | 26,3% | 15,6% | 23,7% | 14,8% | 13,7% | 0,0001 | 100,0% |

O notebook mostra também o `alpha` 0,00000001 com τ = 0,5 e 1 (mais baixos ainda).

**Escolha, com a regra escrita antes de correr:**

| Comparação | Top-5 código | Top-5 classe | Fica |
|---|---|---|---|
| `alpha` 0,0000001 → 0,00000001 (τ = 0) | −7,47 pontos (candidato melhor em 0% das amostras) | −6,41 (0%) | 0,0000001 |
| τ | não se volta a escolher: o `alpha` não mudou | | 0,25 |
| regra por defeito → 3 épocas | −2,48 (0%) | −1,12 (0%) | regra por defeito |
| regra por defeito → 6 épocas | −0,11 (18%: não conta) | −0,21 (5%: não conta) | regra por defeito |
| regra por defeito → 12 épocas | +0,58 (100%) | +0,19 (94%: não conta) | 12 épocas |
| 12 → 18 épocas | −0,05 (29%) | +0,02 (59%) | 12 épocas |
| 12 → 24 épocas | +0,82 (100%) | +0,98 (100%) | **24 épocas** |

Escolhido: **`alpha` = 0,0000001, τ = 0,25 e 24 épocas fixas** (decisão 7 do E4, no `backlog.md`). Na avaliação completa: top-1 na divisão 81,1% e top-5 93,7%; no texto novo, top-1 na divisão 78,3%. Continua a bater a régua nos quatro casos do E4-H1, com o bootstrap (última célula do notebook).

**A curva época a época** (gráfico do notebook, só para ver; não entra na decisão): `scripts/curva_epocas.py` treinou uma época de cada vez e gravou as sugestões depois de cada uma. O top-5 no código completo sobe depressa até à época 6 (65,1% na 1.ª, 76,9% na 6.ª), devagar até à 15.ª (77,7%), e fica entre 77,7% e 77,8% da 15.ª à 24.ª. É uma **aproximação**: nesta curva todas as épocas veem os exemplos pela mesma ordem, e num treino normal a ordem muda de época para época (o script explica porquê). Na época 1 as duas maneiras dão o mesmo modelo (há um teste a isso).

Como ler:
- **O `alpha` mais pequeno piora muito**, e não é por falta de épocas: com a regra por defeito, o 0,00000001 treinou até 30 épocas (contra 12 do 0,0000001). A busca pelo `alpha` fica fechada **dentro da lista testada**: o 0,0000001 deixou de ser o extremo que ganhou, porque o valor seguinte perde. Mas, como dizia a nota da revisão no backlog, o `alpha` também muda o tamanho do passo (`learning_rate="optimal"`) e o momento em que a paragem automática chega, por isso a comparação mistura os três efeitos. A conclusão é "0,00000001, treinado assim, é pior", não "regularização a menos é pior".
- **A paragem automática chegava cedo demais, mas pouco.** Treinar todas as etiquetas mais tempo ganha +0,58 pontos (12 épocas) e +1,41 (24 épocas) no código completo sobre a regra por defeito, e +1,2 na classe com 24. É menos do que a mudança do `alpha` no E4-S4 (+33,7). A suspeita da revisão era que as etiquetas raras paravam cedo demais. O F1 macro quase não muda (0,2565 → 0,2568), o que sugere que o ganho **não** vem sobretudo das etiquetas raras. É uma leitura, não uma medição: nenhuma célula mede o ganho por frequência da etiqueta.
- **Limitação 1: 24 é o extremo da lista.** O mesmo problema do `alpha` no E4-S4: não se sabe se 30 ou 36 épocas seriam melhores. Pela regra escrita antes de correr, não se acrescentaram valores neste spike. A curva aproximada está quase plana da época 15 à 24, o que sugere que o ganho para lá de 24 seria pequeno, mas a curva não é o treino normal.
- **Limitação 2: o salto das 18 para as 24 épocas pode ser, em parte, sorte do treino.** Os treinos normais dão 77,5% com 12 e com 18 épocas, e 78,3% com 24, enquanto a curva sobe de forma suave e nunca passa de 77,8%. O bootstrap só mede a variação por causa da **amostra de contratos da validação**, com cada modelo fixo; não mede a variação que vem de **treinar** com outra ordem dos exemplos (outra semente). Saber quanto vale essa variação pedia treinar o mesmo modelo com várias sementes, o que não estava no plano. A diferença de 12 para 24 conta no bootstrap (100% das amostras), e a regra escrita aplica-se como está; esta limitação diz que a margem real pode ser menor do que +0,82 pontos.
- **Só vale para N = 10 e o treino desde 2022.** Um número fixo de épocas não se adapta a mais ou menos dados, ao contrário da regra por defeito. O E4-S2 (janela) continua a treinar as suas opções com a regra por defeito (`--regra-por-defeito`). O E4-S1 (N) acabou por testar as épocas dentro de uma grelha com o N (secção 16): o número ótimo de épocas depende do N, e a grelha confirmou as 24 épocas para N = 10.
- **Custo de treinar.** Com 24 épocas fixas, cada etiqueta treina sempre 24 épocas; com a regra por defeito, a maior parte parava antes (o máximo foi 12). Neste pod, com 4 treinos ao mesmo tempo e 8 processadores cada, o treino de 24 épocas levou 1 581 s a vetorizar e treinar, e o de 12, 821 s (`treino.json`, célula 3 do notebook). O tempo cresce quase em linha com as épocas. Os tempos da regra por defeito (224 s) foram medidos com 2 treinos ao mesmo tempo e 16 processadores, por isso não se comparam diretamente com estes.
- **Otimista, mas pouco:** o `alpha`, o τ e as épocas foram escolhidos na validação; o teste continua fechado.
- **Custo do spike:** um pod `cpu3g` de 32 processadores e 128 GB, a 1,28 $/h, ligado cerca de 44 minutos: cerca de 0,95 $ (`metrics/e4_s5/pod_info.json`, escrito à mão a partir das leituras da Runpod). O pico de memória do pod inteiro foi 36,0 GB, com 4 treinos ao mesmo tempo. Mais do que a estimativa do backlog (menos de 0,32 $), porque a curva e os 5 números de épocas somam cerca de 90 épocas de treino; o pod maior custa o mesmo por época e acaba 4 vezes mais depressa.
- **Ainda não medido com o modelo novo:** o comando de sugestões (secção 13) continua sem ter corrido com o modelo escolhido. Os treinos deste spike também não gravaram o modelo.

### 16. Grelha N × épocas × τ no TF-IDF (E4-S1)

De [`notebooks/07_tfidf_grelha.ipynb`](../notebooks/07_tfidf_grelha.ipynb), que lê as sugestões gravadas pelos treinos no pod (`metrics/e4_s1/`).

**Porquê:** o E4-S5 escolheu 24 épocas com N = 10, mas o número ótimo de épocas **depende do N** (o N muda quantas etiquetas existem e quantos exemplos tem cada uma), e o τ também (corrige a frequência das etiquetas, que muda com o N). Escolher o N com as épocas e o τ fixos podia escolher um N que não seria o melhor com os seus próprios valores — é a **interação** entre hiperparâmetros, e um eixo de cada vez não a vê. Por isso testou-se o produto **N × épocas × τ** de uma vez. O `alpha` ficou fixo em 0,0000001 (o efeito já tinha sido medido, é grande e claro).

**Como:** as opções foram escritas antes de correr em [`experiencias/e4_s1.yaml`](../experiencias/e4_s1.yaml). N = 5, 10 e 20; épocas = a regra de paragem por defeito do scikit-learn e 3, 6, 12, 18 e 24; τ = 0, 0,25 e 0,5. O τ **não retreina** o modelo — só muda a ordem das sugestões ao sugerir — por isso os 18 treinos (3 N × 6 épocas) deram as 3 × 6 × 3 = 54 combinações.

**Repetível:** o treino com N = 10 e 24 épocas (o modelo do E4-S5) deu sugestões 100% iguais às do E4-S5, com τ = 0 e τ = 0,25 (célula 7 do notebook). Os números da secção 15 confirmam-se.

Resultados na validação (outubro de 2025), todos os contratos. As 54 combinações completas estão no notebook; as primeiras (a régua usa N = 10, e as suas 5 etiquetas são as mesmas com N = 5 e 20, secção 12):

| N | Épocas | τ | Top-5 código | Top-5 classe | Top-5 código (novo) | F1 macro código |
|---|---|---|---|---|---|---|
| **10** | **24** | **0,25** | **78,35%** | **84,11%** | **75,14%** | 0,2568 |
| 10 | 24 | 0 | 78,18% | 83,31% | 74,89% | 0,2469 |
| 5 | 24 | 0,25 | 78,05% | 83,33% | 74,72% | 0,2647 |
| 5 | 18 | 0,25 | 77,94% | 83,26% | 74,54% | 0,2651 |
| régua | - | - | 26,3% | 15,6% | 23,7% | 0,0001 |

**Escolha, com a regra escrita antes de correr:** a melhor das 54 é a combinação do E4-S5 (N = 10, 24 épocas, τ = 0,25), com 78,35% no código completo. Como é a opção que já estava em vigor, o duelo dá diferença 0 e a decisão é mantê-la. Os concorrentes mais próximos perdem com uma diferença que conta no bootstrap: N = 10/24 épocas/τ = 0 (−0,17 pontos), N = 5/24/τ = 0,25 (−0,30) e N = 5/18/τ = 0,25 (−0,41).

Escolhido: **N = 10** (decisão 9), com as 24 épocas e o τ = 0,25 que já estavam em vigor (decisão 7). A avaliação completa do escolhido é a da secção 15; continua a bater a régua nos quatro casos do E4-H1 (última célula do notebook).

Como ler:
- **A grelha confirma o E4-S5, não o substitui.** O N = 10 já era o valor por defeito e nenhuma das outras 53 combinações o bate. Não há razão para mudar.
- **O N = 5 fica perto, mas perde.** Com 24 épocas e τ = 0,25, o N = 5 faz 78,05% no código completo (contra 78,35%), com melhor F1 macro (0,2647 contra 0,2568), porque mantém mais códigos detalhados. A regra decide pelo código completo, e a diferença conta. O N = 20 é o pior dos três (melhor caso: 77,49%).
- **Limitação (viés de seleção):** com 54 combinações, alguma ganharia só por sorte, por isso o número do vencedor fica **otimista**; confirma-se no teste, na fase 5. Aqui o efeito é pequeno, porque o vencedor coincide com a opção que já estava em vigor — não foi escolhido à custa de muitas comparações.
- **A interação N × épocas existe, mas é pequena.** Com N = 10 o melhor é 24 épocas e com N = 5 também; a diferença entre os três N é de décimas de ponto. A suspeita de que o N mudaria muito as épocas ótimas não se confirmou dentro da lista testada.
- **Custo:** um pod `cpu3g` de 32 processadores e 128 GB, a 1,28 $/h, ligado cerca de 80 minutos (15:36–16:56 UTC; os 18 treinos correram das 15:41 às 16:54), com 4 treinos ao mesmo tempo: cerca de 1,66 $ (`metrics/e4_s1/pod_info.json`, escrito à mão a partir das leituras da Runpod). O N = 5 é o mais pesado (3 848 etiquetas): o treino de 24 épocas levou 2 474 s, contra 1 873 s a N = 10 e 1 287 s a N = 20 (`treino.json`). O pico de memória do pod inteiro foi 29,4 GB de 128.
- **Otimista, mas pouco:** o N, as épocas e o τ foram escolhidos na validação; o teste continua fechado.

### 17. A entidade compradora no texto do TF-IDF (E4-S3)

De [`notebooks/08_entidade.ipynb`](../notebooks/08_entidade.ipynb), que lê as sugestões sem entidade (o treino N = 10/24 épocas do E4-S1, `metrics/e4_s1/n_min_10_ep_24/`) e com entidade (o treino deste spike, `metrics/e4_s3/com_entidade/`).

**Porquê:** a `adjudicante` (quem compra) já se conhece antes de escolher o CPV, por isso pode entrar no texto de entrada (decisão 7). Pode ajudar (um hospital e uma câmara compram coisas diferentes), mas o modelo pode aprender os hábitos de cada entidade, incluindo os códigos genéricos que ela costuma usar.

**Como:** a entidade junta-se ao início do texto, no formato `entidade | texto`, no treino e na validação (`src/train_tfidf.py --com-entidade`). O resto do treino é igual ao modelo em vigor (N = 10, `alpha` = 0,0000001, 24 épocas, τ = 0,25), para a entidade ser a única diferença. As opções, os três grupos e a regra de escolha foram escritos antes de correr em [`experiencias/e4_s3.yaml`](../experiencias/e4_s3.yaml). Uma entidade da validação "está no treino" se a sua `adjudicante`, com os espaços limpos, aparece no `treino.parquet`.

Resultados na validação (outubro de 2025, 23 945 contratos). A última coluna é a parte das 1.as sugestões ao nível do grupo ou acima (códigos genéricos como `45200000`):

| Grupo | Contratos | Top-5 código (sem → com) | Top-5 classe (sem → com) | Top-1 código (sem → com) | 1.ª sugestão ≤ grupo (sem → com) |
|---|---|---|---|---|---|
| todos | 23 945 | 78,35% → 80,76% | 84,11% → 85,48% | 57,66% → 62,51% | 30,57% → 33,13% |
| entidades no treino | 23 438 | 78,43% → 80,82% | 84,07% → 85,44% | 57,89% → 62,76% | 30,72% → 33,35% |
| entidades novas | 507 | 71,43% → 75,45% | 86,93% → 89,20% | 37,05% → 40,63% | 23,87% → 22,68% |

**Escolha, com a regra escrita antes de correr:**
- **Passo 1 (regra do E4-T1, em todos):** ganha o "com entidade": top-5 no código completo +2,41 pontos e na classe +1,37, as duas com uma diferença que conta no bootstrap.
- **Guarda 1 (não piorar o top-5 no código completo nas entidades novas):** cumpre-se. Até melhora (+4,02 pontos; o "com entidade" vence em 98% das amostras do bootstrap).
- **Guarda 2 (não aumentar a parte das 1.as sugestões genéricas, em todos):** **falha.** Sobe de 30,57% para 33,13%, e o "com entidade" tem a parte maior em 100% das amostras do bootstrap.

Escolhido: **sem entidade** (decisão 7). O texto de entrada continua o da decisão 5. Como o modelo em vigor não muda, a régua (secção 12) e os resultados do E4-H1 não se recalculam.

Como ler:
- **A entidade acerta mais, mas apoia-se mais em códigos genéricos.** Nas entidades conhecidas, a parte de 1.as sugestões genéricas sobe; nas entidades novas desce. É o que se esperaria se o modelo copiasse o hábito de cada entidade (por exemplo, "esta câmara regista sempre `45200000`").
- **Parte do ganho pode vir desse hábito.** Se a entidade regista um código genérico e o contrato da validação também o tem, sugeri-lo conta como acerto. Com estes números não se separa o ganho real do ganho por copiar códigos genéricos. Uma pergunta mais fina (por exemplo, o top-5 só nos contratos com código específico) teria de ser um spike novo, escrito antes de correr; não se fez.
- **A guarda 2 decide sozinha.** Sem ela, a entidade entrava. Ficou escrita antes de ver os números, por isso a decisão mantém-se.
- **Custo:** um pod `cpu3m` de 8 processadores e 64 GB, a 0,44 $/h (o `cpu3g` de 0,32 $/h não tinha stock), ligado cerca de 44 minutos: cerca de 0,32 $ (`metrics/e4_s3/pod_info.json`, escrito à mão a partir das leituras da Runpod). O treino levou 1 958 s; pico de memória do processo 10,9 GB (`treino.json`).

### 18. A janela de treino do TF-IDF (E4-S2)

De [`notebooks/09_janela.ipynb`](../notebooks/09_janela.ipynb), que lê as sugestões das três janelas: desde 2022 (o treino N = 10 com a regra de paragem por defeito do E4-S1, `metrics/e4_s1/n_min_10_ep_defeito/`), e desde 2025 e desde 2019 (treinos deste spike, `metrics/e4_s2/`).

**Porquê:** a decisão 1 escolheu treinar desde 2022 olhando só para quantos contratos da validação têm um código com exemplos suficientes no treino. Faltava confirmá-lo com um modelo: menos anos dão dados mais recentes, mais anos dão mais exemplos (sobretudo dos códigos raros), mas mais antigos.

**Como:** as janelas, a ordem da comparação e a regra foram escritas antes de correr em [`experiencias/e4_s2.yaml`](../experiencias/e4_s2.yaml). As três acabam em setembro de 2025; só muda o início (`src/prepare.py --inicio-treino`, E4-T4). O treino é o mesmo nas três: N = 10, `alpha` = 0,0000001, τ = 0,25 e a **regra de paragem por defeito** do scikit-learn, não as 24 épocas fixas. As 24 épocas foram escolhidas na janela desde 2022 e favoreciam-na; a regra por defeito para sozinha consoante os dados, por isso trata as três janelas da mesma forma. As janelas desde 2025 e desde 2019 foram preparadas e treinadas no pod; o `teste.parquet` que a preparação cria foi apagado logo lá. A janela desde 2022, preparada no pod, deu ficheiros iguais byte a byte (md5) aos de `data/processed/` (`metrics/e4_s2/pod_md5.txt`).

**O grupo da decisão:** os contratos da validação cujo texto não aparece no treino desde 2019 (20 150 de 23 945). Com mais anos no treino, mais textos da validação já foram vistos (3 706 desde 2022, 3 795 desde 2019), e o modelo acerta neles por decorar; usar o mesmo grupo nas três janelas impede a janela longa de ganhar só por isso.

| Janela | Exemplos de treino | Etiquetas | Épocas até parar | Top-5 código | Top-5 classe | Top-1 código |
|---|---|---|---|---|---|---|
| desde 2025 | 116 996 | 1 540 | 16 | 63,28% | 72,70% | 44,52% |
| desde 2022 | 478 234 | 3 000 | 12 | 73,46% | 80,73% | 51,46% |
| **desde 2019** | **785 542** | **3 621** | 11 | **75,38%** | **81,98%** | **52,04%** |

(Grupo da decisão. Em todos os contratos, só como informação: 67,34%, 76,94% e 78,63% no top-5 do código completo.)

**Escolha, com a regra escrita antes de correr** (a do E4-T1, da janela mais curta para a mais longa; em empate fica a mais curta):
- desde 2022 contra desde 2025: +10,18 pontos no top-5 do código completo e +8,03 na classe; a de 2022 é melhor em 100% das amostras do bootstrap. Fica a de 2022.
- desde 2019 contra desde 2022: +1,92 pontos no código completo e +1,25 na classe, também em 100% das amostras. Fica a de 2019.

Escolhida: **desde 2019**, com a regra de paragem por defeito. **Não passou a valor por defeito:** o modelo em vigor treina 24 épocas, e o spike E4-S6 (secção 19) treinou a janela desde 2019 com 24 épocas. Aí empatou com o modelo em vigor, e a decisão 1 ficou desde 2022.

**Análise completa** (informação, não decide; secção "Análise completa" do notebook). Em todas as medidas do E4-T1, desde 2019 contra desde 2022:
- **Nenhuma piora.** No bootstrap das 18 combinações (código, classe e divisão × top-1, top-5 e F1 macro, no grupo da decisão e em todos os contratos), a de 2019 ganha em 15 com uma diferença que conta e empata nas outras 3 (F1 macro); a de 2022 não ganha em nenhuma. As medidas hierárquicas sobem um pouco (F1 micro em todos 65,74% → 66,38%).
- **Não cai em códigos genéricos** (o que deixou a entidade de fora, secção 17): a parte das 1.as sugestões ao nível do grupo ou acima fica igual (29,89% → 29,98%, diferença que não conta), e a parte das 1.as sugestões que são uma das 5 etiquetas da régua desce um pouco (28,3% → 28,0%).
- **O ganho é maior nos códigos mais específicos** (top-5 no código completo, grupo da decisão, pelo nível da etiqueta): +1,3 pontos no nível 3 (grupo), +1,4 no 4 (classe), +2,0 no 5, +3,9 no 6, +2,5 no 7 e +2,9 no 8. Os anos 2019 a 2021 trazem exemplos dos códigos raros: com N = 10 passam a existir 3 621 etiquetas, em vez de 3 000.
- **A régua muda um pouco:** a 5.ª etiqueta passa de `98300000` para `33696500`. O modelo fica mais acima dela (+52,4 pontos no código completo em todos os contratos, contra +50,6 desde 2022 e +40,3 desde 2025, com a régua de cada janela). O treino desde 2025 não veio do pod; recriou-se no portátil (pico de 1,3 GB) e deu o mesmo md5 que lá.

Como ler:
- **Mais dados ganham, mesmo antigos.** A janela desde 2025 perde muito: com 117 mil exemplos, só conhece 1 540 etiquetas.
- **O ganho de 2019 sobre 2022 é pequeno, mas certo:** cerca de 2 pontos, em 100% das amostras do bootstrap, e em todas as medidas.
- **Custo:** a janela desde 2019 tem 64% mais exemplos. O treino passou de 8,7 para 15,8 minutos (`treino.json`, com 8 processadores) e o pico de memória do processo de 10,5 para 17,9 GB. A preparação desde 2019 tem um pico de 4,3 GB (`/usr/bin/time -v`, `metrics/e4_s2/pod_preparar_2019.log`) e **não cabe no portátil** (cerca de 3 GB livres): passaria a fazer-se no pod. Para o BERTimbau e o Albertina, o tempo de GPU deve crescer mais ou menos na mesma proporção dos exemplos (estimativa, não medida).
- **Verificação corrigida pelo caminho:** a preparação desde 2019 parou na verificação "'None' ou 'nan' escrito no texto". Era um falso alarme: um contrato de 2020 tem a gralha "águas residuais domésticas nan Quinta dos Farias" ("nan" por "na") no próprio Portal BASE. A verificação passou a olhar só para pedaços inteiros do texto entre separadores (`" | "`), que é onde um campo vazio escrito como texto aparece; os ficheiros desde 2022 não mudam.
- **Limitação:** a escolha usou a regra de paragem por defeito (o modelo em vigor usa 24 épocas). É o que o E4-S6 mede.
- **Custo do spike:** um pod `cpu3g` de 16 processadores e 64 GB, a 0,64 $/h, ligado cerca de 34 minutos: cerca de 0,36 $ (`metrics/e4_s2/pod_info.json`). O pico de memória do pod inteiro foi 19,7 GB de 64.

### 19. A janela desde 2019 com 24 épocas (E4-S6)

De [`notebooks/10_janela_24_epocas.ipynb`](../notebooks/10_janela_24_epocas.ipynb), que é o notebook 09 com outro YAML. Lê as sugestões do modelo em vigor (desde 2022, N = 10, 24 épocas: o treino do E4-S1, `metrics/e4_s1/n_min_10_ep_24/`) e as da janela desde 2019 com a mesma configuração (treino deste spike, `metrics/e4_s6/desde_2019/`).

**Porquê:** o E4-S2 (secção 18) escolheu a janela desde 2019 com a regra de paragem por defeito, mas o modelo em vigor treina 24 épocas fixas (decisão 7 do E4). Antes de mudar o `TRAIN_START`, confirmou-se a escolha com o modelo que se usa de facto.

**Como:** as opções e a regra foram escritas antes de correr em [`experiencias/e4_s6.yaml`](../experiencias/e4_s6.yaml) (no git antes do treino, commit `fc4996c`): o mesmo grupo de decisão do E4-S2, a regra do E4-T1, a de 2022 como opção em vigor e, em empate, fica a de 2022. Só a janela desde 2019 treinou, no pod, com os ficheiros de `data/janelas/2019/` (md5 iguais aos do E4-S2).

Resultados no grupo da decisão (20 150 contratos com texto novo para o treino desde 2019), τ = 0,25:

| | desde 2022 (em vigor) | desde 2019 | Diferença | desde 2019 melhor em |
|---|---|---|---|---|
| Top-5 código | 75,09% | 75,40% | +0,31 | 94,8% das amostras |
| Top-5 classe | 82,13% | 81,98% | −0,15 | 21,0% das amostras |
| Top-1 código | 51,99% | 52,18% | +0,19 | 79,9% das amostras |

(Em todos os contratos, só como informação: 78,35% e 78,63% no top-5 do código completo.)

**Escolha, com a regra escrita antes de correr:** nenhuma das duas diferenças do top-5 conta (é preciso 95% das amostras do mesmo lado; o código completo fica nos 94,8%). É empate, e fica a janela em vigor. Escolhida: **desde 2022** (decisão 1). Como o modelo em vigor não muda, a régua (secção 12) e os resultados do E4-H1 não se recalculam.

Porque é que o E4-S2 deu outra coisa:

| Top-5 código (grupo da decisão) | Regra por defeito (E4-S2) | 24 épocas (E4-S6) |
|---|---|---|
| desde 2022 | 73,46% (parou na época 12) | 75,09% |
| desde 2019 | 75,38% (parou na época 11) | 75,40% |

Com a regra por defeito, a janela desde 2022 parava cedo demais e perdia 1,6 pontos. A desde 2019 tem mais exemplos, logo cada época dá mais passos de aprendizagem, e em 11 épocas já tinha chegado ao seu máximo. A vantagem medida no E4-S2 era sobretudo de **treino mais completo**, não de mais dados.

Como ler:
- **A regra por defeito não era neutra.** O E4-S2 usou-a para nenhuma janela partir com vantagem, mas a paragem automática favoreceu a janela com mais exemplos. Lição que generaliza: ao comparar opções de dados, cada opção tem de treinar até ao seu melhor. Senão, mede-se quem aprende mais depressa, e não quem tem melhores dados.
- **Análise completa** (informação, não decide): nas 18 comparações do bootstrap (código, classe e divisão × top-1, top-5 e F1 macro, no grupo e em todos), a desde 2019 ganha em 6 (F1 macro no código; top-1 e F1 macro na divisão) e empata nas outras 12. As sugestões genéricas ficam iguais (30,57% → 30,42%, diferença que não conta). Por nível da etiqueta, a desde 2019 ganha nos códigos específicos (+1,4 pontos no nível 5, +3,1 no 6, +1,4 no 7) e perde na classe (−2,1 pontos no nível 4).
- **O ganho possível não paga o custo.** O treino desde 2019 levou 57,6 minutos, contra 31,2 (`treino.json`, 8 processadores), com um pico de 17,9 GB, e a preparação não cabe no portátil (secção 18). Para o BERTimbau e o Albertina, o tempo de GPU deve crescer mais ou menos na mesma proporção dos exemplos (estimativa, não medida).
- **Limitação:** as 24 épocas foram afinadas na janela desde 2022 (E4-S5/S1). Não se correu a grelha de épocas na janela desde 2019, que talvez tivesse outro ótimo. Decidido não a correr (14 de outubro de 2026): a diferença medida é pequena (+0,31 pontos, que não conta) e cada treino custa quase o dobro.
- **Custo do spike:** um pod `cpu3g` de 8 processadores e 32 GB, a 0,32 $/h (não havia stock com 16 processadores), ligado cerca de 63 minutos: cerca de 0,34 $ (`metrics/e4_s6/pod_info.json`). O pico de memória do pod inteiro foi 19,2 GB de 32.

## Decisões

1. Separação por `dataPublicacao`, lendo todos os ficheiros (ver [`notebooks/02_anos_treino.ipynb`](../notebooks/02_anos_treino.ipynb)):

   | Parte | Período | Para quê |
   |---|---|---|
   | Treino | janeiro de 2022 a setembro de 2025 | O modelo aprende com estes |
   | Validação | outubro de 2025 | Escolher entre modelos e afinações |
   | Teste | novembro e dezembro de 2025 | Medir o resultado final, uma só vez, no fim |
   | Reserva | 2026 (até 4 de outubro) | Segunda verificação no fim, com dados mais recentes |

   Porquê desde 2022: com só 2025, 92,9% dos contratos de validação têm um código com pelo menos 10 exemplos no treino; desde 2022 são 98,2%; recuar até 2012 só sobe para 99,3% e multiplica os dados por 2,4. Confirmar na fase 4: treinar o TF-IDF com as janelas desde 2025, 2022 e 2019 e comparar na validação.

   **Confirmado na fase 4 (14 de outubro de 2026): o treino fica desde 2022.**
   - **E4-S2 (secção 18):** com o TF-IDF e a regra de paragem por defeito, a janela desde 2019 ganhou à de 2022 (+1,92 pontos no top-5 do código completo), e a de 2022 ganhou à de 2025 (+10,18).
   - **E4-S6 (secção 19):** com as 24 épocas do modelo em vigor, a janela desde 2019 empata com a de 2022 (+0,31 pontos no top-5 do código completo, numa diferença que não conta). A regra escrita antes de correr manda ficar com a mais curta. A vantagem do E4-S2 vinha sobretudo de a regra por defeito parar cedo demais o treino desde 2022.
2. Tirar os 3 contratos sem CPV.
3. Ficar com uma só cópia de cada `idcontrato`: a do ficheiro mais recente e, no mesmo ficheiro, a de data de publicação mais recente.
4. Contratos com vários códigos: treinar com o primeiro. Na avaliação, contar como acerto se a sugestão estiver entre os códigos do contrato.
5. Texto de entrada: o `objectoContrato`, mais a `descContrato` só quando for diferente, mais a `DescrAcordoQuadro` sempre que existir.
6. Textos repetidos:
   - **No treino:** um só exemplo por texto (o texto de entrada completo da decisão 5), com o código mais comum desse texto. Acaba com as repetições (o modelo deixa de ver "medicamentos" 1 870 vezes) e com as contradições (cada texto tem uma só resposta). Perde 2,2% dos contratos de treino (14 348 no treino final), muitas vezes ruído.
   - **Na validação e no teste:** manter todos os contratos, para o resultado refletir a realidade. Medir também à parte só os contratos com texto que nunca apareceu no treino, para ver se o modelo aprendeu ou só decorou.
7. A `adjudicante` **não entra** no texto de entrada (fechada no E4-S3, 13 de outubro de 2026, secção 17). Testou-se treinar com e sem ela: com ela, o top-5 no código completo sobe de 78,35% para 80,76%, mas a parte das 1.as sugestões ao nível do grupo ou acima também sobe, de 30,57% para 33,13%, com uma diferença que conta no bootstrap. A regra escrita antes de correr não deixava a entidade entrar nesse caso: o modelo aprende os hábitos de cada entidade, incluindo os códigos genéricos.
8. Códigos genéricos (secção 9):
   - **No treino:** tirar só os contratos com código ao nível da divisão (17,6% em 2025; 18,8% no treino final) e manter todos os outros. O código ao nível da divisão quase nunca ajuda; os do grupo para baixo muitas vezes são escolhas legítimas. **É uma suposição de produto, não medida:** assume-se que um código de divisão não é uma sugestão útil para o técnico, mesmo quando foi a escolha registada. Custo: o modelo nunca sugere divisões, por isso não pode dar a resposta registada nos contratos que só têm códigos de divisão.
   - **Na avaliação:** cada contrato avalia-se em três medidas. No código completo, uma sugestão só acerta se for igual ao código: se a etiqueta é `33600000` (grupo), só `33600000` acerta, e `33651100` acerta na divisão mas não no código completo. (Mudado a 10 de outubro de 2026. Antes acertava qualquer sugestão começada por `336`, mas isso dava acerto total a uma sugestão mais específica do que o código escolhido, o que a literatura de classificação hierárquica conta como erro, Kosmopoulos et al., 2015. O meio acerto mede-se à parte, com as medidas hierárquicas da E4-T1.) Os contratos que só têm códigos de divisão não contam no código completo (o modelo não os pode acertar, ver acima): avaliam-se na divisão, e a avaliação mostra-os numa linha à parte (`so_divisao`), com quantos são e quantas vezes o modelo acerta a divisão. O acerto na classe (4 dígitos) só se mede nos cerca de 56% de contratos com etiqueta da classe para baixo, e isto tem de ficar dito no relatório final.
9. Códigos raros (back-off hierárquico, `src/labels.py`): um código com menos de N exemplos no treino sobe para a classe (4 dígitos); se a classe também for rara, sobe para o grupo (3 dígitos); se o grupo também for raro, o exemplo sai. Nunca se sobe até à divisão.
   - As contagens fazem-se só com o treino (contar com a validação seria fuga de dados).
   - Aplica-se no momento do treino, não no `prepare.py`: o N é um hiperparâmetro e o `treino.parquet` fica com os códigos originais.
   - Na avaliação a etiqueta verdadeira não muda: sugerir a classe quando o código certo é mais detalhado conta como acerto na classe e erro no código completo.
   - N a testar: 5, 10 e 20. **Escolhido N = 10** no E4-S1 (secção 16): a grelha N × épocas × τ mostrou que N = 10 é o melhor dos três, confirmando o valor por defeito. Continua a ser o valor por defeito.
10. Régua "códigos mais comuns" (E4-T2, secção 12): as 5 etiquetas mais frequentes do treino depois da regra dos códigos raros com o N do modelo, contadas **uma vez por exemplo** (um por texto), não pelo `n_contratos`. Primeiro escolheu-se contar contratos, porque a validação tem todos os contratos com as repetições. A medição mostrou que essa régua fica mais fácil de bater no código completo (diferença que conta no bootstrap), e mudou-se a 11 de outubro de 2026 para a mais exigente das duas. Escolher a régua depois de ver a validação não favorece o modelo: só lhe torna a tarefa mais difícil.
