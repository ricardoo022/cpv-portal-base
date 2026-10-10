# Qualidade dos dados

Resultados das verificações de qualidade (fase 2 do [`workflow.md`](workflow.md)) e as decisões que saem delas.

- Dados: Portal BASE, versão de 4 de outubro de 2026.
- **Âmbito das medições (importante):**
  - As secções 1 a 6 foram medidas só com os ficheiros de 2025 e 2026 (433 838 contratos). Depois descobriu-se que o ficheiro de um ano é o ano de celebração (secção 2) e decidiu-se treinar desde 2022 (decisão 1). Servem para as decisões (são problemas técnicos que se repetem em todos os anos), mas os números não são os do treino final.
  - As secções 7 a 10 estão medidas com o **treino final**: contratos publicados de janeiro de 2022 a setembro de 2025, lidos de todos os ficheiros. A primeira versão destas secções usava só janeiro a setembro de 2025; onde a conclusão mudou, está dito.
  - A secção 11 é a saída do `src/prepare.py`.
- De onde vêm os números:
  - secções 1 a 6: [`notebooks/00_qualidade.ipynb`](../notebooks/00_qualidade.ipynb) (só conta);
  - secções 7 a 10: [`notebooks/01_exploracao.ipynb`](../notebooks/01_exploracao.ipynb) (só o treino; a validação só na secção 10). Dois números deste notebook têm de bater com o funil da secção 11, e batem: 794 883 contratos de treino e 14 348 fora do código mais comum do seu texto.
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

## Decisões

1. Separação por `dataPublicacao`, lendo todos os ficheiros (ver [`notebooks/02_anos_treino.ipynb`](../notebooks/02_anos_treino.ipynb)):

   | Parte | Período | Para quê |
   |---|---|---|
   | Treino | janeiro de 2022 a setembro de 2025 | O modelo aprende com estes |
   | Validação | outubro de 2025 | Escolher entre modelos e afinações |
   | Teste | novembro e dezembro de 2025 | Medir o resultado final, uma só vez, no fim |
   | Reserva | 2026 (até 4 de outubro) | Segunda verificação no fim, com dados mais recentes |

   Porquê desde 2022: com só 2025, 92,9% dos contratos de validação têm um código com pelo menos 10 exemplos no treino; desde 2022 são 98,2%; recuar até 2012 só sobe para 99,3% e multiplica os dados por 2,4. Confirmar na fase 4: treinar o TF-IDF com as janelas desde 2025, 2022 e 2019 e comparar na validação.
2. Tirar os 3 contratos sem CPV.
3. Ficar com uma só cópia de cada `idcontrato`: a do ficheiro mais recente e, no mesmo ficheiro, a de data de publicação mais recente.
4. Contratos com vários códigos: treinar com o primeiro. Na avaliação, contar como acerto se a sugestão estiver entre os códigos do contrato.
5. Texto de entrada: o `objectoContrato`, mais a `descContrato` só quando for diferente, mais a `DescrAcordoQuadro` sempre que existir.
6. Textos repetidos:
   - **No treino:** um só exemplo por texto (o texto de entrada completo da decisão 5), com o código mais comum desse texto. Acaba com as repetições (o modelo deixa de ver "medicamentos" 1 870 vezes) e com as contradições (cada texto tem uma só resposta). Perde 2,2% dos contratos de treino (14 348 no treino final), muitas vezes ruído.
   - **Na validação e no teste:** manter todos os contratos, para o resultado refletir a realidade. Medir também à parte só os contratos com texto que nunca apareceu no treino, para ver se o modelo aprendeu ou só decorou.
7. A `adjudicante` entra como experiência: treinar com e sem ela e comparar. Pode ajudar (um hospital e uma câmara compram coisas diferentes), mas o modelo pode aprender os hábitos de cada entidade, incluindo os erros, e falhar com entidades que não viu no treino.
8. Códigos genéricos (secção 9):
   - **No treino:** tirar só os contratos com código ao nível da divisão (17,6% em 2025; 18,8% no treino final) e manter todos os outros. O código ao nível da divisão quase nunca ajuda; os do grupo para baixo muitas vezes são escolhas legítimas.
   - **Na avaliação:** cada contrato de teste avalia-se até ao nível que a etiqueta tem. Se a etiqueta é `33600000` (grupo), só se verifica se o modelo acertou em `336`. Por isso o acerto na classe (4 dígitos) só se mede nos cerca de 56% de contratos com etiqueta da classe para baixo, e isto tem de ficar dito no relatório final.
9. Códigos raros (back-off hierárquico, `src/labels.py`): um código com menos de N exemplos no treino sobe para a classe (4 dígitos); se a classe também for rara, sobe para o grupo (3 dígitos); se o grupo também for raro, o exemplo sai. Nunca se sobe até à divisão.
   - As contagens fazem-se só com o treino (contar com a validação seria fuga de dados).
   - Aplica-se no momento do treino, não no `prepare.py`: o N é um hiperparâmetro e o `treino.parquet` fica com os códigos originais.
   - Na avaliação a etiqueta verdadeira não muda: sugerir a classe quando o código certo é mais detalhado conta como acerto na classe e erro no código completo.
   - N a testar: 5, 10 e 20. Por defeito 10, até a fase 4 escolher com o TF-IDF na validação.
