# Qualidade dos dados

Resultados das verificações de qualidade (fase 2 do [`workflow.md`](workflow.md)) e as decisões que saem delas.

- Dados: ficheiros de 2025 e 2026 do Portal BASE, versão de 4 de outubro de 2026. 433 838 contratos no total.
- De onde vêm os números:
  - secções 1 a 6: [`notebooks/00_qualidade.ipynb`](../notebooks/00_qualidade.ipynb) (todos os dados; só conta);
  - secções 7 a 10: [`notebooks/01_exploracao.ipynb`](../notebooks/01_exploracao.ipynb) (só o treino; a validação só na secção 10).
- O script só conta. Os exemplos que mostra vêm apenas de contratos publicados antes de novembro de 2025, para não espreitar o período de teste.

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
| 2025 | 225 113 | treino, validação e teste |
| 2026 | 208 719 | por agora não se usa |

O ficheiro de um ano não coincide com o ano de publicação: os contratos publicados em 2025 estão espalhados pelos ficheiros de 2025 e 2026.

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

Medido só nos contratos de treino (janeiro a setembro de 2025, 166 112 contratos).

| Coluna | Preenchida | O que traz |
|---|---|---|
| `DescrAcordoQuadro` | 15,5% de todos os contratos, mas 97,7% dos que falam de acordo-quadro | A descrição do acordo-quadro, por exemplo "Concurso público internacional para formação do acordo quadro de medicamentos..." |
| `adjudicante` | 100% | A entidade que compra |

Os textos que falam de acordo-quadro são 9 981 (6,0%). Muitos não dizem o que se compra ("convite à apresentação de propostas ao abrigo do acordo-quadro..."), mas a `DescrAcordoQuadro` diz.

As duas colunas são informação que o técnico já tem quando escolhe o código. Por isso não é fuga de dados.

### 8. O mesmo texto com códigos diferentes: quanto se perde ao escolher o mais comum

Medido só nos contratos de treino (janeiro a setembro de 2025, 166 112 contratos).

Exemplo: o texto "medicamentos" no treino (1 565 contratos, 7 entidades; a secção 6 conta 1 870 porque inclui outubro):

| Código | Vezes |
|---|---|
| `33000000` Equipamento médico, medicamentos e produtos para cuidados pessoais | 1 457 |
| `33600000` Produtos farmacêuticos | 105 |
| `33690000` Medicamentos vários | 2 |
| `32000000` Equipamento de rádio, televisão, comunicação... | 1 |

Só 3 585 contratos de treino (2,2%) têm um código diferente do mais comum do seu texto.

Primeira ideia, errada: um exemplo por par (texto, código). Os códigos de "medicamentos" ficavam com um exemplo cada, e o modelo achava-os igualmente prováveis, apesar de um aparecer 1 457 vezes e outro 2.

### 9. Códigos genéricos: a que nível as entidades escolhem o código

Medido só nos contratos de treino (janeiro a setembro de 2025, 166 112 contratos).

O nível de um código conta-se pelos dígitos antes dos zeros finais: `33600000` tem 3 dígitos significativos (`336`), por isso é um código ao nível do grupo. É uma aproximação: não foi verificado, código a código, que a tabela oficial do CPV segue sempre esta regra.

| Nível do código escolhido | Exemplo | Contratos | % |
|---|---|---|---|
| Divisão (2 dígitos) | `33000000` | 29 281 | 17,6% |
| Grupo (3) | `33600000` Produtos farmacêuticos | 41 737 | 25,1% |
| Classe (4) | `33690000` Medicamentos vários | 40 029 | 24,1% |
| Categoria (5) | `90911000` | 26 997 | 16,3% |
| Mais detalhado (6 a 8) | `90911200` Limpeza de edifícios | 28 068 | 16,9% |

Só 33% dos contratos têm código ao nível da categoria ou mais detalhado.

Por divisão:
- A divisão `33` (equipamento médico e medicamentos) tem 50 405 contratos, 30% do treino. Desses, 26% estão ao nível da divisão e 70% ao nível do grupo ou acima.
- Tirar os códigos ao nível da divisão não apaga nenhuma divisão. A que fica pior é a `41` (água captada): de 28 contratos para 7.
- Tirar também os do nível do grupo custaria 43% do treino e 70% da divisão `33`.

Genérico não é o mesmo que errado. Um contrato que compra muitos medicamentos diferentes pode estar bem classificado como `33600000` Produtos farmacêuticos. Com estes dados não se distingue um genérico legítimo de um preguiçoso.

### 10. Quantos exemplos tem cada código (desequilíbrio de classes)

Medido depois de aplicar as decisões 1 a 4 e 8 (sem os códigos ao nível da divisão), nos contratos de treino (janeiro a setembro de 2025, 136 831 contratos). A validação (outubro de 2025, 18 718 contratos) só entra na última coluna, para ver a cobertura.

Códigos distintos no treino:

| Nível | Códigos distintos |
|---|---|
| Divisão (2 dígitos) | 45 |
| Grupo (3) | 260 |
| Classe (4) | 1 013 |
| Código como foi escolhido (8) | 4 330 |

Exemplos por código (código como foi escolhido):
- Mediana: 4 exemplos. Metade dos códigos tem 4 ou menos.
- 1 103 códigos têm um só exemplo.
- O mais comum, `33600000` Produtos farmacêuticos, tem 18 987 (14% do treino).
- Os 10 códigos mais comuns cobrem 28,0% do treino.

Se só se treinar com códigos que têm pelo menos N exemplos:

| N | Códigos que ficam | % dos contratos de treino | % dos contratos de validação |
|---|---|---|---|
| 1 | 4 330 | 100,0% | 99,0% |
| 2 | 3 227 | 99,2% | 98,1% |
| 5 | 2 089 | 96,9% | 95,7% |
| 10 | 1 418 | 93,7% | 92,2% |
| 20 | 911 | 88,6% | 87,1% |
| 50 | 431 | 77,8% | 76,6% |
| 100 | 229 | 67,4% | 66,6% |

## Decisões

1. Usar só os contratos publicados em 2025 (pela `dataPublicacao`), dos ficheiros de 2025 e 2026.
2. Tirar os 3 contratos sem CPV.
3. Ficar com uma só cópia de cada `idcontrato`.
4. Contratos com vários códigos: treinar com o primeiro. Na avaliação, contar como acerto se a sugestão estiver entre os códigos do contrato.
5. Texto de entrada: o `objectoContrato`, mais a `descContrato` só quando for diferente, mais a `DescrAcordoQuadro` sempre que existir.
6. Textos repetidos:
   - **No treino:** um só exemplo por texto (o texto de entrada completo da decisão 5), com o código mais comum desse texto. Acaba com as repetições (o modelo deixa de ver "medicamentos" 1 870 vezes) e com as contradições (cada texto tem uma só resposta). Perde 2,2% dos contratos de treino, muitas vezes ruído.
   - **No teste:** manter todos os contratos, para o resultado refletir a realidade. Medir também à parte só os contratos com texto que nunca apareceu no treino, para ver se o modelo aprendeu ou só decorou.
7. A `adjudicante` entra como experiência: treinar com e sem ela e comparar. Pode ajudar (um hospital e uma câmara compram coisas diferentes), mas o modelo pode aprender os hábitos de cada entidade, incluindo os erros, e falhar com entidades que não viu no treino.
8. Códigos genéricos (secção 9):
   - **No treino:** tirar só os contratos com código ao nível da divisão (17,6%) e manter todos os outros. O código ao nível da divisão quase nunca ajuda; os do grupo para baixo muitas vezes são escolhas legítimas.
   - **Na avaliação:** cada contrato de teste avalia-se até ao nível que a etiqueta tem. Se a etiqueta é `33600000` (grupo), só se verifica se o modelo acertou em `336`. Por isso o acerto na classe (4 dígitos) só se mede nos cerca de 57% de contratos com etiqueta da classe para baixo, e isto tem de ficar dito no relatório final.
