# Problemas dos dados no início do projeto

Lista para discutir que consequência teve cada problema no modelo e decidir quais vale a pena contar no vídeo.

Os commits são de 10 de outubro de 2026. Os números vêm de [`docs/qualidade_dados.md`](../docs/qualidade_dados.md), [`docs/dados.md`](../docs/dados.md) e [`notebooks/dados_01_ano_do_ficheiro.ipynb`](notebooks/dados_01_ano_do_ficheiro.ipynb). Âmbito de cada número:

- **Só 2025 e 2026:** medido nos ficheiros de 2025 e 2026.
- **Treino final:** contratos publicados de janeiro de 2022 a setembro de 2025.

## Os problemas

| # | Problema | Evidência | Como se resolveu | Commit |
|---|---|---|---|---|
| 1 | O ano do ficheiro é o da assinatura, não o da publicação | 29 898 de 255 011 contratos publicados em 2025 (11,7%) ficaram fora dos ficheiros de 2025 e 2026 | Ler todos os ficheiros, filtrar por `dataPublicacao` e refazer as medições | `b249110`, `9f1ab56` |
| 2 | Uma medição incluía outubro, o mês de validação | A secção 7 estava medida de janeiro a outubro de 2025 | Medir só com o treino | `e1d3246` |
| 3 | O mesmo texto com códigos diferentes | 6 849 textos com mais do que um código (treino final) | Um exemplo por texto, com o código mais comum. Saem 14 348 contratos (2,2%). A primeira ideia, um exemplo por par (texto, código), estava errada | `3df5f48` |
| 4 | Textos duplicados | 139 064 contratos (32%) com texto igual ao de outro (só 2025 e 2026) | Um só exemplo por texto no treino | `3df5f48` |
| 5 | Textos vagos | "medicamentos" ou "convite à apresentação de propostas ao abrigo do acordo-quadro..." não dizem o que se compra | Juntar a `DescrAcordoQuadro` ao texto de entrada, que resolve os do acordo-quadro | `3df5f48` |
| 6 | Códigos genéricos | 149 258 contratos (18,8%) só ao nível da divisão (treino final) | Tirá-los do treino e avaliá-los à parte, só na divisão | `3df5f48` |
| 7 | Códigos com poucos exemplos | 1 026 códigos com um só exemplo; metade dos códigos com 8 ou menos (treino final) | Com menos de 10 exemplos, o código sobe para a classe e depois para o grupo | `3df5f48`, `a3c8edb` |
| 8 | Fuga de dados por textos repetidos | 3 706 contratos de validação (15,5%) têm texto que já aparece no treino | Manter os repetidos na validação e no teste, e medir também só os textos novos | `3df5f48` |
| 9 | `descContrato` repetida | Igual ao `objectoContrato` em 313 691 contratos (72%, só 2025 e 2026) | Só entra no texto quando é diferente | `3df5f48` |
| 10 | Vários códigos na mesma célula | 5 242 contratos (1,2%, só 2025 e 2026) | Treinar com o primeiro; na avaliação, acerta se a sugestão estiver entre os códigos do contrato | `3df5f48` |
| 11 | Contratos repetidos com o mesmo `idcontrato` | 1 199 linhas (0,3%, só 2025 e 2026) | Ficar com uma só cópia de cada contrato | `3df5f48`, `80e9a78` |
| 12 | Contratos sem CPV | 3 contratos (só 2025 e 2026) | Retirá-los | `3df5f48` |
| 13 | Memória | A conversão lê linha a linha "para não esgotar a RAM" | Ler linha a linha, gravar aos blocos e processar ficheiro a ficheiro | `3df5f48` |
| 14 | Colunas que só existem depois da adjudicação | Por exemplo, `adjudicatarios` e `concorrentes` | Só usar colunas que existem quando o técnico escolhe o CPV | `12d2cdc` |
| 15 | Sem dicionário de dados | O dados.gov.pt só tem os ficheiros | Descrever cada coluna em `docs/dados.md` e marcar com (?) o que não é certo | `12d2cdc` |

À parte, porque foi decidido mais à frente: **quantos anos usar no treino** (`b249110`, `f62bc5c`).

## O que vamos contar no vídeo

Decidido: vamos falar destes 4 problemas. As consequências são raciocínio, não medições, porque o modelo nunca foi treinado com estes problemas por resolver. No vídeo dizem-se como risco, não como resultado.

| # | Problema | Consequência no modelo |
|---|---|---|
| 1 | Ano do ficheiro | Treino, validação e teste ficavam sem parte dos contratos, e as medições estavam erradas |
| 3 | Mesmo texto, códigos diferentes | A mesma pergunta tinha várias respostas, e a primeira solução tornava igualmente prováveis códigos muito desiguais |
| 6 | Códigos genéricos | O modelo aprendia a sugerir códigos que quase não ajudam o técnico. Com a solução, não consegue acertar o código exato nos 16,9% de contratos de validação só com divisão |
| 8 | Fuga por textos repetidos | O acerto parecia melhor do que é, porque o modelo podia ter só decorado a resposta |

Os outros 11 problemas ficam fora do vídeo.
