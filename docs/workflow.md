# Workflow

O processo que seguimos neste projeto. É baseado no CRISP-DM, um processo de referência para projetos de dados e ML criado no fim dos anos 90 por um consórcio de empresas (DaimlerChrysler, SPSS, NCR, entre outras).

O CRISP-DM não é uma lei, é uma referência. E é iterativo: anda-se para trás e para a frente entre fases. Um mau resultado na avaliação pode obrigar a voltar à preparação dos dados.

Cada fase tem duas partes: o que é geral a qualquer projeto de ML e o que fazemos neste.

Os épicos e as user stories (o que já está feito e o próximo épico, com critérios de aceitação) estão em [`backlog.md`](backlog.md).

## 1. Perceber o problema

**Geral:** definir que problema queremos resolver, para quem, e como sabemos se o resolvemos (critério de sucesso).

**Neste projeto:** feito. Ver as secções "O problema" e "A tarefa, em termos de código" no [`readme.md`](../readme.md).
- Problema: sugerir o código CPV a partir do texto do contrato.
- Sucesso: o código certo aparece entre as 5 sugestões, no código completo e na classe (4 dígitos).

## 2. Perceber os dados

**Geral:** recolher os dados, verificar que são os certos e explorá-los com números antes de treinar seja o que for (análise exploratória, em inglês EDA).

Cuidado: separar os dados de teste antes de explorar a fundo. Se olharmos para os dados de teste, podemos tomar decisões com base no que lá vimos, e depois o modelo parece bom no teste mas falha com dados novos. Chama-se a isto data snooping ("espreitar os dados").

**Neste projeto:**
- [x] Descarregar os dados (`scripts/download_data.py`).
- [x] Verificar que são os certos: formato, colunas, número de linhas, anos.
- [x] Converter para Parquet sem transformar nada (`src/convert_to_parquet.py`).
- [x] Verificações de qualidade em todos os dados, em [`notebooks/00_qualidade.ipynb`](../notebooks/00_qualidade.ipynb) (são problemas técnicos, não dependem do resultado):
  - percentagem de contratos com texto e CPV preenchidos;
  - contratos com vários códigos na mesma célula;
  - textos duplicados.
  - Resultados e decisões em [`qualidade_dados.md`](qualidade_dados.md).
- [x] Não olhar para o teste (contratos publicados em novembro e dezembro de 2025) até à fase 5. Até aqui filtrou-se pela data; os ficheiros separados criam-se na fase 3.
- [x] Exploração só nos dados de treino, em [`notebooks/01_exploracao.ipynb`](../notebooks/01_exploracao.ipynb) (resultados nas secções 7 a 10 de [`qualidade_dados.md`](qualidade_dados.md)):
  - códigos distintos em cada nível da árvore (divisão, grupo, classe, código completo);
  - quantos exemplos tem cada código (desequilíbrio de classes).
- [x] Quantos anos usar no treino, em [`notebooks/02_anos_treino.ipynb`](../notebooks/02_anos_treino.ipynb).
- [x] Voltar a medir as secções 7 a 10 com o treino final (janeiro de 2022 a setembro de 2025, todos os ficheiros). A primeira versão usava só janeiro a setembro de 2025 e só os ficheiros de 2025 e 2026. As conclusões mantêm-se; a cobertura da validação com N = 10 subiu de 92,2% para 97,9%.

## 3. Preparar os dados

**Geral:** limpar e transformar os dados no formato que o modelo precisa, e separá-los em treino, validação e teste. Os dados originais nunca se alteram: a limpeza é um script que se pode corrigir e voltar a correr.

**Neste projeto:**
- [x] Extrair o código do `CPV` (`90611000-3 - Serviços de limpeza de ruas` → `90611000`) e os níveis da árvore.
- [x] Contratos com vários códigos: treinar com o primeiro, guardar os restantes para a avaliação.
- [x] Texto de entrada: objeto + descrição (se diferente) + descrição do acordo-quadro (se existir).
- [x] No treino: tirar os códigos ao nível da divisão e juntar os textos repetidos num só exemplo. Na validação e no teste: ficam todos, com a marca `visto_no_treino`.
- [x] Separar por `dataPublicacao`, lendo todos os ficheiros: treino (janeiro de 2022 a setembro de 2025), validação (outubro de 2025), teste (novembro e dezembro de 2025). 2026 fica de reserva. Decisão em [`notebooks/02_anos_treino.ipynb`](../notebooks/02_anos_treino.ipynb).
- [x] Script de preparação `src/prepare.py`, com verificações automáticas no fim.
- [x] Testes unitários e de integração (`.venv/bin/pytest`), verificados com erros introduzidos de propósito.
- [x] Regra para os códigos raros: sobem para a classe ou o grupo (`src/labels.py`, decisão 9 de [`qualidade_dados.md`](qualidade_dados.md)).
- [x] Escolher o N (5, 10 ou 20) com o TF-IDF na validação, numa grelha com as épocas e o τ (spike E4-S1): fica N = 10, o valor por defeito ([`qualidade_dados.md`](qualidade_dados.md), secção 16).
- [x] Testar a entidade compradora (`adjudicante`) no texto de entrada com o TF-IDF (spike E4-S3): não entra. Acerta mais, mas aumenta as 1.as sugestões genéricas, e a regra escrita antes de correr proibia isso ([`qualidade_dados.md`](qualidade_dados.md), secção 17, decisão 7).

## 4. Modelar

**Geral:** começar por um modelo simples (baseline), para ter com que comparar. Depois treinar os modelos a sério e escolher o melhor pela validação, nunca pelo teste.

**Neste projeto:**
- [x] Avaliação comum a todos os modelos (`src/evaluation.py`, `src/comparison.py`, tarefa E4-T1): top-1 e top-5 no código completo, na classe e na divisão, F1 macro, medidas hierárquicas, e o bootstrap emparelhado para decidir entre dois modelos.
- [x] Régua "códigos mais comuns" (`src/baselines.py`, tarefa E4-T2): sugerir sempre as 5 etiquetas mais frequentes do treino. Na validação, top-5 de 26,3% no código completo e 15,6% na classe ([`qualidade_dados.md`](qualidade_dados.md), secção 12).
- [x] Medir o custo de um treino do TF-IDF numa máquina alugada (spike E4-S0): com o treino completo, 8 a 10 minutos e 6 a 7 GB de memória num pod da Runpod com 32 GB, a 0,32 $/h. Guia da máquina em [`maquina_runpod.md`](maquina_runpod.md).
- [x] Baseline: TF-IDF com regressão logística (treinada com o `SGDClassifier(loss="log_loss")`, decisão 5 do E4), num pod da Runpod com 32 GB de memória (história E4-H1). Na validação, top-5 de 43,2% no código completo e 37,9% na classe, contra 26,3% e 15,6% da régua ([`qualidade_dados.md`](qualidade_dados.md), secção 13).
- [x] Afinar o TF-IDF (spike E4-S4): `alpha` = 0,0000001 e correção das etiquetas frequentes com τ = 0,25. Na validação, top-5 de 76,9% no código completo e 82,9% na classe ([`qualidade_dados.md`](qualidade_dados.md), secção 14).
- [x] Afinar as épocas do TF-IDF e testar um `alpha` mais pequeno (spike E4-S5): o `alpha` 0,00000001 perde; ganham 24 épocas fixas. Na validação, top-5 de 78,3% no código completo e 84,1% na classe ([`qualidade_dados.md`](qualidade_dados.md), secção 15).
- [x] Confirmar a janela de treino: TF-IDF com treino desde 2025, 2022 e 2019, comparar na validação. Fica desde 2022 (E4-S2 e E4-S6, secções 18 e 19 do `qualidade_dados.md`).
- [x] Medir o comprimento dos textos em *tokens* com o tokenizador de cada modelo (E5-T1): 256 *tokens* para o BERTimbau e para o Albertina ([`qualidade_dados.md`](qualidade_dados.md), secção 20).
- [ ] Tunar o BERTimbau base e o Albertina PT-PT 100M como classificadores, numa GPU alugada.
- [ ] Uma grelha de hiperparâmetros para cada um (taxa de aprendizagem × N × épocas × τ, 9 treinos por modelo), registados em MLflow (E5-S1 no [`backlog.md`](backlog.md)).
- [ ] Escolher o melhor de cada pela validação.
- [ ] Testes ao modelo: consegue decorar poucos exemplos (se não conseguir, há um erro no treino); só sugere códigos que existem; bate o "escolher sempre o código mais comum".
- [x] Portão de qualidade (*quality gate*): o CI falha se um modelo ficar abaixo do limite. Feito no E4-T5 (decisão 8 do E4): a referência é o campeão, o melhor modelo até agora, e qualquer descida num top-5 recusa o modelo novo (`src/model_gate.py`, `metrics/validacao.json`, registo no MLflow).
  - O limite mede-se na **validação**, nunca no teste (senão o teste passa a servir para escolher, e deixa de ser uma medição honesta).
  - O limite é relativo, não um número inventado: o TF-IDF tem de bater o "escolher sempre o código mais comum"; o BERTimbau e o Albertina têm de bater o TF-IDF; um modelo novo não pode ficar mais do que alguns pontos abaixo do melhor guardado. Os números definem-se depois dos primeiros resultados.
  - O CI não treina os modelos (não tem GPU nem os dados reais). O treino a sério escreve as métricas da validação num ficheiro pequeno no git (por exemplo `metrics/validacao.json`), e um teste no CI lê esse ficheiro e falha se estiver abaixo do limite.
  - No CI corre também um teste rápido de treino com dados inventados minúsculos.

## 5. Avaliar

**Geral:** medir com dados que o modelo nunca viu (o teste), uma só vez, no fim. Olhar para os erros, não só para os números.

**Neste projeto:**
- [ ] Métricas no teste: top-1 e top-5 no código completo e na classe, acerto na divisão, F1 macro em todos os níveis, e as medidas hierárquicas. Com a mesma avaliação da validação (`src/evaluation.py`).
- [ ] Duas situações, com o mesmo modelo:
  1. O modelo sugere 5 códigos entre todos.
  2. O técnico diz a divisão e o modelo sugere 5 códigos só dentro dela.
- [ ] Comparar com o BERT multilingue já tunado com concursos europeus, só nas divisões.
- [ ] Análise de erros: ler 100 casos em que o modelo discorda da etiqueta e decidir quem tem razão.
- [ ] Tabela final com todos os modelos e conclusão.

Se os resultados forem maus, voltar à fase 3 ou 4. Se o modelo único falhar muito dentro das divisões, testar um modelo especializado por divisão.

## 6. Pôr em uso

**Geral:** integrar o modelo num sistema real ou apresentar os resultados.

**Neste projeto:** fora do âmbito, por agora. O resultado é a tabela e a conclusão da fase 5.

## Termos

Vamos acrescentando à medida que aparecem.

- **Etiqueta:** a resposta certa que o modelo aprende a dar. Aqui, o código CPV que a entidade escolheu.
- **Treino, validação, teste:** três partes separadas dos dados. Com o treino o modelo aprende. Com a validação escolhemos entre modelos e hiperparâmetros. Com o teste medimos o resultado final, uma só vez.
- **Hiperparâmetros:** as escolhas que fazemos antes de treinar e que o modelo não aprende sozinho (por exemplo, quantas vezes passa pelos dados de treino).
- **Baseline:** um modelo simples que serve de comparação. Se o modelo complexo não o bater, não vale a pena.
- **Data snooping:** tomar decisões com base nos dados de teste, o que faz o resultado no teste parecer melhor do que é.
- **Fuga de dados (data leakage):** quando o modelo vê no treino informação que não devia, e por isso parece melhor no teste do que vai ser na realidade. Por exemplo, o mesmo texto no treino e no teste: o modelo acerta porque decorou, não porque aprendeu.
- **Informação disponível no momento da previsão:** só se dá ao modelo informação que existe quando ele vai ser usado. Se usarmos uma coluna que só se preenche depois (por exemplo o preço final), o modelo parece bom no teste e falha na realidade. É uma forma de fuga de dados.
- **Top-k:** o contrato conta como acerto se o código certo estiver entre as k primeiras sugestões. Top-1: só a 1.ª; top-5: qualquer uma das 5.
- **Sobre-especialização:** o modelo sugere um código mais detalhado do que o certo (o técnico escolheu `33600000`, o modelo diz `33651100`). Conta como erro no código completo, como na literatura (Kosmopoulos et al., 2015).
- **Medidas hierárquicas:** dão meio acerto a uma sugestão perto do código certo, contando os níveis da árvore em comum. Exemplo: certo `45233200`, sugerido `45200000` partilham 2 de 5 níveis. Só se mostram; não decidem.
- **Bootstrap emparelhado:** para saber se a diferença entre dois modelos é real ou sorte, tira-se 1 000 vezes, à sorte, uma amostra dos contratos, a mesma para os dois, e vê-se quem ganha. A diferença só conta se o mesmo ganhar em pelo menos 95% das amostras.
- **A avaliação só é tão boa como as etiquetas:** quando as etiquetas são incompletas (por exemplo um código genérico), mede-se só o que elas permitem medir e diz-se claramente que parte ficou de fora.
- **Teste unitário:** testa uma função sozinha, com dados pequenos inventados em que se sabe a resposta certa.
- **Teste de integração:** testa várias peças a funcionar juntas, do princípio ao fim. Aqui: o pipeline inteiro, de .xlsx a treino/validação/teste.
- **Determinismo:** correr duas vezes com os mesmos dados dá exatamente o mesmo resultado. Sem isto não se consegue reproduzir nem comparar experiências.
- **CI (integração contínua):** uma máquina do GitHub corre os testes automaticamente em cada envio e em cada PR (`.github/workflows/testes.yml`).
- **Portão de qualidade (quality gate):** uma regra automática que impede um modelo pior de entrar, por exemplo "o modelo novo tem de bater o TF-IDF na validação".
- **Mutation testing:** introduzir erros de propósito no código para confirmar que os testes os apanham. Um teste que passa sempre não prova nada.
- **Funil:** quantos dados ficam depois de cada passo da preparação. Mostra quanto custa cada decisão.
- **Impressão digital (hash):** um número calculado a partir de um texto. Textos iguais dão o mesmo número, e um número ocupa muito menos memória do que o texto.
- **Back-off hierárquico:** quando um código tem poucos exemplos, usa-se o código do nível de cima da árvore (a classe ou o grupo), que tem mais.
- **Desequilíbrio de classes:** quando algumas etiquetas têm muitos exemplos e outras muito poucos.
- **Top-1 e top-5:** se a resposta certa é a primeira sugestão (top-1) ou está entre as 5 primeiras (top-5).
- **F1 macro:** o F1 mede, para cada etiqueta, um equilíbrio entre não dar falsos positivos e não falhar casos verdadeiros. O "macro" faz a média entre todas as etiquetas com o mesmo peso, por isso as etiquetas raras contam tanto como as frequentes. Explicamos em detalhe na fase 5.
- **Tunar (fine-tuning):** pegar num modelo já treinado (que já sabe português) e continuar a treiná-lo com os nossos dados para uma tarefa específica.
