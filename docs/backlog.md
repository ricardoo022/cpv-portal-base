# Backlog: épicos e user stories

Todos os épicos do projeto num só sítio: os que já estão feitos e o próximo. O plano por fases (CRISP-DM) e as caixas de verificação estão no [`workflow.md`](workflow.md); as medições e as decisões numeradas no [`qualidade_dados.md`](qualidade_dados.md). Cada épico novo acrescenta-se aqui; quando acaba, passa a "feito".

| Épico | Fase do workflow | Estado |
|---|---|---|
| E1 · Saber que problema resolvemos | 1 | feito |
| E2 · Conhecer os dados antes de treinar | 2 | feito |
| E3 · Treino, validação e teste em que se pode confiar | 3 | feito |
| E4 · Primeiras sugestões de CPV com um modelo simples | 4 | próximo |
| Tunar o BERTimbau e o Albertina | 4 | por escrever, depois do E4 |
| Avaliar no teste e analisar os erros | 5 | por escrever |

## Glossário

- **Épico:** um resultado grande que se quer atingir, partido em itens mais pequenos. Tem objetivo, medida de sucesso e o que fica de fora.
- **User story (história):** um item que entrega algo a um utilizador real. Escreve-se "Como `<quem>`, quero `<o quê>`, para `<porquê>`".
- **Persona:** o utilizador real. Aqui é o **técnico de contratação** que prepara um procedimento e escolhe o CPV à mão.
- **Tarefa técnica:** trabalho necessário que nenhum utilizador vê diretamente (dados, avaliação, testes, CI).
- **Spike:** uma pergunta em aberto com prazo máximo. O resultado é uma decisão escrita, não código para ficar.
- **Timebox:** o prazo máximo de um spike.
- **Critério de aceitação:** uma condição que se verifica com sim ou não para dar o item por feito. Diz o que se vê, não como se programa.
- **INVEST:** seis qualidades de uma boa história: Independente, Negociável, Valiosa, Estimável, pequena (*Small*), Testável.
- **Régua "códigos mais comuns":** sugerir sempre os mesmos 5 códigos, os mais frequentes no treino, sem ler o texto. É o mínimo que qualquer modelo tem de bater.
- **Smoke test:** um teste rápido que só confirma que o treino corre e faz algo com sentido, com dados minúsculos.
- **Portão de qualidade:** um teste no CI que falha se um modelo ficar abaixo de um limite.
- **Tamanhos:** P = até 1 dia; M = até 3 dias. Um item maior divide-se.

**Porque há poucas user stories nos E1 a E3:** quase todo esse trabalho é sobre dados e testes, que nenhum técnico vê. Por isso são tarefas técnicas e spikes, não histórias. Chamar história a tudo é um erro comum: esconde que o utilizador ainda não recebeu nada. A primeira capacidade para o técnico é a história E4-H1.

---

## E1 · Saber que problema resolvemos (feito)

**Objetivo:** saber que problema resolvemos, para quem, e como sabemos que está resolvido, antes de tocar em dados.

**Medida de sucesso:** o [`readme.md`](../readme.md) diz o problema, a persona, a medida de sucesso, o que está fora do âmbito e que modelos se vão treinar. ✅

**Fora do âmbito:** sugestão de preço base, matching concurso-empresa, API e demo.

### E1-S1 · Spike: o problema existe e já foi resolvido noutros países? (feito)

**Pergunta:** há quem tenha tentado sugerir o CPV a partir do texto, e com que resultados?

**Resultado:** três casos no `readme.md` (Zwolle, estudo italiano, artigo sobre qualidade de dados). As réguas publicadas são todas nas divisões: cerca de 75% a 79% de accuracy, com F1 macro de 0,30 a 0,40. Decisão: medir também nos níveis detalhados, com top-5, que é onde o problema existe.

### E1-T1 · Tarefa: definir a medida de sucesso (feito)

**Porque é precisa:** sem ela não se sabe quando um modelo é bom o suficiente.

**Critérios de aceitação:**
- [x] O `readme.md` diz que a medida principal é o top-1 e o top-5 no código completo e na classe (4 dígitos).
- [x] A divisão (2 dígitos) mede-se só para comparar com as réguas publicadas.
- [x] O F1 macro reporta-se sempre, porque as classes são desequilibradas.

### E1-S2 · Spike: que modelos treinar? (feito)

**Pergunta:** criar um modelo de raiz ou tunar um que já sabe português? Quais?

**Resultado:**
- Tunar o BERTimbau base e o Albertina PT-PT 100M como classificadores, com o TF-IDF e a regressão logística como baseline.
- AMALIA e Qwen ficam de fora.
- Um só modelo para todos os códigos. Um modelo por divisão só se testa se o modelo único falhar muito dentro das divisões.

---

## E2 · Conhecer os dados antes de treinar (feito)

**Objetivo:** saber o que os dados têm, que problemas trazem e que decisões isso obriga, sem espreitar o teste.

**Medida de sucesso:**
- Cada número do `qualidade_dados.md` sai de uma célula de notebook. ✅
- Cada medição levou a uma decisão numerada. ✅
- O teste (novembro e dezembro de 2025) não foi usado em nenhuma decisão. ✅

**Fora do âmbito:** transformar os dados (isso é o E3).

### E2-T1 · Tarefa: descarregar os dados (feito)

**Porque é precisa:** sem dados não há nada.

**Critérios de aceitação:**
- [x] Os 15 ficheiros .xlsx do Portal BASE (2012 a 2026) estão em `data/raw/`, com as mesmas 39 colunas em todos.
- [x] Voltar a correr o download não descarrega outra vez os ficheiros que não mudaram.
- [x] Um download interrompido não deixa um ficheiro meio escrito com o nome final.

### E2-T2 · Tarefa: converter para Parquet sem transformar nada (feito)

**Porque é precisa:** ler um .xlsx demora cerca de 1 minuto; o Parquet lê-se em segundos.

**Critérios de aceitação:**
- [x] Cada ficheiro .xlsx dá um Parquet com as mesmas linhas e as mesmas 39 colunas, todas como texto.
- [x] As datas em número de série do Excel aparecem como datas normais.

### E2-T3 · Tarefa: descrever as 39 colunas (feito)

**Porque é precisa:** é preciso saber que colunas podem entrar no modelo sem fuga de dados.

**Critérios de aceitação:**
- [x] O [`dados.md`](dados.md) diz, para cada coluna, o que é, quanto está preenchida, exemplos, se existe quando o técnico escolhe o código e se a usamos.

### E2-S1 · Spike: os dados têm problemas técnicos? (feito)

**Pergunta:** há campos vazios, contratos repetidos, vários códigos na mesma célula, textos duplicados?

**Resultado:** secções 1 a 6 do `qualidade_dados.md` e decisões 2 a 4:
- tirar os contratos sem CPV;
- uma só cópia de cada `idcontrato`;
- treinar com o primeiro código e avaliar com todos.

### E2-S2 · Spike: o que os dados dizem sobre o texto e os códigos? (feito)

**Pergunta:** que colunas ajudam os textos vagos? O mesmo texto tem sempre o mesmo código? A que nível da árvore se escolhem os códigos? Quantos exemplos tem cada código?

**Resultado:** secções 7 a 10 do `qualidade_dados.md`, medidas com o treino final, e decisões 5 a 8:
- o texto de entrada junta a descrição do acordo-quadro;
- a `adjudicante` fica só como experiência;
- no treino, um exemplo por texto, com o código mais comum;
- tiram-se do treino os códigos ao nível da divisão (18,8% dos contratos).

### E2-S3 · Spike: quantos anos usar no treino? (feito)

**Pergunta:** juntar anos anteriores a 2025 ajuda?

**Resultado:** decisão 1. O treino vai de janeiro de 2022 a setembro de 2025. A validação é outubro de 2025 e o teste novembro e dezembro de 2025. Fica por confirmar com o TF-IDF (E4-S2).

---

## E3 · Treino, validação e teste em que se pode confiar (feito)

**Objetivo:** três conjuntos separados por data, criados por um script que se pode voltar a correr, que para sozinho se uma regra for violada, e com testes que provam que as regras se cumprem.

**Medida de sucesso:**
- A preparação cria treino (478 384 exemplos), validação (23 945 contratos) e teste (37 361 contratos). ✅
- 70 testes passam no CI em cada envio. ✅
- Os testes apanharam os 9 erros introduzidos de propósito (a lista dos erros não ficou guardada no repositório). ✅

**Fora do âmbito:** treinar modelos (E4).

### E3-T1 · Tarefa: separar por data de publicação (feito)

**Porque é precisa:** o modelo vai ser usado com contratos futuros, por isso avalia-se com contratos mais recentes do que os do treino.

**Critérios de aceitação:**
- [x] O treino só tem contratos publicados de janeiro de 2022 a setembro de 2025; a validação, de outubro de 2025; o teste, de novembro e dezembro de 2025.
- [x] Nenhum contrato aparece em mais do que uma parte.

### E3-T2 · Tarefa: texto de entrada e etiqueta (feito)

**Porque é precisa:** é o que o modelo lê e o que aprende a responder (decisões 4 e 5).

**Critérios de aceitação:**
- [x] O texto junta o objeto, a descrição (só se for diferente) e a descrição do acordo-quadro (só se existir).
- [x] A etiqueta é o primeiro código; os restantes ficam guardados para a avaliação.
- [x] Campos vazios nunca aparecem escritos como "None" ou "nan" no texto.

### E3-T3 · Tarefa: limpar o treino sem tocar na avaliação (feito)

**Porque é precisa:** o treino não deve ter contradições nem códigos genéricos; a avaliação tem de refletir a realidade (decisões 6 e 8).

**Critérios de aceitação:**
- [x] No treino não há códigos ao nível da divisão nem textos repetidos. Cada texto tem o seu código mais comum (empate: o menor código).
- [x] Na validação e no teste ficam todos os contratos, marcados quando o texto também aparece no treino.

### E3-T4 · Tarefa: verificações automáticas no fim da preparação (feito)

**Porque é precisa:** um erro nos dados não dá erro nenhum sozinho; só dá um modelo pior.

**Critérios de aceitação:**
- [x] A preparação para com erro se houver contratos repetidos entre partes, datas fora do período, códigos ao nível da divisão no treino, textos repetidos no treino ou códigos mal formados.
- [x] As verificações não se desligam com o modo otimizado do Python (`-O`).

### E3-T5 · Tarefa: regra para os códigos raros (feito)

**Porque é precisa:** um modelo não aprende um código com 1 ou 2 exemplos (decisão 9).

**Critérios de aceitação:**
- [x] Um código com menos de N exemplos no treino sobe para a classe; se a classe também for rara, para o grupo; se o grupo também for raro, o exemplo sai.
- [x] Nunca se sobe até à divisão.
- [x] As contagens fazem-se só com o treino.
- [x] O N é um parâmetro (por defeito 10) e o treino guardado fica com os códigos originais.

### E3-T6 · Tarefa: testes unitários, de integração e CI (feito)

**Porque é precisa:** para mudar o código sem medo de partir uma regra sem dar por isso.

**Critérios de aceitação:**
- [x] Cada regra das tarefas E3-T1 a E3-T5 tem pelo menos um teste.
- [x] Um teste de integração corre o pipeline inteiro, de .xlsx a treino/validação/teste, com 15 linhas inventadas (14 contratos: um aparece em dois ficheiros, de propósito), e verifica onde acaba cada um.
- [x] Correr duas vezes dá exatamente os mesmos ficheiros.
- [x] Cada teste novo foi visto a falhar com o erro que guarda introduzido de propósito.
- [x] O GitHub Actions corre todos os testes em cada envio e em cada PR, sem os dados reais.

---

## E4 · Primeiras sugestões de CPV com um modelo simples (próximo)

**Objetivo:** o técnico de contratação dá a descrição de uma compra e recebe 5 códigos CPV, nenhum ao nível da divisão, e sabemos, com números da validação, quanto valem essas sugestões. Esses números servem de régua para o BERTimbau e o Albertina. Com o mesmo modelo, fecham-se as duas decisões que ficaram em aberto: o N dos códigos raros e a janela de treino.

**Persona:** o técnico de contratação. Ainda não há interface gráfica nem API. As sugestões entregam-se por um comando, que é a forma mais simples de entregar a capacidade sem a interface (não é a demo que o `readme.md` põe fora do âmbito). A interface fica para depois.

**Medida de sucesso** (tudo na validação, outubro de 2025):

1. O TF-IDF tem top-5 maior do que a régua "códigos mais comuns" no código completo e na classe, em todos os contratos e também só nos contratos com texto novo.
2. O N (5, 10 ou 20) e se a entidade compradora entra no modelo estão escolhidos e escritos. O N está escolhido e escrito no [`qualidade_dados.md`](qualidade_dados.md), com a tabela de resultados que o justifica. A janela de treino (desde 2025, 2022 ou 2019) também, ou, se o E4-S2 for cortado, o `qualidade_dados.md` diz que a decisão 1 se mantém sem confirmação.
3. O CI falha se as métricas guardadas do TF-IDF ficarem abaixo da régua, ou se o treino com dados minúsculos deixar de funcionar.
4. O treino e a avaliação só conseguem carregar o teste com uma autorização explícita, e um teste no CI confirma que sem ela o carregamento é recusado.

**Fora do âmbito deste épico:**

- tunar o BERTimbau e o Albertina, e alugar a GPU;
- MLflow;
- comparar com o modelo multilingue já tunado com concursos europeus;
- medir no teste e fazer a análise de erros (fase 5);
- interface gráfica e API.

**Aviso:** se o TF-IDF não bater a régua (E4-H1), para-se aqui. Isso quer dizer que há um erro nos dados, no treino ou na avaliação, e os itens seguintes esperam até se perceber qual.

**Tamanho:** a soma das estimativas é de 14 dias de trabalho, no máximo (2 itens M de 3 dias, 4 itens P de 1 dia e 4 spikes de 1 dia). Não há prazo fixo. Se for preciso cortar, o E4-S2 (janela de treino) é o primeiro a sair, e fica a decisão 1 como está.

### E4-T1 · Tarefa: avaliação comum a todos os modelos (M)

**Porque é precisa:** todos os modelos (a régua, o TF-IDF, o BERTimbau, o Albertina) vão ser comparados com esta avaliação. Se tiver um erro, todas as comparações ficam erradas sem ninguém dar por isso.

**Desbloqueia:** E4-T2, E4-H1 e os spikes.

**Regras de contagem** (desenvolvem as decisões 4 e 8):

- **Nível de um código:** os dígitos antes dos zeros finais. `33600000` tem 3 (grupo); `90911200` tem 6.
- **Três medidas, de cada vez com as sugestões do modelo:**
  - **código completo (no nível da etiqueta):** uma sugestão acerta se tiver os mesmos dígitos que o código até ao nível dele. Para `33600000`, acerta qualquer sugestão que comece por `336` (é a decisão 8). Para `90911200`, só `90911200`;
  - **classe:** os mesmos 4 primeiros dígitos. Só contam os contratos com um código ao nível da classe ou abaixo;
  - **divisão:** os mesmos 2 primeiros dígitos. Contam todos os contratos.
- **Vários códigos (decisão 4):** um contrato conta numa medida se algum dos seus códigos chegar a esse nível, e acerta se uma sugestão acertar em algum desses códigos. Exemplo: um contrato com `33000000` e `33690000` conta na classe (por causa do segundo código) e acerta se uma sugestão começar por `3369`.
- **F1 macro:** calcula-se com a 1.ª sugestão contra a etiqueta (o 1.º código), no nível de cada medida. A média faz-se sobre as etiquetas presentes na validação nesse nível.

**Critérios de aceitação:**

- [ ] Dadas as 5 sugestões de cada contrato e os seus códigos, a avaliação dá o top-1 e o top-5 em cada nível e diz quantos contratos entraram em cada nível.
- [ ] Dá também o F1 macro em cada medida, como descrito acima.
- [ ] Os resultados aparecem para todos os contratos e, em separado, só para os contratos com texto que não aparece no treino.
- [ ] Há um teste com contratos inventados, com o resultado calculado à mão, para cada regra de contagem acima (incluindo uma etiqueta de grupo, uma de categoria, e um contrato com um código de divisão e outro de classe).
- [ ] Cada teste falha quando se estraga de propósito a regra que ele guarda.

### E4-T2 · Tarefa: régua "códigos mais comuns" medida na validação (P)

**Porque é precisa:** sem ela, não há com que comparar o TF-IDF.

**Depende de:** E4-T1.

**Critérios de aceitação:**

- [ ] A régua sugere sempre as 5 etiquetas mais comuns do treino, contadas depois da regra dos códigos raros e com o mesmo N que o modelo usa. Assim a régua e o modelo escolhem entre as mesmas etiquetas.
- [ ] Os seus números na validação, calculados com a avaliação E4-T1, estão escritos no `qualidade_dados.md`.

### E4-S0 · Spike: o TF-IDF com regressão logística cabe no portátil? (timebox: 1 dia)

**Pergunta:** o treino cabe no portátil (cerca de 5,8 GB de memória, partilhados com o editor)? Se não couber, o que se faz: limitar o vocabulário, usar outro método de treino ou alugar uma máquina com mais memória?

**Como:** com um treino de ensaio, que se deita fora no fim, medir dois casos com a janela atual (desde 2022): o caso por defeito (N = 10) e o pior caso do E4-S1 (N = 5, que tem mais códigos). A janela desde 2019 tem mais dados; a sua memória mede-se dentro do E4-S2.

**Limites propostos** (ver "Decisões pendentes"): pico de memória abaixo de 4 GB e um treino abaixo de 1 hora.

**Resultado:** uma decisão escrita (tamanho do vocabulário e máquina a usar), com o pico de memória e o tempo de treino medidos nos dois casos.

### E4-H1 · História: 5 sugestões de CPV a partir da descrição (M)

**Como** técnico de contratação, **quero** escrever a descrição da compra e receber 5 códigos CPV ordenados, **para** escolher entre 5 em vez de procurar na tabela de cerca de 9 500 códigos.

**Depende de:** E4-T1, E4-T2 e E4-S0. Usa o N por defeito (10) e a janela atual (desde 2022).

**Critérios de aceitação:**

- [ ] **Dado** uma descrição escrita à mão, **quando** corro o comando de sugestões, **então** recebo 5 códigos distintos de 8 dígitos, ordenados do mais provável para o menos provável.
- [ ] Os 5 códigos são todos etiquetas do treino (depois da regra dos códigos raros), e nenhum está ao nível da divisão.
- [ ] Dada a mesma descrição duas vezes, recebo as mesmas 5 sugestões pela mesma ordem.
- [ ] Na validação, o top-5 é maior do que o da régua E4-T2 no código completo e na classe. Os resultados completos da avaliação E4-T1 estão escritos no `qualidade_dados.md`.
- [ ] Um teste confirma que nenhum contrato da validação entra no treino do modelo nem nas contagens do N.
- [ ] Nos contratos da validação com texto que não aparece no treino, o top-5 também é maior do que o da régua, no código completo e na classe, medidos nesses mesmos contratos. O `qualidade_dados.md` mostra lado a lado os resultados com texto novo e com texto já visto, para se ver quanto do resultado vem de decorar textos.
- [ ] O comando mostra as sugestões para pelo menos 3 descrições escritas à mão, por exemplo "serviços de limpeza de três escolas básicas durante 12 meses".

### E4-T3 · Tarefa: testes ao treino que correm no CI (P)

**Porque é precisa:** um treino com um erro pode correr sem dar erro nenhum e só produzir um modelo mau. Estes testes apanham isso antes de se gastar tempo com dados reais.

**Depende de:** E4-H1.

**Critérios de aceitação:**

- [ ] Com um treino minúsculo inventado (poucos textos bem distintos), sem a regra dos códigos raros (N = 1) e com regularização fraca, o modelo acerta em todos os textos com que treinou, no top-1. Se não acertar, há um erro no treino.
- [ ] As sugestões só contêm etiquetas que existem no treino.
- [ ] Os testes correm no CI em menos de 1 minuto, sem dados reais.
- [ ] Cada teste falha quando se estraga de propósito o que ele guarda.

### E4-S1 · Spike: que N usar nos códigos raros? (timebox: 1 dia)

**Pergunta:** com N = 5, 10 ou 20 (decisão 9), qual dá melhor resultado na validação?

**Depende de:** E4-H1, e a decisão pendente 2 fechada antes de começar.

**Regra de escolha, escrita antes de correr:** ganha o maior top-5 no código completo. Se dois ficarem a menos de 0,5 pontos percentuais, ganha o maior top-5 na classe. Se continuar empatado, fica o 10, que já é o valor por defeito. A regra fica escrita antes de se verem os resultados, para os resultados não a influenciarem.

**Resultado:**

- uma tabela com o top-5 no código completo e na classe, e o F1 macro, para os três valores de N;
- o N escolhido, escrito na decisão 9 do `qualidade_dados.md`, e passa a ser o valor por defeito;
- se o N mudar, a régua (E4-T2) e os resultados do E4-H1 voltam a calcular-se com o novo N.

### E4-S3 · Spike: a entidade compradora ajuda o modelo? (timebox: 1 dia)

**Pergunta:** juntar a entidade (`adjudicante`) ao texto de entrada melhora as sugestões, ou faz o modelo aprender os hábitos de cada entidade, incluindo os códigos genéricos? (decisão 7)

**Depende de:** E4-S1 (usa o N escolhido) e E4-H1.

**Como:** treinar o mesmo modelo com e sem a entidade e comparar na validação. A comparação faz-se em três grupos:

- todos os contratos;
- os contratos de entidades que aparecem no treino;
- os contratos de entidades que não aparecem no treino.

**Regra de escolha, escrita antes de correr:** a entidade entra se as três condições se cumprirem:

- melhorar o top-5 no código completo em todos os contratos em pelo menos 0,5 pontos percentuais;
- não piorar mais de 0,5 pontos nos contratos de entidades novas;
- não aumentar a parte das 1.as sugestões ao nível do grupo ou acima.

**Resultado:**

- uma tabela com e sem entidade nos três grupos, com a parte das 1.as sugestões ao nível do grupo ou acima;
- a decisão 7 fechada no `qualidade_dados.md`;
- se a entidade entrar, a régua (E4-T2) e os resultados do E4-H1 voltam a calcular-se.

### E4-T4 · Tarefa: escolher a data de início do treino na preparação (P)

**Porque é precisa:** hoje o treino começa sempre em janeiro de 2022. O E4-S2 precisa de preparar treinos com outras datas de início.

**Desbloqueia:** E4-S2. Não depende de nada.

**Critérios de aceitação:**

- [ ] A preparação aceita a data de início do treino. Sem a indicar, a data continua a vir do mesmo sítio de hoje (a constante partilhada) e dá exatamente os mesmos ficheiros e o mesmo funil (478 384 exemplos de treino).
- [ ] Com outra data de início, a validação e o teste têm os mesmos contratos e as mesmas etiquetas. Só muda a marca "texto também aparece no treino", que depende do treino.
- [ ] As verificações automáticas usam a data de início indicada e continuam a passar.

### E4-S2 · Spike: a janela de treino desde 2022 é a certa? (timebox: 1 dia)

**Pergunta:** com o N escolhido no E4-S1, treinar desde 2025, 2022 ou 2019 muda o resultado na validação?

**Depende de:** E4-S1 e E4-T4.

**Regra de escolha, escrita antes de correr:** compara-se o top-5 no código completo sempre no mesmo grupo de contratos: os da validação cujo texto não aparece no treino da janela mais longa (desde 2019). Fica a janela mais curta que esteja a menos de 0,5 pontos percentuais da melhor. É mais barato treinar com menos dados.

**Porquê esse grupo:** com mais anos no treino, mais textos da validação já foram vistos, e o modelo acerta neles por decorar. Comparar em todos os contratos favorecia as janelas longas sem elas serem melhores. O resultado em todos os contratos mostra-se também, só como informação.

**Também:** medir o pico de memória do treino desde 2019 (o E4-S0 só mediu desde 2022).

**Resultado:**

- uma tabela com as três janelas;
- a decisão 1 confirmada ou mudada no `qualidade_dados.md`.

Se mudar, o `readme.md` e este ficheiro são atualizados, a preparação volta a correr com a nova data, e a régua (E4-T2) e os resultados do E4-H1 voltam a calcular-se.

### E4-T5 · Tarefa: portão de qualidade no CI (P)

**Porque é precisa:** impede que uma mudança futura piore o modelo, ou passe a usar o teste, sem ninguém reparar.

**Depende de:** E4-H1. Os valores guardados atualizam-se depois do E4-S1 e do E4-S2.

**Critérios de aceitação:**

- [ ] As métricas da validação do TF-IDF e da régua ficam guardadas no git, num ficheiro pequeno. Cada bloco de métricas diz de que conjunto vem e o período de datas.
- [ ] O CI falha se o top-5 guardado do TF-IDF não for maior do que o da régua, no código completo e na classe.
- [ ] O CI falha se algum bloco de métricas disser que vem de outro conjunto que não a validação (outubro de 2025).
- [ ] O treino e a avaliação carregam os conjuntos sempre da mesma maneira, que recusa o teste sem uma autorização explícita. Um teste no CI confirma a recusa.
- [ ] Cada verificação falha quando se estraga de propósito o que ela guarda.

### Ordem e dependências do E4

```
E4-T1 ──> E4-T2 ──┐
E4-S0 ────────────┴──> E4-H1 ──┬──> E4-T3
                               ├──> E4-T5
                               └──> E4-S1 ──┬──> E4-S3 ──> (atualizar E4-T2, E4-H1 e E4-T5)
E4-T4 ──────────────────────────────────────┴──> E4-S2 ──> (atualizar E4-T2, E4-H1 e E4-T5)
```

- E4-T1, E4-S0 e E4-T4 podem começar já e não dependem uns dos outros.
- O E4-T5 cria o portão logo a seguir ao E4-H1. Os valores guardados voltam a atualizar-se no fim, com o N, a janela e a entidade decididos.

### Decisões tomadas para o E4

- A entidade compradora testa-se já com o TF-IDF (spike E4-S3), não depois do BERTimbau. Mais barato: se não ajudar no TF-IDF, provavelmente não vale a pena gastar GPU a testá-la.

### Decisões pendentes do E4

1. **Regras de escolha do E4-S1, do E4-S2 e do E4-S3** (métrica principal: top-5 no código completo; margem de 0,5 pontos): concordas, ou preferes outra métrica principal, por exemplo o top-5 na classe? Tem de ficar fechado antes do E4-S1.
2. **Limites do E4-S0** (memória abaixo de 4 GB, treino abaixo de 1 hora): servem, ou preferes alugar uma máquina logo à partida?
