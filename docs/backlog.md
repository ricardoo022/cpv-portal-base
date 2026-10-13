# Backlog: épicos e user stories

Todos os épicos do projeto num só sítio: os que já estão feitos e o próximo. O plano por fases (CRISP-DM) e as caixas de verificação estão no [`workflow.md`](workflow.md); as medições e as decisões numeradas no [`qualidade_dados.md`](qualidade_dados.md). Cada épico novo acrescenta-se aqui; quando acaba, passa a "feito".

| Épico | Fase do workflow | Estado |
|---|---|---|
| E1 · Saber que problema resolvemos | 1 | feito |
| E2 · Conhecer os dados antes de treinar | 2 | feito |
| E3 · Treino, validação e teste em que se pode confiar | 3 | feito |
| E4 · Primeiras sugestões de CPV com um modelo simples | 4 | em curso (E4-T1, E4-T2, E4-S0, E4-H1, E4-S4, E4-T3, E4-S5, E4-S1 e E4-S3 feitos) |
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
- **Bootstrap:** tirar muitas vezes, à sorte e com repetição, uma amostra dos contratos de validação e recalcular a métrica em cada uma. Mostra quanto o resultado muda só por acaso.
- **Métrica de guarda:** uma métrica que não escolhe o vencedor, mas elimina uma opção que a piore.
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

- tunar o BERTimbau e o Albertina, e treinar em GPU (o TF-IDF só usa o processador e a memória do pod, mesmo que o pod tenha GPU);
- MLflow;
- comparar com o modelo multilingue já tunado com concursos europeus;
- medir no teste e fazer a análise de erros (fase 5);
- interface gráfica e API.

**Aviso:** se o TF-IDF não bater a régua (E4-H1), para-se aqui. Isso quer dizer que há um erro nos dados, no treino ou na avaliação, e os itens seguintes esperam até se perceber qual.

**Tamanho:** a soma das estimativas é de 16 dias de trabalho, no máximo (2 itens M de 3 dias, 4 itens P de 1 dia e 6 spikes de 1 dia). Não há prazo fixo. Se for preciso cortar, o E4-S2 (janela de treino) é o primeiro a sair, e fica a decisão 1 como está.

### E4-T1 · Tarefa: avaliação comum a todos os modelos (M, feito)

**Porque é precisa:** todos os modelos (a régua, o TF-IDF, o BERTimbau, o Albertina) vão ser comparados com esta avaliação. Se tiver um erro, todas as comparações ficam erradas sem ninguém dar por isso.

**Desbloqueia:** E4-T2, E4-H1 e os spikes.

**Feito** no PR #7 (11 de outubro de 2026): `src/evaluation.py` (as notas) e `src/comparison.py` (bootstrap e regra de escolha). Descrição completa em [`prs/pr_E4-T1.md`](prs/pr_E4-T1.md).

**Regras de contagem** (desenvolvem as decisões 4 e 8):

- **Nível de um código:** os dígitos antes dos zeros finais. `33600000` tem 3 (grupo); `90911200` tem 6.
- **Três medidas, de cada vez com as sugestões do modelo:**
  - **código completo:** só contam os contratos com algum código abaixo da divisão, e só esses códigos servem para acertar; o modelo nunca sugere divisões (decisão 8). Uma sugestão acerta só se for igual ao código. Para `33600000`, só `33600000` acerta; `33651100` é erro, porque é mais específica do que o código escolhido (na literatura chama-se *sobre-especialização* e conta como erro, Kosmopoulos et al., 2015). Essa sugestão acerta na divisão (e na classe, quando o código é da classe ou abaixo);
  - **classe:** os mesmos 4 primeiros dígitos. Só contam os contratos com um código ao nível da classe ou abaixo;
  - **divisão:** os mesmos 2 primeiros dígitos. Contam todos os contratos.
- **Contratos só com códigos de divisão** (por exemplo, só `45000000`): não contam no código completo nem na classe. Mostram-se numa linha à parte, `so_divisao`, com quantos são e o top-1, o top-5 e o F1 macro na divisão, para não ficarem escondidos.
- **Vários códigos (decisão 4):** um contrato conta numa medida se algum dos seus códigos chegar a esse nível, e acerta se uma sugestão acertar em algum desses códigos. Exemplo: um contrato com `33000000` e `33690000` conta na classe (por causa do segundo código) e acerta se uma sugestão começar por `3369`.
- **F1 macro:** calcula-se com a 1.ª sugestão contra a etiqueta (o 1.º código), no nível de cada medida. A média faz-se sobre as etiquetas presentes na validação nesse nível.
- **Medidas hierárquicas** (as do artigo *Zero-Shot Hierarchical Classification on the CPV Taxonomy*, 2024, que vêm de Kiritchenko et al., 2005): dão meio acerto a uma sugestão perto do código certo. Mostram-se ao lado, mas **não entram na regra de comparação**.
  - Cada código vale o conjunto dele e dos seus antepassados, do nível 2 até ao nível dele. `45233200` (nível 6) é {`45`, `452`, `4523`, `45233`, `452332`}, 5 nós.
  - Com a 1.ª sugestão contra a etiqueta: **precisão hierárquica** = nós em comum / nós da sugestão; **recall hierárquico** = nós em comum / nós da etiqueta. Exemplo: etiqueta `45233200` (5 nós) e sugestão `45233223` (7 nós) têm 5 em comum, logo precisão 5/7 e recall 5/5. Com a sugestão `45200000` (o avô, 2 nós): precisão 2/2 e recall 2/5.
  - **Micro:** somam-se os nós em comum e os nós de todos os contratos antes de dividir. **Macro:** como no artigo, a precisão (e o recall) de cada contrato faz-se média por etiqueta, e depois média sobre as etiquetas.
  - **F1 hierárquico** = 2 · precisão · recall / (precisão + recall), em micro e em macro. É o resumo: castiga tanto a sugestão vaga (o avô) como a demasiado específica.
  - Não se usam para decidir porque a precisão sozinha dá nota máxima a uma sugestão vaga (o avô acima tem precisão 1), e o objetivo do projeto é o código exato.

**Regra de comparação entre duas opções** (decidida a 10 de outubro de 2026, usada nos spikes E4-S1, E4-S2, E4-S3, E4-S4 e E4-S5):

- **Quando uma diferença conta:** usa-se o bootstrap emparelhado. Tira-se 1 000 vezes, à sorte e com repetição, uma amostra dos contratos de validação, sempre a mesma amostra para as duas opções, e calcula-se a diferença em cada uma. A diferença só conta se for do mesmo lado em pelo menos 95% das amostras. Assim a margem vem dos dados, não de um número inventado.
- **Métrica principal:** o top-5 no código completo. É o objetivo: sugerir o código exato.
- **Métrica de guarda:** o top-5 na classe. Uma sugestão na classe certa já ajuda o técnico a escolher, por isso uma opção que piore a classe, com uma diferença que conte, é eliminada, mesmo que ganhe no código completo.
- **Ganha** a opção com melhor top-5 no código completo, se a diferença contar e se não tiver sido eliminada pela guarda.
- **Empate:** se nenhuma diferença no código completo contar, desempata o top-5 na classe, se essa diferença contar. Se também não contar, fica a opção por defeito de cada spike.

**Critérios de aceitação:**

- [x] Dadas as 5 sugestões de cada contrato e os seus códigos, a avaliação dá o top-1 e o top-5 em cada nível e diz quantos contratos entraram em cada nível.
- [x] Dá também o F1 macro em cada medida, como descrito acima.
- [x] Dá a precisão, o recall e o F1 hierárquicos da 1.ª sugestão, em micro e em macro, para todos os contratos e só para os de texto novo.
- [x] Dadas as sugestões de duas opções para os mesmos contratos, dá a diferença entre elas em cada medida e diz se essa diferença conta, segundo a regra de comparação acima.
- [x] Com a mesma semente aleatória, o bootstrap dá sempre o mesmo resultado. Duas opções com sugestões iguais dão diferença zero, que não conta.
- [x] Os resultados aparecem para todos os contratos e, em separado, só para os contratos com texto que não aparece no treino.
- [x] Há um teste com contratos inventados, com o resultado calculado à mão, para cada regra de contagem acima (incluindo uma etiqueta de grupo, uma de categoria, e um contrato com um código de divisão e outro de classe).
- [x] Cada teste falha quando se estraga de propósito a regra que ele guarda.

### E4-T2 · Tarefa: régua "códigos mais comuns" medida na validação (P, feito)

**Porque é precisa:** sem ela, não há com que comparar o TF-IDF.

**Depende de:** E4-T1.

**Feito** no PR #10 (11 de outubro de 2026), descrição completa em [`prs/pr_E4-T2.md`](prs/pr_E4-T2.md): `src/baselines.py` (a régua), [`notebooks/03_regua.ipynb`](../notebooks/03_regua.ipynb) (os números) e a secção 12 do [`qualidade_dados.md`](qualidade_dados.md). A régua conta exemplos de treino (um por texto); contar contratos dava uma régua mais fácil de bater (decisão 10). A régua a bater no E4-H1: top-5 de 26,3% no código completo e 15,6% na classe (23,7% e 14,8% nos contratos com texto novo). As 5 etiquetas são as mesmas com N = 5, 10 e 20, por isso a régua não muda se o E4-S1 mudar o N.

**Critérios de aceitação:**

- [x] A régua sugere sempre as 5 etiquetas mais comuns do treino, contadas depois da regra dos códigos raros e com o mesmo N que o modelo usa. Assim a régua e o modelo escolhem entre as mesmas etiquetas.
- [x] Os seus números na validação, calculados com a avaliação E4-T1, estão escritos no `qualidade_dados.md`.

### E4-S0 · Spike: quanto custa cada treino na máquina alugada? (timebox: 1 dia, feito)

**Decisão já tomada (mudada a 11 de outubro de 2026):** o TF-IDF treina num pod alugado na Runpod, com pelo menos 32 GB de memória (só processador, ou com uma GPU barata se não houver pods só com processador; o TF-IDF não usa a GPU). A Hetzner CX53, a escolha anterior, estava indisponível na página oficial a 11 de outubro de 2026, e a conta nova pedia um pagamento adiantado. A Runpod é onde vão treinar o BERTimbau e o Albertina, por isso o saldo carregado serve às duas fases. O preço por hora vê-se na consola no dia. Com 32 GB, a memória do portátil deixa de ser um limite.

**Pergunta:** quanto tempo e quanta memória gasta um treino nessa máquina, e quanto vão custar todos os treinos do épico?

**Como:**

- preparar a máquina: o ambiente com as mesmas versões das bibliotecas, os dados preparados, e os testes todos a passar lá;
- com um treino de ensaio, que se deita fora no fim, medir dois casos com a janela atual (desde 2022): o caso por defeito (N = 10) e o pior caso do E4-S1 (N = 5, que tem mais códigos). A janela desde 2019 tem mais dados; a sua memória mede-se dentro do E4-S2.

**Resultado:**

- o pico de memória e o tempo de treino medidos nos dois casos;
- o custo estimado de todos os treinos do épico;
- o tipo de pod confirmado (memória, processadores e preço por hora);
- os passos para preparar a máquina e para a apagar no fim, escritos em [`maquina_runpod.md`](maquina_runpod.md). Um pod parado continua a pagar o disco de volume; só apagado deixa de ser cobrado.

**Feito** no PR #13 (11 de outubro de 2026), descrição completa em [`prs/pr_E4-S0.md`](prs/pr_E4-S0.md). O script é o [`scripts/ensaio_treino.py`](../scripts/ensaio_treino.py), e os números vêm dos ficheiros em `metrics/e4_s0/`:

- `portatil.jsonl` e `portatil_vocabulario.jsonl`: amostras no portátil. Foram geradas por uma versão anterior do mesmo script, com a mesma lógica mas caminhos fixos e sem o campo `maquina`;
- `portatil_logreg30k_parada.txt`: a corrida da `LogisticRegression` que foi parada;
- `pod.jsonl`: treino completo no pod;
- `pod_info.json`: o pod, escrito à mão a partir das leituras da Runpod e do cgroup;
- `pod_memoria.log`: a memória do pod lida a cada minuto.

1. **Classificador.** No portátil, com amostras, a `LogisticRegression` (solver `saga`) não convergiu em 100 iterações com 10 000 exemplos (170 s). Com 30 000 exemplos e o vocabulário cortado a 20 000 palavras, foi parada ao fim de 600 s sem acabar. O `SGDClassifier(loss="log_loss")` convergiu sempre. Os dois treinam regressão logística, mas de maneira diferente:
   - a `LogisticRegression` treina um só modelo para todas as etiquetas;
   - o SGD treina um modelo "sim/não" por etiqueta (um contra todos), ajustando os pesos exemplo a exemplo, com a regularização `alpha` = 0,0001 (o valor por defeito).

   Decisão 5 do E4: o TF-IDF usa o `SGDClassifier(loss="log_loss")`. Escolheu-se pela velocidade, sem medir a qualidade. Confirmada no E4-H1: bateu a régua na validação.
2. **Treino completo no pod** (TF-IDF com palavras e pares de palavras, `min_df=2`, valores em float32; `SGDClassifier(loss="log_loss", n_jobs=-1)` num pod com 8 processadores):

   | | N = 10 | N = 5 |
   |---|---|---|
   | exemplos de treino | 478 234 | 478 334 |
   | etiquetas | 3 000 | 3 848 |
   | vocabulário | 402 084 | 402 154 |
   | vetorizar | 7,9 s | 8,8 s |
   | treinar | 462 s (7,7 min) | 616 s (10,3 min) |
   | épocas até convergir | 6 | 6 |
   | pesos do modelo | 4,8 GB | 6,2 GB |
   | pico de memória do processo | 5,7 GB | 7,2 GB |

   Todas as memórias estão em GB (mil milhões de bytes). O pico do processo vem do `pico_rss_mb.final` do `pod.jsonl`, que está em MiB: por exemplo, 5 480 MiB × 1 048 576 bytes ≈ 5,7 GB. O pico de memória do pod inteiro (`memory.peak` do cgroup, em `pod_info.json`) foi 8,3 GB, de 32 GB. O campo `nucleos` do `pod.jsonl` diz 192, mas esse é o número de núcleos da máquina física; o pod tinha 8. Não ficou registado quantos núcleos o treino usou de facto.
3. **Tipo de pod confirmado:** `cpu3g` (General Purpose) com 8 processadores e 32 GB, a 0,32 $/h. Os 32 GB chegam com folga, e o portátil (cerca de 5,8 GB) não chegava. Custo do ensaio, pelas horas do `pod_info.json`:
   - o pod esteve ligado das 09:54:30 às cerca de 10:15:30 (UTC), cerca de 21 minutos (0,35 h);
   - 0,35 h × 0,32 $/h ≈ 0,11 $;
   - a conta foi feita com o preço por hora, porque a faturação da Runpod ainda não tinha o registo.
4. **Custo estimado do resto do épico.** É uma estimativa, e esta é a conta:
   - **Treinos:** 6 a 10 (o E4-H1, mais 2 valores de N no E4-S1, 1 ou 2 no E4-S3, 2 janelas no E4-S2, e os recálculos que essas decisões pedirem). A 8 a 11 minutos cada, dão 0,8 a 1,8 h.
   - **Preparar os pods:** cerca de 10 minutos por pod (criar, copiar, `make setup` e testes), com 5 pods (um por item), dão cerca de 0,8 h.
   - **Total:** 1,6 a 2,6 h de pod, ou seja, cerca de 0,50 a 0,85 $.
   - **Com margem:** dobrando o tempo, para avaliar na validação, gravar modelos e esperas, fica abaixo de 2 $.
   - A janela desde 2019 (E4-S2) tem mais dados; a sua memória e o seu tempo medem-se lá.
   - Esta conta é de antes do E4-S4 (que custou cerca de 0,21 $) e do E4-S5 (estimativa no próprio item).

### E4-H1 · História: 5 sugestões de CPV a partir da descrição (M, feito)

**Como** técnico de contratação, **quero** escrever a descrição da compra e receber 5 códigos CPV ordenados, **para** escolher entre 5 em vez de procurar na tabela de cerca de 9 500 códigos.

**Nota:** a capacidade fica a existir, mas o técnico só a recebe quando houver interface, que está fora deste épico. Por agora, quem a usa somos nós, para ter uma régua para o BERTimbau e o Albertina.

**Depende de:** E4-T1, E4-T2 e E4-S0. Usa o N por defeito (10) e a janela atual (desde 2022).

**Feito** no PR #14 (11 de outubro de 2026), descrição completa em [`prs/pr_E4-H1.md`](prs/pr_E4-H1.md):

- o modelo está em `src/tfidf.py`, o treino em `src/train_tfidf.py` e o comando em `src/suggest.py`;
- os números estão na secção 13 do [`qualidade_dados.md`](qualidade_dados.md), calculados com o [`notebooks/04_tfidf.ipynb`](../notebooks/04_tfidf.ipynb), e os ficheiros do treino no pod estão em `metrics/e4_h1/`;
- na validação, o top-5 é 43,2% no código completo e 37,9% na classe (régua: 26,3% e 15,6%), e 38,5% e 34,2% no texto novo (régua: 23,7% e 14,8%); o bootstrap dá o TF-IDF à frente em 100% das amostras nos quatro casos;
- o treino demorou 454 s num pod `cpu3g` de 8 processadores e 32 GB. O pico de memória do processo foi 9 975 MiB, cerca de 10,5 GB, e o do pod inteiro (cgroup) foi 16,4 GB, de 32 GB. É mais do que no E4-S0 (5,7 GB): este processo também grava o modelo e sugere para a validação, mas não se mediu quanto pesa cada passo.

**Critérios de aceitação:**

- [x] **Dado** uma descrição escrita à mão, **quando** corro o comando de sugestões, **então** recebo 5 códigos distintos de 8 dígitos, ordenados do mais provável para o menos provável.
- [x] Os 5 códigos são todos etiquetas do treino (depois da regra dos códigos raros), e nenhum está ao nível da divisão.
- [x] Dada a mesma descrição duas vezes, recebo as mesmas 5 sugestões pela mesma ordem.
- [x] Na validação, o top-5 é maior do que o da régua E4-T2 no código completo e na classe. Os resultados completos da avaliação E4-T1 estão escritos no `qualidade_dados.md`.
- [x] Um teste confirma que nenhum contrato da validação entra no treino do modelo nem nas contagens do N.
- [x] Nos contratos da validação com texto que não aparece no treino, o top-5 também é maior do que o da régua, no código completo e na classe, medidos nesses mesmos contratos. O `qualidade_dados.md` mostra lado a lado os resultados com texto novo e com texto já visto, para se ver quanto do resultado vem de decorar textos.
- [x] O comando mostra as sugestões para pelo menos 3 descrições escritas à mão, por exemplo "serviços de limpeza de três escolas básicas durante 12 meses".

### E4-S4 · Spike: afinar o TF-IDF (regularização e etiquetas frequentes) (timebox: 1 dia, feito)

**Porque é precisa:** no E4-H1 o TF-IDF bateu a régua, mas a 1.ª sugestão é quase sempre uma das 5 etiquetas mais frequentes e o F1 macro é quase zero (secção 13 do `qualidade_dados.md`). Uma baseline mal afinada fazia o BERTimbau e o Albertina parecerem melhores do que são.

**Pergunta:** um `alpha` (a força da regularização) mais pequeno, ou a correção das etiquetas frequentes de Menon et al. (2021), melhoram as sugestões na validação?

**Depende de:** E4-H1. Usa N = 10 e a janela desde 2022.

**Como** (com o treino completo, no pod; o melhor `alpha` depende do número de exemplos, por isso uma amostra no portátil não serve):

- `alpha` = 0,0001 (o do E4-H1), 0,00001 e 0,000001 (intervalo recomendado pela documentação do scikit-learn);
- **acrescentado depois de ver os resultados** (11 de outubro de 2026): `alpha` = 0,0000001, porque o 0,000001 ganhou e era o extremo da lista. Quando o melhor valor fica no extremo, não se sabe se um valor mais pequeno seria ainda melhor. Entra no fim da lista, com a mesma regra;
- para cada `alpha`, a correção das etiquetas frequentes: ordenar por pontuação − τ · log(frequência da etiqueta no treino), com τ = 0 (sem correção), 0,25, 0,5 e 1.

**Regra de escolha, escrita antes de correr:** primeiro escolhe-se o `alpha` com τ = 0, depois o τ com esse `alpha`. Em cada passo começa-se com a opção por defeito (`alpha` = 0,0001 no 1.º passo, τ = 0 no 2.º) e compara-se cada candidato com a opção em vigor, pela ordem da lista acima, com a regra de comparação do E4-T1. Se o candidato ganhar, passa a ser a opção em vigor; em empate, fica a que já estava. Como o `alpha` e o τ se escolhem na validação, os números do vencedor ficam um pouco otimistas; o teste continua fechado e mede isso no fim.

**Também se mostra** (sem entrar na decisão): o F1 macro e a parte das 1.as sugestões que são uma das 5 etiquetas da régua.

**Resultado:**

- uma tabela com todas as opções;
- o `alpha` e o τ escolhidos, escritos como decisão do E4; se mudarem, passam a ser o valor por defeito em `src/tfidf.py` e os resultados do E4-H1 voltam a calcular-se.

**Feito** no PR #15 (11 de outubro de 2026), descrição completa em [`prs/pr_E4-S4.md`](prs/pr_E4-S4.md). A tabela e as comparações estão na secção 14 do [`qualidade_dados.md`](qualidade_dados.md), calculadas com o [`notebooks/05_tfidf_afinacao.ipynb`](../notebooks/05_tfidf_afinacao.ipynb), e os ficheiros dos treinos no pod estão em `metrics/e4_s4/`.

- **Escolhido:** `alpha` = 0,0000001 e τ = 0,25 (decisão 6 abaixo). Passam a ser o valor por defeito em `src/tfidf.py` (`ALPHA`, `TAU`), no `train_tfidf.py` e no comando de sugestões.
- **Na validação:** top-5 de 76,9% no código completo e 82,9% na classe (E4-H1: 43,2% e 37,9%); 73,5% e 80,8% no texto novo; F1 macro 0,2565 (E4-H1: 0,0107). Os resultados do E4-H1 com os valores novos são estes: o notebook 05 mostra a avaliação completa do escolhido ao lado da do E4-H1.
- **Custo:** cerca de 40 minutos de pod, perto de 0,21 $ (um pod calhou outra vez na máquina de EU-SE-1 cuja porta SSH não respondia, e foi apagado sem uso; ver `metrics/e4_s4/pod_info.json`).
- **Por fazer:** o comando de sugestões ainda não correu com o modelo novo, porque os treinos do spike não gravaram o modelo. Faz-se no próximo treino no pod.

### E4-T3 · Tarefa: testes ao treino que correm no CI (P, feito)

**Porque é precisa:** um treino com um erro pode correr sem dar erro nenhum e só produzir um modelo mau. Estes testes apanham isso antes de se gastar tempo com dados reais.

**Depende de:** E4-H1.

**Feito** no PR #16 (12 de outubro de 2026), descrição completa em [`prs/pr_E4-T3.md`](prs/pr_E4-T3.md): dois testes unitários em `tests/unit/test_tfidf.py` e dois de integração em `tests/integration/test_tfidf_pipeline.py` (do `.xlsx` ao modelo gravado). Não guardam o `alpha` nem as épocas: com textos tão distintos, o modelo acerta com qualquer valor (o `alpha` já tem o seu teste; as épocas medem-se no E4-S5).

**Critérios de aceitação:**

- [x] Com um treino minúsculo inventado (poucos textos bem distintos), sem a regra dos códigos raros (N = 1) e com regularização fraca, o modelo acerta em todos os textos com que treinou, no top-1. Se não acertar, há um erro no treino.
- [x] As sugestões só contêm etiquetas que existem no treino.
- [x] Os testes correm no CI em menos de 1 minuto, sem dados reais.
- [x] Cada teste falha quando se estraga de propósito o que ele guarda.

### E4-S5 · Spike: épocas e um `alpha` mais pequeno no TF-IDF (timebox: 1 dia, feito)

**Porque é precisa:** o TF-IDF vai ser a régua do BERTimbau e do Albertina. Ficaram duas coisas por medir que podem mudar o seu resultado:

- **as épocas** (passagens completas pelo treino): hoje o treino para sozinho, pela regra por defeito do scikit-learn, quando o erro **no treino** não baixa pelo menos 0,001 em 5 épocas seguidas. Há um classificador por etiqueta e cada um para quando a regra lho diz; o mais lento parou ao fim de 6 épocas no E4-H1 e de 12 com o `alpha` do E4-S4 (`metrics/e4_s4/alpha_1e-7/treino.json` guarda só esse máximo). Nunca se mediu se outro número de épocas dá melhor resultado na **validação**;
- **o `alpha`** = 0,00000001: o 0,0000001 ganhou no E4-S4 e é o extremo da lista testada (limitação da secção 14 do `qualidade_dados.md`).

As duas fazem-se numa só ida ao pod.

**Duas notas da revisão do E4-H1 por um agente juiz** (12 de outubro de 2026):

- **Porque a paragem automática pode chegar cedo demais.** No scikit-learn 1.7.2, cada classificador para quando a soma do erro sobre **todos** os exemplos de treino não desce mais do que 0,001 × o número de exemplos (cerca de 478 numa soma sobre ~478 mil exemplos; `_sgd_fast.pyx.tp`, linha 571). Numa etiqueta rara, quase todos os exemplos são "não" e acerta-se neles depressa, por isso a soma pode ficar abaixo desse limite logo nas primeiras épocas, mesmo que os poucos exemplos "sim" ainda estejam mal aprendidos. É uma suspeita, não uma medição: é o passo 3 que a confirma ou não.
- **O `alpha` não muda só a regularização.** O treino usa a taxa de aprendizagem por defeito, `learning_rate="optimal"`, em que o tamanho de cada passo é 1 / (`alpha` × (t + t0)). Um `alpha` mais pequeno dá passos maiores e pode mudar quando a paragem automática chega. Por isso, no passo 1 a comparação de `alpha` mistura três efeitos (regularização, tamanho do passo e número de épocas), e o mesmo vale para o ganho de 43% para 77% no E4-S4, que a secção 14 atribui à regularização. A conclusão sobre o `alpha` escreve-se com este aviso; separar os efeitos só é possível com épocas fixas (passo 3).

**Pergunta:** um `alpha` de 0,00000001, ou parar o treino noutra época, melhoram as sugestões na validação?

**Depende de:** E4-S4 e E4-T3 (o E4-S5 mexe no treino, e os testes do E4-T3 confirmam que continua sem erros). Usa N = 10 e a janela desde 2022.

**Como** (com o treino completo, no pod), em três passos, pela ordem:

1. **`alpha`**, com τ = 0: o 0,00000001 contra o 0,0000001, que é a opção em vigor. Se o 0,00000001 ganhar, adota-se, e escreve-se que continua a ser o extremo da lista; não se acrescentam valores mais pequenos neste spike.
2. **τ**, só se o `alpha` mudar: volta a escolher-se o τ com o `alpha` novo, com a lista e a regra do E4-S4 (0, 0,25, 0,5 e 1, a começar em τ = 0).
3. **Épocas**, com o `alpha` e o τ em vigor. A opção em vigor é a regra por defeito (cada etiqueta para sozinha). Os candidatos são um número fixo de épocas, igual para todas as etiquetas e sem paragem automática, desta lista curta, escrita antes de correr: 3, 6, 12, 18 e 24 épocas. Cada candidato compara-se com a opção em vigor, por esta ordem.

**Regra de escolha, escrita antes de correr:** a regra de comparação do E4-T1. Em cada passo, o candidato só substitui a opção em vigor se ganhar; em empate, fica a que já estava. Como o `alpha`, o τ e as épocas se escolhem todos na validação, os números do vencedor ficam um pouco otimistas; o teste continua fechado e mede isso no fim.

**Também se mostra** (sem entrar na decisão): o F1 macro e, se for barato de obter, um gráfico do top-5 na validação em todas as épocas de 1 a 24. Só a lista curta entra na decisão: com 24 comparações, alguma época ganharia só por sorte.

**Resultado:**

- uma tabela com as opções de cada passo e o gráfico das épocas;
- confirmação de que o treino com a regra por defeito, repetido com o mesmo código, dá o mesmo top-5 que o E4-S4 (76,9% no código completo e 82,9% na classe); se não der, a diferença fica explicada antes de se tirar qualquer conclusão;
- o `alpha`, o τ e as épocas escolhidos, escritos como decisão do E4. Se algum mudar, passa a ser o valor por defeito do treino do TF-IDF, e os resultados do E4-H1 voltam a calcular-se;
- **se ganhar um número fixo de épocas**, ele só vale para este treino (N = 10, desde 2022): com outro N ou outra janela há mais ou menos dados e etiquetas, e um número fixo não se adapta, ao contrário da regra por defeito. Por isso o E4-S2 treina todas as opções com a regra por defeito, para nenhuma ficar favorecida, e as épocas voltam a medir-se no fim, com a mesma lista, para o N e a janela escolhidos (o E4-S1 passou depois a testar as épocas dentro de uma grelha com o N — mudado a 13 de outubro de 2026, ver o E4-S1). O `alpha` e o τ seguem a mesma lógica do E4-S4: usam-se nos spikes seguintes, sabendo que foram afinados com N = 10;
- **se o timebox de 1 dia acabar a meio**, o passo 1 (`alpha`) e o 2 (τ) têm prioridade; as épocas ficam escritas como por medir;
- o custo em tempo de pod e em dólares (estimativa antes de correr: menos de 1 hora a 0,32 $/h, não medida).

**Feito** no PR #17 (13 de outubro de 2026), descrição completa em [`prs/pr_E4-S5.md`](prs/pr_E4-S5.md). A tabela, as comparações e a curva estão na secção 15 do [`qualidade_dados.md`](qualidade_dados.md), calculadas com o [`notebooks/06_tfidf_epocas.ipynb`](../notebooks/06_tfidf_epocas.ipynb); os ficheiros dos treinos no pod estão em `metrics/e4_s5/`.

- **As opções foram escritas antes de correr num ficheiro**, [`experiencias/e4_s5.yaml`](../experiencias/e4_s5.yaml), que o treino no pod ([`scripts/correr_experiencia.py`](../scripts/correr_experiencia.py)) e o notebook leem. Um teste escreve à mão a lista do backlog e falha se o YAML mudar.
- **Repetível:** a regra por defeito, repetida, deu sugestões 100% iguais às do E4-S4, nos 4 τ.
- **Passo 1:** o `alpha` 0,00000001 perde (−7,5 pontos no top-5 do código completo, −6,4 na classe); fica o 0,0000001, que deixa de estar no extremo da lista. **Passo 2** não se fez (o `alpha` não mudou); o τ fica 0,25.
- **Passo 3:** ganharam **24 épocas fixas** (regra por defeito → 12 → 24). Na validação, top-5 de 78,3% no código completo e 84,1% na classe (E4-S4: 76,9% e 82,9%); 75,1% e 82,2% no texto novo. Continua a bater a régua nos quatro casos.
- **Duas limitações escritas na secção 15:** 24 é o extremo da lista (não se acrescentaram valores, como a regra dizia); e o bootstrap não mede a variação entre treinos com outra semente, por isso parte do ganho de 18 para 24 épocas pode ser sorte do treino.
- **O valor por defeito do treino muda:** `EPOCAS = 24` em `src/tfidf.py`, ao lado do `ALPHA` e do `TAU`. Para treinar com a regra por defeito do scikit-learn, como o E4-S2 vai fazer, usa-se `--regra-por-defeito` (ou `epocas=None`).
- **A curva época a época** (aproximação, só para ver) fica quase plana da época 15 à 24.
- **Custo:** um pod `cpu3g` de 32 processadores, 4 treinos ao mesmo tempo, cerca de 44 minutos a 1,28 $/h: cerca de 0,95 $ (`metrics/e4_s5/pod_info.json`).

### E4-S1 · Spike: que N, épocas e τ usar nos códigos raros? (timebox: 1 dia)

**Pergunta:** que combinação de N (5, 10 ou 20, decisão 9), de épocas (a regra por defeito ou um número fixo de 3, 6, 12, 18 ou 24) e de τ (0, 0,25 ou 0,5) dá melhor resultado na validação?

**Porque uma grelha, e não só o N** (mudado a 13 de outubro de 2026): o número ótimo de épocas **depende do N** — o N muda quantas etiquetas existem e quantos exemplos tem cada uma, e por isso quanto cada etiqueta precisa de treinar. O τ também depende do N, porque corrige a frequência das etiquetas, que muda com o N. Escolher o N com as épocas e o τ fixos podia escolher um N que não seria o melhor com os seus próprios valores: é a **interação** entre hiperparâmetros, e um eixo de cada vez não a vê. Por isso testam-se as combinações de uma vez. O `alpha` fica **fixo** em 0,0000001 (E4-S4/S5): o seu efeito já foi medido, é grande e claro (o 0,00000001 perdeu), e reabri-lo multiplicaria a grelha sem ganho provável.

**Depende de:** E4-H1 e E4-S5.

**Como** (com o treino completo, no pod; 3 N × 6 épocas = **18 treinos**, cada um com `--taus 0 0.25 0.5`):

- **N** = 5, 10 e 20;
- **épocas** = a regra de paragem por defeito do scikit-learn (`null`) e um número fixo, igual para todas as etiquetas, de 3, 6, 12, 18 e 24;
- **τ** = 0, 0,25 e 0,5 (o 1,0 perdeu claramente no E4-S4). O τ só muda a ordem das sugestões ao sugerir, não retreina o modelo: cada treino grava os três ficheiros de sugestões, e daí as 3 × 6 × 3 = 54 combinações avaliadas saírem de 18 treinos.

**Regra de escolha, escrita antes de correr:** as opções ficam em [`experiencias/e4_s1.yaml`](../experiencias/e4_s1.yaml), escritas antes de correr, e a decisão aplica a regra de comparação do E4-T1 (no [`notebooks/07_tfidf_grelha.ipynb`](../notebooks/07_tfidf_grelha.ipynb)). O vencedor é a combinação com o melhor top-5 no **código completo**, desde que não perca o top-5 na **classe** (métrica de guarda, por bootstrap) e desde que bata a opção em vigor — o modelo do E4-S5, N = 10, 24 épocas e τ = 0,25 — pela regra de comparação do E4-T1. Em caso de empate, fica a opção em vigor (N = 10, 24 épocas, τ = 0,25). Com 54 combinações, alguma ganharia só por sorte, por isso o número do vencedor fica **otimista** (viés de seleção): escreve-se como limitação no `qualidade_dados.md` e confirma-se no teste, na fase 5.

**Resultado:**

- uma tabela com as 54 combinações (top-5 no código completo e na classe, e F1 macro) e a combinação escolhida;
- a limitação do viés de seleção escrita no `qualidade_dados.md`;
- o N escolhido na decisão 9 do `qualidade_dados.md`, com as épocas e o τ escolhidos, e passa a ser o valor por defeito;
- se o N mudar, a régua (E4-T2) e os resultados do E4-H1 voltam a calcular-se com o novo N;
- o custo em tempo de pod e em dólares (estimativa: 18 treinos; mede-se com o primeiro).

**Feito** no PR #18 (13 de outubro de 2026), descrição completa em [`prs/pr_E4-S1.md`](prs/pr_E4-S1.md). As 54 combinações estão no [`notebooks/07_tfidf_grelha.ipynb`](../notebooks/07_tfidf_grelha.ipynb), e as primeiras na secção 16 do [`qualidade_dados.md`](qualidade_dados.md); os ficheiros dos treinos no pod estão em `metrics/e4_s1/`.

- **Repetível:** o treino com N = 10 e 24 épocas deu sugestões 100% iguais às do E4-S5, com τ = 0 e τ = 0,25.
- **Vencedor:** a combinação do E4-S5 (N = 10, 24 épocas, τ = 0,25), com 78,35% no top-5 do código completo e 84,11% na classe. Como já era a opção em vigor, a decisão é mantê-la: **N = 10** (decisão 9), com as 24 épocas e o τ = 0,25 (decisão 7). Os concorrentes mais próximos (N = 5 com 24 épocas, N = 10 com τ = 0) perdem com uma diferença que conta.
- **A grelha confirmou o E4-S5**, em vez de o substituir: nenhuma das outras 53 combinações bate a que já estava em vigor.
- **Limitação escrita na secção 16:** com 54 combinações, o vencedor fica otimista (viés de seleção); confirma-se no teste, na fase 5.
- **Ferramenta:** o `scripts/correr_experiencia.py` ganhou o modo `grelha` (`modo: grelha` no YAML): o produto cartesiano dos eixos, um treino por (N, épocas) e um ficheiro de sugestões por τ.
- **Custo:** um pod `cpu3g` de 32 processadores e 128 GB, ligado cerca de 80 minutos (15:36–16:56 UTC; os 18 treinos correram das 15:41 às 16:54): cerca de 1,66 $ (`metrics/e4_s1/pod_info.json`).

### E4-S3 · Spike: a entidade compradora ajuda o modelo? (timebox: 1 dia)

**Pergunta:** juntar a entidade (`adjudicante`) ao texto de entrada melhora as sugestões, ou faz o modelo aprender os hábitos de cada entidade, incluindo os códigos genéricos? (decisão 7)

**Depende de:** E4-S1 (usa o N escolhido) e E4-H1.

**Como:** treinar o mesmo modelo com e sem a entidade e comparar na validação. A comparação faz-se em três grupos:

- todos os contratos;
- os contratos de entidades que aparecem no treino;
- os contratos de entidades que não aparecem no treino.

**Regra de escolha, escrita antes de correr:** a regra de comparação do E4-T1, em todos os contratos, com "sem entidade" como opção por defeito (é mais simples). Além disso, a entidade só entra se as duas condições se cumprirem:

- não piorar o top-5 no código completo nos contratos de entidades novas;
- não aumentar a parte das 1.as sugestões ao nível do grupo ou acima.

Em ambas, "piorar" e "aumentar" querem dizer uma diferença que conte segundo o bootstrap.

**Resultado:**

- uma tabela com e sem entidade nos três grupos, com a parte das 1.as sugestões ao nível do grupo ou acima;
- a decisão 7 fechada no `qualidade_dados.md`;
- se a entidade entrar, a régua (E4-T2) e os resultados do E4-H1 voltam a calcular-se.

**Feito** (13 de outubro de 2026), descrição completa em [`prs/pr_E4-S3.md`](prs/pr_E4-S3.md). A entidade **não entra**: ganha no top-5 (80,76% contra 78,35% no código completo) e não piora nas entidades novas, mas aumenta a parte das 1.as sugestões ao nível do grupo ou acima (33,13% contra 30,57%), e a regra escrita antes de correr proibia isso. A tabela dos três grupos está no [`notebooks/08_entidade.ipynb`](../notebooks/08_entidade.ipynb) e na secção 17 do [`qualidade_dados.md`](qualidade_dados.md); a decisão 7 está fechada. Como o modelo em vigor não muda, a régua e o E4-H1 não se recalculam.

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

**Regra de escolha, escrita antes de correr:** a regra de comparação do E4-T1, aplicada sempre ao mesmo grupo de contratos: os da validação cujo texto não aparece no treino da janela mais longa (desde 2019). Em caso de empate, fica a janela mais curta, porque é mais barato treinar com menos dados.

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

- [ ] As métricas da validação do TF-IDF e da régua ficam guardadas no git, num ficheiro pequeno. Cada bloco de métricas diz de que conjunto vem e o período de datas. As métricas do TF-IDF são as do modelo por defeito (`alpha` e τ da decisão 6): o ficheiro de sugestões a ler é o que o `treino.json` indica em `ficheiro_do_modelo_por_defeito`, e não o `sugestoes_validacao.parquet`, que é o de τ = 0.
- [ ] O CI falha se o top-5 guardado do TF-IDF não for maior do que o da régua, no código completo e na classe.
- [ ] O CI falha se algum bloco de métricas disser que vem de outro conjunto que não a validação (outubro de 2025).
- [ ] O treino e a avaliação carregam os conjuntos sempre da mesma maneira, que recusa o teste sem uma autorização explícita. Um teste no CI confirma a recusa.
- [ ] Cada verificação falha quando se estraga de propósito o que ela guarda.

### Ordem e dependências do E4

```
E4-T1 ──> E4-T2 ──┐
E4-S0 ────────────┴──> E4-H1 ──┬──> E4-T3 ────────┐
                               ├──> E4-T5        │
                               └──> E4-S4 ───────┴──> E4-S5 ──> E4-S1 ──┬──> E4-S3 ──> (atualizar E4-T2, E4-H1 e E4-T5)
E4-T4 ──────────────────────────────────────────────────────────────────┴──> E4-S2 ──> (atualizar E4-T2, E4-H1 e E4-T5)
```

- O E4-T1, o E4-T2, o E4-S0, o E4-H1, o E4-S4, o E4-T3, o E4-S5, o E4-S1 e o E4-S3 estão feitos (a entidade não entra). Segue-se o E4-T4 (a data de início na preparação, que desbloqueia o E4-S2); o E4-T5 não depende de mais nada.
- O E4-S1 testa o N, as épocas e o τ numa só grelha, porque interagem (ver o E4-S1). O E4-S2 continua a treinar com a regra por defeito das épocas: a janela muda a quantidade de dados e um número fixo de épocas não se adapta (as épocas voltam a medir-se no fim, para o N e a janela escolhidos).
- O E4-S5 vem antes do E4-S1 para os spikes seguintes (N, entidade, janela) já usarem o TF-IDF afinado.
- O E4-T5 cria o portão logo a seguir ao E4-H1. Os valores guardados voltam a atualizar-se no fim, com o N, a janela e a entidade decididos.

### Decisões tomadas para o E4 (10 de outubro de 2026)

1. **Entidade compradora:** testa-se já com o TF-IDF (spike E4-S3), não depois do BERTimbau. É mais barato: se não ajudar no TF-IDF, provavelmente não vale a pena gastar GPU a testá-la.
2. **Como se escolhe entre opções:** decide o top-5 no código completo, com o top-5 na classe como métrica de guarda. Uma diferença só conta se o bootstrap mostrar que não é acaso. Regra completa no E4-T1.
3. **Máquina:** alugar um pod na Runpod com pelo menos 32 GB de memória para o TF-IDF (mudado a 11 de outubro de 2026; antes era a Hetzner CX53). Ver o E4-S0.
4. **A E4-H1 fica como história**, com uma nota a dizer que o técnico só a recebe quando houver interface.
5. **Classificador do TF-IDF** (11 de outubro de 2026): `SGDClassifier(loss="log_loss")`, porque a `LogisticRegression` com o solver `saga` não treina em tempo útil com cerca de 3 000 etiquetas. Foi escolhida só pela velocidade e ficou confirmada no E4-H1, porque bateu a régua na validação (secção 13 do `qualidade_dados.md`). Ver o E4-S0.
6. **Regularização e correção das etiquetas frequentes do TF-IDF** (11 de outubro de 2026, spike E4-S4): `alpha` = 0,0000001 (em vez do valor por defeito do scikit-learn, 0,0001) e τ = 0,25. Escolhidos na validação com a regra do E4-T1, escrita antes de correr. O `alpha` 0,0000001 foi acrescentado à lista depois de ver os resultados, porque o 0,000001 ganhou e era o extremo da lista. O 0,0000001 voltou a ganhar e continua no extremo; o 0,00000001 não foi testado (limitação escrita na secção 14 do `qualidade_dados.md`).

7. **Épocas do TF-IDF** (12 de outubro de 2026, spike E4-S5): 24 épocas fixas, iguais para todas as etiquetas, sem a paragem automática do scikit-learn (`EPOCAS = 24` em `src/tfidf.py`). O `alpha` e o τ da decisão 6 mantêm-se: o `alpha` 0,00000001 perdeu. Escolhidos na validação com a regra do E4-T1 e a lista escrita antes de correr (`experiencias/e4_s5.yaml`). Só vale para N = 10 e o treino desde 2022. O E4-S2 continua a comparar as suas opções com a regra por defeito, e as épocas voltam a medir-se no fim, para o N e a janela escolhidos. O E4-S1 reabre as épocas dentro de uma grelha N × épocas × τ, porque o número ótimo de épocas depende do N (mudado a 13 de outubro de 2026 — ver o E4-S1). 24 é o extremo da lista testada (limitação escrita na secção 15 do `qualidade_dados.md`).

### Decisões pendentes do E4

Nenhuma.

---

## Ideias para depois (ainda não são épicos)

- **Interface por níveis da árvore** (proposta a 10 de outubro de 2026): em vez de uma lista de 5 códigos, o técnico escolhe a divisão, depois o grupo, depois a classe, com o modelo a ordenar as opções em cada passo. Ajudaria quando o técnico só sabe a área da compra. O projeto de Zwolle/Haskoning treinou um modelo por nível (ver `readme.md`). Fica para quando houver interface, que está fora do E4.
