## E4-S4 · Spike: afinar o TF-IDF (regularização e etiquetas frequentes)

## Em poucas palavras

O primeiro TF-IDF (E4-H1) batia a régua, mas sugeria quase sempre os mesmos códigos de saúde, mesmo para descrições que não tinham nada a ver. Este spike encontrou a causa: o modelo estava travado demais, e as etiquetas raras quase não aprendiam com o texto. Testámos 4 forças de travão e 4 forças de uma correção que tira vantagem às etiquetas frequentes, e escolhemos com a regra escrita antes de correr. O código certo passa a estar nas 5 sugestões em 76,9% dos contratos de validação, contra 43,2% antes. O modelo escolhido passa a ser o valor por defeito do projeto.

| Na validação (outubro de 2025) | Régua | TF-IDF antes (E4-H1) | TF-IDF afinado |
|---|---|---|---|
| O código certo está nas 5 sugestões | 26,3% | 43,2% | **76,9%** |
| A classe certa está nas 5 sugestões | 15,6% | 37,9% | **82,9%** |
| O mesmo, só texto que não aparece no treino (código / classe) | 23,7% / 14,8% | 38,5% / 34,2% | **73,5% / 80,8%** |
| A 1.ª sugestão é um dos 5 códigos da régua | 100% | 84,7% | 28,7% |

## Porquê

Um TF-IDF mal afinado fazia o BERTimbau e o Albertina parecerem melhores do que são: batê-lo não provava nada. E para o técnico, o código certo entre as 5 sugestões em 2 de cada 5 compras ajuda pouco. Em 3 de cada 4, ajuda.

> Pergunta do spike: um `alpha` (a força da regularização) mais pequeno, ou a correção das etiquetas frequentes de Menon et al. (2021), melhoram as sugestões na validação?

Item: [`E4-S4`](docs/backlog.md) · Desbloqueia: E4-S1 (o N dos códigos raros, agora com o modelo afinado), E4-T5 (portão de qualidade, com os números do modelo afinado)

## O que mudou no código

Dois termos usados abaixo:
- **Regularização (`alpha`):** um travão que impede o modelo de dar pesos grandes às palavras, para não decorar o treino. Com o travão forte demais, o modelo deixa de usar o texto. O valor por defeito do scikit-learn é 0,0001; a documentação recomenda procurá-lo entre 0,1 e 0,000001.
- **Correção das etiquetas frequentes (τ, "tau"):** em vez de ordenar as etiquetas pela pontuação do modelo, ordena-se por pontuação − τ · log(frequência da etiqueta no treino). Exemplo com τ = 1: uma etiqueta com frequência 0,1 ganha +2,3 e uma com frequência 0,5 ganha só +0,7. Com τ = 0 nada muda.

### `src/tfidf.py`

- `ALPHA = 1e-7` e `TAU = 0.25` (`src/tfidf.py:39-40`): os valores escolhidos neste spike, agora por defeito.
- `train` (`src/tfidf.py:43`): passa a receber `alpha` (por defeito `ALPHA`) e guarda no modelo a frequência de cada etiqueta no treino (`log_prior_`, linha 68).
- `log_prior` (`src/tfidf.py:84`): log da frequência de cada etiqueta, contada só no treino e depois da regra dos códigos raros.
- `suggest_many` (`src/tfidf.py:97`): as sugestões para vários τ de uma vez. As pontuações calculam-se uma só vez, o que poupou tempo no pod.
- `suggest` (`src/tfidf.py:128`): passa a usar `TAU` por defeito; `tau=0.0` dá a ordem só pela pontuação.

```python
label = collapse_rare_codes(codigo, n_min=n_min).dropna()
counts = label.value_counts()
...
return np.log(counts.reindex(classes).to_numpy(dtype=float) / len(label))
```
A frequência conta-se só com os códigos do treino, depois da regra dos códigos raros. Contar com a validação seria fuga de dados.

```python
if any(t < 0 for t in taus):
    raise ValueError(f"τ tem de ser >= 0: {list(taus)}")
...
adjusted = scores if t == 0 else scores - t * prior
out[t].extend(top_k(adjusted, classes, k))
```
Recusa τ negativos e tira a cada pontuação τ vezes o log da frequência. Como esse log é negativo, as etiquetas raras sobem mais do que as frequentes.

### `src/train_tfidf.py`

- `run` (`src/train_tfidf.py:79`): recebe `alpha` e `taus` (por defeito `ALPHA` e τ = 0 e 0,25) e grava um ficheiro de sugestões por τ.
- `suggestions_file` (`src/train_tfidf.py:74`): o nome de cada ficheiro. τ = 0 mantém o nome do E4-H1 (`sugestoes_validacao.parquet`); os outros levam o τ (`sugestoes_validacao_tau0.25.parquet`).
- `SAIDA` (`src/train_tfidf.py:46`): por defeito grava em `metrics/tfidf/`. As pastas `metrics/e4_h1/` e `metrics/e4_s4/` ficam como registo fixo de cada item.
- `main`: opções `--alpha`, `--taus` e `--sem-modelo`. Sem gravar o modelo poupam-se cerca de 5 GB de disco no pod.

```python
"ficheiros": {suggestions_file(t): t for t in taus},
"ficheiro_do_modelo_por_defeito": suggestions_file(tfidf.TAU) if tfidf.TAU in taus else None,
```
O `treino.json` diz o τ de cada ficheiro e qual é o do modelo escolhido. Sem isto, quem lesse o ficheiro principal avaliava o modelo sem correção sem dar por isso.

### `src/suggest.py`

- `format_suggestions` (`src/suggest.py:32`) e a opção `--tau` (linha 51): o comando ordena com o τ escolhido. A probabilidade mostrada continua a ser a do modelo, por isso nem sempre desce de linha para linha.

### Decisões tomadas pelo caminho

- **Treinar com os dados todos, no pod:** o melhor `alpha` depende do número de exemplos, por isso uma amostra no portátil escolhia um valor errado.
- **Regra de escolha escrita no backlog antes de correr:** primeiro o `alpha` (com τ = 0), depois o τ. Cada candidato é comparado com a opção em vigor, com a regra do E4-T1: decide o top-5 no código completo, e a classe serve de guarda.
- **O `alpha` 0,0000001 entrou depois de ver os resultados:** o 0,000001 tinha ganho e era o extremo da lista. Está escrito assim no backlog.
- **Parou-se no 0,0000001, que voltou a ser o extremo:** o 0,00000001 não foi testado. Está escrito como limitação, porque esse critério de paragem não estava escrito antes.
- **Os nomes dos ficheiros ficam como estavam:** as pastas guardadas usam-nos. O `treino.json` diz qual é o do modelo escolhido, e o E4-T5 (no backlog) diz para ler esse.

## Outras mudanças

- `notebooks/05_tfidf_afinacao.ipynb` (novo, `make afinacao`): lê as 16 opções e verifica que o `alpha` antigo repete o E4-H1. Depois aplica a regra de escolha passo a passo e compara o escolhido com o E4-H1 e com a régua. Nunca lê o teste.
- `metrics/e4_s4/`: as sugestões dos 4 treinos no pod (4 `alpha` × 4 τ), os `treino.json`, os registos de memória e o `pod_info.json`.
- `docs/qualidade_dados.md`: secção 14 nova, com a tabela das 16 opções, as comparações e as limitações (`alpha` no extremo, valores um pouco otimistas por terem sido escolhidos na validação, o mecanismo do travão como hipótese). A secção 13 fica marcada como substituída.
- `docs/backlog.md`: o spike E4-S4, com a regra escrita antes de correr, o resultado e a decisão 6; o E4-T5 passa a dizer que ficheiro ler.
- `docs/workflow.md` e `notebooks/readme.md`: linhas do spike e do notebook 05.
- `CLAUDE.md`: os valores por defeito, a pasta `metrics/tfidf/` e as pastas que ficam fixas.
- `Makefile`: `make afinacao`; o `make treinar-tfidf` grava em `metrics/tfidf/`.
- Testes: `tests/unit/test_tfidf.py`, `tests/unit/test_train_tfidf.py` (novo), `tests/integration/test_tfidf_pipeline.py`, `tests/integration/test_tfidf_afinacao_dados_reais.py` (novo) e `tests/integration/readme.md` (ver as provas).

### Ficou igual de propósito

- **A régua e a avaliação:** o modelo afinado é medido com as mesmas regras do E4-H1.
- **O N dos códigos raros (10), os pares de palavras e o `min_df`:** só se mexeu em dois botões, para saber o que ajudou. O N é o E4-S1.
- **`metrics/e4_h1/`, a secção 13 e o notebook 04:** ficam como registo do primeiro modelo. O notebook 05 mostra o escolhido ao lado.
- **Os nomes dos ficheiros de sugestões:** ver "Decisões".

## Validação e provas

### Critérios de aceitação

O spike não tem caixas de critérios. Usei o "Como", a "Regra de escolha", o "Também se mostra" e o "Resultado" do backlog.

| Pedido no backlog (copiado) | Cumprido? | Prova (comando → resultado) |
|---|---|---|
| `alpha` = 0,0001 (o do E4-H1), 0,00001 e 0,000001 (e depois 0,0000001), com o treino completo, no pod | ✅ | `metrics/e4_s4/alpha_*/treino.json` (N = 10, 3 000 etiquetas, convergiram); `test_ficheiros_de_cada_alpha_sao_da_validacao_atual` → 4 passed |
| para cada `alpha`, a correção das etiquetas frequentes com τ = 0, 0,25, 0,5 e 1 | ✅ | 16 ficheiros em `metrics/e4_s4/`; notebook 05, célula 9 |
| Regra de escolha: primeiro o `alpha` com τ = 0, depois o τ; cada candidato contra a opção em vigor, com a regra do E4-T1 | ✅ | notebook 05, células 11 e 13; `test_regra_de_escolha_da_os_vencedores_do_doc_e_o_valor_por_defeito` → passa (refaz a escolha com o bootstrap sobre os dados reais) |
| Também se mostra o F1 macro e a parte das 1.as sugestões que são uma das 5 etiquetas da régua | ✅ | notebook 05, célula 9; tabela da secção 14 |
| Uma tabela com todas as opções | ✅ | secção 14 do `qualidade_dados.md` (16 opções e a régua) |
| O `alpha` e o τ escolhidos, escritos como decisão do E4 | ✅ | decisão 6 do backlog |
| Se mudarem, passam a ser o valor por defeito em `src/tfidf.py` | ✅ | `src/tfidf.py:39-40`; `test_train_usa_o_alpha_escolhido_por_defeito`, `test_suggest_usa_o_tau_escolhido_por_defeito` → passam |
| e os resultados do E4-H1 voltam a calcular-se | ✅ / ❌ | ✅ A avaliação completa na validação e o "bate a régua" foram refeitos para o escolhido (notebook 05, células 17 e 18). ❌ O comando de sugestões para as 3 descrições à mão não voltou a correr, porque os treinos do spike não gravaram o modelo. Fica para o próximo treino no pod (E4-S1), escrito no backlog. |

Caixas do `docs/backlog.md` marcadas como `[x]` neste PR: nenhuma, porque o spike não tem caixas. A caixa da afinação no `workflow.md` está marcada.

### Testes focados

`.venv/bin/pytest -q tests/unit/test_tfidf.py tests/unit/test_train_tfidf.py tests/integration/test_tfidf_pipeline.py tests/integration/test_tfidf_afinacao_dados_reais.py` → 68 passed

### O comportamento que já existia continua igual

- Suíte inteira: `.venv/bin/pytest -q` → 243 passed (`main` antes deste PR: 207 passed).
- Com `alpha` = 0,0001 e τ = 0, o treino repete exatamente o E4-H1 (100% das sugestões iguais): `test_alpha_antigo_sem_correcao_repete_o_e4_h1` → passa; notebook 05, célula 7.
- Os números do E4-H1 (secção 13) continuam certos: `test_tfidf_dados_reais.py` → passa.
- A régua não mudou: `test_regua_dados_reais.py` → passa.
- CI: `success` no commit `59348d6` (o de cima, com estas correções).

### Os testes novos falham com o código estragado

Cada estrago foi feito à mão, com o teste corrido e o código reposto a seguir. Vários foram repetidos pelos revisores, numa cópia.

| Teste | Estrago introduzido | Resultado com o estrago |
|---|---|---|
| `test_tau_tira_vantagem_a_etiqueta_frequente` | `scores - t * prior` trocado por `+`; τ ignorado | falhou ✅ |
| `test_log_prior_usa_as_etiquetas_depois_dos_codigos_raros` | `log_prior` sem a regra dos códigos raros; sem dividir pelo total | falhou ✅ |
| `test_log_prior_com_etiqueta_que_nao_esta_no_treino_da_erro` | sem o erro de etiqueta em falta | falhou ✅ |
| `test_train_guarda_as_frequencias_do_treino_no_modelo`, `test_modelo_gravado_leva_as_frequencias_do_treino` | modelo sem frequências; frequências com outro N | falhou ✅ |
| `test_sem_prior_usa_as_frequencias_do_modelo` | `suggest` sem usar as frequências do modelo | falhou ✅ |
| `test_train_usa_o_alpha_escolhido_por_defeito` | `alpha` não passado ao classificador; `ALPHA = 1e-6` | falhou ✅ |
| `test_suggest_usa_o_tau_escolhido_por_defeito` | `suggest` volta a τ = 0 por defeito | falhou ✅ (depois da 3.ª revisão; antes só falhava outro teste, por acaso) |
| `test_tau_zero_ignora_as_frequencias_e_ordena_so_pela_pontuacao` | com τ = 0 aplica-se uma correção | falhou ✅ |
| `test_tau_negativo_da_erro` | τ negativo aceite | falhou ✅ |
| `test_nome_do_ficheiro_de_sugestoes_por_tau` | `{tau}` em vez de `{tau:g}` ("tau1.0"); regra do τ = 0 | falhou ✅ |
| `test_mudar_a_validacao_nao_muda_as_frequencias_nem_as_sugestoes_corrigidas` | frequências contadas com treino + validação (fuga) | falhou ✅ |
| `test_comando_de_treino_com_alpha_taus_e_sem_modelo` | ignorar `--sem-modelo`, `--alpha` ou `--taus` | falhou ✅ |
| `test_comando_de_sugestoes_aceita_outro_tau`, `test_comando_ordena_com_o_tau_escolhido` | o comando ignora o τ | falhou ✅ |
| `test_treino_por_defeito_grava_numa_pasta_propria_e_nao_no_registo_do_e4_h1` | `SAIDA` volta a `metrics/e4_h1` | falhou ✅ |
| `test_treino_json_tem_as_chaves_esperadas` | `ficheiro_do_modelo_por_defeito` aponta para o ficheiro de τ = 0 | falhou ✅ |
| `test_regra_de_escolha_da_os_vencedores_do_doc_e_o_valor_por_defeito` | `TAU = 0.5`; `ALPHA = 1e-6`; a regra sem o desempate pela classe | falhou ✅ |
| `test_top5_do_escolhido_e_o_do_doc` | a avaliação da classe a comparar 3 dígitos | falhou ✅ |
| `test_ficheiros_de_cada_alpha_sao_da_validacao_atual` | um ficheiro com as linhas por ordem inversa | falhou ✅ |
| `test_alpha_antigo_sem_correcao_repete_o_e4_h1` | a ordem das sugestões de 1 contrato trocada | falhou ✅ |
| `test_notebook_da_afinacao_nao_abre_o_teste` | uma célula do notebook 05 a ler `teste.parquet` | falhou ✅ |

### Números que provam que funciona

Na validação de outubro de 2025, com o notebook 05. O teste não foi lido.

| Medida | E4-H1 (antes) | Escolhido (depois) | Fonte |
|---|---|---|---|
| Top-5 código completo, todos | 43,2% | 76,9% | célula 9 |
| Top-5 classe, todos | 37,9% | 82,9% | célula 9 |
| Top-5 código completo, texto novo | 38,5% | 73,5% | célula 9 |
| Top-5 classe, texto novo | 34,2% | 80,8% | célula 9 |
| Top-1 código completo | 30,8% | 57,0% | célula 9 |
| F1 macro, código completo | 0,0107 | 0,2565 | célula 9 |
| 1.ª sugestão é um dos 5 da régua | 84,7% | 28,7% | célula 9 |
| Top-1 na divisão | 43,3% | 81,0% | célula 17 |
| Custo dos treinos no pod | — | cerca de 40 min, perto de 0,21 $ | `metrics/e4_s4/pod_info.json` |

**Limitações, ditas às claras:**
- O `alpha` 0,0000001 é o extremo da lista testada. Não se sabe se o 0,00000001 seria melhor.
- O `alpha` e o τ foram escolhidos na validação, por isso estes números ficam um pouco otimistas. O teste continua fechado e vai mostrar quanto.
- A combinação `alpha` = 0,000001 com τ = 0,5 tem mais 0,87 pontos no top-5 do código completo, mas menos 0,96 na classe. A guarda da classe elimina-a, também numa comparação direta (célula 15).

### Revisão de código

`/requesting-code-review`, três agentes revisores novos, sem contexto da tarefa:
1. `4e4c85f..f6d006d`: **com correções**, nada crítico. 3 pontos importantes, todos corrigidos em `8586547`:
   - o treino gravava por cima de `metrics/e4_h1/`;
   - o `alpha` continuava no extremo da lista sem isso estar dito;
   - o `suggest` usava τ = 0 por defeito.

   Também corrigidos os pontos menores baratos: o tempo do teste no readme, o nome da pasta na docstring, o CLAUDE.md, o mecanismo do travão escrito como hipótese, a chave explícita no teste.
2. `4e4c85f..8586547`: **com correções**, nada crítico. 1 ponto importante corrigido em `0adea2a`: o ficheiro principal de sugestões é o de τ = 0, e agora o `treino.json` diz qual é o do modelo escolhido. Também corrigidos: a conta "0,00002" sem célula, a extrapolação sobre o 0,00000001, e `len(ids)` em vez da variável do último ciclo.
3. `4e4c85f..0adea2a`: **com correções**, nada crítico. 2 pontos importantes, corrigidos no commit seguinte:
   - `test_suggest_usa_o_tau_escolhido_por_defeito` não falhava com τ = 0 por defeito, porque no modelo minúsculo os dois τ davam a mesma ordem;
   - `test_tau_zero_e_igual_a_sem_correcao` comparava sem querer τ = 0 com τ = 0,25.

   Os dois foram reescritos com um modelo falso de pontuações fixas, em que o τ muda mesmo a ordem, e cada um foi visto a falhar com a regra estragada. Também corrigidos: o tempo dos treinos (7 a 9 minutos com a sugestão, citando os logs) e a mensagem de erro para modelos gravados antes do E4-S4.

   Não houve uma 4.ª revisão: as últimas mudanças são só testes, uma mensagem de erro e texto, e cada teste reescrito foi visto a falhar.

**Não corrigido, e porquê:**
- **O comando de sugestões com o modelo novo** fica para o próximo treino no pod, como está no backlog.
- **O 0,00000001 não foi testado:** está escrito como limitação.
- **O `log_prior_` é um atributo solto no modelo:** sobrevive ao gravar e ler com joblib, que é o que o projeto usa, mas perde-se com `sklearn.clone`, que o projeto não usa.
- **Um modelo antigo, sem frequências, falha no `make sugerir`** com uma mensagem que diz para treinar de novo ou usar `--tau 0`. Não há modelos antigos guardados: os pods foram apagados.
- **Os 40 minutos e os 0,21 $** vêm do `pod_info.json`, escrito à mão a partir das leituras da Runpod, e não de uma célula. Está dito na secção 14.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
