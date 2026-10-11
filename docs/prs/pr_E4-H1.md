## E4-H1 · 5 sugestões de CPV a partir da descrição

## Em poucas palavras

Este PR cria o primeiro modelo do projeto que lê a descrição de uma compra e devolve 5 códigos CPV ordenados. O modelo aprende com os contratos de janeiro de 2022 a setembro de 2025 e treina num computador alugado, porque o portátil não tem memória suficiente. Nos contratos de outubro de 2025 (a validação), o código certo está entre as 5 sugestões em 43,2% dos contratos, contra 26,3% da régua que sugere sempre os mesmos 5 códigos sem ler nada. Isto cumpre o mínimo pedido. Mas o modelo ainda puxa demasiado para os códigos mais frequentes, e o spike seguinte (E4-S4, noutro PR) corrige isso.

| Na validação (outubro de 2025) | O código certo está nas 5 sugestões | A classe certa (4 dígitos) está nas 5 sugestões |
|---|---|---|
| Modelo, todos os contratos | **43,2%** | **37,9%** |
| Régua, todos os contratos | 26,3% | 15,6% |
| Modelo, só texto que não aparece no treino | **38,5%** | **34,2%** |
| Régua, só texto que não aparece no treino | 23,7% | 14,8% |

## Porquê

Até aqui havia dados limpos, uma avaliação comum e uma régua, mas nenhum modelo. Esta é a primeira capacidade para o técnico de contratação: escrever a descrição da compra e escolher entre 5 códigos, em vez de procurar na tabela de cerca de 9 500. O técnico só a recebe quando houver interface (fora do E4). Por agora, os números servem de régua para o BERTimbau e o Albertina.

> Como técnico de contratação, quero escrever a descrição da compra e receber 5 códigos CPV ordenados, para escolher entre 5 em vez de procurar na tabela de cerca de 9 500 códigos.

Item: [`E4-H1`](docs/backlog.md) · Desbloqueia: E4-T3 (testes ao treino no CI), E4-T5 (portão de qualidade), E4-S1 (o N dos códigos raros) e E4-S4 (afinar o TF-IDF, PR seguinte)

## O que mudou no código

Dois termos usados abaixo:
- **TF-IDF:** transforma cada texto num vetor com um peso por palavra e por par de palavras. Uma palavra pesa mais se aparece neste texto e é rara nos outros. Exemplo: em "limpeza de escolas", "limpeza" pesa muito e "de" quase nada.
- **Regressão logística um-contra-todos:** um modelo "sim/não" por etiqueta (cerca de 3 000), que dá uma pontuação a cada etiqueta. As 5 sugestões são as 5 pontuações mais altas.

### `src/tfidf.py` (novo)

- `train` (`src/tfidf.py:32`): treina o TF-IDF e o classificador com o treino. As etiquetas passam primeiro pela regra dos códigos raros (decisão 9), com o mesmo N da régua.
- `top_k` (`src/tfidf.py:61`): escolhe as k etiquetas de maior pontuação; num empate fica o menor código.
- `suggest` (`src/tfidf.py:71`): dá as 5 sugestões a cada texto, em lotes de 2 000 textos para não gastar memória demais.

**O treino recusa o que não devia lá estar.** Antes de treinar, verifica três coisas:

```python
if codigo.str.endswith("000000").any():
    # O prepare.py tira-os do treino (decisão 8); se aparecerem, o modelo podia sugeri-los.
    raise ValueError("o treino tem códigos ao nível da divisão (XX000000)")
label = collapse_rare_codes(codigo, n_min=n_min)
keep = label.notna()
n_labels = label[keep].nunique()
if n_labels < TOP_K:
    raise ValueError(...)
```
Recusa códigos ao nível da divisão (o modelo nunca os deve sugerir), aplica a regra dos códigos raros, tira os exemplos que saíram e exige pelo menos 5 etiquetas. Antes disto, verifica também que o texto e o código têm o mesmo índice.

**A mesma descrição dá sempre as mesmas 5 sugestões.** A semente aleatória é fixa (`SEED = 0`), e os empates resolvem-se assim:

```python
order = np.argsort(-scores, axis=1, kind="stable")[:, :k]
return [list(row) for row in classes[order]]
```
A ordenação "estável" mantém os empates pela ordem das colunas. Como as etiquetas estão por ordem crescente, num empate fica o menor código.

### `src/train_tfidf.py` (novo)

- `check_not_test` (`src/train_tfidf.py:48`): dá erro se lhe passarem um ficheiro chamado `teste.parquet`.
- `run` (`src/train_tfidf.py:70`): treina, grava o modelo, sugere 5 códigos para cada contrato da validação e grava `sugestoes_validacao.parquet` e `treino.json` (datas, N, vocabulário, tempos, memória, versões).
- `main` (`src/train_tfidf.py:139`): o comando (`make treinar-tfidf`); grava o modelo em `models/tfidf_n10.joblib`.

```python
check_not_test(train_path)
check_not_test(val_path)
...
treino = pd.read_parquet(train_path, columns=["data", "texto", "codigo"])
```
O treino e as contagens do N só leem o treino. A validação só se abre depois de o modelo estar treinado, para lhe dar sugestões.

```python
out = pd.DataFrame({"idcontrato": val["idcontrato"].astype(str).to_numpy(),
                    "sugestoes": [[str(c) for c in s] for s in sugestoes]})
out.to_parquet(out_dir / SUGESTOES, index=False)
```
O ficheiro de sugestões só tem o id e os 5 códigos, sem texto. Fica pequeno (242 KB) para ir para o git e ser lido no portátil, onde o modelo (4,8 GB de pesos) não cabe.

### `src/suggest.py` (novo)

- `format_suggestions` (`src/suggest.py:30`): para cada descrição, os 5 códigos numerados, com a probabilidade que o modelo dá a cada um.
- `main` (`src/suggest.py:46`): o comando (`make sugerir`). Sem descrições, usa 3 exemplos escritos à mão, entre eles "serviços de limpeza de três escolas básicas durante 12 meses".

### Decisões tomadas pelo caminho

- **Treinar no pod e avaliar no portátil:** o treino precisa de cerca de 10 GB e o portátil tem 5,8 GB. O pod grava só as sugestões, e o notebook avalia-as no portátil.
- **A decisão 5 do E4 (classificador `SGDClassifier`) fica confirmada:** bateu a régua.
- **`alpha` (a força da regularização) ficou no valor por defeito, 0,0001:** era o do ensaio do E4-S0. O resultado mostrou que é forte demais. O E4-S4 mediu e mudou-o; vem no PR seguinte.

## Outras mudanças

- `notebooks/04_tfidf.ipynb` (novo, `make tfidf`): lê as sugestões do pod, compara com a régua (todos os contratos, texto novo, texto já visto), corre o bootstrap e dá o veredicto do critério. Nunca lê o teste.
- `metrics/e4_h1/`: o que o pod escreveu (sugestões da validação, `treino.json`, a saída do comando para as 3 descrições, a memória do pod e o `pod_info.json`).
- `docs/qualidade_dados.md`: secção 13 com os resultados; a origem dos nomes CPV; os 4,8 GB passam a ser descritos como os pesos, não o ficheiro.
- `docs/backlog.md`: E4-H1 feito, com os 7 critérios marcados; decisão 5 confirmada.
- `docs/workflow.md`: caixa da baseline TF-IDF marcada.
- `docs/maquina_runpod.md`: o que fazer quando o SSH direto de um pod não responde (aconteceu em EU-SE-1).
- `notebooks/readme.md`: linha do notebook 04.
- `CLAUDE.md`: parágrafo sobre o modelo, o treino e o comando.
- `Makefile`: `make test-tfidf`, `make treinar-tfidf`, `make sugerir` e `make tfidf`.
- `.gitignore`: `models/` (o modelo tem 4,8 GB de pesos).
- `tests/conftest.py` (novo): os ajudantes `run_pipeline` e `contratos_regua` passam do `test_pipeline.py` para aqui, como fixtures do pytest. Antes, o teste do TF-IDF importava-os de outro ficheiro de teste.
- `tests/integration/test_pipeline.py`: usa as fixtures do `conftest.py`; os testes são os mesmos.
- `tests/unit/test_tfidf.py`, `tests/integration/test_tfidf_pipeline.py` e `tests/integration/test_tfidf_dados_reais.py` (novos): ver as provas abaixo.
- `tests/integration/readme.md`: os ficheiros novos e o `conftest.py`.

### Ficou igual de propósito

- **O `prepare.py` e os três conjuntos:** o modelo lê o treino e a validação tal como estão.
- **A régua e a avaliação** (`baselines.py`, `evaluation.py`, `comparison.py`): o modelo é avaliado com as mesmas regras da régua.
- **O N dos códigos raros fica em 10:** escolhê-lo é o E4-S1.
- **O teste só se recusa pelo nome do ficheiro, não pelas datas.** Recusar uma cópia do teste com outro nome é o E4-T5 (carregar o teste só com autorização).

## Validação e provas

### Critérios de aceitação

| Critério (copiado do backlog) | Cumprido? | Prova (comando → resultado) |
|---|---|---|
| **Dado** uma descrição escrita à mão, **quando** corro o comando de sugestões, **então** recebo 5 códigos distintos de 8 dígitos, ordenados do mais provável para o menos provável. | ✅ | `test_sugere_5_codigos_distintos_de_8_digitos_por_ordem_da_pontuacao`, `test_comando_sem_descricoes_mostra_os_3_exemplos_com_5_codigos`, `test_probabilidades_por_ordem_decrescente` → passam; no pod, `metrics/e4_h1/sugestoes_exemplos.txt` |
| Os 5 códigos são todos etiquetas do treino (depois da regra dos códigos raros), e nenhum está ao nível da divisão. | ✅ | `test_as_sugestoes_sao_sempre_etiquetas_de_treino`, `test_codigo_raro_so_aparece_como_a_sua_classe`, `test_nunca_sugere_divisao`, `test_treino_com_codigo_de_divisao_da_erro` → passam |
| Dada a mesma descrição duas vezes, recebo as mesmas 5 sugestões pela mesma ordem. | ✅ | `test_mesmo_texto_duas_vezes_da_as_mesmas_sugestoes`, `test_dois_treinos_dao_o_mesmo_modelo[1,2]`, `test_empate_fica_o_menor_codigo`, `test_mesma_descricao_duas_vezes_da_o_mesmo` → passam |
| Na validação, o top-5 é maior do que o da régua E4-T2 no código completo e na classe. Os resultados completos da avaliação E4-T1 estão escritos no `qualidade_dados.md`. | ✅ | notebook 04, células 7 e 11 → 43,2% contra 26,3% e 37,9% contra 15,6%, bootstrap 100%; secção 13; `test_top5_do_tfidf_na_validacao_e_o_do_doc` → passa com os dados reais |
| Um teste confirma que nenhum contrato da validação entra no treino do modelo nem nas contagens do N. | ✅ | `test_palavra_so_da_validacao_nao_esta_no_vocabulario`, `test_etiquetas_contadas_so_no_treino`, `test_mudar_a_validacao_nao_muda_o_modelo` → passam, e falham se a validação for junta ao treino |
| Nos contratos da validação com texto que não aparece no treino, o top-5 também é maior do que o da régua, no código completo e na classe, medidos nesses mesmos contratos. O `qualidade_dados.md` mostra lado a lado os resultados com texto novo e com texto já visto, para se ver quanto do resultado vem de decorar textos. | ✅ | notebook 04, células 9 e 11 → 38,5% contra 23,7% e 34,2% contra 14,8%, bootstrap 100%; 2.ª tabela da secção 13 (texto já visto: 66,4% no código completo) |
| O comando mostra as sugestões para pelo menos 3 descrições escritas à mão, por exemplo "serviços de limpeza de três escolas básicas durante 12 meses". | ✅ | `make sugerir` no pod → `metrics/e4_h1/sugestoes_exemplos.txt` (3 descrições, 5 códigos cada); `test_comando_sem_descricoes_mostra_os_3_exemplos_com_5_codigos` → passa |

Caixas do `docs/backlog.md` marcadas como `[x]` neste PR: os 7 critérios do E4-H1.

### Testes focados

`make test-tfidf` → 35 passed (18 unitários, 14 de integração com dados inventados e 3 com os dados reais; estes saltam no CI)

### O comportamento que já existia continua igual

- Suíte inteira: `.venv/bin/pytest -q` → 207 passed. Antes do trabalho, o `main` tinha 172 testes (numa cópia sem `data/`: 167 passed e 5 skipped, os testes com dados reais).
- Os 172 testes que já existiam continuam a passar sem mudar o que verificam. O `test_pipeline.py` só passou a receber os ajudantes do `conftest.py`.
- A régua continua igual: `test_regua_dados_reais.py` → passa, com as mesmas 5 etiquetas e o mesmo top-5 da secção 12.
- CI do commit `47def09`: <a preencher com o estado no momento de abrir o PR>

### Os testes novos falham com o código estragado

Cada estrago foi feito à mão, com o teste corrido e o código reposto a seguir. Os estragos M1 a M7 foram repetidos de forma independente pelo revisor, numa cópia.

| Teste | Estrago introduzido | Resultado com o estrago |
|---|---|---|
| `test_palavra_so_da_validacao_nao_esta_no_vocabulario`, `test_etiquetas_contadas_so_no_treino`, `test_mudar_a_validacao_nao_muda_o_modelo` | validação junta ao treino no `run` (fuga) | falharam ✅ |
| `test_etiquetas_contadas_so_no_treino` | contagens do N com os códigos da validação | falhou ✅ |
| `test_empate_fica_o_menor_codigo` | ordenação não estável / empate para o maior código | falhou ✅ |
| `test_caminho_do_teste_da_erro[treino/validacao]` | `check_not_test` sem efeito | falharam ✅ |
| `test_treino_com_codigo_de_divisao_da_erro` | sem a recusa dos códigos `XX000000` | falhou ✅ |
| `test_dois_treinos_dao_o_mesmo_modelo` | sem `random_state=SEED` | falhou ✅ |
| `test_codigo_raro_so_aparece_como_a_sua_classe` | sem a regra dos códigos raros | falhou ✅ |
| `test_uma_linha_por_contrato_da_validacao_pela_mesma_ordem` | sugestões gravadas por outra ordem | falhou ✅ |
| `test_cinco_codigos_distintos_de_8_digitos_do_treino_sem_divisoes` | 4 sugestões em vez de 5; um código de divisão nas sugestões | falhou ✅ |
| `test_probabilidades_por_ordem_decrescente` | ordem aleatória / probabilidade de outro código | falhou ✅ |
| `test_comando_sem_descricoes_mostra_os_3_exemplos_com_5_codigos` | só 2 exemplos | falhou ✅ |
| `test_comando_de_treino_grava_o_modelo_onde_o_comando_de_sugestoes_o_procura` | outro nome de ficheiro no `train_tfidf.py`; outro caminho no `Makefile` | falhou ✅ |
| `test_sugestoes_guardadas_sao_da_validacao_atual` | ficheiro de sugestões com as linhas por ordem inversa | falhou ✅ |
| `test_top5_do_tfidf_na_validacao_e_o_do_doc` | a avaliação da classe a comparar 3 dígitos em vez de 4 | falhou ✅ |
| `test_notebook_do_tfidf_nao_abre_o_teste` | uma célula do notebook 04 a ler `teste.parquet` | falhou ✅ |

### Números que provam que funciona

Treino completo num pod da Runpod (`cpu3g`, 8 processadores, 32 GB, 0,32 $/h), avaliado na validação de outubro de 2025 com o notebook 04. O teste não foi lido.

| Medida | Régua (antes) | Modelo (depois) | Fonte |
|---|---|---|---|
| Top-5 código completo, todos | 26,3% | 43,2% | notebook 04, célula 7 |
| Top-5 classe, todos | 15,6% | 37,9% | célula 7 |
| Top-5 código completo, texto novo | 23,7% | 38,5% | célula 7 |
| Top-5 classe, texto novo | 14,8% | 34,2% | célula 7 |
| F1 macro, código completo | 0,0001 | 0,0107 | célula 7 |
| Top-5 código completo, texto já visto | 39,2% | 66,4% | célula 9 |
| Tempo de treino | — | 454 s | `metrics/e4_h1/treino.json` |
| Pico de memória do processo | — | 9 975 MiB | `treino.json` |
| Custo do pod | — | cerca de 0,09 $ (16 minutos, dois pods) | `metrics/e4_h1/pod_info.json` |

**O ponto fraco, dito às claras:**
- O F1 macro é quase zero, porque o modelo acerta sobretudo nas etiquetas frequentes.
- Nas descrições escritas à mão, os códigos de saúde da régua aparecem à frente do código certo. Exemplo: nas refeições escolares, `55524000` só aparece em 4.º lugar.
- O spike E4-S4 mostrou que a causa é a regularização forte demais. Com `alpha` = 0,0000001 e uma correção para as etiquetas frequentes, o top-5 no código completo sobe para 76,9%. Isso vem no PR seguinte.

### Revisão de código

`/requesting-code-review` sobre `8f02665..47def09` (agente revisor novo, sem contexto da tarefa): **pronto para merge com correções**. Não encontrou nada crítico; confirmou os 7 critérios e repetiu 7 estragos (todos apanhados).
- **Corrigido** (commit seguinte ao `47def09`):
  - escrita esta descrição em `docs/prs/pr_E4-H1.md` (o ponto importante);
  - os 4,8 GB passam a ser descritos como os pesos, não o ficheiro;
  - indicada a origem dos nomes CPV;
  - o `make test-tfidf` passa a incluir o teste com dados reais.
- **Não corrigido, e porquê:**
  - **Recusar o teste pelas datas e não só pelo nome:** é o âmbito do E4-T5.
  - **O teste com dados reais só guarda os 4 top-5 da secção 13:** são os números do critério; o resto da tabela vem do notebook, que se volta a correr.
  - **`alpha` forte demais:** foi tratado no E4-S4 (PR seguinte).
  - **A linha "Feito" do backlog ainda não tem o número do PR:** acrescenta-se depois de criar o PR, como no E4-S0.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
