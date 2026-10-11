## E4-T2 · Régua "códigos mais comuns" medida na validação

## Em poucas palavras

Este PR cria a régua mais simples possível: sugerir sempre os mesmos 5 códigos CPV, os mais frequentes no treino, sem ler a descrição do contrato. Mede quanto essa régua acerta nos contratos de outubro de 2025 (a validação). Esses números são o mínimo que o primeiro modelo a sério (TF-IDF, história E4-H1) tem de bater. Sem eles, "o modelo acerta 30%" não dizia se o modelo é bom. Com eles sabe-se que, sem ler nada, já se acerta o código exato em 1 de cada 4 contratos, porque as compras de saúde dominam.

| Na validação (outubro de 2025) | O código certo está nas 5 sugestões | A classe certa (4 dígitos) está nas 5 sugestões |
|---|---|---|
| Todos os contratos | 26,3% | 15,6% |
| Só contratos com texto que não aparece no treino | 23,7% | 14,8% |

## Porquê

O TF-IDF, o BERTimbau e o Albertina vão ser comparados entre si, mas primeiro têm de provar que aprenderam alguma coisa com o texto. A forma de o provar é bater uma régua que não lê texto nenhum. Se o TF-IDF não a bater, há um erro nos dados, no treino ou na avaliação, e o épico E4 para aí (aviso do backlog).

Item: [`E4-T2`](docs/backlog.md) · Desbloqueia: E4-H1 (precisa também do E4-S0) e o portão de qualidade E4-T5

## O que mudou no código

### `src/baselines.py` (novo)

- `most_common_labels` (`src/baselines.py:19`): devolve as k etiquetas mais frequentes do treino (por defeito 5), da mais para a menos frequente.
- `suggest_most_common` (`src/baselines.py:41`): dá essas mesmas 5 sugestões a cada contrato, no formato que a avaliação comum (E4-T1) lê.

**A régua conta as etiquetas depois da regra dos códigos raros, com o mesmo N que o modelo.** A regra dos códigos raros (decisão 9): um código com menos de N exemplos no treino sobe para a classe; se a classe também for rara, sobe para o grupo; se o grupo também for raro, o exemplo sai. Assim a régua e o modelo escolhem entre as mesmas etiquetas.

```python
label = collapse_rare_codes(codigo, n_min=n_min)
keep = label.notna()
counts = weight[keep].groupby(label[keep]).sum().rename_axis("etiqueta").reset_index(name="n")
counts = counts.sort_values(["n", "etiqueta"], ascending=[False, True])
```
Aplica a regra dos raros, tira os exemplos que saíram, conta cada etiqueta e ordena da mais frequente para a menos; num empate fica primeiro o menor código, para o resultado não depender da ordem das linhas.

**Conta-se um por exemplo de treino, não por contrato.** O treino tem um exemplo por texto (decisão 6): se 1 870 contratos dizem "medicamentos", fica 1 exemplo com `n_contratos = 1870`. Por defeito cada exemplo vale 1. Passar o `n_contratos` faz contar contratos; isso só se usa para comparar no notebook (ver "Decisões" abaixo).

```python
weight = pd.Series(1, index=codigo.index) if n_contratos is None else n_contratos
if not weight.index.equals(codigo.index):
    raise ValueError("`n_contratos` e `codigo` têm índices diferentes")
if weight.isna().any() or (weight < 1).any():
    raise ValueError("`n_contratos` tem de ser >= 1 em todos os exemplos")
```
Sem pesos, cada exemplo vale 1; com pesos, recusa índices desalinhados e valores em falta, zero ou negativos.

Se depois da regra dos raros sobrarem menos de k etiquetas, dá erro em vez de devolver menos de 5 sugestões (`src/baselines.py:36-37`).

### Decisões tomadas pelo caminho

- **Contar exemplos, não contratos (decisão 10 do `qualidade_dados.md`).** Primeiro escolheu-se contar contratos, porque a validação tem todos os contratos com as repetições. Medido, essa régua acerta menos no código completo (26,0% contra 26,3%; 23,3% contra 23,7% no texto novo), e o bootstrap mostra que não é acaso. Ficou a que conta exemplos, a mais difícil de bater. Escolhê-la depois de ver a validação não favorece o modelo: só lhe torna a tarefa mais difícil.
- **Bootstrap**, usado acima: tira-se 1 000 vezes, à sorte e com repetição, uma amostra dos contratos de validação, a mesma para as duas réguas, e vê-se quantas vezes cada uma ganha. Aqui a régua por exemplos ganhou em 100% das amostras.
- **Empate pelo menor código**, para o resultado ser sempre o mesmo.

## Outras mudanças

- `notebooks/03_regua.ipynb` (novo, `make regua`): calcula a régua e os seus números na validação; três verificações que explicam a tabela; compara contar exemplos com contar contratos (com bootstrap); mostra a régua com N = 5, 10 e 20. Lê só o treino e a validação.
- `docs/qualidade_dados.md`: secção 12 com a régua e os resultados; decisão 10; o âmbito no topo diz de onde vem a secção 12.
- `docs/backlog.md`: E4-T2 marcada como feita, com os números a bater no E4-H1; estado do E4 na tabela.
- `docs/workflow.md`: caixa da régua marcada na fase 4.
- `CLAUDE.md`: parágrafo sobre `src/baselines.py`.
- `Makefile`: `make regua` (volta a correr o notebook); `make test-avaliacao` inclui os testes da régua.
- `src/readme.md`, `notebooks/readme.md`, `tests/unit/readme.md`, `tests/integration/readme.md`: uma linha para cada ficheiro novo.
- `tests/unit/test_baselines.py` (novo): 16 testes unitários.
- `tests/integration/test_pipeline.py`: 8 testes de integração novos (ver provas); `run_pipeline` passa a aceitar outra lista de contratos.
- `tests/integration/test_regua_dados_reais.py` (novo): os números da secção 12 contra os dados reais, e o notebook nunca abre o teste.

### Ficou igual de propósito

- `src/prepare.py` e `src/labels.py` — a régua só lê o que eles escrevem; não muda nenhuma regra dos dados.
- `src/evaluation.py` e `src/comparison.py` — a régua é avaliada com a avaliação comum do E4-T1, sem a mudar.
- Os contratos inventados que já existiam em `test_pipeline.py` — o cenário novo usa uma lista à parte, para não mexer nos testes antigos.
- O N continua 10 — escolhê-lo é o spike E4-S1. A régua dá as mesmas 5 etiquetas com N = 5, 10 e 20, por isso não muda quando o N for escolhido.

## Validação e provas

### Critérios de aceitação

| Critério (copiado do backlog) | Cumprido? | Prova (comando → resultado) |
|---|---|---|
| A régua sugere sempre as 5 etiquetas mais comuns do treino, contadas depois da regra dos códigos raros e com o mesmo N que o modelo usa. Assim a régua e o modelo escolhem entre as mesmas etiquetas. | ✅ | `.venv/bin/pytest -q tests/integration -k regua` → passam; `test_regua_com_o_n_do_modelo_sobe_os_raros_e_ignora_divisoes` (12 códigos raros sobem juntos para `33140000` com N = 10 por defeito) e `test_n_por_defeito_e_o_do_modelo` |
| Os seus números na validação, calculados com a avaliação E4-T1, estão escritos no `qualidade_dados.md`. | ✅ | secção 12, saída das células do `notebooks/03_regua.ipynb` (`ev.report`); `test_top5_da_regua_na_validacao_e_o_do_doc` → passa com os dados reais |

Caixas do `docs/backlog.md` marcadas como `[x]` neste PR: as duas do E4-T2.

### Testes focados

`.venv/bin/pytest -q tests/unit/test_baselines.py tests/integration -k "regua or baselines or test_baselines"` → 30 passed, 15 deselected

### O comportamento que já existia continua igual

- Suíte inteira: `.venv/bin/pytest -q` → 172 passed (no `main`: 142 passed)
- Os 15 testes de integração antigos passam sem mudança nas suas asserções; o funil da preparação não muda (o `prepare.py` não foi tocado).
- Sem `data/` (como no CI), os 5 testes com dados reais saltam: `pytest -rs tests/integration/test_regua_dados_reais.py` numa cópia sem dados → 1 passed, 5 skipped.
- CI: `de4cb54` e `93a506f` → success (`gh run list`); o commit final deste PR é verificado antes do merge.

### Os testes novos falham com o código estragado

| Teste | Estrago introduzido | Resultado com o estrago |
|---|---|---|
| `test_regua_com_o_n_do_modelo_sobre_o_treino_da_preparacao` e outros | N por defeito trocado para 1 | falhou ✅ (4 testes) |
| `test_regua_com_o_n_do_modelo_sobre_o_treino_da_preparacao` e outros | N por defeito trocado para 5 | falhou ✅ (3 testes, sem dados reais) |
| idem | N por defeito trocado para 20 | falhou ✅ (5 testes, sem dados reais) |
| `test_regua_com_o_n_do_modelo_sobe_os_raros_e_ignora_divisoes` e outros | regra dos raros desligada (`label = codigo`) | falhou ✅ (4 testes) |
| `test_regua_conta_exemplos_e_nao_contratos_no_pipeline` e outros | pesar sempre pelo `n_contratos` | falhou ✅ (8 testes) |
| `test_regua_com_o_n_do_modelo_avaliada_na_validacao` e outros | empate pelo maior código | falhou ✅ (3 testes) |
| `test_regua_com_o_n_do_modelo_sem_etiquetas_suficientes_da_erro` | exemplos que saem continuam a contar | falhou ✅ |
| `test_regua_com_o_n_do_modelo_sem_etiquetas_suficientes_da_erro` | tirar o erro de "menos de k etiquetas" | falhou ✅ |
| `test_regua_avaliada_sobre_a_validacao_da_preparacao` | sugestões pela ordem inversa | falhou ✅ |
| `test_regua_nunca_sugere_divisoes` | uma etiqueta sobe até à divisão (`90000000`) | falhou ✅ |
| `test_regua_real_e_a_do_doc_com_qualquer_n` e outros | 5.ª etiqueta do doc trocada (doc desatualizado) | falhou ✅ (4 testes) |
| `test_notebook_da_regua_nao_abre_o_teste` | notebook passa a ler `teste.parquet` | falhou ✅ |
| `test_notebook_da_regua_nao_abre_o_teste` | notebook lê um ficheiro preparado por variável (`DATA / f"{nome}.parquet"`) | falhou ✅ |
| `test_notebook_da_regua_nao_abre_o_teste` | `load_contracts` sem `until=cd.VAL_START` | falhou ✅ |
| `test_n_por_defeito_e_o_do_modelo` (unitário) | N por defeito trocado para 5 | falhou ✅ |
| `test_n_contratos_invalido_da_erro` (3 casos) | verificação do `n_contratos` desligada | falhou ✅ (3 testes) |

### Números que provam que funciona

Validação (outubro de 2025, 23 945 contratos), `notebooks/03_regua.ipynb`. Não há "antes": é a primeira medição de um modelo no projeto.

| Medida | Todos os contratos | Texto novo | Fonte |
|---|---|---|---|
| Top-5 código completo | 26,3% (19 906 contratos) | 23,7% (16 565) | célula `ev.report` |
| Top-5 classe | 15,6% (14 047) | 14,8% (11 959) | célula `ev.report` |
| Top-1 código completo | 13,7% | 11,4% | célula `ev.report` |
| Top-1 classe | 0,0% | 0,0% | célula `ev.report` |
| F1 macro código completo | 0,0001 | 0,0001 | célula `ev.report` |

- **As 5 sugestões:** `33600000` Produtos farmacêuticos, `33140000` Material médico de consumo, `33690000` Medicamentos vários, `33100000` Equipamento médico, `98300000` Serviços diversos. Juntas são 17,6% dos exemplos de treino.
- **Top-1 na classe é 0%** porque a 1.ª sugestão é um código de grupo (`33600000`), e nenhum contrato da validação tem um código da classe para baixo que comece por `3360` (0 contratos, contados no notebook). Na classe só acertam `33140000` (1 753 contratos) e `33690000` (438).
- **F1 macro quase zero:** é a média do F1 sobre todas as etiquetas da validação, e a régua só acerta em 5. É o esperado.
- **Texto novo abaixo de todos**, embora a régua não leia o texto: a etiqueta é uma das 5 em 35,4% dos contratos com texto já visto e só em 19,4% dos de texto novo.

### Revisão de código

`/requesting-code-review` sobre `main..regua/e4-t2`, três revisores novos (só com o diff e o backlog) e antes deles um agente juiz pedido pelo utilizador. Último veredicto: **pronto para merge com uma pequena correção** (commitar esta descrição), sem problemas críticos.
- Corrigido:
  - contar exemplos que saem, desempate e lista partilhada sem teste → testes novos (1.º juiz);
  - frases interpretativas no `qualidade_dados.md` sem célula que as medisse → células novas no notebook e frases reescritas;
  - diferença entre as duas formas de contar sem bootstrap → bootstrap no notebook, decisão 10;
  - N por defeito não guardado (N = 1, depois N = 5 passavam) → cenário de integração e teste unitário que distinguem N = 5 de N = 10;
  - nomes dos códigos lidos de um ficheiro que pode ter contratos do período de teste → só publicados antes da validação;
  - guarda do notebook só procurava texto → passa a exigir que cada leitura seja do treino ou da validação e que cada `load_contracts` pare antes da validação;
  - "0,0%" arredondado na classe → a célula mostra a contagem exata (0 contratos).
- Não corrigido, e porquê:
  - o filtro `keep` e o desempate pelo menor código ficam garantidos duas vezes pelo pandas (estragos equivalentes, o resultado não muda) — ficam explícitos de propósito;
  - o número deste PR na linha "Feito" do backlog — só existe depois de o PR ser criado; acrescenta-se num commit a seguir.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
