## E5-T7 · Perda na validação em cada época

## Em poucas palavras

O treino do BERT passa a medir, no fim de cada época, a perda na validação. A perda na validação é a mesma conta da perda no treino, feita nos contratos de outubro de 2025. Para cada contrato, a conta é −log da probabilidade que o modelo dá à etiqueta certa: se der 0,6, o contrato custa −log(0,6) ≈ 0,51, e a perda é a média dessas contas. Com as duas curvas lado a lado no MLflow vê-se o sobreajuste, quando a perda no treino continua a descer e a da validação pára ou sobe. É só informação: quem escolhe o modelo continua a ser o top-5 no código completo.

Que contratos da validação entram na média, num exemplo inventado em que as etiquetas do treino são só 90911200, 33140000 e 45200000:

| Primeiro código do contrato | Etiqueta usada na perda | Porquê |
|---|---|---|
| 90911200 | 90911200 | o código é etiqueta |
| 33141100 | 33140000 | o código não é etiqueta, a classe é |
| 45233000 | 45200000 | nem o código nem a classe 45230000 são etiquetas, o grupo é |
| 34110000 | fica de fora | nem o código, nem a classe, nem o grupo são etiquetas |
| 45000000 (divisão) | fica de fora | o treino não tem divisões |

## Porquê

Até aqui o treino media, em cada época, a perda no treino e o top-1 e o top-5 na validação, mas não a perda na validação, porque o `Trainer` corre com `eval_strategy="no"`. Sem ela, os treinos da grelha do E5-S1 não mostravam se o modelo começa a decorar o treino. O utilizador pediu esta medida a 7 de novembro de 2026, com o treino do E5-T5 já a arrancar. Por isso entra só nos treinos seguintes, e o código que treinou o E5-T5 fica o do seu commit.

Item: [`E5-T7`](../backlog.md) · Desbloqueia: E5-S1 (os treinos da grelha já gravam a perda)

## O que mudou no código

### `src/bert.py`

- `validation_labels` (`src/bert.py:393`): dá a etiqueta de cada contrato da validação entre as etiquetas do treino. Fica o código, senão a classe, senão o grupo; se nenhum for etiqueta, o contrato fica com NaN.

```python
for nivel in (texto, to_level(texto, 4), to_level(texto, 3)):  # código -> classe -> grupo
    falta = etiqueta.loc[texto.index].isna() & nivel.isin(no_treino)
    etiqueta.loc[falta[falta].index] = nivel[falta]
```
Tenta os três níveis por ordem e fica com o primeiro que é etiqueta do treino. É a regra dos códigos raros do treino: um código que não é etiqueta ou teve menos de N exemplos no treino ou não apareceu lá, e nos dois casos o treino subia-o para a classe e depois para o grupo. Na validação não se conta nada, só se procura nas etiquetas do treino, por isso não há fuga de dados.

- `cross_entropy` (`src/bert.py:421`): devolve a entropia cruzada média e quantos contratos entraram, `(perda, n)`. Se não entrar nenhum, devolve `(None, 0)` sem dar erro.

```python
x = pontuacoes[entram].astype(np.float64)
maximo = x.max(axis=1, keepdims=True)
log_soma = np.log(np.exp(x - maximo).sum(axis=1)) + maximo[:, 0]
perdas = log_soma - x[np.arange(len(x)), colunas]  # −log softmax na etiqueta certa
```
Calcula o −log da probabilidade softmax da etiqueta certa, em float64. Tira primeiro o maior *logit* de cada linha, para o `exp` não dar infinito. Um *logit* é a pontuação crua de cada etiqueta, antes de virar probabilidade.

### `src/train_bert.py`

- `run` (`src/train_bert.py:301` e `:314`) lê também a coluna `codigo` da validação e calcula as etiquetas uma vez, antes do treino. Em cada época calcula a perda a partir das pontuações que já calculava para as sugestões, sem segunda passagem pela validação.

```python
perda, n_perda = bert.cross_entropy(pontuacoes, etiquetas_val, classes)
info = {"epoca": k, "checkpoint": checkpoint, "pontuacoes": scores_file(k),
        "segundos_desde_o_inicio": round(time.perf_counter() - t0, 2),
        "perda_validacao": perda, "n_contratos_perda": n_perda}
```
Os dois campos novos vão para o `epoca.json` de cada época e, por isso, para o `por_epoca` do `treino.json`. Num treino retomado, as épocas antigas trazem-nos do seu `epoca.json`.

### `src/registo_bert.py`

- `epoch_metrics` (`src/registo_bert.py:119`) regista no MLflow a `perda_validacao` e o `n_contratos_perda` de cada época, com o passo igual à época. Um treino sem estes campos, como os do E5-T4 e do E5-T5, regista-se na mesma, sem as duas métricas.
- A docstring do módulo diz que a perda é só informação, e que a perda e o top-5 podem discordar. Diz também porque as duas perdas não se comparam ao valor: a do treino é a média dos passos da época, com o *dropout* ligado; a da validação mede-se no fim da época, sem *dropout*. O *dropout* desliga à sorte parte dos neurónios em cada passo do treino.

## Outras mudanças

- `tests/unit/test_bert.py`: testes da tradução das etiquetas (fica o código, sobe para a classe, sobe para o grupo, sai, divisão, sem código; prefere o nível mais fino) e da conta da perda (conta à mão, *logits* enormes, caso vazio, formas erradas).
- `tests/integration/test_bert_pipeline.py`: a perda de cada época confere com a do `torch` a partir do `.npy`; o contrato com etiqueta fora do treino fica de fora (`n_contratos_perda` = 3 de 5); uma validação sem contratos com etiqueta grava `None`; o teste de retoma confirma também os dois campos novos.
- `tests/unit/test_registo_bert.py`: a curva `perda_validacao` (e `n_contratos_perda`) tem uma entrada por época, com o passo igual à época; um `treino.json` antigo regista-se sem ela; com `perda_validacao: null` não se cria a métrica; o conjunto exato de nomes de métricas inclui as duas novas.
- `docs/backlog.md`: E5-T7 marcado como feito, critérios marcados, resultado escrito; o estado do E5 passa a "a seguir o E5-S1".
- `docs/workflow.md`: a linha do E5 diz que a perda na validação já se grava e que o próximo passo é o E5-S1.
- `docs/qualidade_dados.md` (secção 23): a linha "Perda na validação" do E5-T5 já não diz que o treino não a calcula; diz que não foi medida nesse treino, que desde o E5-T7 se grava, e que as pontuações no portátil chegam para a calcular sem GPU.
- `CLAUDE.md`: o estado do E5 e uma linha sobre o E5-T7.
- `docs/prs/pr_E5-T7.md`: esta descrição.

### Ficou igual de propósito

- **O treino em si:** a perda calcula-se a partir das pontuações que já existiam e não muda os pesos, as sugestões nem os *checkpoints*.
- **O ensaio de custo** (`--max-passos`): não pontua a validação, por isso também não calcula a perda.
- **A perda no treino:** continua a vir do registo do `Trainer`.
- **O E5-T4 e o E5-T5:** ficam sem esta curva no MLflow. As pontuações do E5-T4 perderam-se com o pod, e os treinos do E5-S1 repetem as duas configurações.
- **Quem decide:** continua a ser o top-5 no código completo (E5-S1 e E5-T6). A perda não entra em nenhuma regra.

## Validação e provas

### Critérios de aceitação

| Critério (copiado do backlog) | Cumprido? | Prova (comando → resultado) |
|---|---|---|
| No fim de cada época, o treino calcula a perda na validação a partir dos *logits* que já calcula para as pontuações (sem uma segunda passagem pela validação), e grava `perda_validacao` e `n_contratos_perda` no `epoca.json` dessa época. | ✅ | `test_perda_na_validacao_de_cada_epoca_confere_com_a_conta_a_mao` → passa; `src/train_bert.py:314` usa as `pontuacoes` do `on_epoch` |
| Só entram os contratos da validação cuja etiqueta existe no treino, depois da mesma regra dos códigos raros do treino (`collapse_rare_codes` com o N do treino, contado no treino); os outros ficam de fora sem dar erro, e o `n_contratos_perda` diz quantos entraram. | ✅ | `test_etiqueta_da_validacao_sobe_como_no_treino` (6 casos) e `test_contrato_com_etiqueta_fora_do_treino_fica_de_fora_da_perda` → passam; `n_contratos_perda` = 3 de 5 |
| O registo no MLflow (`registo_bert.py`) grava a `perda_validacao` de cada época com o passo = a época, ao lado da `perda_treino`. | ✅ | `test_curva_tem_uma_entrada_por_epoca_com_o_passo_da_epoca` → passa |
| Um teste com o modelo minúsculo confirma o valor contra uma conta feita à mão, e outro confirma que um contrato com uma etiqueta fora do treino fica de fora. Cada teste falha quando se estraga de propósito o que ele guarda. | ✅ | 14 estragos, todos apanhados (tabela abaixo) |
| Fica escrito que é só informação: quem decide continua a ser o top-5 no código completo (E5-S1 e E5-T6). A perda e o top-5 podem discordar (a perda pode subir com o top-5 ainda a melhorar, quando o modelo fica confiante demais nos erros). | ✅ | docstring de `src/registo_bert.py`, e o resultado do E5-T7 em `docs/backlog.md` |

Caixas do `docs/backlog.md` marcadas como `[x]` neste PR: as 5 do E5-T7.

### Testes focados

`.venv/bin/pytest tests/unit/test_bert.py tests/integration/test_bert_pipeline.py tests/integration/test_suggest_bert.py tests/integration/test_experiencias_bert.py` → 147 passed. `.venv/bin/pytest tests/unit/test_registo_bert.py` → 22 passed.

### O comportamento que já existia continua igual

- Suíte inteira: `.venv/bin/pytest -q` → 734 passed, 1 skipped (o teste do `$VALIDACAO_MAIN`, que só corre no CI). Antes do trabalho, a suíte não foi corrida nesta sessão.
- Os testes que já existiam do pipeline BERT (sugestões, *checkpoints*, retoma, mesma semente dá as mesmas pontuações) continuam a passar sem mudanças, além das verificações novas acrescentadas ao teste de retoma.
- CI do commit de cima: a correr quando esta descrição foi escrita; o estado final fica no PR.

### Os testes novos falham com o código estragado

| Teste | Estrago introduzido | Resultado com o estrago |
|---|---|---|
| unitário da perda, integração (conta à mão, fica de fora) | a perda usa a coluna errada | falhou ✅ |
| unitário da perda, integração, retoma | os contratos de fora contam na média | falhou ✅ |
| unitários das etiquetas, integração, retoma | sem subir para a classe | falhou ✅ |
| `test_etiqueta_da_validacao_sobe_como_no_treino` (caso do grupo) | sem subir para o grupo | falhou ✅ |
| `test_perda_com_logits_enormes_nao_da_infinito` | sem tirar o máximo | falhou ✅ |
| os dois testes do caso vazio | sem o `(None, 0)` | falhou ✅ |
| integração, retoma | `train_bert` sem a regra dos códigos raros | falhou ✅ |
| integração, retoma | perda acrescentada depois de gravar o `epoca.json` | falhou ✅ |
| integração | pontuações de outras linhas | falhou ✅ |
| conjunto exato de métricas, curva | o registo não grava `perda_validacao` | falhou ✅ |
| conjunto exato, curva, teste do None | o registo não grava `n_contratos_perda` | falhou ✅ |
| treino antigo, teste do None | sem a guarda: None ou campo em falta passa a 0 | falhou ✅ |
| curva | passo = época − 1 | falhou ✅ |
| treino antigo | sem a guarda de `n_contratos_perda` (KeyError) | falhou ✅ |

O estrago "testes do caso vazio" conta os dois testes que o apanharam. Só o caso do grupo é apanhado por um único teste unitário, porque a validação do pipeline minúsculo não tem nenhum contrato que suba para o grupo.

### Números que provam que funciona

Verificação com dados reais, feita nesta sessão com um script fora do repositório e ainda não num notebook, por isso estes números não estão nos `docs/`. As funções novas foram aplicadas às pontuações das 3 épocas do Albertina (E5-T5), guardadas no portátil. O SHA-256 do `validacao.parquet` do portátil é o do `treino.json` do pod. Dos 23 945 contratos da validação entram 19 889 (83%).

| Época | Perda no treino (do `treino.json`) | Perda na validação | Contratos na média |
|---|---|---|---|
| 1 | 3,96 | 2,43 | 19 889 |
| 2 | 2,50 | 2,12 | 19 889 |
| 3 | 2,17 | 2,03 | 19 889 |

As duas perdas descem até à 3.ª época, por isso não há sinal de sobreajuste em 3 épocas. A da validação fica abaixo da do treino pelas razões escritas no `registo_bert.py`: média da época contra fim da época, e *dropout* ligado contra desligado.

### Revisão de código

`/requesting-code-review` sobre `f471d5b..0d33744`, com um revisor independente: **pronto com correções**. Os 5 critérios estão cumpridos e os testes que o revisor estragou falharam, exceto o caso indicado abaixo.
- Corrigido (importante): `docs/qualidade_dados.md`, secção 23, dizia que o treino não calculava a perda (tarefa E5-T7) e contradizia o backlog.
- Corrigido (menor): a docstring do `registo_bert.py` explica porque as duas perdas não se comparam ao valor.
- Não corrigido, e porquê:
  - Memória: a conta em float64 cria duas matrizes de ~0,6 GB por época no pod. Um pod de GPU tem muito mais memória; não foi medido.
  - Alinhamento pelo índice em `validation_labels`: falharia com um índice repetido, mas o `validacao.parquet` tem índice 0, 1, 2, …
  - Um estrago do revisor (a perda usar o *logit* máximo) não fez falhar o teste "fica de fora", porque o modelo minúsculo já acerta as 3 etiquetas. É apanhado pelos dois testes da conta à mão.
- A correção depois da revisão mexeu só em texto (docs e uma docstring), por isso não se lançou um segundo revisor.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
