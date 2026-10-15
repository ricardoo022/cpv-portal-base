## E5-T3 · Registar cada treino do BERT no MLflow

## Em poucas palavras

Cada treino do BERT passa a ficar registado no MLflow, o caderno de registo dos treinos, com uma página no browser que os mostra lado a lado. O treino corre no pod; no fim copia-se a pasta pequena de resultados para o portátil, e um comando regista-a. O registo guarda os hiperparâmetros, o top-5 de cada época (uma curva), a versão do código, a configuração da experiência e uma impressão digital dos dados. Antes de registar, confirma que os dados do portátil são exatamente os que o pod usou. Na GPU, um treino sem a configuração da experiência ou sem a versão do código recusa-se a arrancar, para não se pagar um treino que depois não se consegue identificar.

| Situação | Resultado |
|---|---|
| Pasta copiada do pod, dados iguais aos do pod | um run novo na experiência `treinos_bert` |
| A mesma pasta registada outra vez | não regista (não duplica) |
| `validacao.parquet` do portátil com um só byte diferente | recusa: "não é a validação que o pod usou" |
| Treino na GPU sem `--experiencia` | para no primeiro segundo, antes de ler os dados |

## Porquê

O E5-S1 tem 18 treinos, cada um com uma curva por época. Comparar isso à mão, em pastas e notebooks, como no E4, é lento e dá erros. Com o MLflow, cada treino fica com os parâmetros e as métricas, e a página mostra-os numa tabela, com a curva de cada um. Para os números serem de confiança, o registo também tem de provar com que código e com que dados cada treino foi feito.

Item: [`E5-T3`](docs/backlog.md) · Desbloqueia: E5-T4, E5-T5 e E5-S1

## O que mudou no código

Dois termos usados abaixo:
- **SHA-256 (impressão digital):** um código de 64 carateres calculado a partir do conteúdo de um ficheiro. Um só byte diferente dá outro código. Ex.: se o `validacao.parquet` do portátil e o do pod dão o mesmo SHA-256, são o mesmo ficheiro.
- **Commit do git:** o identificador da versão do código (ex. `d836321…`). Com ele, `git checkout d836321` devolve o código exato de um treino.

**Decisão do utilizador (15 de outubro de 2026), que muda a decisão do backlog:** o `mlflow.db` do pod **não** se copia para o portátil, porque copiado por cima apagava os registos do portão e do histórico. Copia-se só a pasta `--saida` (pequena, como no TF-IDF) e o registo refaz-se no portátil a partir dela.

### `src/train_bert.py` (o treino, no pod)

- `read_git` (`src/train_bert.py:85`): lê o commit (`git rev-parse HEAD`) e se há alterações por gravar (`git status --porcelain --untracked-files=no`). Sem git devolve `None`, em vez de parar.

```python
commit = run(["git", "rev-parse", "HEAD"], cwd=raiz, capture_output=True, text=True)
estado = run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=raiz, capture_output=True,
             text=True)
...
return {"commit": commit.stdout.strip(), "alteracoes_por_gravar": estado.stdout.strip() != ""}
```
O `--untracked-files=no` ignora os ficheiros que o git não segue. Sem ele, a pasta de resultados que aparece a meio do treino contava como "alteração", e uma retoma depois de uma queda era sempre recusada.

- `read_experiment` (`src/train_bert.py:112`): lê o YAML dado com `--experiencia` (caminho, SHA-256 e texto). Recusa um ficheiro que não existe, que não é um dicionário ou que tem mais de 8000 carateres (o limite de uma etiqueta do MLflow), antes de treinar.
- `run` (`src/train_bert.py:158`): na GPU recusa arrancar sem YAML ou sem commit. Depois grava `sha256`, `git` e `experiencia` no `treino.json`.

```python
if dispositivo == "cuda" and experiencia is None:
    raise ValueError("na GPU o YAML da experiência é obrigatório (--experiencia experiencias/<item>.yaml), ...")
...
git = git_info()  # no início: o código que treina é o de agora, mesmo que mude durante o treino
if git["commit"] is None:
    if dispositivo == "cuda":
        raise ValueError("não se leu o commit do git (...); na GPU o commit é obrigatório")
```
No processador (nos testes) a falta do commit só dá um aviso. O commit lê-se no arranque, não no fim, para ser o do código que de facto treinou.

- A retoma (`--retomar`) passa a recusar outro commit, outras alterações por gravar, outro YAML ou outros dados (`src/train_bert.py:211`):

```python
"experiencia_sha256": experiencia["sha256"] if experiencia else None,
"treino_sha256": sha256["treino"], "validacao_sha256": sha256["validacao"],
"git_commit": git["commit"], "git_alteracoes_por_gravar": git["alteracoes_por_gravar"]}
```
Estas chaves entram no `argumentos.json`, que a retoma compara. Sem isto, um `rsync` de código novo a meio deixava um treino feito com duas versões, e o registo dizia que era todo da última.

### `src/registo_bert.py` (novo: o registo, no portátil)

- `log_training` (`src/registo_bert.py:173`): regista a pasta como um run. Avalia tudo antes de abrir o run, por isso um erro não deixa um run a meio.

```python
for k, metricas in sorted(por_epoca.items()):
    mlflow.log_metrics(metricas, step=k)  # no MLflow o passo é a época: dá a curva
```
Cada época é um ponto da curva na página.

- `epoch_metrics` (`src/registo_bert.py:78`): avalia as sugestões de cada `epoca_<k>/` com o `evaluation.report` do E4. Dá o top-1 e o top-5 no código e na classe, para todos os contratos e só para os textos novos, mais a perda do treino.
- `_check` (`src/registo_bert.py:147`): as verificações do portão (nada do teste, datas, janela). Confirma também que o SHA-256 do `validacao.parquet` e do `treino.parquet` do portátil é o que o pod gravou.

```python
if file_sha256(val_path) != sha:
    raise RegistoError(f"{val_path} não é a validação que o pod usou (o SHA-256 é outro): use o "
                       "validacao.parquet que foi copiado para o pod")
```

- `_already_logged` (`src/registo_bert.py:137`): não regista se já houver um run acabado com o mesmo `treino.json`. Um run que rebentou a meio não conta.
- `training_params` e `training_cost`: os hiperparâmetros, o modelo e a sua revisão, os SHA-256, o commit e o YAML como parâmetros (um valor que falta fica "nenhum"), mais o tempo e a memória do treino.

### `src/cpv_data.py`

- `file_sha256` (`src/cpv_data.py:95`) e `MAX_YAML` (`src/cpv_data.py:92`): estão aqui e não no `train_bert.py`, para o registo no portátil não carregar o `torch`.

## Outras mudanças

- `Makefile`: alvo `registar-bert` (`make registar-bert PASTA=metrics/bert/...`); o `test-bert` inclui os testes novos.
- `tests/conftest.py`: uma fixture substitui o git verdadeiro em todos os testes (commit `000…`), para os testes não dependerem do estado do repositório.
- `tests/unit/test_registo_bert.py` (novo, 18 testes): parâmetros, métricas, curva, duplicados e cada recusa, sobre uma pasta feita à mão.
- `tests/unit/test_train_bert_git.py` (novo, 5 testes): o `read_git` com um git fingido e o `file_sha256` igual ao do `hashlib`.
- `tests/integration/test_bert_pipeline.py`: o treino minúsculo grava `sha256`, `git` e `experiencia`; é copiado para outra pasta e registado no MLflow, de ponta a ponta; a retoma com outro código ou outros dados é recusada; na GPU sem YAML ou sem commit não treina.
- `tests/integration/test_makefile_bert.py`: o `registar-bert` passa a pasta e os outros argumentos.
- `docs/maquina_runpod.md` (secção 7): `--experiencia` no comando de treino; confirmar `git rev-parse HEAD` no pod (e o `safe.directory` se aparecer "dubious ownership"); gravar o código num commit antes de treinar; o passo "Registar no MLflow".
- `docs/backlog.md`: E5-T3 marcado feito, com o resultado e as duas decisões do utilizador; novo critério no E5-T4 (ver o primeiro treino real na página).
- `docs/workflow.md`, `CLAUDE.md`, `src/readme.md`, `tests/unit/readme.md`, `tests/integration/readme.md`: o estado do projeto e os ficheiros novos.

### Ficou igual de propósito

- O `mlflow.db` do pod não se copia — apagava os registos do portão e do histórico no portátil (decisão do utilizador).
- Os modelos treinados não vão para o MLflow — são grandes (decisão do backlog); ficam no pod.
- O `--experiencia` não muda nenhum hiperparâmetro — só se grava. Os hiperparâmetros continuam a vir das opções do comando; o E5-T4 decide como o YAML os define.
- O portão (`model_gate.py`) e o `metrics/` não mudaram — o MLflow é um caderno de registo, não substitui os ficheiros pequenos que os docs e o CI leem.
- Com alterações por gravar no arranque e na retoma, o treino não consegue ver se elas mudaram entre as duas — o guia pede para gravar um commit antes de treinar.

## Validação e provas

### Critérios de aceitação

| Critério (copiado do backlog) | Cumprido? | Prova (comando → resultado) |
|---|---|---|
| Cada treino do BERT fica registado com: os hiperparâmetros, o top-5 no código completo e na classe na validação no fim de cada época, o YAML da experiência, o commit do git e a impressão digital (SHA-256) dos ficheiros de treino e de validação. | ✅ | `test_treino_minusculo_copiado_fica_registado_no_mlflow_com_todos_os_campos_e_a_curva` e `test_treino_com_yaml_copiado_fica_no_mlflow_com_o_texto_do_yaml` → passed. Treino minúsculo real no portátil (git verdadeiro): `git_commit = d8363216…`, `git_alteracoes_por_gravar = True`, `experiencia_sha256`, `treino_sha256`, tag `experiencia_yaml` com o texto do YAML, 23 métricas |
| Depois de copiar a pasta do pod, a página do MLflow abre no portátil e mostra todos os treinos de uma experiência numa só tabela. | ✅ pelo servidor da página; falta ver no browser | `mlflow ui` (porta 5001) sobre um `mlflow.db` à parte → `runs/search` da experiência `treinos_bert` devolve `['minusculo_cpu']` |
| Um teste no CI confirma que um treino minúsculo deixa um registo com todos os campos acima. | ✅ | o mesmo teste de integração; corre sem `data/`, por isso corre no CI |
| A versão do MLflow fica fixa no `requirements.txt` (feito no E4-T5). | ✅ | `mlflow==3.17.1` (E4-T5) |
| A página mostra, para cada treino, a curva do top-5 por época. | ✅ pelo servidor da página; falta ver no browser | `metrics/get-history` → `top5_codigo_todos` com os passos 1, 2, 3, 4; `perda_treino` 2,22 → 1,62 → 1,27 → 1,03 |

Caixas do `docs/backlog.md` marcadas como `[x]` neste PR: os critérios 1 e 3 (o 4 já estava). O 2 e o 5 ficam `[ ]` até o utilizador ver a página no browser.

### Testes focados

`.venv/bin/pytest -q tests/integration/test_bert_pipeline.py tests/unit/test_registo_bert.py tests/unit/test_train_bert_git.py` → 83 passed

### O comportamento que já existia continua igual

- Suíte inteira: `.venv/bin/pytest -q` → 620 passed, 1 skipped (antes do trabalho: 577 testes no `main`, `pytest --collect-only` em `d836321`; agora 621, ou seja, 44 testes novos).
- Os testes antigos do treino não mudaram, incluindo os da retoma e o de "sem GPU, recusa antes de ler os dados" (este falhou numa versão intermédia, que verificava o YAML antes da GPU; a ordem foi reposta).
- Um treino sem `--experiencia` no processador continua a treinar como antes, com o YAML "nenhum" no registo.
- CI do commit `385a734`: ainda não corrido (o ramo ainda não foi enviado).

### Os testes novos falham com o código estragado

| Teste | Estrago introduzido | Resultado com o estrago |
|---|---|---|
| `test_na_gpu_sem_yaml_ou_sem_commit_recusa_antes_de_treinar[experiencia]` | a verificação do YAML na GPU trocada por `if False:` | falhou ✅ |
| `test_na_gpu_sem_yaml_ou_sem_commit_recusa_antes_de_treinar[commit]` | a recusa sem commit na GPU trocada por `if False:` | falhou ✅ |
| `test_train_bert_git.py` | sem `--untracked-files=no` | falhou ✅ |
| `test_commit_e_o_do_inicio_do_treino_mesmo_que_o_codigo_mude_a_meio` | o commit lido no fim do treino | falhou ✅ |
| `test_sem_commit_avisa_logo_no_inicio_mas_treina` | sem o aviso | falhou ✅ |
| `test_yaml_da_experiencia_errado_e_recusado_antes_de_treinar` | sem o limite de 8000 carateres no treino | falhou ✅ |
| `test_treino_com_outro_conteudo_recusa` | sem a verificação do SHA-256 do treino | falhou ✅ |
| `test_treino_de_outra_janela_recusa` | sem o `check_window` | falhou ✅ |
| `test_yaml_maior_que_uma_tag_do_mlflow_recusa_sem_run` | sem o limite de 8000 carateres no registo | falhou ✅ |
| `test_epoca_sem_ficheiro_de_sugestoes_recusa_sem_run` | sem a verificação do ficheiro da época | falhou ✅ |
| `test_pasta_sem_treino_json_recusa` | sem a verificação do `treino.json` | falhou ✅ |
| `test_treino_com_yaml_copiado_fica_no_mlflow_com_o_texto_do_yaml` | a etiqueta do YAML sempre "nenhum" | falhou ✅ |
| `test_registar_bert_passa_a_pasta_e_os_outros_argumentos` | o Makefile sem `$(ARGS)` | falhou ✅ |

### Números que provam que funciona

Não se aplica com dados reais: o E5-T3 não treina nem avalia nada de novo, só regista. O primeiro treino real é o E5-T4, que tem um critério para ver esse treino na página. A prova com um treino de verdade (minúsculo, no processador, com o git verdadeiro) está nos critérios acima.

| Medida | Antes | Depois | Fonte |
|---|---|---|---|
| Testes recolhidos | 577 | 621 (620 passed, 1 skipped) | `.venv/bin/pytest -q` |
| Registar a mesma pasta duas vezes | — | 1 run (a segunda devolve `None`) | treino minúsculo no portátil |

### Revisão de código

Três revisões independentes antes deste commit, e uma final sobre `d836321..385a734`: **com correções, nenhum problema crítico**. O único importante era esta descrição, que ainda não estava escrita.
- Corrigido (revisão 1): a retoma gravava o commit e os SHA-256 da retoma, não do treino original → passaram para o `argumentos.json`.
- Corrigido (revisão 2, crítico): o `git status` contava a pasta de resultados como alteração e toda a retoma no pod seria recusada → `--untracked-files=no`. O commit podia ficar vazio sem aviso → aviso no arranque e um passo no guia.
- Corrigido (revisão 3): na GPU, YAML e commit passam a obrigatórios (decisão do utilizador); o registo já não carrega o `torch`; as mensagens de erro já não sugerem `make preparar`; o guia pede um commit antes de treinar.
- Corrigido (revisão final): o guia diz que o registo recusa também um `treino.parquet` com outro SHA-256; o índice `docs/prs/readme.md` tem a entrada do E5-T3.
- Não corrigido, e porquê (menores, para não mexer no código depois da revisão final):
  - um ficheiro novo, ainda sem commit, não conta como "alteração por gravar" (`--untracked-files=no`). O guia já pede um commit antes de treinar.
  - o comando de registo mostra o erro completo do Python em vez de só a mensagem;
  - sem o `sha256.treino` no `treino.json` (um treino anterior ao E5-T3), a mensagem diz "não é o treino que o pod usou" em vez de "não tem o SHA-256";
  - um treino sem τ = 0,25 não se regista (falta o ficheiro do modelo por defeito); o comando grava sempre τ = 0,25, e o E5-S1 deve mantê-lo;
  - todos os treinos vão para a experiência `treinos_bert`; para ver só um spike filtra-se pelo parâmetro `experiencia_ficheiro`. Isto decide-se no E5-S1.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
