## E4-S0 · Spike: quanto custa cada treino na máquina alugada?

## Em poucas palavras

Antes de treinar o primeiro modelo a sério, medimos quanto tempo e quanta memória gasta um treino com todos os dados. O treino completo demora 8 a 10 minutos e precisa de 6 a 7 GB de memória. Isso não cabe no portátil, mas cabe com folga numa máquina alugada de 32 GB, que custa 0,32 $ por hora. O ensaio custou cerca de 0,11 $, e todos os treinos do resto do épico devem custar menos de 2 $. Pelo caminho trocámos a Hetzner pela Runpod e escolhemos uma forma mais rápida de treinar a regressão logística.

## Porquê

O plano era treinar o TF-IDF numa máquina alugada, mas não sabíamos quanta memória nem quanto tempo cada treino gasta. Sem isso, não se escolhe a máquina certa nem se sabe o custo do épico. O risco era alugar uma máquina pequena demais e o treino morrer por falta de memória a meio, ou deixar uma máquina cara ligada sem saber quanto tempo era preciso.

Um **spike** é uma tarefa curta cujo produto é uma resposta, não código para ficar. Aqui a pergunta era: "quanto tempo e quanta memória gasta um treino, e quanto vão custar todos os treinos do épico?".

Item: [`E4-S0`](docs/backlog.md) · Desbloqueia: E4-H1 (o primeiro modelo TF-IDF)

## O que mudou no código

### `scripts/ensaio_treino.py` (novo)

Um script de ensaio: treina uma vez, mede e escreve uma linha JSON. Não avalia a qualidade, não grava o modelo e não lê a validação nem o teste. O treino a sério nasce no E4-H1. Fica em `scripts/` e não em `src/`, porque não é código do pipeline.

- `main` (`scripts/ensaio_treino.py:47`): lê o treino, aplica a regra dos códigos raros, vetoriza, treina e escreve o resultado.
- `pico_rss_mb` (`scripts/ensaio_treino.py:38`): o máximo de memória que o processo usou até agora.

A regra dos códigos raros aplica-se ao treino **completo** e só depois se tira a amostra. Assim, cada exemplo da amostra tem a mesma etiqueta que teria no treino real:

```python
codigo = pd.read_parquet(TREINO, columns=["codigo"])["codigo"]
etiqueta = collapse_rare_codes(codigo, a.n_min)
...
if a.amostra == 0:
    idx = validas
else:
    idx = np.sort(rng.choice(validas, size=min(a.amostra, len(validas)), replace=False))
```
`--amostra 0` quer dizer "treino completo". Qualquer outro número tira essa quantidade de exemplos, sempre com a mesma semente (0), para o resultado ser igual de cada vez.

O modelo medido:

```python
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                      max_features=a.max_features, dtype=np.float32)
...
clf = SGDClassifier(loss="log_loss", n_jobs=-1, random_state=0)
```
O **TF-IDF** transforma cada texto numa lista de números, um por palavra ou par de palavras. Por exemplo, em "aquisição de papel A4", "papel a4" ganha um número alto, porque é raro e diz muito, e "aquisição" ganha um número baixo, porque aparece em quase todos os contratos. Os valores ficam em float32 (4 bytes cada) em vez de float64 (8 bytes), o que corta a memória para metade. O **`SGDClassifier(loss="log_loss")`** é uma regressão logística treinada por descida de gradiente estocástica: ajusta os pesos exemplo a exemplo, com um modelo "sim/não" por etiqueta, e usa todos os processadores.

## Outras mudanças

- `requirements.txt`: passa a ter o `scikit-learn==1.7.2`, que o script usa e que o E4-H1 vai usar.
- `metrics/e4_s0/`: os resultados guardados, para cada número dos docs ter origem num ficheiro.
  - `portatil.jsonl` e `portatil_vocabulario.jsonl`: amostras no portátil, geradas por uma versão anterior do mesmo script (a mesma lógica, mas com caminhos fixos e sem o campo `maquina`).
  - `portatil_logreg30k_parada.txt`: a corrida da `LogisticRegression` que foi parada.
  - `pod.jsonl` e `pod_ensaio.log`: o treino completo no pod.
  - `pod_info.json`: o pod (tipo, preço, horas, pico de memória), escrito à mão a partir das leituras da Runpod e do pod.
  - `pod_memoria.log`: a memória do pod, lida a cada minuto.
- `docs/maquina_runpod.md` (novo): como criar, preparar e apagar o pod, com o que se confirmou no ensaio.
- `docs/backlog.md`:
  - a máquina passa da Hetzner CX53 para a Runpod, porque o CX53 estava indisponível e a conta nova pedia pagamento adiantado;
  - o E4-S0 fica marcado como feito, com os resultados;
  - entra a decisão 5 do E4, provisória: o classificador é o `SGDClassifier(loss="log_loss")`.
- `docs/workflow.md`: a linha do baseline diz Runpod e SGD.
- `docs/readme.md`, `scripts/readme.md` e `docs/prs/readme.md`: uma linha nova em cada índice.

### Ficou igual de propósito

- **`src/`:** não muda. O script do ensaio não é pipeline, e o treino a sério é o E4-H1.
- **Testes:** não há testes novos. O script é para medir uma vez. O revisor estragou-o de propósito (ler a validação em vez do treino; aplicar a regra dos códigos raros a 50 linhas) e a suíte continuou verde. Isso é aceitável para um ensaio, mas o E4-H1 não pode copiar este código para `src/` sem testes.
- **`numpy`, `scipy` e `joblib`:** continuam sem versão fixa no `requirements.txt`, porque são instalados pelo `scikit-learn`. No pod saíram as mesmas versões do portátil (numpy 2.2.6, scipy 1.15.3, registadas no `pod.jsonl`). Fixá-las fica para quando for preciso.
- **Janela desde 2019:** o ensaio só mediu a janela atual (desde 2022). A de 2019 mede-se no E4-S2, como o backlog já dizia.
- **Qualidade do modelo:** não foi medida. Este spike só mede o custo.

## Validação e provas

### Critérios de aceitação

O E4-S0 é um spike: em vez de critérios "dado/quando/então", o backlog lista o resultado esperado.

| Resultado pedido (copiado do backlog) | Cumprido? | Prova (comando → resultado) |
|---|---|---|
| o pico de memória e o tempo de treino medidos nos dois casos | ✅ | `scripts/ensaio_treino.py --n-min {10,5} --amostra 0 --modelo sgd` no pod → `metrics/e4_s0/pod.jsonl`: N = 10, 462 s e 5 480 MiB; N = 5, 616 s e 6 854 MiB |
| o custo estimado de todos os treinos do épico | ✅ | conta no `docs/backlog.md` (E4-S0, ponto 4): 1,6 a 2,6 h de pod, cerca de 0,50 a 0,85 $, abaixo de 2 $ com margem |
| o tipo de pod confirmado (memória, processadores e preço por hora) | ✅ | leitura `create-pod` da Runpod → `cpu3g`, 8 processadores, 32 GB, `cost: 0.32`; em `metrics/e4_s0/pod_info.json` |
| os passos para preparar a máquina e para a apagar no fim, escritos em `maquina_runpod.md` | ✅ | `docs/maquina_runpod.md`, secções 2 a 5, seguidas no ensaio; no fim, `list-pods` → `[]` e `list-network-volumes` → `[]` |

O "Como" do backlog também se cumpriu:
- **Preparar a máquina:** no pod, `make setup` e `.venv/bin/pytest -q` → 172 passed, com Python 3.10.12 e scikit-learn 1.7.2, os mesmos do portátil.
- **Treino de ensaio deitado fora:** o script não grava o modelo, e o pod foi apagado.

Caixas do `docs/backlog.md` marcadas como `[x]` neste PR: nenhuma, porque o E4-S0 não tem caixas. Fica marcado "(feito)" no título.

### Testes focados

Não há testes novos (ver "Ficou igual de propósito"). O script correu de ponta a ponta 8 vezes no portátil (`metrics/e4_s0/portatil.jsonl`) e 2 vezes no pod (`metrics/e4_s0/pod.jsonl`). Depois de o mudar para caminhos relativos, voltei a corrê-lo no portátil: `.venv/bin/python scripts/ensaio_treino.py --n-min 10 --amostra 3000 --modelo sgd` → `"convergiu": true`.

### O comportamento que já existia continua igual

- Suíte inteira: `.venv/bin/pytest -q` → 172 passed (antes do trabalho: 172 passed). Também 172 passed no pod.
- Nenhum ficheiro em `src/`, `tests/` ou `notebooks/` mudou: `git diff --stat main...HEAD` só tem `docs/`, `metrics/`, `scripts/` e `requirements.txt`.
- CI do commit de cima: ver a secção "Revisão de código".

### Os testes novos falham com o código estragado

Não há testes novos. O revisor estragou o script duas vezes, e a suíte não reparou, como era de esperar:

| Teste | Estrago introduzido | Resultado com o estrago |
|---|---|---|
| suíte inteira | o script lê `validacao.parquet` em vez de `treino.parquet` | não falhou (167 passed, 5 skipped): nada guarda o script |
| suíte inteira | a regra dos códigos raros é aplicada só a 50 linhas | não falhou (167 passed, 5 skipped): nada guarda o script |

### Números que provam que funciona

Treino completo (478 mil exemplos) num pod `cpu3g` com 8 processadores e 32 GB. As memórias estão em GB (mil milhões de bytes):

| Medida | N = 10 | N = 5 | Fonte |
|---|---|---|---|
| etiquetas | 3 000 | 3 848 | `pod.jsonl`, `n_classes_amostra` |
| vocabulário | 402 084 | 402 154 | `pod.jsonl`, `n_features` |
| vetorizar | 7,9 s | 8,8 s | `pod.jsonl`, `tempo_vetorizar_s` |
| treinar | 7,7 min | 10,3 min | `pod.jsonl`, `tempo_treinar_s` |
| pesos do modelo | 4,8 GB | 6,2 GB | `pod.jsonl`, `coef_bytes` |
| pico de memória do processo | 5,7 GB | 7,2 GB | `pod.jsonl`, `pico_rss_mb.final` (em MiB) |
| pico de memória do pod | 8,3 GB, de 32 GB | | `pod_info.json`, `cgroup_memory_peak_bytes` |

**Porque o SGD e não a `LogisticRegression`.** Numa amostra no portátil:

| Modelo | 10 000 exemplos | 30 000 exemplos | Fonte |
|---|---|---|---|
| `LogisticRegression` (solver `saga`) | 170 s, não convergiu | parada aos 600 s, sem acabar (vocabulário cortado a 20 000) | `portatil.jsonl`, `portatil_logreg30k_parada.txt` |
| `SGDClassifier(loss="log_loss")` | 6 s, convergiu | 44 s, convergiu | `portatil.jsonl` |

**Converge** quer dizer que os pesos deixaram de mudar, ou seja, o treino chegou ao fim. A decisão é **provisória**: escolheu-se pela velocidade, e confirma-se no E4-H1 se o modelo bater a régua na validação.

### Revisão de código

`/requesting-code-review` sobre `c027200..5684a64`: **com correções**. Não houve problemas críticos e não há fuga de dados: o script só lê o treino.

Corrigido, no segundo commit:
- **Decisão 5:** passou a provisória (escolhida só pela velocidade; confirma-se no E4-H1), e o `alpha` usado ficou registado.
- **Números sem ficheiro de origem:**
  - a corrida parada da `LogisticRegression` passou a ter registo em `portatil_logreg30k_parada.txt`;
  - o número exato do `free -h` saiu do guia, porque não ficou guardado.
- **Custo do épico:** agora mostra a conta, parcela a parcela, e a margem somada.
- **Memória:** as unidades passaram todas a GB (5,7 e 7,2 GB, em vez de 5,5 e 6,9).
- **Texto do classificador:** em vez de "a mesma regressão logística", explica a diferença: um modelo para todas as etiquetas, ou um modelo "sim/não" por etiqueta.
- **Referência à decisão:** passou a "decisão 5 do E4", para não se confundir com a decisão 5 do `qualidade_dados.md`.
- **Guia da máquina:** "parado à espera" passou a "sem nada para fazer", para não se confundir com o "Stop".
- **Núcleos:** a frase "treinado em 8 processadores" passou a "num pod com 8 processadores", porque não ficou registado quantos o treino usou.
- **IP do pod:** saiu do `pod_memoria.log`.
- **Descrição do PR:** é este ficheiro.

Não corrigido, e porquê:
- **O script não tem testes:** é um ensaio para medir uma vez, como o revisor também aceitou.
- **`numpy` e `scipy` sem versão fixa:** ver "Ficou igual de propósito".

🤖 Generated with [Claude Code](https://claude.com/claude-code)
