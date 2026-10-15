## E5-T2 · Código de treino do BERT, testado no CI

## Em poucas palavras

Este PR escreve o código que tuna o BERTimbau e o Albertina para sugerir códigos CPV, e os testes que provam que o código liga bem cada texto à sua etiqueta. "Tunar" é continuar o treino de um modelo que já sabe português, agora para escolher entre as ~3 000 etiquetas CPV. Os testes usam um modelo minúsculo, de pesos aleatórios, que corre em segundos no processador: se o código trocasse etiquetas, o modelo minúsculo errava e o CI falhava, antes de se pagar uma hora de GPU. No fim de cada época o treino grava as sugestões, as pontuações de todas as etiquetas e o modelo; se o pod cair a meio, o treino continua da última época gravada. Ainda não se treina com os dados reais: isso é o E5-T4.

| Se o código tiver este erro… | …este teste falha em segundos |
|---|---|
| etiquetas desalinhadas dos textos | o modelo minúsculo deixa de acertar os 5 contratos que decorou |
| tradução número → código partida no modelo gravado | o modelo recarregado dá outros códigos |
| o conjunto de teste entra no treino | o treino recusa-o pelo nome e pelas datas |
| a retoma recomeça do zero, ou com outros argumentos | o treino retomado não é igual ao que nunca parou, ou é recusado |

## Porquê

Um erro no código de treino (textos e etiquetas desalinhados, a tradução de número para código partida) pode correr horas na GPU sem dar erro nenhum e só produzir um modelo mau. Nesse caso não se sabe se o BERT é mau ou se o código está errado. Este item escreve o código de treino e de sugestões do BERT e os testes que apanham esses erros em segundos, sem GPU.

Item: [`E5-T2`](docs/backlog.md) · Desbloqueia: E5-T3 (registo no MLflow) e E5-T4 (primeiro treino do BERTimbau)

## O que mudou no código

### `src/bert.py` (novo): o modelo, sem ler nem gravar ficheiros de dados

- `check_gpu` (`src/bert.py:53`): erro se o `torch` não vir nenhuma GPU; senão devolve o nome da placa.
- `training_labels` (`src/bert.py:70`): a etiqueta de cada exemplo com a regra dos códigos raros (N), e a lista de etiquetas por ordem crescente.

```python
    if codigo.str.endswith("000000").any():
        raise ValueError("o treino tem códigos ao nível da divisão (XX000000)")
    etiqueta = collapse_rare_codes(codigo, n_min=n_min)
    classes = np.sort(etiqueta.dropna().unique()).astype(object)
```
As etiquetas contam-se só no treino e ficam por ordem crescente, como no TF-IDF: em empate de pontuação ganha o menor código. Um código de divisão no treino é recusado, para o modelo nunca o sugerir.

- `model_max_length` (`src/bert.py:87`): o comprimento máximo em *tokens*; por defeito o do E5-T1 (256).
- `load_model` (`src/bert.py:101`): carrega o modelo na versão fixa, com uma camada de classificação nova (uma saída por etiqueta) e a tradução número → código gravada dentro do modelo (`id2label`). Carrega sempre em `float32`, porque o Albertina está gravado em `bfloat16` e o transformers 5 manteria esse tipo.
- `predict_scores` (`src/bert.py:184`): a pontuação de cada etiqueta para cada texto, com o *dropout* desligado. O *dropout* desliga ao acaso 10% das ligações durante o treino; ao pontuar tem de estar desligado, senão as pontuações teriam ruído.
- `EpochScores` (`src/bert.py:210`): no fim de cada época, pontua a validação e entrega as pontuações a quem grava.
- `training_arguments` (`src/bert.py:226`): os argumentos do `Trainer` do Hugging Face. Por defeito, a receita do artigo do BERT (Devlin et al., 2019): taxa 2e-5, lotes de 32, 3 épocas, 10% de aquecimento, decaimento dos pesos 0,01. O E5-T4 escreve os valores a sério num YAML.

```python
        save_strategy="epoch",
        save_only_model=False,
```
Um *checkpoint* (o modelo gravado) no fim de cada época, com o estado do otimizador, para se poder retomar.

- `train` (`src/bert.py:268`): treina; com `retomar=True` continua do último *checkpoint* (`resume_from_checkpoint`).
- `suggest_from_scores` (`src/bert.py:315`): as 5 sugestões, ordenadas por pontuação − τ · log(frequência da etiqueta no treino).

```python
        pontuacoes = pontuacoes - tau * log_prior
    return [[str(c) for c in s] for s in tfidf.top_k(pontuacoes, classes, k)]
```
É a correção das etiquetas frequentes do TF-IDF (Menon et al., 2021), aqui na forma original do artigo, para um modelo com *softmax*. Reutiliza o `top_k` do TF-IDF, por isso os empates resolvem-se da mesma maneira.

### `src/train_bert.py` (novo): o comando que corre no pod

- `run` (`src/train_bert.py:95`): por esta ordem, confirma a GPU (antes de ler qualquer ficheiro), passa o treino e a validação pelo `check_inputs` do E4-T5 (recusa o teste pelo nome e pelas datas), treina e grava por época.

```python
    def on_epoch(k, checkpoint, pontuacoes):
        np.save(pontuacoes_dir / scores_file(k), pontuacoes)
        ...
        for tau in taus:
            sug = bert.suggest_from_scores(pontuacoes, classes, prior, tau)
            bert.suggestions_frame(ids, sug).to_parquet(pasta / suggestions_file(tau), index=False)
```
Em cada época grava as pontuações de todas as etiquetas (ficam no pod, para aplicar outro τ sem treinar outra vez) e as sugestões com τ = 0 e τ = 0,25, no formato do TF-IDF (vêm para o portátil).

- Retoma: o treino grava os argumentos em `argumentos.json`, e `--retomar` recusa outros.

```python
        diferentes = sorted(k for k in argumentos if guardados.get(k) != argumentos[k])
        if diferentes:
            raise ValueError(f"para retomar, os argumentos têm de ser os do treino interrompido ...")
```
Com outro lote ou outro número de épocas, o `Trainer` saltaria os passos errados sem dar erro. Por isso compara os caminhos completos dos ficheiros e das pastas, o modelo, o comprimento, o N, os τ e os hiperparâmetros.

- `main` (`src/train_bert.py:248`): o comando, com `--dispositivo cuda` por defeito e a saída por defeito em `metrics/bert/<modelo>`.

## Outras mudanças

- `requirements-cpu.txt` (novo): o projeto mais o `torch` 2.14.1 só para processador (~200 MB, índice do PyTorch) e o `accelerate` 1.15.0. É o que o CI e o portátil instalam.
- `requirements-gpu.txt` (novo): o mesmo, com o `torch` do PyPI, que traz o CUDA 13.0 (precisa de um *driver* NVIDIA 580 ou mais recente). É o que o pod instala.
- `requirements.txt`: fixa também o `huggingface_hub` e o `safetensors`, que carregam os modelos. Continua sem `torch`.
- `.github/workflows/testes.yml`: o CI instala o `requirements-cpu.txt`.
- `Makefile`: `setup` passa a usar o `requirements-cpu.txt`; alvos novos `setup-gpu`, `verificar-gpu` (mostra a GPU ou falha), `treinar-bert` e `test-bert`.
- `tests/conftest.py`: `make_tiny_model`, um modelo minúsculo de pesos aleatórios (BERT ou DeBERTa, a arquitetura do Albertina) com um tokenizador feito com as palavras dos testes.
- `tests/unit/test_bert.py`, `tests/integration/test_bert_pipeline.py` (novos): os testes do treino (lista abaixo).
- `tests/unit/test_requisitos.py` (novo): as versões fixas nas duas listas, sem `torch` no `requirements.txt`, e o CI a instalar a lista do processador.
- `tests/integration/test_makefile_bert.py` (novo): os alvos do Makefile, vistos com `make -n`, e o `verificar-gpu` a falhar sem GPU.
- `docs/maquina_runpod.md`: secção 7, o pod com GPU. O projeto vai para `/workspace` (o volume), porque o disco do contentor apaga-se quando o pod para ou reinicia (docs da Runpod), e os *checkpoints* têm de sobreviver para se retomar.
- `docs/backlog.md`: E5-T2 "feito", os 10 critérios marcados e o resultado escrito.
- `docs/workflow.md`: caixa nova `[x]` da fase 4 para o código de treino do BERT.
- `docs/prs/pr_E5-T2.md` e `docs/prs/readme.md`: esta descrição e a sua linha na tabela.
- `CLAUDE.md`, `readme.md`, `src/readme.md`, `tests/*/readme.md`: os ficheiros e comandos novos.

### Ficou igual de propósito

- Nenhum treino com os dados reais — é o E5-T4, depois do registo no MLflow (E5-T3) e do teto de gastos.
- Sem registo no MLflow (`report_to="none"`) — é o E5-T3.
- Sem precisão mista (bf16) na GPU — o treino é todo em fp32. Decide-se no E5-T4, se o custo medido o justificar.
- O `requirements.txt` continua sem `torch` — senão o `pip` instalaria a versão com CUDA (vários GB) também no CI.
- O TF-IDF, a avaliação e o portão não mudaram — as sugestões do BERT têm o mesmo formato, e um teste confirma que a avaliação e o portão as leem sem mudanças.

## Validação e provas

### Critérios de aceitação

| Critério (copiado do backlog) | Cumprido? | Prova (comando → resultado) |
|---|---|---|
| Com um modelo minúsculo de pesos aleatórios (poucas camadas, sem descarregar o modelo verdadeiro) e 5 contratos inventados com temas bem distintos, o treino acerta no top-1 em todos os 5. | ✅ | `test_treino_minusculo_pelo_pipeline_acerta_os_5_contratos_no_top_1[bert,deberta]` → passa nas duas arquiteturas |
| As sugestões são sempre 5, só contêm etiquetas que existem no treino, e nunca um código ao nível da divisão. | ✅ | `test_sugestoes_sempre_5_do_treino_sem_divisoes_pela_ordem_da_validacao` (todos os ficheiros, da raiz e de cada época) → passa |
| Dois treinos no processador com a mesma semente aleatória dão exatamente as mesmas sugestões. | ✅ | `test_mesma_semente_da_exatamente_as_mesmas_sugestoes_e_pontuacoes` e, como controlo, `test_outra_semente_da_outras_pontuacoes` → passam |
| O treino passa o treino e a validação pelas verificações do E4-T5 (`check_inputs`), e por isso recusa o teste pelo nome e pelas datas. | ✅ | `test_ficheiro_chamado_teste_*`, `test_copia_do_teste_com_outro_nome_*`, `test_validacao_passada_como_treino_*` → passam, e não fica nada gravado |
| No fim de cada época, para cada contrato da validação, o treino grava as pontuações de todas as etiquetas, para se aplicar qualquer τ depois sem treinar outra vez. Ficam no pod (cerca de 0,3 GB por época e por treino), fora do git; para o portátil vêm só as sugestões e as métricas. O treino também grava as sugestões com τ = 0 e τ = 0,25 no mesmo formato das do TF-IDF. A avaliação do E4-T1 lê-as sem mudanças. | ✅ | `test_pontuacoes_de_cada_epoca_*`, `test_sugestoes_de_cada_epoca_sao_o_top_5_das_pontuacoes_com_cada_tau`, `test_sugestoes_avaliam_se_com_a_avaliacao_comum_e_com_o_portao` → passam. Os 0,3 GB são uma estimativa (24 mil contratos × 3 000 etiquetas × 4 bytes), por medir no E5-T4 |
| O treino grava o modelo no fim de cada época (*checkpoint*), para se poder usar o da época escolhida no E5-S1. | ✅ | `test_um_checkpoint_por_epoca_com_a_traducao_numero_codigo`: cada *checkpoint* recarrega e reproduz as pontuações dessa época → passa |
| As versões do `torch`, do `transformers` e das bibliotecas novas ficam fixas. O CI instala o `torch` só para processador e o pod a versão para GPU, a partir de listas de requisitos separadas, e o CI completo continua abaixo de 5 minutos. | ✅ | `tests/unit/test_requisitos.py` → passa; CI do commit `3e9c87a`: **3 min 4 s** ao todo |
| No pod, antes de treinar, uma verificação confirma que o `torch` vê a GPU. | ✅ | `make verificar-gpu`; `test_sem_gpu_o_treino_em_cuda_recusa_antes_de_ler_os_dados`, `test_verificar_gpu_falha_sem_gpu` → passam |
| Cada teste falha quando se estraga de propósito o que ele guarda. | ✅ | tabela "Os testes novos falham com o código estragado" |
| Fica escrito no teste que ele guarda a ligação entre textos, etiquetas e sugestões, não a qualidade do modelo (como no E4-T3). | ✅ | descrição no topo de `tests/integration/test_bert_pipeline.py` e de `tests/unit/test_bert.py` |

Caixas do `docs/backlog.md` marcadas como `[x]` neste PR: as 10 do E5-T2.

### Testes focados

`make test-bert` (4 ficheiros) → 84 passed

### O comportamento que já existia continua igual

- Suíte inteira no portátil: `.venv/bin/pytest -q` → 576 passed, 1 skipped. Sem os 4 ficheiros de testes novos (`--ignore`): 492 passed, 1 skipped, ou seja, os outros testes continuam todos a passar.
- O TF-IDF, a avaliação e o portão não mudaram de código; os seus testes, incluindo os que leem os dados reais, passam.
- CI do commit `3e9c87a`: success, 554 passed e 23 skipped (os que leem dados reais), em 3 min 4 s.

### Os testes novos falham com o código estragado

Cada estrago foi feito à mão, um de cada vez, corrido o teste, e o ficheiro reposto (confirmado com `cmp`). Foram 74 estragos ao todo, nas várias rondas; estes são os principais:

| Teste | Estrago introduzido | Resultado com o estrago |
|---|---|---|
| `test_treino_minusculo_..._top_1` | etiquetas rodadas uma posição em relação aos textos | falhou ✅ |
| `test_um_checkpoint_por_epoca_...` | sem `id2label` / `id2label` com uma posição de diferença | falhou ✅ |
| testes de recusa do teste | sem `check_inputs`; só o nome, sem as datas | falhou ✅ |
| `test_sugestoes_de_cada_epoca_..._com_cada_tau` | τ ignorado; `+ τ` em vez de `− τ` | falhou ✅ |
| `test_mesma_semente_...` | sem `set_seed`; semente fixa a 0 no `Trainer` | falhou ✅ |
| `test_pontuacoes_de_cada_epoca_...` | pontuações pela ordem inversa da validação | falhou ✅ |
| `test_pontuar_desliga_o_dropout_...` | sem `model.eval()`; sem voltar ao modo de treino | falhou ✅ |
| `test_treino_retomado_da_o_mesmo_que_sem_interrupcao` | otimizador não gravado; `--retomar` ignorado (recomeça do zero) | falhou ✅ |
| `test_retomar_com_outros_argumentos_e_recusado` | sem comparar os argumentos; sem a pasta das pontuações | falhou ✅ |
| `test_retomar_com_o_treino_de_outra_pasta_e_recusado` | só o nome do ficheiro em vez do caminho completo | falhou ✅ |
| `test_retomar_depois_da_ultima_epoca_...` | a retoma começa uma época antes | falhou ✅ |
| `test_retomar_salta_um_checkpoint_gravado_so_a_meio` | aceitar um *checkpoint* sem `trainer_state.json`; voltar ao `resume_from_checkpoint=True` do transformers | falhou ✅ |
| `test_sem_gpu_o_treino_em_cuda_...` | verificação da GPU depois de ler os dados | falhou ✅ |
| `test_verificar_gpu_falha_sem_gpu` | o alvo do Makefile não verifica nada | falhou ✅ |
| `test_requisitos.py` | `torch` no `requirements.txt`; outro `torch` no pod; CI a instalar a lista sem `torch` | falhou ✅ |
| `test_treinar_bert_passa_o_modelo_e_a_pasta_de_saida` | outra pasta de saída; `MODELO` ignorado | falhou ✅ |

Um estrago passou na primeira versão dos testes (a semente fixa a 0 nos argumentos do `Trainer`); acrescentou-se um teste e agora falha.

### Números que provam que funciona

Não se aplica a este item: o código ainda não treinou com os dados reais, e por isso não há números da validação. O primeiro treino a sério, com o custo de uma época, é o E5-T4.

| Medida | Antes | Depois | Fonte |
|---|---|---|---|
| Testes no portátil | 492 passed (sem os ficheiros novos) | 576 passed | `.venv/bin/pytest -q` |
| Tempo do CI (instalar + testes) | 2 min 30 s (E5-T1, commit `5af6013`) | 3 min 4 s | `gh run view` do commit `3e9c87a` |
| Treino retomado vs. treino sem interrupção | — | pontuações iguais (tolerância 1e-5) em todas as épocas | `test_treino_retomado_da_o_mesmo_que_sem_interrupcao` |

### Revisão de código

Revisão independente (um agente novo de cada vez, só com o id, a base e o topo do diff), quatro rondas:

1. `603ab8c..89a69e2`: com correções. Nada crítico. O CI do mesmo commit falhou num teste (no CI não há `.venv`).
2. `603ab8c..d00a748`: com correções. Três importantes: o `model.eval()` sem teste (os modelos minúsculos não tinham *dropout*), a retoma a comparar só o nome dos ficheiros, e o guia a pôr os *checkpoints* num disco que se apaga ao reiniciar o pod.
3. `603ab8c..df75316`: com correções. Nada crítico. Faltavam esta descrição e o tempo do CI no commit final.
4. `603ab8c..98f84a5`: com correções pequenas. Nada crítico; o importante era juntar esta descrição ao commit. Menor corrigido: uma retoma depois de o pod cair a meio de gravar um *checkpoint* partia dessa pasta incompleta.
5. `603ab8c..3e9c87a`: interrompida a pedido do utilizador, para abrir o PR. A última correção (saltar um *checkpoint* incompleto) tem o seu teste e foi vista a falhar com o código estragado.

- Corrigido:
  - o teste do `verificar-gpu` usa o Python que corre os testes (`d00a748`);
  - a retoma recusa argumentos diferentes (`d00a748`);
  - o `verificar-gpu` mostra a GPU, e o `huggingface_hub` e o `safetensors` ficam fixos (`d00a748`);
  - um teste com *dropout* guarda o `model.eval()` (`df75316`);
  - a retoma compara caminhos completos e o comprimento já resolvido (`df75316`);
  - o projeto no pod com GPU fica em `/workspace` (`df75316`);
  - a retoma confirma as pastas dos *checkpoints* e das pontuações, e o `treino.json` grava o n.º de GPUs (`98f84a5`);
  - a retoma salta um *checkpoint* gravado só a meio, e os números estimados dos docs dizem que são estimativas (`3e9c87a`).
- Não corrigido, e porquê:
  - os 0,3 GB por época e o "3 vezes o modelo" são estimativas, escritas como tal → medem-se no E5-T4;
  - com várias GPUs o lote efetivo multiplica-se → fica registado (`n_gpus`), e o E5-T4 escolhe um pod com 1 GPU;
  - sem bf16 → decide-se no E5-T4;
  - o `ignore_mismatched_sizes=True` pode recriar outra camada de tamanho diferente → com as revisões fixas não acontece;

🤖 Generated with [Claude Code](https://claude.com/claude-code)
