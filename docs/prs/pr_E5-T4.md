## E5-T4 · Primeiro treino do BERTimbau

## Em poucas palavras

Este PR faz o primeiro treino a sério do BERTimbau, numa GPU alugada, e grava os resultados. Antes do treino, um ensaio curto mediu quanto custa uma época (cerca de 18 minutos e 0,27 $). O treino completo, de 3 épocas, levou 56 minutos e custou cerca de 0,83 $. Na validação, o código certo está nas 5 sugestões em **73,39%** dos contratos. Isso fica bem acima da régua de 26,3%, mas abaixo dos 78,35% do TF-IDF. O teste de "decorar 100 contratos" falhou nas condições escritas antes de correr. Uma experiência à parte mostrou que a causa era a falta de passos de treino, e não um erro encontrado no código.

| Medida (validação, outubro de 2025) | Régua "códigos mais comuns" | BERTimbau (este PR) | TF-IDF em vigor |
|---|---|---|---|
| Top-5, código completo, todos os contratos | 26,3% | **73,39%** | 78,35% |
| Top-5, classe (4 dígitos), todos os contratos | | 79,59% | 84,11% |

## Porquê

É o primeiro treino a sério de um modelo tunado. Mostra três coisas: se o código funciona com os dados reais, quanto custa um treino, e qual é a opção em vigor com que a grelha do E5-S1 compara. Sem o custo de uma época não se sabe se a grelha cabe no teto de 20 $.

Item: [`E5-T4`](docs/backlog.md) · Desbloqueia: E5-T5 e E5-S1

## O que mudou no código

Três termos usados abaixo:
- **Passo:** um lote de 32 textos. O modelo calcula o erro nesses 32 e muda os pesos uma vez. Uma época do treino tem 14 945 passos.
- **Ensaio de custo:** um treino que para ao fim de 200 passos. Mede o tempo de cada passo e a memória da GPU, e daí calcula-se o custo de uma época sem a pagar.
- **Sinal de decorar:** treina-se o modelo só com 100 contratos e vê-se se ele os acerta quase todos (95 ou mais). Se nem isso conseguir, há quase de certeza um erro no código, por exemplo textos ligados às etiquetas erradas.

Os commits `e98d2ae` e `8403606` trazem o código, e os commits `7c10d1d` e `b6e175b` os resultados e as correções da revisão. O commit `c9805f1` ("changes") é do utilizador e não é do E5-T4 (ver "Outras mudanças").

### `src/bert.py`

- `StepTimer` (`src/bert.py:228`): mede os segundos por passo e deixa de fora o 1.º, que inclui o arranque. Na GPU espera que o passo acabe (`synchronize`), porque as operações da GPU correm em fila.

```python
def segundos_por_passo(self):
    """Média dos passos depois do 1.º (None com menos de 2 passos)."""
    if self.passos < 2:
        return None
    return (self.ultimo - self.primeiro) / (self.passos - 1)
```
O intervalo entre o fim do 1.º passo e o fim do último, a dividir pelos passos que os separam.

- `training_arguments` (`src/bert.py:258`): ganha `max_passos`, que para o treino a meio de uma época, e `gravar_checkpoints=False`, para o ensaio e o sinal não gravarem nada que não se usa.
- `train` (`src/bert.py:319`): aceita `classes` de outro treino. O sinal de decorar treina com 100 contratos mas com as 3 000 etiquetas do treino completo, para testar a ligação real entre textos e etiquetas.

```python
externas = classes is not None
if not externas:
    etiqueta, classes = training_labels(codigo, n_min)
else:
    ...
    fora = sorted(set(codigo) - set(classes))
    if fora:
        raise ValueError(f"etiquetas que não estão nas `classes`: {fora[:3]}")
    etiqueta = codigo
```
Com `classes` dadas não se aplica a regra dos códigos raros, e uma etiqueta que não esteja nessa lista dá erro.

### `src/train_bert.py`

- `--max-passos` (`src/train_bert.py:425`) e `run` (`src/train_bert.py:185`): o ensaio de custo grava só o `treino.json`, com `"ensaio": true`, e recusa-se a retomar ou a escrever por cima de um treino a sério.
- `gpu_memory_mb` (`src/train_bert.py:159`): o pico de memória da GPU, a alocada e a reservada. A reservada é a que tem de caber na placa.

### `src/registo_bert.py`

- `log_training` (`src/registo_bert.py:179`, a recusa na linha 186): um ensaio de custo é recusado no MLflow, porque na tabela passaria por um treino a sério.
- `training_cost` (`src/registo_bert.py:130`): regista também os segundos por passo, os passos por época e a memória da GPU.

### `scripts/sinal_decorar.py`

- `first_epoch_meeting` (`scripts/sinal_decorar.py:77`): o sinal passa se **alguma** época chegar ao critério, como o YAML diz, e não só a última.

```python
def first_epoch_meeting(por_epoca, criterio):
    """A primeira época com pelo menos `criterio` acertos, ou None se nenhuma chegar lá."""
    return next((e["epoca"] for e in por_epoca if e["acertos"] >= criterio), None)
```
Percorre as épocas por ordem e devolve a primeira com acertos suficientes.

### `src/suggest.py`

- `load_bert` (`src/suggest.py:72`) e `format_bert_suggestions` (`src/suggest.py:122`): o comando de sugestões aceita um checkpoint do BERT (`--bert`). Recusa um checkpoint cujas etiquetas não sejam as das pontuações gravadas (de outro treino), e recusa etiquetas ao nível da divisão.

## Outras mudanças

- `experiencias/e5_t4.yaml`: a receita do treino, do ensaio e dos dois sinais, escrita antes de correr, com a fonte de cada valor. O artigo do BERTimbau não dá receita para esta tarefa, por isso usa-se a do BERT original (Devlin et al., 2019), dito como tal. A GPU e o preço foram preenchidos no commit `8403606`, antes de correr.
- `metrics/bert/e5_t4_ensaio/`, `e5_t4_sinal_decorar/`, `e5_t4_sinal_decorar_diag_100ep/`, `e5_t4_bertimbau/`: os resultados copiados do pod. O treino tem uma pasta por época, o `pod_info.json` (máquina, tempos, custo), o `pod_treino.log` e o `sugestoes_exemplos.txt`.
- `tests/integration/test_registo_e5_t4.py` (novo): guarda estes registos fixos e os números que os documentos citam deles (ver "Validação e provas").
- `tests/integration/test_sinal_decorar.py`, `test_suggest_bert.py`, `test_experiencia_e5_t4.py`, `tests/unit/test_bert.py`, `test_registo_bert.py`, `tests/integration/test_makefile_bert.py`, `test_bert_pipeline.py`: testes do código novo.
- `docs/qualidade_dados.md`: secção 21 (o sinal de decorar que falhou e o diagnóstico) e secção 22 (o treino completo, época a época, contra o TF-IDF).
- `docs/backlog.md`: critérios do E5-T4 marcados e o resultado escrito. Escreve também a regra nova do sinal de decorar no épico e no E5-T5 (até 100 épocas, mudança feita depois de ver o resultado), e o teto de gastos fechado em 20 $.
- `docs/maquina_runpod.md`: o que se aprendeu no primeiro pod com GPU (`rsync --no-owner --no-group` no disco de rede, instalação lenta, stock que muda) e as 100 épocas do sinal a partir do E5-T5.
- `docs/workflow.md`, `CLAUDE.md`, `metrics/readme.md`, `tests/integration/readme.md`, `src/readme.md`, `scripts/readme.md`, `experiencias/readme.md`: estado e descrição dos ficheiros novos.
- `Makefile`: os atalhos `sinal-decorar` e `sugerir-bert`. A ajuda do `mlflow` passa a dizer `http://127.0.0.1:5000`, porque neste Mac `localhost:5000` cai no AirPlay e dá 403.
- Commit `c9805f1` do utilizador, **fora do âmbito do E5-T4:** os vídeos (`videos/`, 2 notebooks e 3 `.md`), o `make setup` com uv, o marcador `sys_platform` no `requirements-cpu.txt` (com o teste ajustado em `test_requisitos.py`) e edições ao `CLAUDE.md`. Entra neste PR porque foi feito neste ramo.

### Ficou igual de propósito

- O campeão do portão (`metrics/validacao.json`) continua a ser o TF-IDF: não se correu o `make portao`, porque o BERTimbau desce nas 4 métricas e seria recusado. A comparação a sério, com bootstrap, é da E5-T6.
- O `EPOCAS = 30` por defeito do `scripts/sinal_decorar.py`: o YAML do E5-T5 passa `--epocas 100` explicitamente, e assim o comando do E5-T4 continua a dar o que deu.
- Os checkpoints e as pontuações por época não vieram para o portátil e perderam-se com o pod, como o E5-T2 já previa. Refazê-los custa cerca de 0,83 $, e a grelha do E5-S1 repete esta configuração.
- O `experiencias/e5_t4.yaml` não mudou depois de correr. Um teste confirma que o SHA-256 (a impressão digital do ficheiro) gravado nos 4 resultados é o de hoje.

## Validação e provas

### Critérios de aceitação

| Critério (copiado do backlog) | Cumprido? | Prova (comando → resultado) |
|---|---|---|
| **Antes do treino completo:** um treino curto, de alguns minutos, mede o tempo por lote e a memória da GPU. Daí sai o tempo e o custo em dólares de uma época completa, ao preço por hora lido na Runpod no dia. Esses números são mostrados e aprovados antes de lançar o treino completo, e servem depois para calcular o custo da grelha do E5-S1. | ✅ | `metrics/bert/e5_t4_ensaio/treino.json`: 200 passos, 0,073 s por passo, 1 090,7 s por época estimados (cerca de 0,27 $ a 0,89 $/h), 9 318 MB de pico reservado. O utilizador aprovou antes de lançar o treino. |
| O treino usa a janela de 2022, N = 10, o comprimento máximo do E5-T1, e a taxa de aprendizagem, as épocas e os restantes hiperparâmetros recomendados no artigo do BERTimbau. Se o artigo não der essa receita, usa-se a do artigo original do BERT (Devlin et al., 2019), e diz-se. Os valores ficam num YAML escrito antes de correr, com a fonte de cada um. | ✅ | `experiencias/e5_t4.yaml`. `test_treino_completo_tem_a_receita_do_yaml_e_nao_le_o_teste` e `test_correu_com_o_yaml_de_hoje_num_commit_limpo` → passed. |
| **Sinal de erro 1, antes do treino completo:** decora 100 contratos, nas condições escritas no épico, e a taxa usada fica registada. | ❌ | `metrics/bert/e5_t4_sinal_decorar/sinal_decorar.json`: 86 de 100 em 30 épocas, `passou: false`, taxa 1e-4 registada. O diagnóstico com 100 épocas chegou aos 95 na época 27 e acabou em 100 de 100 (secção 21). O utilizador aceitou a causa e mandou avançar. |
| **Sinal de erro 2:** o treino completo tem top-5 no código completo acima de 26,3% na validação. | ✅ | `test_top5_da_secao_22_volta_a_dar_se_com_a_validacao_atual` → 73,39%. |
| Dada uma descrição de compra, o comando de sugestões devolve 5 códigos deste modelo, nenhum ao nível da divisão. | ✅ | `make sugerir-bert CHECKPOINT=models/bert/e5_t4_bertimbau/checkpoints/checkpoint-44835 MODELO=bertimbau`, no pod → 5 códigos por descrição, nenhum acabado em `000000` (`metrics/bert/e5_t4_bertimbau/sugestoes_exemplos.txt`). |
| Os ficheiros de treino e de validação no pod têm a mesma impressão digital (SHA-256) que no portátil, e nenhum caminho aponta para o teste. | ✅ | `make registar-bert` confirma o SHA-256 antes de registar e registou: `0769501a…` (treino) e `f0cce1a0…` (validação). Os caminhos são `treino.parquet` e `validacao.parquet`. |
| Depois de copiar a pasta do `--saida` e de `make registar-bert`, a página do MLflow (`make mlflow`) mostra o treino na tabela da experiência `treinos_bert`, com a curva do top-5 por época, e o commit gravado é o do código que treinou (critério 2 do E5-T3, que só se vê com um treino real). | ❌ por ver | Pela API da página (`mlflow ui`, porta 5001): o run `e5_t4_bertimbau` está na `treinos_bert`, FINISHED, com o commit `8403606…` e a curva `[(1, 0,6803), (2, 0,7237), (3, 0,7339)]`. Falta o utilizador ver a página aberta, em `http://127.0.0.1:5000`. |
| O tempo, a memória da GPU e o custo do treino ficam num ficheiro em `metrics/`, e o pod é terminado logo a seguir a copiar os resultados. | ✅ | `metrics/bert/e5_t4_bertimbau/pod_info.json`, verificado por `test_custo_e_memoria_do_pod_info_saem_do_treino_json`. `delete-pod` → 204 e `list-pods` → `[]`. |

Caixas do `docs/backlog.md` marcadas como `[x]` neste PR: o ensaio de custo, a receita no YAML, o sinal de erro 2, o comando de sugestões, o SHA-256 e o ficheiro de custo com o pod terminado. Ficam `[ ]` o sinal de erro 1 (falhou) e a página do MLflow (por ver).

### Testes focados

`.venv/bin/pytest -q tests/integration/test_sinal_decorar.py tests/integration/test_suggest_bert.py tests/integration/test_experiencia_e5_t4.py tests/integration/test_registo_e5_t4.py tests/unit/test_bert.py tests/unit/test_registo_bert.py tests/integration/test_makefile_bert.py tests/integration/test_bert_pipeline.py tests/unit/test_requisitos.py` → 190 passed (antes dos 6 testes juntados depois da revisão). `.venv/bin/pytest -q tests/integration/test_registo_e5_t4.py` → 14 passed.

### O comportamento que já existia continua igual

- Suíte inteira: `.venv/bin/pytest -q` → 694 passed, 1 skipped (antes dos testes novos deste commit: 680 passed, 1 skipped). Não medido no `main`.
- O campeão do portão continua igual: `metrics/validacao.json` não mudou, e `test_metricas_do_campeao_sao_as_da_validacao_atual` passa.
- CI do commit `b6e175b`: ver abaixo, antes do merge.

### Os testes novos falham com o código estragado

| Teste | Estrago introduzido | Resultado com o estrago |
|---|---|---|
| `test_correu_com_o_yaml_de_hoje_num_commit_limpo` | uma linha a mais no `experiencias/e5_t4.yaml` | falhou ✅ (5 testes) |
| `test_custo_e_memoria_do_pod_info_saem_do_treino_json` | custo 0,83 → 0,73 no `pod_info.json` | falhou ✅ |
| `test_top5_da_secao_22_volta_a_dar_se_com_a_validacao_atual` | sugestões da época 1 no lugar das da época 3 | falhou ✅ |
| `test_sinal_oficial_falhou_e_o_diagnostico_passou_na_epoca_27` | uma etiqueta da amostra do diagnóstico mudada | falhou ✅ |
| `test_os_14_erros_do_sinal_oficial_…` | um contrato com etiqueta repetida passa a errado | falhou ✅ |
| `test_curva_e_tau0_da_secao_22` | sugestões da época 1 no lugar das da época 2 | falhou ✅ |
| `test_taxa_de_aprendizagem_no_fim_da_epoca_27` | ler a taxa na época 26 em vez da 27 | falhou ✅ (os 2 casos) |
| `test_passa_se_uma_epoca_cumprir_mesmo_que_a_ultima_desca` | o sinal só olha para a última época | falhou ✅ |

Os ficheiros estragados voltaram ao original: o `git status` não mostra nenhuma alteração em `metrics/`.

### Números que provam que funciona

Treino completo, RTX 4090, commit `8403606`, validação de outubro de 2025 (23 945 contratos), τ = 0,25. Fonte: o run `e5_t4_bertimbau` no MLflow e `test_curva_e_tau0_da_secao_22`, que os recalcula das sugestões gravadas.

| Medida | Época 1 | Época 2 | Época 3 | TF-IDF |
|---|---|---|---|---|
| Top-5, código completo, todos | 68,03% | 72,37% | **73,39%** | 78,35% |
| Top-5, classe, todos | 73,41% | 78,24% | 79,59% | 84,11% |
| Top-5, código completo, texto novo | 65,05% | 69,30% | 70,36% | 75,14% |
| Top-5, classe, texto novo | 71,44% | 76,27% | 77,76% | 82,18% |

- **Custo** (`pod_info.json`): o treino levou 3 370,85 s, ou seja cerca de 0,83 $. O pod inteiro, com a instalação, os testes, o ensaio e os sinais, custou cerca de 1,86 $ por cálculo. A fatura lida logo a seguir a apagar mostrava 1,00 $; a diferença não está explicada.
- **Teto de 20 $:** faltam 23 dos 24 treinos planeados. A 0,83 $ cada dão cerca de 19,2 $, e sobram cerca de 18,1 $. A grelha do E5-S1 vai ter de encolher, e decide-se quanto depois do ensaio de custo do Albertina.
- **Limitação:** é um treino só, sem bootstrap, por isso não decide se o BERTimbau é pior do que o TF-IDF. A taxa de aprendizagem desce até zero no fim da época 3, por isso a curva também não mostra que mais épocas não ajudavam.

### Revisão de código

Revisor independente (agente novo) sobre `752493c..7c10d1d`: **pronto para merge com correções**, sem problemas críticos. Estragou 6 testes de propósito numa cópia, e todos falharam.
- Corrigido em `b6e175b`:
  - A regra das 100 épocas estava só no resultado; passou para o épico, para os critérios do E5-T5 e para o guia do pod.
  - O E5-T4 aparecia como "feito" com 2 critérios por cumprir; passou a "a fechar" no backlog, no workflow e no `CLAUDE.md`.
  - Os números da secção 21 e as épocas 1 e 2 da secção 22, contados à mão, passaram a ter teste.
  - O `sugestoes_exemplos.txt` ganhou um cabeçalho com o comando e o checkpoint.
  - O `metrics/readme.md` diz agora que o diagnóstico de 100 épocas não foi pré-registado.
  - Corrigida a frase do custo, que se lia como uma gama de preços.
- Não corrigido, e porquê:
  - O commit `c9805f1` fica neste PR, dito acima: é do utilizador e já está no ramo.
  - O `--base` do `suggest.py` não é confrontado com o checkpoint: hoje é inofensivo, porque os dois modelos têm 256 tokens.
  - O `segundos_por_epoca_estimados` do treino completo inclui pontuar a validação e gravar o checkpoint: está comentado no código, e o tempo real está no `tempo_treinar_s`.
- Não se lançou um segundo revisor: depois da revisão só mudaram documentos e testes, não código do pipeline.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
