## E4-S3 · Spike: a entidade compradora ajuda o modelo?

## Em poucas palavras

Este PR testa se juntar ao texto do contrato o nome de quem compra (a entidade) melhora as sugestões de CPV do modelo simples (TF-IDF). Com a entidade, o modelo acerta mais: o código certo aparece nas 5 sugestões em 80,8% dos contratos, contra 78,4% sem ela. Mas a 1.ª sugestão passa a ser um código genérico mais vezes (33,1% contra 30,6%), porque o modelo aprende o hábito de cada entidade. A regra de escolha, escrita antes de correr, proibia esse aumento. Por isso **a entidade não entra**, e o modelo em vigor fica igual.

| Entidade | Texto que o modelo lê |
|---|---|
| sem (fica assim) | `Pavimentação da Rua X` |
| com (testado, rejeitado) | `Município de Y \| Pavimentação da Rua X` |

## Porquê

A `adjudicante` (quem compra) já se conhece antes de escolher o CPV, por isso pode entrar na entrada do modelo sem fuga de dados. Podia ajudar (um hospital e uma câmara compram coisas diferentes), ou podia ensinar o modelo a repetir os códigos genéricos que cada entidade costuma registar, como `45200000` ("obras de construção") em vez de `45233220` ("pavimentação de estradas"). O projeto quer sugerir códigos específicos, por isso este risco importa. A decisão 7 do `qualidade_dados.md` estava em aberto à espera deste teste.

Item: [`E4-S3`](docs/backlog.md) · Desbloqueia: nada (o E4-S2 depende do E4-T4)

## O que mudou no código

### `src/train_tfidf.py`

- `with_entity` (`src/train_tfidf.py:90`): junta a entidade ao início do texto, no formato `entidade | texto`.

```python
ent = adjudicante.astype("object").fillna("").str.split().str.join(" ")
return texto.where(ent == "", ent + sep + texto)
```
A 1.ª linha passa a entidade para texto e limpa espaços a mais. A 2.ª junta-a ao texto, menos quando a entidade está vazia (aí fica só o texto).

- `run` (`src/train_tfidf.py:103`): ganhou `com_entidade=False`. Quando é `True`, lê a coluna `adjudicante` e usa `with_entity` **no treino e na validação** (linhas 134 e 155). Se só um dos lados levasse a entidade, o modelo treinava com uma entrada e sugeria com outra.

```python
texto_treino = with_entity(treino["texto"], treino["adjudicante"]) if com_entidade else treino["texto"]
...
texto_val = with_entity(val["texto"], val["adjudicante"]) if com_entidade else val["texto"]
```

- O resumo `treino.json` regista `"com_entidade": true/false` (linha 172), e o comando ganhou a opção `--com-entidade` (linhas 220 e 225). Sem a opção, o treino não lê a `adjudicante`: o texto é o da decisão 5, como antes.

### `scripts/correr_experiencia.py`

- `ler` (`scripts/correr_experiencia.py:99`): aceita um 3.º modo de YAML, `entidade`, com as chaves `com_entidade` (nome da opção → pasta das sugestões) e `grupos`. Valida-as como nos outros modos (números com ponto, τ ≥ 0, sem repetidos, sem chaves a mais).
- `comandos` (`scripts/correr_experiencia.py:187`): recusa o modo `entidade`. Este spike é um único treino (`train_tfidf.py --com-entidade`), não uma grelha, e o script não o corre.

## Outras mudanças

- `experiencias/e4_s3.yaml` (novo): as duas opções, os três grupos e a regra de escolha, escritos antes de correr. A opção "sem entidade" reusa o treino do E4-S1 (N = 10, 24 épocas), que já tinha provado ser repetível. Assim poupa-se um treino e a entidade é a única diferença.
- `metrics/e4_s3/` (novo): as sugestões e o `treino.json` do treino com entidade, o log do pod e `pod_info.json` (o custo, escrito à mão a partir das leituras da Runpod).
- `notebooks/08_entidade.ipynb` (novo): compara as duas opções nos três grupos e aplica a regra do YAML. Corre com `make entidade` (alvo novo no `Makefile`).
- `docs/qualidade_dados.md`: secção 17 nova (tabela e leitura dos números) e decisão 7 fechada: "a `adjudicante` não entra".
- `docs/backlog.md`: o E4-S3 fica "Feito" e entra na lista do que está feito.
- `docs/workflow.md`: uma caixa `[x]` para o teste da entidade.
- `experiencias/readme.md`, `notebooks/readme.md`, `CLAUDE.md`: uma linha cada sobre o modo `entidade`, o notebook 08 e a pasta `metrics/e4_s3/`.
- `tests/`: 27 testes novos (lista abaixo).

### Ficou igual de propósito

- **O modelo em vigor e o texto de entrada** — a entidade foi rejeitada, por isso `--com-entidade` fica desligado por defeito e nada em `src/tfidf.py` muda.
- **A régua (E4-T2) e os números do E4-H1** — o backlog só os mandava recalcular se a entidade entrasse.
- **O treino "sem entidade"** — não se voltou a treinar; é o do E4-S1, para as duas opções lerem os mesmos ficheiros e só diferirem na entidade.
- **`prepare.py`** — a coluna `adjudicante` já estava no `treino.parquet` e no `validacao.parquet`; a entidade junta-se no momento do treino, como os códigos raros (decisão 9).

## Validação e provas

### Critérios de aceitação

O E4-S3 não tem caixas: tem um "Resultado" com três pontos, copiados do backlog.

| Critério (copiado do backlog) | Cumprido? | Prova (comando → resultado) |
|---|---|---|
| uma tabela com e sem entidade nos três grupos, com a parte das 1.as sugestões ao nível do grupo ou acima; | ✅ | `make entidade` → célula "Os números nos três grupos" do notebook 08; copiada para a secção 17 do `qualidade_dados.md` |
| a decisão 7 fechada no `qualidade_dados.md`; | ✅ | decisão 7 reescrita: "A `adjudicante` **não entra** no texto de entrada (fechada no E4-S3…)" |
| se a entidade entrar, a régua (E4-T2) e os resultados do E4-H1 voltam a calcular-se. | ✅ (não se aplica) | a entidade não entrou → nada a recalcular |

Caixas do `docs/backlog.md` marcadas como `[x]` neste PR: nenhuma, porque o E4-S3 não tem caixas; o item ficou marcado "Feito".

### Testes focados

`.venv/bin/pytest -q tests/unit/test_train_tfidf.py tests/unit/test_correr_experiencia.py tests/integration/test_tfidf_pipeline.py -k "entidade or with_entity or e4_s3 or comando_passa or adjudicante"` → 27 passed

### O comportamento que já existia continua igual

- Suíte inteira: `.venv/bin/pytest -q` → 366 passed (antes do trabalho: 339, ou seja 366 menos os 27 testes novos; não foi corrida antes nesta sessão).
- Sem `--com-entidade`, o treino não precisa da coluna `adjudicante` e grava `"com_entidade": false`: `test_por_defeito_a_coluna_adjudicante_nao_e_precisa` e `test_com_entidade_entra_no_vocabulario_e_no_resumo_e_por_defeito_nao`.
- Um YAML sem `modo` continua a ser lido como `fases`: `test_yaml_da_entidade_sem_modo_continua_a_ser_fases`.
- CI: corre no push deste ramo; o merge só se faz com o CI verde.

### Os testes novos falham com o código estragado

Cada estrago foi feito numa cópia do repositório no scratchpad, um de cada vez, e reposto a seguir.

| Teste | Estrago introduzido | Resultado com o estrago |
|---|---|---|
| `test_with_entity_junta_a_entidade_no_inicio_do_texto` | entidade depois do texto (`texto + sep + ent`) | falhou ✅ |
| `test_with_entity_com_entidade_vazia_ou_em_falta_fica_so_o_texto` | junta a entidade mesmo vazia | falhou ✅ (3 de 3) |
| `test_with_entity_limpa_os_espacos_da_entidade_category` | tirar a limpeza dos espaços | falhou ✅ |
| `test_com_entidade_junta_a_adjudicante_no_treino_e_nas_sugestoes` | a validação sugere com o texto sem entidade | falhou ✅ |
| testes `-k com_entidade` da integração | o treino usa o texto sem entidade | falhou ✅ (2) |
| `test_com_entidade_entra_no_vocabulario_e_no_resumo_e_por_defeito_nao` | o resumo grava sempre `com_entidade: false` | falhou ✅ |
| `test_por_defeito_a_coluna_adjudicante_nao_e_precisa` | lê sempre a coluna `adjudicante` | falhou ✅ |
| `test_comando_passa_a_entidade_ao_run` | `--com-entidade` não chega ao `run` | falhou ✅ (2) |
| `test_a_experiencia_da_entidade_nao_corre_pelo_script` | tirar a recusa do modo `entidade` em `comandos` | falhou ✅ |
| `test_yaml_da_entidade_invalido_da_erro` | aceitar `com_entidade` com uma só opção | falhou ✅ |
| `test_yaml_da_entidade_invalido_da_erro` | aceitar `grupos` vazio ou repetido | falhou ✅ (2) |

### Números que provam que funciona

Validação (outubro de 2025, 23 945 contratos), N = 10, 24 épocas, τ = 0,25; `make entidade`, corrido nesta sessão (deu os mesmos números que a versão gravada antes). Nenhum número vem do teste.

Três termos:
- **top-5:** o código certo está entre as 5 sugestões.
- **1.ª sugestão ≤ grupo:** a 1.ª sugestão é um código genérico, ao nível do grupo (3 dígitos, ex. `45200000`) ou acima.
- **bootstrap:** tiram-se 1 000 amostras à sorte dos contratos e vê-se em quantas uma opção ganha; a diferença só conta se ganhar em pelo menos 95%.

| Grupo | Contratos | Top-5 código (sem → com) | Top-5 classe (sem → com) | 1.ª sugestão ≤ grupo (sem → com) |
|---|---|---|---|---|
| todos | 23 945 | 78,35% → 80,76% | 84,11% → 85,48% | 30,57% → 33,13% |
| entidades no treino | 23 438 | 78,43% → 80,82% | 84,07% → 85,44% | 30,72% → 33,35% |
| entidades novas | 507 | 71,43% → 75,45% | 86,93% → 89,20% | 23,87% → 22,68% |

A regra do `experiencias/e4_s3.yaml`, passo a passo:

| Passo | Resultado | Conta no bootstrap? |
|---|---|---|
| 1. Regra do E4-T1 em todos (por defeito: sem entidade) | ganha "com entidade" (+2,41 pontos no código, +1,37 na classe) | sim |
| Guarda 1: não piorar o top-5 código nas entidades novas | cumpre-se (até melhora 4,02 pontos; ganha em 98% das amostras) | — |
| Guarda 2: não aumentar a 1.ª sugestão ≤ grupo, em todos | **falha** (+2,55 pontos; maior em 100% das amostras) | sim |
| **Decisão** | **a entidade não entra** | |

- **Custo:** um pod `cpu3m` de 8 processadores e 64 GB, a 0,44 $/h (o `cpu3g` de 0,32 $/h não tinha stock), ligado cerca de 44 minutos: cerca de 0,32 $. O treino levou 1 958 s, com pico de memória de 10,9 GB (`metrics/e4_s3/pod_info.json`, `treino.json`). O pod foi apagado.
- **Limitação 1 (o ganho pode vir de copiar hábitos):** se a entidade costuma registar um código genérico e o contrato da validação também o tem, sugeri-lo conta como acerto. Estes números não separam o ganho real do ganho por copiar códigos genéricos. Para isso seria preciso um spike novo, escrito antes de correr; não se fez.
- **Limitação 2 (pré-registo sem prova no git):** o `experiencias/readme.md` pede que o YAML entre no git **antes** do treino, para o histórico mostrar que a regra foi escrita às cegas. Aqui o YAML só entra neste commit, depois do treino. A regra foi escrita antes de ver os números, e foi ela que deixou a entidade de fora, apesar de ganhar no acerto. Mas o git não o prova.

### Revisão de código

`/requesting-code-review` sobre `main..spike/e4-s3`: **não feita.** O utilizador pediu para abrir o PR e fazer o merge sem esperar pelo revisor independente. A revisão automática da Claude GitHub App (`claude-code-review.yml`) corre neste PR como em todos.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
