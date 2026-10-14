## E4-T4 · Escolher a data de início do treino na preparação

## Em poucas palavras

A preparação dos dados passa a aceitar a data em que o treino começa. Até agora o treino começava sempre em janeiro de 2022. O próximo spike (E4-S2) quer comparar treinos que começam em 2025, 2022 e 2019, e para isso precisa de gerar os três. Sem indicar nada, tudo fica exatamente como antes. Com outra data, os ficheiros vão para outra pasta, e a validação e o teste não mudam.

| Comando | Treino começa em | Onde grava |
|---|---|---|
| `make preparar` (sem opções) | 2022-01-01 | `data/processed/`, como sempre |
| `--inicio-treino 2019-01-01 --saida data/janelas/2019` | 2019-01-01 | `data/janelas/2019/` |
| `--inicio-treino 2019-01-01` sem `--saida` | — | recusa, com erro, e não grava nada |
| `--inicio-treino 2025-10-01` (ou depois) | — | recusa: o treino tem de acabar antes de a validação começar |

## Porquê

Hoje o treino começa sempre em janeiro de 2022 (decisão 1 do `qualidade_dados.md`). Essa data foi escolhida com uma **medida substituta** (*proxy*): a parte dos contratos da validação com um código que tem pelo menos 10 exemplos no treino. Não se treinou nenhum modelo. O spike E4-S2 confirma-a ou muda-a, treinando o TF-IDF com três janelas. Para isso, a preparação tem de conseguir gerar um treino a partir de qualquer data, sem mudar a validação nem o teste.

Item: [`E4-T4`](docs/backlog.md) · Desbloqueia: E4-S2

## O que mudou no código

### `src/cpv_data.py`

- `period` (`src/cpv_data.py:76`): diz a que parte (treino, validação, teste) pertence cada contrato. Passa a aceitar a data de início do treino. Sem ela usa a constante `TRAIN_START`, lida no momento da chamada.

```python
if train_start is None:
    train_start = TRAIN_START
out = pd.Series(None, index=data.index, dtype="object")
out[(data >= train_start) & (data < VAL_START)] = "treino"
```
Só a fronteira de baixo do treino muda. A validação e o teste continuam presos às constantes.

### `src/prepare.py`

- `load_one` (`src/prepare.py:64`): o início por defeito (`since=None` → `cd.TRAIN_START`) passa a ler-se no momento da chamada, como no `period`. Antes ficava fixo quando o módulo era importado. Não muda nenhum resultado, porque o `load_all` passa sempre a data (pedido da 1.ª revisão).
- `load_all` (`src/prepare.py:92`): lê os contratos publicados a partir da data indicada, em vez de 2022.
- `add_columns` (`src/prepare.py:113`): passa a data ao `period`, para marcar os contratos antigos como treino.
- `check_sets` (`src/prepare.py:160`): a verificação "treino fora do período" usa a data indicada. Sem isso, um treino desde 2019 daria sempre erro.
- `run` (`src/prepare.py:183`): resolve a data uma só vez e passa a mesma a todos os passos. As linhas do funil mostram a data real em vez de "jan 2022".

```python
train_start = pd.Timestamp(cd.TRAIN_START if train_start is None else train_start)
if not train_start < cd.VAL_START:
    raise ValueError(f"o início do treino ({train_start:%Y-%m-%d}) tem de ser antes de "
                     f"{cd.VAL_START:%Y-%m-%d} (início da validação)")
```
Uma data a partir de outubro de 2025 deixaria o treino vazio ou sobreposto à validação, por isso dá erro logo no início.

- `main` (`src/prepare.py:225`): é a linha de comando, nova, com `--inicio-treino` e `--saida`.

```python
if args.inicio_treino != cd.TRAIN_START and args.saida is None:
    parser.error("com --inicio-treino diferente de "
                 f"{cd.TRAIN_START:%Y-%m-%d} é preciso indicar --saida (para não sobrescrever {OUT_DIR})")
run(out_dir=args.saida or OUT_DIR, train_start=args.inicio_treino)
```
Com outra data, é obrigatório dizer onde gravar. Assim, o treino desde 2019 nunca substitui por engano o `data/processed/`, que todos os notebooks e treinos leem.

## Outras mudanças

- `Makefile`: alvo novo `make preparar-janela INICIO=2019-01-01 SAIDA=data/janelas/2019`.
- `tests/unit/test_cpv_data.py`: 2 testes do `period` com outra data (8 casos de fronteira) e sem data.
- `tests/unit/test_prepare.py`: 9 testes:
  - `load_all`, `add_columns` e `check_sets` com outra data;
  - o erro para datas a partir da validação (4 casos);
  - a linha de comando com outra data e sem `--saida`;
  - a linha de comando sem argumentos e com a data por defeito, que tem de gravar em `data/processed/`.
- `tests/integration/test_pipeline.py`: 7 testes que correm o pipeline inteiro, de `.xlsx` inventados a ficheiros gravados, com a data por defeito, com 2022 dito explicitamente e com 2019. Usam 3 contratos novos (dois de 2020 e um de 2018), só nos testes novos. A lista `CONTRATOS` dos testes antigos não mudou.
- `tests/conftest.py`: o ajudante `run_pipeline` passa a aceitar `train_start`.
- `tests/unit/readme.md`, `tests/integration/readme.md`, `src/readme.md`: descrevem os testes novos e a opção nova.
- `CLAUDE.md`: diz onde está a opção e porque exige `--saida`.
- `docs/backlog.md`: E4-T4 marcado como feito, e o E4-S2 ganha 6 verificações de que o pod corre o que foi decidido: pré-registo, comandos mostrados antes de correr, impressões digitais dos ficheiros, a janela de 2022 como controlo, contagens do funil, e validação e teste iguais em todas as janelas. A última veio da 1.ª revisão. Estas verificações não fazem parte do E4-T4: foram pedidas durante esta tarefa e ficam declaradas aqui.

### Ficou igual de propósito

- **A constante `TRAIN_START` (2022-01-01).** Mudá-la é a decisão do E4-S2, não desta tarefa.
- **A regra de deduplicação (uma cópia por `idcontrato`, decisão 3).** Com um início mais antigo, um contrato da validação podia, em teoria, perder para uma cópia antiga com o mesmo `idcontrato` e sair da avaliação. Nos dados reais, com início em 2019, isso acontece em 0 contratos. É uma contagem feita no terminal, não num notebook, por isso não entra nos docs.
- **Os modelos, as métricas e os notebooks.** Esta tarefa só cria a ferramenta. Gerar as janelas e treinar é o E4-S2.

## Validação e provas

### Critérios de aceitação

| Critério (copiado do backlog) | Cumprido? | Prova (comando → resultado) |
|---|---|---|
| A preparação aceita a data de início do treino. Sem a indicar, a data continua a vir do mesmo sítio de hoje (a constante partilhada) e dá exatamente os mesmos ficheiros e o mesmo funil (478 384 exemplos de treino). | ✅ | `.venv/bin/python src/prepare.py --saida <pasta temporária>` com os dados reais → funil com **478 384** exemplos de treino. O SHA-256 dos três ficheiros é igual ao de `data/processed/` (treino `0769501ae2a306db…`, validação `f0cce1a06d532124…`, teste `bf3c1b80f10cd30c…`; do teste só se calculou o hash, sem abrir o conteúdo). Com dados inventados: `test_inicio_por_defeito_da_a_mesma_saida_que_antes`, `test_inicio_por_defeito_grava_os_mesmos_ficheiros`, `test_period_sem_inicio_do_treino_usa_train_start`. |
| Com outra data de início, a validação e o teste têm os mesmos contratos e as mesmas etiquetas. Só muda a marca "texto também aparece no treino", que depende do treino. | ✅ | `test_outro_inicio_nao_muda_validacao_nem_teste_so_visto_no_treino` (todas as colunas iguais, menos `visto_no_treino`) e `test_outro_inicio_muda_visto_no_treino_dos_textos_antigos` (os contratos 11 e 12 passam a "visto" com 2019). Não medido com os dados reais: gerar a janela desde 2019 é o primeiro passo do E4-S2. |
| As verificações automáticas usam a data de início indicada e continuam a passar. | ✅ | `test_check_sets_usa_o_inicio_do_treino_indicado`, `test_check_sets_aceita_treino_antigo_com_inicio_anterior` e `test_check_sets_por_defeito_apanha_treino_antes_de_2022`. A corrida com 2019 no teste de integração passa as verificações todas. |

Caixas do `docs/backlog.md` marcadas como `[x]` neste PR: os três critérios do E4-T4.

### Testes focados

`.venv/bin/pytest -q tests/unit/test_prepare.py tests/unit/test_cpv_data.py tests/integration/test_pipeline.py` → 95 passed

### O comportamento que já existia continua igual

- Suíte inteira: `.venv/bin/pytest -q` → **394 passed**. O número antes do trabalho não foi contado no `main`. Nenhum teste antigo foi alterado: o diff de `tests/` só acrescenta testes, e o `run_pipeline` do `conftest.py` ganhou um argumento opcional.
- Sem a opção nova, os três ficheiros reais saem byte a byte iguais aos de `data/processed/` (impressões digitais acima), com o mesmo funil de 478 384 exemplos.
- CI do commit `<sha>`: <por preencher depois do push>

### Os testes novos falham com o código estragado

Cada erro foi injetado no código real, um de cada vez, e reposto no fim (`cmp` confirma o ficheiro original):

| Teste(s) | Estrago introduzido | Resultado com o estrago |
|---|---|---|
| `test_check_sets_usa_o_inicio_do_treino_indicado` e testes de integração com 2019 | `check_sets` volta a usar `cd.TRAIN_START` | 2 falharam e 7 deram erro ✅ |
| `test_add_columns_com_inicio_anterior_…`, `test_build_train_com_inicio_anterior_…`, integração | `add_columns` ignora a data | 4 falharam ✅ |
| `test_inicio_em_2019_poe_contratos_antigos_no_treino`, `test_outro_inicio_muda_visto_no_treino_…` | `run` não passa a data ao `load_all` | 2 falharam ✅ |
| `test_run_inicio_do_treino_nao_anterior_a_validacao_da_erro` | a verificação da data desaparece | 4 falharam ✅ |
| `test_linha_de_comando_com_inicio_por_defeito_grava_em_data_processed` | sem `--saida` dá sempre erro (partia o `make preparar`) | 2 falharam ✅ (antes deste teste, os outros 62 continuavam verdes) |
| o mesmo | a linha de comando grava noutra pasta | 2 falharam ✅ |

O agente que escreveu os testes também injetou 12 erros numa cópia do projeto, e cada um fez falhar o teste correspondente.

### Números que provam que funciona

| Medida | Antes | Depois | Fonte |
|---|---|---|---|
| exemplos de treino (data por defeito) | 478 384 | 478 384 | funil do `prepare.py` com os dados reais |
| contratos de validação / de teste | 23 945 / 37 361 | 23 945 / 37 361 | mesmo funil |
| SHA-256 dos 3 ficheiros | `data/processed/` | iguais | `sha256sum` |

As janelas desde 2019 e desde 2025 não foram medidas com os dados reais: isso é o E4-S2.

### Revisão de código

Dois revisores independentes (agentes novos, sem o contexto da tarefa), sobre `main` contra a árvore de trabalho.

**1.ª revisão: pronto para merge com correções menores.** Os três critérios estão cumpridos. Não há problemas críticos nem importantes. Estragou 6 linhas numa cópia do projeto, e os testes falharam em todas.
- Corrigido:
  - a garantia "validação e teste iguais em todas as janelas" depende de nenhum `idcontrato` aparecer em dois ficheiros, e nada no código o verifica. O revisor contou 0 casos com início em 2019 ou em 2025. A comparação entrou como verificação do E4-S2 no `backlog.md`;
  - o `since` do `load_one` passa a ler-se no momento da chamada.
- Não corrigido, e porquê: `--saida data/processed` com outra data é aceite e sobrescreve. É um pedido explícito de quem corre o comando, e a proteção serve para evitar o engano, não a escolha.

**2.ª revisão** (depois da mudança no `load_one`): <por preencher>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
