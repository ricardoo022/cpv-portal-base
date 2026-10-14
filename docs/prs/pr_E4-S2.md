## E4-S2 e E4-S6 · A janela de treino desde 2022 é a certa?

## Em poucas palavras

O modelo aprende com contratos publicados desde janeiro de 2022. Este PR testa se começar noutra data dá melhores sugestões: desde 2025 (menos dados, mais recentes) ou desde 2019 (mais dados, mais antigos). Na primeira medição ganhou a janela desde 2019. Mas essa medição deixava o treino parar sozinho, e o treino desde 2022 parava cedo demais. Com o mesmo treino do modelo que usamos (24 passagens pelos dados), 2019 e 2022 empatam, e **o treino fica desde 2022**. Nada muda para quem usa o projeto: o modelo em vigor é o mesmo.

| Janela de treino | Exemplos | Top-5 código, paragem automática | Top-5 código, 24 épocas |
|---|---|---|---|
| desde 2025 | 116 996 | 63,28% | não medido |
| desde 2022 (em vigor) | 478 234 | 73,46% | 75,09% |
| desde 2019 | 785 542 | 75,38% | 75,40% |

(Top-5 no código completo: o código certo está entre as 5 sugestões. Medido nos 20 150 contratos da validação cujo texto não aparece no treino desde 2019.)

## Porquê

A decisão 1 escolheu treinar desde 2022 olhando só para quantos contratos da validação têm um código com exemplos suficientes no treino. Faltava confirmá-lo com um modelo, antes de treinar o BERTimbau e o Albertina, porque a janela muda os dados com que eles treinam, e repetir treinos em GPU é a parte cara.

Item: [`E4-S2`](docs/backlog.md) e o spike que dele saiu, [`E4-S6`](docs/backlog.md) · Desbloqueia: os treinos em GPU do E5 (que esperavam pela janela).

**Termos usados abaixo:**
- **Janela de treino:** o período de onde vêm os exemplos de treino. As três janelas acabam em setembro de 2025; só muda o início.
- **Época:** uma passagem completa por todos os exemplos de treino. O modelo em vigor treina 24 épocas fixas. A "regra de paragem por defeito" do scikit-learn para sozinha quando o erro no treino quase deixa de descer.
- **Bootstrap:** tira-se 1 000 vezes, à sorte, uma amostra dos contratos de validação e vê-se em quantas uma opção ganha. Uma diferença só conta se uma opção ganhar em pelo menos 95% das amostras.
- **Pré-registo:** escrever as opções e a regra de escolha num ficheiro (`experiencias/*.yaml`) antes de ver resultados, para não ajustar a regra ao que sair.

## O que mudou no código

### `scripts/correr_experiencia.py`

- `_ler_janela` (`scripts/correr_experiencia.py:102`): valida o novo modo de experiência `janela` (o mesmo treino em várias datas de início). Exige pelo menos 2 janelas, cada uma com `nome`, `inicio` e `pasta`, nomes distintos e datas por ordem decrescente (da janela mais curta, a que fica em empate, para a mais longa).
- `_data` (`scripts/correr_experiencia.py:88`): uma data do YAML tem de vir como texto `AAAA-MM-DD`. Sem aspas, o PyYAML lê-a como data, e o erro pede as aspas.
- `comandos` (`scripts/correr_experiencia.py:258`): recusa o modo `janela`, porque estas janelas preparam-se e treinam-se à mão no pod.

```python
inicios = [j["inicio"] for j in out["janelas"]]
if any(a <= b for a, b in zip(inicios, inicios[1:])):
    raise ValueError(f"janelas: as datas de início têm de ser distintas e por ordem decrescente "
                     f"(da janela mais curta para a mais longa): {inicios}")
```
A primeira janela é a opção em vigor de partida; a ordem garante que, em empate, fica a mais curta.

### `src/prepare.py`

- `check_sets` (`src/prepare.py:172`): a verificação "'None' ou 'nan' escrito no texto" passa a olhar só para pedaços inteiros do texto entre separadores.

```python
pedacos = x["texto"].str.split(SEP, regex=False).explode().str.strip()
check(not pedacos.isin(["None", "nan"]).any(), f"'None' ou 'nan' escrito no texto em {nome}")
del pedacos  # é uma cópia do texto partido; não fica em memória (CLAUDE.md)
```
O texto de entrada junta campos com `" | "`. Um campo vazio escrito por engano como texto aparece como um pedaço inteiro (`"objeto | None"`). A regra antiga procurava a palavra em qualquer sítio e parava a preparação desde 2019 por causa de uma gralha real de 2020 no Portal BASE:

| Texto | Regra antiga | Regra nova |
|---|---|---|
| `jardins \| None` | para | para |
| `jardins \| nan \| AQ` | para | para |
| `águas domésticas nan Quinta dos Farias` | **para** (falso alarme) | passa |
| `jardins \| None of the above` | **para** (falso alarme) | passa |

Os ficheiros desde 2022 não mudam (md5 iguais, ver as provas).

## Outras mudanças

- `experiencias/e4_s2.yaml`: o pré-registo do E4-S2. Três janelas (desde 2025, 2022 e 2019), N = 10, `alpha` = 0,0000001, a regra de paragem por defeito e τ = 0,25. Decide nos contratos com texto novo para o treino desde 2019. A janela desde 2022 reusa o treino do E4-S1.
- `experiencias/e4_s6.yaml`: o pré-registo do E4-S6. A janela desde 2019 com 24 épocas contra o modelo em vigor; em empate fica a de 2022.
- `notebooks/09_janela.ipynb` (novo): a decisão do E4-S2 e uma "análise completa" com todas as medidas dos spikes anteriores, mais as verificações do controlo e dos funis.
- `notebooks/10_janela_24_epocas.ipynb` (novo): o notebook 09 com o YAML do E4-S6.
- `metrics/e4_s2/` e `metrics/e4_s6/` (novos): sugestões, `treino.json`, logs do pod (com o `/usr/bin/time -v` de cada passo), md5 dos ficheiros preparados no pod e `pod_info.json`.
- `tests/unit/test_correr_experiencia.py`: os YAML do E4-S2 e do E4-S6 escritos à mão (falham se alguém mudar a lista), e 16 casos de YAML de janelas inválido.
- `tests/unit/test_prepare.py` e `tests/integration/test_pipeline.py`: os casos da tabela acima, do `.xlsx` ao `treino.parquet`.
- `docs/qualidade_dados.md`: secções 18 (E4-S2) e 19 (E4-S6), e a decisão 1 confirmada.
- `docs/backlog.md`: E4-S2 e E4-S6 feitos, com as verificações do E4-S2 marcadas com o que se cumpriu e o que não; estado do E4.
- `docs/workflow.md`: a caixa "Confirmar a janela de treino" marcada.
- `Makefile`: `make janela` e `make janela-24` voltam a correr os notebooks 09 e 10.
- `.gitignore`: `data/janelas/` fica fora do git.
- `CLAUDE.md`, `data/readme.md`, `experiencias/readme.md` e `notebooks/readme.md`: os novos ficheiros e o resultado.

### Ficou igual de propósito

- O `TRAIN_START` (2022-01-01) e o `data/processed/`: a janela fica desde 2022.
- A régua (E4-T2) e os resultados do E4-H1 não se recalculam, porque o modelo em vigor não muda.
- A grelha N × épocas × τ não se repetiu na janela desde 2019. O `e4_s2.yaml` previa-a se ganhasse outra janela; o E4-S6 substituiu-a por um só treino, para confirmar primeiro se a janela ganhava com o modelo em vigor. Como empatou, a grelha deixou de ser precisa.
- O E5 no `docs/backlog.md` não mexe: o épico continua em escrita.

## Validação e provas

### Critérios de aceitação

**E4-S2** (copiados do backlog):

| Critério | Cumprido? | Prova |
|---|---|---|
| Antes de correr: as janelas, o N e a regra das épocas estão em `experiencias/e4_s2.yaml`, escrito antes de correr, e um teste falha se o YAML for diferente da lista deste item. | ⚠️ em parte | O teste existe: `test_o_yaml_do_e4_s2_tem_as_opcoes_do_backlog` falha se se mudar o YAML (visto pelo revisor). O YAML foi escrito antes de criar o pod, mas só entrou no git depois dos treinos (commit `fc4996c` às 09:23 UTC; treinos das 08:46 às 09:08 UTC). O git não o prova. |
| Antes de correr: a lista exata dos comandos de treino (caminhos e parâmetros) é mostrada e aprovada antes de os lançar. Todos têm `--regra-por-defeito` e nenhum caminho aponta para o teste. | ❌ (aceite pelo utilizador) | Os comandos não foram mostrados antes. A sessão trabalhava sobre uma cópia antiga do `backlog.md` (gravada pelo editor por cima da do git) que não tinha estas verificações. Depois de correr: os `Command being timed` dos `metrics/e4_s2/pod_*.log` têm todos `--regra-por-defeito`, e nenhum caminho aponta para o teste. |
| Antes de correr: a impressão digital (SHA-256) de cada ficheiro de treino e de validação é igual no portátil e no pod. A do treino desde 2022 é igual à do `treino.parquet` atual. | ⚠️ em parte | Comparou-se com md5, não SHA-256 (`metrics/e4_s2/pod_md5.txt`). O treino e a validação desde 2022 preparados no pod têm o mesmo md5 que `data/processed/`. As janelas desde 2019 e desde 2025 foram preparadas no pod; as cópias no portátil têm o mesmo md5. |
| Depois de correr (controlo): a janela desde 2022 é o mesmo treino do E4-S4 (…), por isso dá sugestões 100% iguais às de `metrics/e4_s4/alpha_1e-7/`, com τ = 0 e τ = 0,25. (…) | ✅ | Célula "Verificações do E4-S2" do notebook 09 → 23 945 contratos, 100% iguais, nos dois τ. |
| Depois de correr: o número de exemplos de treino no `treino.json` de cada janela é igual ao do funil da preparação dessa janela. | ✅ | Mesma célula → 117 357, 478 384 e 785 660, iguais à linha "8b" de cada `pod_preparar_*.log`. |
| Resultado: uma tabela com as três janelas; a decisão 1 confirmada ou mudada no `qualidade_dados.md`. | ✅ | Notebook 09; secções 18 e 19 e decisão 1. |
| Também: medir o pico de memória do treino desde 2019. | ✅ | 17,9 GB no treino e 4,3 GB na preparação (`/usr/bin/time -v`, notebook 09, secção "Memória"). |

**E4-S6:**

| Critério | Cumprido? | Prova |
|---|---|---|
| Regra escrita antes de correr, num YAML | ✅ | `experiencias/e4_s6.yaml` no commit `fc4996c` (09:23:44 UTC), antes do treino (09:27:15 UTC); `test_o_yaml_do_e4_s6_tem_as_opcoes_do_backlog`. |
| Tabela das duas opções, com a análise completa do E4-S2 | ✅ | Notebook 10; secção 19. |
| Se não ganhar, a decisão 1 fica desde 2022 | ✅ | Diferença +0,31 pontos, desde 2019 melhor em 94,8% das amostras (< 95%): empate, fica desde 2022. |
| Custo em tempo de pod e em dólares | ✅ | `metrics/e4_s6/pod_info.json`: cerca de 63 min a 0,32 $/h, cerca de 0,34 $. |

Caixas do `docs/backlog.md` marcadas como `[x]` neste PR: as verificações do controlo e do funil do E4-S2. As outras três ficam `[ ]`, com o que se cumpriu e o que não; o utilizador aceitou-as por cumprir a 14 de outubro de 2026, e o título do item di-lo.

### Testes focados

`.venv/bin/pytest -q tests/unit/test_correr_experiencia.py tests/unit/test_prepare.py tests/integration/test_pipeline.py tests/integration/test_correr_experiencia_pipeline.py` → 152 passed.

### O comportamento que já existia continua igual

- Suíte inteira: `.venv/bin/pytest -q` → 427 passed (antes do trabalho, no pod: 395 passed e 22 saltados por falta de `data/processed/`). O revisor também correu a suíte inteira no portátil: 427 passed.
- Os ficheiros desde 2022 não mudam com a verificação nova: o treino e a validação desde 2022 preparados no pod têm o mesmo md5 que `data/processed/` (`295d1a89…` e `c514da9c…`).
- CI: `ae7a968`, `6bdd5a5` e `37440d2` → success (o commit que só atualiza esta linha volta a correr antes do merge).

### Os testes novos falham com o código estragado

| Teste | Estrago introduzido | Resultado com o estrago |
|---|---|---|
| `test_palavra_nan_no_meio_de_uma_frase_nao_para_a_preparacao`, `test_check_sets_aceita_none_ou_nan_dentro_de_uma_frase` | voltar à regra antiga (`\b(?:None\|nan)\b` em qualquer sítio) | 3 falharam ✅ |
| `test_campo_escrito_como_none_ou_nan_para_a_preparacao`, `test_check_sets_apanha_none_escrito_no_texto` | verificação desligada (`check(True, …)`) | 7 falharam ✅ |
| `test_check_sets_apanha_none_escrito_no_texto[jardins \|  nan ]` | tirar o `.str.strip()` | falhou ✅ |
| `test_yaml_das_janelas_invalido_da_erro` (ordem crescente, datas repetidas) | verificação da ordem desligada | 2 falharam ✅ |
| `test_data_da_janela_sem_aspas_da_erro_a_pedir_aspas` | aceitar `datetime.date` | falhou ✅ |
| `test_yaml_das_janelas_invalido_da_erro` (janela repetida) | verificação de nomes distintos desligada | falhou ✅ |
| `test_o_yaml_do_e4_s2_tem_as_opcoes_do_backlog` | `epocas: null` passa a `24` no YAML | falhou ✅ (revisor) |

### Números que provam que funciona

Tudo na validação (outubro de 2025), nunca no teste. Grupo da decisão: 20 150 contratos com texto novo para o treino desde 2019.

| Medida | desde 2025 | desde 2022 | desde 2019 | Fonte |
|---|---|---|---|---|
| Top-5 código, regra de paragem por defeito (E4-S2) | 63,28% | 73,46% | 75,38% | notebook 09 |
| Top-5 classe, regra de paragem por defeito | 72,70% | 80,73% | 81,98% | notebook 09 |
| Top-5 código, 24 épocas (E4-S6) | — | 75,09% | 75,40% | notebook 10 |
| Top-5 classe, 24 épocas | — | 82,13% | 81,98% | notebook 10 |
| Treino (min) / pico de memória do processo | 0,5 / 2,1 GB | 31,2 / 10,5 GB (24 épocas) | 57,6 / 17,9 GB (24 épocas) | `treino.json` |

- **E4-S2:** a de 2022 ganha à de 2025 (+10,18 pontos) e a de 2019 ganha à de 2022 (+1,92); as duas diferenças contam em 100% das amostras.
- **E4-S6:** a de 2019 fica +0,31 pontos à frente, mas só ganha em 94,8% das amostras: é empate e fica a de 2022. Na classe fica −0,15 (não conta).
- **Porque mudou:** com a regra por defeito, o treino desde 2022 parava na época 12 e perdia 1,6 pontos. Com estes hiperparâmetros, a vantagem do E4-S2 era em boa parte de treino mais completo; a desde 2019 ainda ganha no F1 macro do código (99,3% das amostras) e nos códigos de nível 5 a 7. Lição que generaliza: ao comparar opções de dados, cada uma tem de treinar até ao seu melhor.

**Limitações** (escritas nas secções 18 e 19):
- o pré-registo do E4-S2 não está provado pelo git;
- as 24 épocas foram afinadas na janela de 2022, na mesma validação, o que lhe dá vantagem;
- os tempos de 2022 e 2019 vêm de pods diferentes;
- os logs da preparação no pod trazem as contagens do funil do teste (quantos contratos e quantos já vistos). São contagens da preparação, não resultados, e nenhuma decisão as usou.

**Custo:** E4-S2 cerca de 0,36 $ (pod de 16 processadores, 34 min); E4-S6 cerca de 0,34 $ (pod de 8 processadores, 63 min). Os dois pods foram apagados (lista de pods vazia).

### Revisão de código

Revisor independente sobre `149e646..ae7a968`: **pronto, com correções**. Não encontrou nada crítico nem fuga de dados.
- Corrigido em `6bdd5a5`:
  - o 1.º critério do E4-S2 deixou de estar `[x]`;
  - o controlo e os funis passaram a ter células no notebook 09;
  - saíram os números do portátil sem fonte;
  - `del pedacos` no `prepare.py`;
  - frases sobre o plano da grelha substituído, a vantagem das 24 épocas e os pods diferentes.
- Não corrigido, e porquê: os logs com as contagens do funil do teste ficam como estão (são saída da preparação, que sempre as mostrou; nenhuma decisão as usou).

Segundo revisor (agente novo) sobre `149e646..6bdd5a5`: **pronto, com correções**. Nada crítico; conferiu todos os números das secções 18 e 19 contra as saídas dos notebooks, e estragou 8 vezes o código numa cópia (todos os testes falharam como deviam).
- Corrigido no commit seguinte:
  - os dois desvios ficam escritos como **aceites pelo utilizador a 14 de outubro de 2026**: as 3 verificações do E4-S2 por cumprir (no título do item) e a troca da grelha pré-registada por um só treino de 24 épocas (E4-S6 e secção 19);
  - a conclusão ficou mais moderada ("em boa parte de treino mais completo, com estes hiperparâmetros"), porque a desde 2019 ainda ganha no F1 macro do código e nos níveis 5 a 7;
  - "grelha de épocas na janela desde 2019" entrou nas ideias para depois do backlog;
  - os notebooks verificam o `grupo` do YAML e o mês de início de cada treino;
  - notas sobre o `teste.parquet` apagado no portátil, a correção do `check_sets` que correu no pod antes de estar no git, a chave `informacao_epocas` retirada antes de qualquer treino, e os números do `pod_info.json` (escrito à mão, como nos spikes anteriores).
- Não corrigido, e porquê: a memória do novo `check_sets` (o revisor mediu +166 MB, abaixo do pico de 2,7 GB da preparação; não é preciso mudar).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
