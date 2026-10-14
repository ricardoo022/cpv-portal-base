## E5-T1 · Medir o comprimento dos textos em *tokens*

## Em poucas palavras

Antes de tunar o BERTimbau e o Albertina é preciso escolher o comprimento máximo de cada texto: o número de *tokens* a partir do qual o texto é cortado. Um *token* é um pedaço de texto que o modelo conhece: "Câmara" é um *token*, "Aquisição" parte-se em três (`Aqui` `##si` `##ção`). Este PR conta os *tokens* de cada texto do treino com o tokenizador de cada modelo e aplica a regra escrita antes de medir: o menor de 128, 256 e 512 em que cabem inteiros pelo menos 99% dos textos de treino. Deu **256 *tokens* para os dois modelos**. O nome exato de cada modelo, a versão e o comprimento ficam guardados num só sítio, que o código de treino vai ler.

| Modelo | Cabem em 128 | Cabem em 256 | Escolha |
|---|---|---|---|
| BERTimbau | 98,17% | 99,81% | 256 |
| Albertina | 96,21% | 99,63% | 256 |

## Porquê

O comprimento máximo decide quanto custa cada treino em GPU, e um texto cortado perde o que estava no fim. A única medição que existia era em palavras (12 de mediana por campo, só em 2025/2026), e um modelo não lê palavras, lê *tokens*.

Item: [`E5-T1`](docs/backlog.md) · Desbloqueia: E5-T4 e E5-T5 (o primeiro treino de cada modelo)

## O que mudou no código

### `src/bert_models.py` (novo)

- `MODELOS` (`src/bert_models.py:14`): o nome de cada modelo no Hugging Face e a revisão, que é o commit exato do repositório do modelo. Se os autores mudarem o tokenizador, o projeto continua a usar a versão medida.

```python
MODELOS = {
    "bertimbau": dict(
        nome="neuralmind/bert-base-portuguese-cased",
        revisao="94d69c95f98f7d5b2a8700c420230ae10def0baa",
    ),
    "albertina": dict(
        nome="PORTULAN/albertina-100m-portuguese-ptpt-encoder",
        revisao="d15e086005cdab26d746d9113984cc654439df2b",
    ),
}
```
Estas linhas fixam os dois modelos. O Albertina é a variante PT-PT, e não a PT-BR.

- `CANDIDATOS`, `QUOTA` e `COMPRIMENTO_MAXIMO` (`src/bert_models.py:27-31`): a regra do backlog e o resultado.

```python
CANDIDATOS = (128, 256, 512)
QUOTA = 0.99
COMPRIMENTO_MAXIMO = {"bertimbau": 256, "albertina": 256}
```
O notebook lê `CANDIDATOS` e `QUOTA` para escolher, e para com erro se `COMPRIMENTO_MAXIMO` não for o que escolheu.

### `notebooks/11_comprimento_tokens.ipynb` (novo)

Lê só a coluna `texto` do treino e da validação. Carrega os dois tokenizadores na revisão fixa, sem os pesos dos modelos e sem GPU, e conta os *tokens* de cada texto já com os 2 especiais (`[CLS]` no início, `[SEP]` no fim). A escolha:

```python
cabe = [L for L in bm.CANDIDATOS if (n <= L).mean() >= bm.QUOTA]
escolha[m] = cabe[0] if cabe else max(bm.CANDIDATOS)
...
assert bm.COMPRIMENTO_MAXIMO == escolha, f"atualizar src/bert_models.py: {escolha}"
```
Escolhe o menor candidato em que cabe a quota dos textos de treino (`n` é o comprimento de cada um), ou 512 se nenhum chegar, e confirma que é o valor guardado. Outras duas guardas: a função que conta dá, na frase de exemplo, o mesmo número que o tokenizador com os especiais, e cada modelo aceita pelo menos 512 posições (`max_position_embeddings`, lido do `config.json` na revisão fixa).

## Outras mudanças

- `docs/qualidade_dados.md`: secção 20 nova, com a tabela de treino e validação, a escolha e como a ler.
- `docs/workflow.md`: caixa nova `[x]` da fase 4 para a medição do E5-T1.
- `docs/backlog.md`: E5 passa a "em curso", E5-T1 "feito", as 4 caixas marcadas e um parágrafo com o resultado.
- `requirements.txt`: `transformers==5.17.0` e `tokenizers==0.23.2`, versões com mais de 2 semanas que funcionam sem `torch`.
- `Makefile`: alvo `make tokens`, que volta a correr o notebook 11 (cerca de 3 minutos).
- `CLAUDE.md`: o `make tokens`, o E5 em curso e um parágrafo sobre `src/bert_models.py` (nomes, revisão, 256, e a nota de que o tokenizador do BERTimbau não corta aos 512 sem `max_length`).
- `tests/unit/test_bert_models.py` (novo): os nomes escritos à mão, a revisão é um commit (40 caracteres hexadecimais, não `main`), a regra e o 256 de cada modelo. Guarda só as constantes (uma revisão com o formato certo mas inexistente passa); a medição é guardada pelos `assert` do notebook, que precisam de rede e de `data/` e não correm no CI. Está escrito no `tests/unit/readme.md`.
- `notebooks/readme.md`, `src/readme.md` e `tests/unit/readme.md`: uma linha para cada ficheiro novo.
- `docs/prs/pr_E5-T1.md` e `docs/prs/readme.md`: esta descrição e a sua linha na tabela.

### Ficou igual de propósito

- O `prepare.py` e os ficheiros de `data/processed/`: o comprimento só decide onde o treino corta o texto, e isso faz-se no código de treino (E5-T2).
- A regra dos códigos raros não se aplica na contagem: o N do BERT só se escolhe no E5-S1, e a regra tira poucos exemplos.
- O `huggingface_hub` não fica fixado no `requirements.txt` (instalado: 1.33.0): vem com o `transformers`, e a revisão de cada modelo já está fixa.
- O `torch` não entra no `requirements.txt`: contar *tokens* não o precisa, e entra com o código de treino (E5-T2).
- Não se mediu o custo do treino: o custo de uma época mede-se no ensaio curto do E5-T4.

## Validação e provas

### Critérios de aceitação

| Critério (copiado do backlog) | Cumprido? | Prova (comando → resultado) |
|---|---|---|
| Para cada modelo, com o seu próprio tokenizador, a percentagem dos textos de treino que cabem inteiros em 128, 256 e 512 *tokens*. A contagem inclui os *tokens* especiais que o modelo acrescenta a cada texto, porque também ocupam lugar. | ✅ | `make tokens` → BERTimbau 98,17 / 99,81 / 99,98%; Albertina 96,21 / 99,63 / 99,96%. A célula dos especiais confirma 2 por texto nos dois (`assert`). |
| A mesma tabela na validação, só como informação (a regra usa o treino). | ✅ | Mesma célula → BERTimbau 94,22 / 98,10 / 99,91%; Albertina 91,84 / 97,81 / 99,92% |
| O comprimento escolhido para cada modelo, pela regra acima, está escrito numa secção nova do [`qualidade_dados.md`](docs/qualidade_dados.md), com os números vindos do notebook. | ✅ | Secção 20; a última célula do notebook imprime 256 para os dois e confirma `src/bert_models.py` |
| Os nomes exatos dos dois modelos (os do Hugging Face) estão escritos e são os mesmos que o treino vai usar. | ✅ | Secção 20 e `src/bert_models.py:14`, o único sítio de onde o notebook e o treino os leem |

Caixas do `docs/backlog.md` marcadas como `[x]` neste PR: as 4 do E5-T1.

### Testes focados

`.venv/bin/pytest -q tests/unit/test_bert_models.py` → 3 passed

### O comportamento que já existia continua igual

- Suíte inteira: `.venv/bin/pytest -q` → 492 passed, 1 skipped (com `data/processed/`, no commit `9725659`). A suíte não foi corrida antes do trabalho; o diff só junta os 3 testes do `test_bert_models.py` e não mexe em nenhum teste existente.
- Nenhum ficheiro do pipeline (`prepare.py`, `tfidf.py`, `evaluation.py`, …) mudou.
- `.venv/bin/pip check` → No broken requirements found (o `transformers` convive com o `mlflow` fixado).
- CI do commit `9725659` (o último com código; o seguinte só junta esta descrição): success

### Os testes novos falham com o código estragado

| Teste | Estrago introduzido | Resultado com o estrago |
|---|---|---|
| `test_comprimento_escolhido_para_cada_modelo_e_um_candidato` | `"albertina": 256` → `128` | falhou ✅ |
| `test_os_dois_modelos_com_nome_e_revisao_fixa` | revisão do BERTimbau → `"main"` | falhou ✅ |
| `test_os_dois_modelos_com_nome_e_revisao_fixa` | `ptpt-encoder` → `ptbr-encoder` | falhou ✅ |
| `test_regra_do_comprimento_e_a_do_backlog` | `QUOTA = 0.99` → `0.95` | falhou ✅ |
| guarda dos especiais no notebook | contar com `add_special_tokens=False` (frase de exemplo: 16 em vez de 18 no BERTimbau, 33 em vez de 35 no Albertina) | falharia ✅ (verificado fora do notebook, com os mesmos tokenizadores) |
| `assert` da última célula do notebook (pelos revisores) | `cabe[0]` → `cabe[-1]`; `COMPRIMENTO_MAXIMO["albertina"] = 512` | falhou ✅ |

### Números que provam que funciona

Do notebook 11 (treino e validação, nunca o teste):

| Medida | BERTimbau | Albertina | Fonte |
|---|---|---|---|
| Mediana de *tokens* por texto (treino) | 31 | 40 | célula "Quantos textos cabem inteiros" |
| Percentil 99 (treino / validação) | 154 / 316 | 190 / 372 | mesma célula |
| *Tokens* por palavra (mediana, treino) | 2,05 | 2,72 | mesma célula |
| Textos de treino cortados com 256 | 890 | 1 786 | célula "Os textos mais compridos" |

O Albertina usa o tokenizador original do DeBERTa, feito para inglês (diz o cartão do modelo), e por isso parte mais as palavras portuguesas: "manutenção" dá 1 *token* no BERTimbau e 4 no Albertina. Limitação escrita na secção 20: segundo o cartão, o Albertina foi pré-treinado com textos cortados a 128 *tokens*, e não se sabe se perde qualidade entre 128 e 256.

### Revisão de código

Duas revisões independentes (agentes novos, só com o id, a base e o topo):

- **1.ª, sobre `7ed0b85..1874ebf`:** "pronto para merge, com correções menores opcionais"; nenhum problema crítico ou importante. Corrigido em `e65424d`:
  - a ligação ao cartão do Albertina;
  - o custo por texto só com *padding* dinâmico;
  - a frase do custo por lote;
  - a guarda dos *tokens* especiais no notebook;
  - o readme dos testes;
  - o `CLAUDE.md`;
  - um espaço a mais no fim de uma linha.
- **2.ª, sobre `7ed0b85..e65424d`:** "pronto para merge: sim, com correções menores opcionais". Corrigido em `9725659`:
  - a frase do custo por lote, que dava como certo "pelo menos o dobro";
  - as 512 posições, que agora saem de uma célula;
  - a caixa do `workflow.md`.
- **Não corrigido, e porquê:**
  - `CANDIDATOS` e `QUOTA` estão em `src/` e só o notebook os usa: são a regra pré-registada e ficam junto do valor escolhido.
  - O `huggingface_hub` não fica fixado: a revisão de cada modelo já está.
  - Os testes não guardam a medição: precisaria de rede e de `data/` no CI.
- As correções de `9725659` são só texto e uma célula que lê o `config.json`: não houve terceira revisão.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
