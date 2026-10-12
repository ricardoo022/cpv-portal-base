## E4-T3 · Testes ao treino que correm no CI

## Em poucas palavras

Este PR acrescenta testes que apanham um treino com erro, mesmo quando o treino corre sem dar erro nenhum. A ideia é dar ao modelo um treino minúsculo e exigir que o decore: com 13 frases sem palavras em comum, se o modelo não acerta o código de cada frase, há um erro no caminho. Outro teste confirma que o modelo só sugere códigos que existem no treino. Os testes novos correm em cerca de 1 s no portátil, com dados inventados, e a suíte inteira demora 8 s no CI, por isso cada PR fica protegido sem precisar dos dados reais nem do pod. Não muda nenhum código do modelo: só testes e documentação.

| Erro injetado de propósito no código | Corre sem dar erro? | Os testes novos apanham? |
|---|---|---|
| Etiquetas baralhadas antes do treino | sim | sim |
| O comando de treino ignora N = 1 e usa N = 2 | sim | sim |
| O treino lê a coluna com todos os códigos em vez do 1.º | sim | sim |
| O treino usa os códigos originais em vez das etiquetas dos códigos raros | sim | sim |

## Porquê

Um treino com um erro pode correr sem dar erro nenhum e só produzir um modelo mau. Por exemplo, se as etiquetas ficarem desalinhadas dos textos, o treino termina, grava o modelo e o top-5 só desce, o que parece "o modelo é assim". Estes testes apanham isso antes de se gastar tempo e dinheiro com dados reais no pod. São também a rede de segurança do spike seguinte (E4-S5), que vai mexer no treino.

Item: [`E4-T3`](docs/backlog.md) · Desbloqueia: E4-S5

## O que mudou no código

Nenhum ficheiro de `src/` mudou. Só testes.

**Termo novo, sobreajuste (*overfitting*):** o modelo decorar o treino em vez de aprender a generalizar. Normalmente é um defeito; aqui é um teste. Se o modelo nem consegue decorar 13 frases bem distintas, o treino tem um erro.

### `tests/conftest.py`

- `TEMAS_DECORAR` e o fixture `temas_decorar` (`tests/conftest.py:95`): 7 códigos, cada um com frases sem palavras de conteúdo em comum. Os testes unitários e de integração usam o mesmo dicionário.

```python
TEMAS_DECORAR = {
    "90911200": ["Limpeza das escolas", "Lavagem dos pavilhões escolares"],
    "33600000": ["Medicamentos hospitalares", "Antibióticos e analgésicos"],
    ...
    "66512100": ["Seguro de acidentes pessoais"],
}
```
O último código tem uma só frase: com N = 1 fica como etiqueta, com N = 2 sairia. Isto deixa o teste distinguir N = 1 de N = 2.

### `tests/integration/test_tfidf_pipeline.py`

- `test_treino_minusculo_pelo_pipeline_acerta_todos_os_textos_no_top_1` (`tests/integration/test_tfidf_pipeline.py:322`): escreve `.xlsx` inventados, corre a preparação, treina com o comando de treino (N = 1), lê o modelo gravado em disco e exige que o 1.º código sugerido para cada frase do treino seja o seu.

```python
treino = [(2024, f"d-{c}-{i}", datetime(2024, 3, 1), t, None, None, f"{c}-0 - Tema\n15800000-0 - Outro")
          for c, textos in temas_decorar.items() for i, t in enumerate(textos)]
...
top1 = tt.tfidf.suggest(joblib.load(modelo), treino_df["texto"], tau=0.0).str[0]
errados = top1[top1 != treino_df["codigo"]]
```
Cada contrato tem um 2.º código CPV, para que ler a coluna errada (todos os códigos em vez do 1.º) dê um resultado diferente. O `tau=0.0` ordena só pela pontuação do modelo: a correção das etiquetas frequentes (τ) mexeria na ordem e o teste deixaria de medir só o que o modelo aprendeu.

- `test_sugestoes_gravadas_sao_as_etiquetas_do_treino_gravado` (`tests/integration/test_tfidf_pipeline.py:306`), com N = 6 e N = 10: as sugestões gravadas só podem ter etiquetas contadas a partir do `treino.parquet`, não a partir do que o modelo diz ter.

```python
permitidas = set(collapse_rare_codes(treino_df["codigo"], n_min).dropna())
...
    assert "33140000" in permitidas and "33141100" not in permitidas  # pré-condição
```
Com N = 10, dois códigos raros da classe 3314 sobem juntos para a classe 33140000. Se o treino usasse os códigos originais, apareceria 33141100 nas sugestões e o teste falharia.

### `tests/unit/test_tfidf.py`

- `test_treino_minusculo_sem_codigos_raros_acerta_todos_os_textos_no_top_1` (`tests/unit/test_tfidf.py:291`): o mesmo teste de sobreajuste, só com a função de treino, sem `.xlsx`. Serve para localizar um erro depressa: se este falhar, o erro está no modelo e não no caminho até ele.

```python
assert lb.collapse_rare_codes(codigo, n_min=1).equals(codigo)
assert lb.collapse_rare_codes(codigo, n_min=2).isna().any()
m = tfidf.train(texto, codigo, n_min=1, min_df=1, alpha=tfidf.ALPHA)
```
As duas pré-condições confirmam que N = 1 desliga a regra dos códigos raros e que N = 2 mudaria o resultado. O `alpha` é o do modelo em uso (1e-7, regularização fraca).

- `test_as_sugestoes_sao_sempre_etiquetas_de_treino` (`tests/unit/test_tfidf.py:68`), reescrito e agora com N = 2 e N = 3: pede tantas sugestões quantas as etiquetas do modelo, para todas aparecerem. Assim uma etiqueta a mais ou a menos é apanhada, sejam quais forem os textos. Com N = 3 há exemplos que saem do treino e também não podem aparecer.

```python
permitidas = set(lb.collapse_rare_codes(codigo, n_min).dropna())
sug = tfidf.suggest(m, pd.concat([texto, pd.Series(["xyz", ""])]), k=len(m.classes_))
assert all(set(s) == permitidas for s in sug)
```

## Outras mudanças

- `docs/backlog.md`: E4-T3 marcado como feito, com os 4 critérios a `[x]` e uma nota sobre o que os testes não guardam; o estado do E4 e o "segue-se" atualizados.
- `docs/backlog.md`, **commit à parte, não é do E4-T3**: duas notas no spike E4-S5, vindas de uma revisão do E4-H1 por um agente. (1) A paragem automática do treino pode chegar cedo demais nas etiquetas raras, porque o limite é 0,001 × o número de exemplos na soma do erro. (2) Com a taxa de aprendizagem por defeito, o `alpha` também muda o tamanho de cada passo, por isso comparar valores de `alpha` mistura efeitos. As duas estão escritas como suspeitas; o E4-S5 confirma-as ou não.
- `docs/prs/pr_E4-T3.md`: esta descrição.

### Ficou igual de propósito

- O código do modelo (`src/tfidf.py`, `src/train_tfidf.py`, `src/suggest.py`): esta tarefa só acrescenta testes. Nenhum teste encontrou um erro no código.
- Os testes não guardam o `alpha` nem o número de épocas. Com frases tão distintas, o modelo acerta com qualquer `alpha` de 1e-7 a 100 e com uma só época de treino. Um revisor tentou dados mais difíceis e o resultado foi o mesmo. O `alpha` por defeito já tem o seu teste (`test_train_usa_o_alpha_escolhido_por_defeito`); as épocas medem-se na validação, no E4-S5.
- Os testes com dados reais (`*_dados_reais.py`) não mudaram: continuam a correr só no portátil.

## Validação e provas

### Critérios de aceitação

| Critério (copiado do backlog) | Cumprido? | Prova (comando → resultado) |
|---|---|---|
| Com um treino minúsculo inventado (poucos textos bem distintos), sem a regra dos códigos raros (N = 1) e com regularização fraca, o modelo acerta em todos os textos com que treinou, no top-1. Se não acertar, há um erro no treino. | ✅ | `.venv/bin/pytest -q tests/unit/test_tfidf.py tests/integration/test_tfidf_pipeline.py -k "minusculo"` → passa; os dois testes de sobreajuste, unitário e de integração |
| As sugestões só contêm etiquetas que existem no treino. | ✅ | `-k "sempre_etiquetas or gravadas"` → 4 casos passam (N = 2, 3 no unitário; N = 6, 10 na integração) |
| Os testes correm no CI em menos de 1 minuto, sem dados reais. | ✅ | `--durations=0` → os 6 casos novos somam cerca de 0,7 s; nenhum lê `data/`. CI do commit `3c03f80`: passo "Correr os testes" (a suíte inteira) em 8 s, success |
| Cada teste falha quando se estraga de propósito o que ele guarda. | ✅ | tabela "Os testes novos falham com o código estragado" abaixo |

Caixas do `docs/backlog.md` marcadas como `[x]` neste PR: os 4 critérios do E4-T3.

### Testes focados

`.venv/bin/pytest -q tests/unit/test_tfidf.py tests/integration/test_tfidf_pipeline.py -k "sempre_etiquetas or minusculo or gravadas"` → 6 passed, 54 deselected in 1.38s

### O comportamento que já existia continua igual

- Suíte inteira: `.venv/bin/pytest -q` → 248 passed in 32.63s. Antes do trabalho eram 243: o reescrito `test_as_sugestoes_sao_sempre_etiquetas_de_treino` passou de 1 caso para 2, e entraram 1 teste unitário e 3 casos de integração novos (243 − 1 + 2 + 1 + 3 = 248). O número 243 é deduzido, não foi medido no `main`.
- `src/` não mudou, por isso o modelo, o treino e as sugestões são os mesmos.
- CI do commit `3c03f80`: success (passo dos testes em 8 s).

### Os testes novos falham com o código estragado

Cada estrago foi feito à mão em `src/`, um de cada vez, e reposto a seguir. No fim, `cmp` confirmou que os ficheiros são iguais aos originais.

| Teste | Estrago introduzido | Resultado com o estrago |
|---|---|---|
| sobreajuste (unitário e integração) | etiquetas baralhadas antes do treino (`np.roll`) | falhou ✅ |
| sobreajuste (integração) | o comando de treino passa N = 2 quando recebe 1 | falhou ✅ |
| sobreajuste (integração) | o comando de treino lê a coluna `codigos` em vez de `codigo` | falhou ✅ (e os 2 casos das etiquetas dão erro) |
| sobreajuste (integração) | treino só com metade dos exemplos (um sim, outro não) | falhou ✅ |
| sobreajuste (unitário e integração) | classes por ordem invertida no top-5 | falhou ✅ |
| etiquetas (unitário N = 2, integração N = 10) | treino com os códigos originais em vez das etiquetas dos códigos raros | falhou ✅ |
| etiquetas (todos os casos) | uma etiqueta falsa (99999999) nas sugestões | falhou ✅ |
| etiquetas (integração N = 6) | exemplos que saem a entrar com o código original | falhou ✅ |

### Números que provam que funciona

Não se aplica: o PR só acrescenta testes e não mexe no código que transforma dados, treina ou avalia. Os números da validação são os do E4-S4.

### Revisão de código

Duas revisões por agentes "juiz", sem o contexto da conversa, sobre as mudanças por fazer commit:
- 1.ª revisão: o τ no teste das etiquetas não acrescentava nada, o teste antigo ficava contido no novo, o `alpha` do teste (1e-6) era mais forte do que o do modelo, e um comentário era falso. Corrigido.
- 2.ª revisão (13 estragos tentados): o teste de sobreajuste não apanhava a coluna errada nem o N ignorado, e o teste das etiquetas não apanhava o treino com os códigos originais. Corrigido com o 2.º código CPV, o tema de uma só frase e o caso N = 10.
- Não corrigido, e porquê: os testes não guardam o `alpha` nem as épocas (ver "Ficou igual de propósito").

`/requesting-code-review` sobre `5600dee..3c03f80`: **pronto para merge**, sem problemas críticos nem importantes. Confirmou os 4 critérios e estragou o código em 8 sítios; só o `alpha = 1,0` fixo passa, que é a limitação já declarada.
- Corrigido: a descrição dizia que o CI estava por correr e que os testes demoravam "menos de 1 s no GitHub" sem medição; agora tem o tempo do CI.
- Não corrigido, e porquê: o `alpha=tfidf.ALPHA` explícito no teste unitário fica, para o teste dizer à vista que usa a regularização fraca do modelo; o link para o ficheiro do scikit-learn nas notas do E4-S5 fica de fora, porque não é do E4-T3.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
