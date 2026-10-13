## E4-S1 · Spike: que N, épocas e τ usar nos códigos raros

## Em poucas palavras

O TF-IDF sugere 5 códigos CPV a partir da descrição de uma compra. Este PR fecha a escolha do **N** — o número mínimo de exemplos para um código não subir para a classe — testando as combinações de N, épocas de treino e τ numa só grelha, em vez de um eixo de cada vez. A melhor das 54 combinações é a que já estava em vigor (N = 10, 24 épocas, τ = 0,25), e o treino repetido dá sugestões 100% iguais às do spike anterior. Para quem usa o projeto nada muda: o modelo é o mesmo, agora com mais confiança de que é o melhor dentro da lista testada.

## Porquê

O número ótimo de épocas de treino **depende do N** — o N muda quantas etiquetas existem e quantos exemplos tem cada uma — e o τ também, porque corrige a frequência das etiquetas. Escolher o N com as épocas e o τ fixos podia escolher um N que não seria o melhor com os seus próprios valores: é a **interação** entre hiperparâmetros, e um eixo de cada vez não a vê. O item E4-S1 pedia "que N usar"; alargou-se a uma grelha N × épocas × τ (mudança aprovada a 13 de outubro de 2026, escrita no backlog) para fechar essa interação.

Item: [`E4-S1`](../backlog.md) · Desbloqueia: E4-S3 (a entidade)

## O que mudou no código

### `scripts/correr_experiencia.py`

O leitor dos YAML dos spikes ganhou um segundo modo, `grelha`, além do modo `fases` do E4-S5.

- `ler` (`scripts/correr_experiencia.py:80`): lê o YAML conforme o `modo`. Em `grelha`, cada eixo (`n_min`, `epocas`, `tau`) é uma lista de valores e `alpha` é um número fixo.

```python
    if modo == "grelha":
        out["alpha"] = _numero(exp["alpha"], "alpha")
        ...
        out["n_min"] = [_inteiro_positivo(v, "n_min") for v in _lista(exp["n_min"], "n_min")]
        out["epocas"] = [None if v is None else _inteiro_positivo(v, "epocas")
                         for v in _lista(exp["epocas"], "epocas")]
        return out
```

Estas linhas validam cada eixo como uma lista de valores distintos; `None` nas épocas é a regra de paragem por defeito do scikit-learn.

- `comandos` (`scripts/correr_experiencia.py:159`): na fase `grelha`, gera o produto cartesiano de N × épocas — um treino por par — e passa a lista de τ a cada um.

```python
    if exp.get("modo") == "grelha":
        taus = [f"{t:g}" for t in exp["tau"]]
        out = [(f"n_min_{n:02d}_ep_{'defeito' if k is None else f'{k:02d}'}",
                [PY, "src/train_tfidf.py", "--n-min", str(n), "--alpha", f"{exp['alpha']:g}",
                 "--taus", *taus, *args_epocas(k), "--sem-modelo", *comum,
                 "--saida", str(pasta_grelha(exp, n, k))], 12 if k is None else k)
               for n in exp["n_min"] for k in exp["epocas"]]
```

Um treino por (N, épocas) grava um ficheiro de sugestões por τ: o τ não retreina o modelo, só reordena as sugestões ao sugerir. Daí os 18 treinos darem as 54 combinações.

### `experiencias/e4_s1.yaml`

Pré-registo das opções, escrito antes de correr: `alpha` = 1e-7 fixo, `n_min` [5, 10, 20], `epocas` [defeito, 3, 6, 12, 18, 24], `tau` [0, 0.25, 0.5].

## Outras mudanças

- `notebooks/07_tfidf_grelha.ipynb`: a decisão — tabela das 54 combinações, confirmação de que N = 10/24 épocas reproduz o E4-S5 (100%), duelo contra a opção em vigor e o "bate a régua".
- `metrics/e4_s1/`: as 18 pastas de treino (sugestões por τ + `treino.json`), os logs e o `pod_info.json`.
- `docs/qualidade_dados.md`: secção 16 (resultados) e decisão 9 (N = 10).
- `docs/backlog.md`: o item E4-S1 passou a grelha (13/10/2026) e ficou marcado como feito; ordem e dependências atualizadas.
- `docs/workflow.md`, `CLAUDE.md`, `Makefile` (`make grelha`), `scripts/readme.md`, `notebooks/readme.md`, `experiencias/readme.md`, `tests/unit/readme.md`.
- `tests/unit/test_correr_experiencia.py`: testes novos (ler o YAML da grelha, comandos, erros de validação).
- `src/labels.py`: comentário do `DEFAULT_N_MIN` atualizado (N = 10 escolhido no E4-S1).

### Ficou igual de propósito

- O modelo por defeito (`src/tfidf.py`): `ALPHA`, `TAU` e `EPOCAS` **não** mudam — a grelha confirmou os valores do E4-S5, não os substituiu.
- O `prepare.py` e o `treino.parquet`: o N aplica-se no treino, não na preparação (decisão 9).
- O modo `fases` do E4-S5: o `correr_experiencia.py` continua a lê-lo igual e o `e4_s5.yaml` não mudou.

## Validação e provas

### Critérios de aceitação

| Critério (copiado do backlog) | Cumprido? | Prova (comando → resultado) |
|---|---|---|
| Uma tabela com as 54 combinações (top-5 no código completo e na classe, e F1 macro) e a combinação escolhida | ✅ | `notebooks/07_tfidf_grelha.ipynb`, célula 5 (54 linhas + régua) |
| A limitação do viés de seleção escrita no `qualidade_dados.md` | ✅ | secção 16, "Limitação (viés de seleção)" |
| O N escolhido na decisão 9, com as épocas e o τ escolhidos, e passa a ser o valor por defeito | ✅ | decisão 9 (N = 10) e decisão 7 (24 épocas, τ = 0,25) |
| Se o N mudar, a régua (E4-T2) e os resultados do E4-H1 voltam a calcular-se com o novo N | ✅ (n/a) | N não mudou (10 → 10); o escolhido bate a régua nos 4 casos (célula 14) |
| O custo em tempo de pod e em dólares | ✅ | `metrics/e4_s1/pod_info.json` → ~80 min, 1,28 $/h, ~1,66 $ |
| (pré-registo) as opções escritas antes de correr | ✅ | `experiencias/e4_s1.yaml` (ver a limitação no fim) |
| Repetível: N = 10/24 épocas = E4-S5 | ✅ | célula 7 → 100,00% com τ = 0 e τ = 0,25 |

Caixas do `docs/backlog.md` marcadas como `[x]` neste PR: nenhuma, porque os spikes não têm caixas.

### Testes focados

`.venv/bin/pytest tests/unit/test_correr_experiencia.py -q` → 35 passed

### O comportamento que já existia continua igual

- Suíte inteira: `.venv/bin/pytest -q` no pod → **338 passed**. No portátil, `tests/unit` → 233 passed e `tests/integration -m "not data"` → 83 passed (as 22 marcadas `data` leem o `treino.parquet` grande e não cabem no portátil).
- O modo `fases` do E4-S5: `test_o_yaml_do_e4_s5_tem_as_opcoes_do_backlog` e os testes de `alpha`/`epocas` continuam a passar.
- CI do commit `<sha>`: <a correr>

### Os testes novos falham com o código estragado

| Teste | Estrago introduzido | Resultado |
|---|---|---|
| `test_fase_grelha_um_treino_por_n_e_epocas_com_os_taus` | `--taus` só com o 1.º τ | falhou ✅ |
| `test_uma_grelha_so_corre_a_fase_grelha` | remover a verificação da fase | falhou ✅ |
| `test_fase_grelha_usa_a_regra_por_defeito_quando_as_epocas_sao_null` | `args_epocas(None)` a devolver `--epocas 12` | falhou ✅ |
| `test_o_yaml_do_e4_s1_tem_as_opcoes_do_backlog` | tirar o 24 das épocas no YAML | falhou ✅ |
| `test_yaml_de_grelha_invalido_da_erro[...]` | remover a verificação de repetidos em `_lista` | falhou ✅ |

### Números que provam que funciona

| Medida | E4-S5 (em vigor) | Melhor da grelha | Fonte |
|---|---|---|---|
| Top-5 no código completo (validação) | 78,3% | 78,35% | notebook 07, célula 10 |
| Top-5 na classe | 84,1% | 84,11% | idem |
| Repetibilidade vs E4-S5 | — | 100,00% | célula 7 |

O melhor da grelha é exatamente a configuração do E4-S5: a grelha confirma-a. O concorrente mais próximo (N = 5, 24 épocas, τ = 0,25) faz 78,05% e perde com uma diferença que conta no bootstrap.

### Revisão de código

Duas revisões independentes (agentes novos, com só o id do item, o `BASE` e o `HEAD`).
- 1.ª (`main..8d6e294`): **com correções** — faltava o `pr_E4-S1.md` e a secção 16 usava decimais que nenhuma célula imprimia. Corrigido.
- 2.ª (`main..63b8cfb`): **com correções** — os mesmos dois pontos e mais quatro menores (frase do backlog, janela de tempo do pod, teste de `--treino`/`--validacao` na grelha, comentário do `DEFAULT_N_MIN`). Corrigido neste commit.

Limitação assumida: o `experiencias/e4_s1.yaml` foi escrito antes de correr (a data do ficheiro prova-o), mas entrou no mesmo commit dos resultados. A partir de agora, o YAML deve ser commitado **antes** da ida ao pod, como promete o `experiencias/readme.md`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
