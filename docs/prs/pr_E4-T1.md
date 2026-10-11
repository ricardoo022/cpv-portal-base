## E4-T1 · Avaliação comum a todos os modelos

## Em poucas palavras

Este PR cria a **forma de dar nota** aos modelos. Ainda não há nenhum modelo: isto é a régua com que todos vão ser medidos (a régua "códigos mais comuns", o TF-IDF, o BERTimbau, o Albertina), sempre da mesma maneira.

O modelo dá 5 códigos CPV por contrato. Cada contrato vale **1 ponto se acerta e 0 se falha**, e a nota é a percentagem de contratos certos. Mede-se a três níveis de exigência. Exemplo, com o técnico a ter escolhido `45233200`:

| 1.ª sugestão do modelo | Código completo (igual) | Classe (4 dígitos) | Divisão (2 dígitos) |
|---|---|---|---|
| `45233200` (igual) | 1 | 1 | 1 |
| `45233223` (mais específica) | 0 | 1 | 1 |
| `45200000` (mais vaga) | 0 | 0 | 1 |
| `33600000` (outra área) | 0 | 0 | 0 |

- **Top-1** só olha para a 1.ª sugestão; **top-5** dá o ponto se qualquer uma das 5 acertar.
- **Meio acerto:** à parte, mostram-se as medidas hierárquicas dos papers, que dão uma nota entre 0 e 1 a uma sugestão perto do código certo. Só se mostram; não decidem.
- **Decidir entre dois modelos:** se A tem 61% e B 60%, foi sorte? Tira-se 1 000 vezes, à sorte, uma amostra dos contratos; a diferença só conta se A ganhar em pelo menos 95% delas. Decide o top-5 no código completo, e um modelo que piore a classe é eliminado.
- **Contratos só com código de divisão** (por exemplo, só `45000000`): o modelo nunca os pode acertar no código completo, porque não aprendeu divisões. Contam só na divisão e aparecem numa linha à parte.

## Porquê

Todos os modelos (a régua, o TF-IDF, o BERTimbau, o Albertina) vão ser comparados com esta avaliação. Se tiver um erro, todas as comparações ficam erradas sem ninguém dar por isso. Até aqui não havia forma de medir sugestões nem de dizer se a diferença entre duas opções é real ou acaso.

Item: [`E4-T1`](docs/backlog.md) · Desbloqueia: E4-T2, E4-H1 e os spikes (E4-S1, E4-S2, E4-S3)

**Âmbito alterado neste PR (decidido a 10 de outubro de 2026):**
- **Decisão 8 revista:** no "código completo", uma sugestão passa a acertar só se for **igual** ao código. Antes, qualquer sugestão que começasse pelos dígitos do código acertava (`33600000` aceitava `33651100`). A literatura de classificação hierárquica conta como erro a sugestão mais específica do que o código verdadeiro (*sobre-especialização*; Kosmopoulos et al., 2015).
- **Contratos só com códigos de divisão** (por exemplo, só `45000000`): deixam de contar no código completo, porque o modelo nunca sugere divisões (tiradas do treino, decisão 8) e por isso nunca os pode acertar. Avaliam-se na divisão, numa linha à parte (`so_divisao`). A decisão 8 passa a dizer de forma explícita que tirar as divisões do treino é uma suposição de produto, não medida.
- **Critério novo:** as medidas hierárquicas do artigo *Zero-Shot Hierarchical Classification on the CPV Taxonomy* (2024), que vêm de Kiritchenko et al. (2005). Dão meio acerto a uma sugestão perto do código, mas só se mostram: não entram na regra de comparação.

## O que mudou no código

### `src/evaluation.py` (novo): dar nota às sugestões

- `hits` (`src/evaluation.py:79`): para cada contrato, 1 se alguma das k primeiras sugestões acerta, 0 se não, e vazio (NaN) se o contrato não conta nessa medida.
- `evaluate` (`src/evaluation.py:170`): a tabela final, com uma linha por medida (código, classe, divisão, `so_divisao`) e as colunas contratos, top-1, top-5 e F1 macro.
- `report` (`src/evaluation.py:262`): o `evaluate` duas vezes, para todos os contratos e só para os de texto novo.
- `f1_pairs` e `f1_macro` (`src/evaluation.py:95` e `:123`): o F1 macro da 1.ª sugestão contra a etiqueta, com pesos opcionais para o bootstrap.
- `hierarchical` e `hierarchical_report` (`src/evaluation.py:231` e `:267`): o meio acerto dos papers (precisão, recall e F1 hierárquicos).
- `_check_codigos` e `_check_suggestions` (`src/evaluation.py:48` e `:54`): dão erro claro se a entrada vier mal formada (códigos sem 8 dígitos, menos de 5 sugestões, índices diferentes).

A regra das três medidas está numa só tabela:

```python
# Por medida: (nível mínimo do código para contar, dígitos comparados)
_RULES = {"codigo": (3, 8), "classe": (4, 4), "divisao": (2, 2)}
```
Código completo: só contam códigos de nível 3 ou mais (os de divisão ficam de fora) e comparam-se os 8 dígitos, ou seja, tem de ser igual. Classe: códigos de nível 4 ou mais, 4 dígitos. Divisão: todos, 2 dígitos.

Cada contrato ganha 1 ou 0 assim:

```python
float(any(s.startswith(p) for s in sugg[:k] for p in prefs)) if prefs else np.nan
```
Acerta (1.0) se alguma das k primeiras sugestões começar pelo pedaço exigido de algum dos códigos do contrato. Se o contrato não tem nenhum código que conte nesta medida, fica NaN e não entra na média.

Os contratos só com divisão vão para a linha à parte:

```python
only = codigos.str.split().map(lambda cs: all(c.endswith("000000") for c in cs))
```
Um contrato vai para `so_divisao` se **todos** os seus códigos acabarem em `000000` (por exemplo, só `45000000`). Aí só se mede se o modelo acertou a divisão.

O meio acerto conta os níveis da árvore em comum:

```python
return max(0, min(prefix, level_a, level_b) - 1)
```
`prefix` é quantos dígitos iniciais são iguais. Limita-se ao nível do código mais curto e tira-se 1 porque o 1.º nó é a divisão (2 dígitos). Exemplo: `45233200` e `45200000` partilham 2 nós (`45`, `452`).

### `src/comparison.py` (novo): decidir entre dois modelos

- `paired_bootstrap` (`src/comparison.py:77`): compara A e B em 1 000 amostras e diz se a diferença conta.
- `bootstrap_weights` (`src/comparison.py:22`): gera cada amostra como "quantas vezes saiu cada contrato".
- `weighted_mean` e `f1_macro_stat` (`src/comparison.py:33` e `:51`): calculam o top-k e o F1 numa amostra. O `f1_macro_stat` é a versão rápida do F1: baixou o tempo de 300 s para cerca de 40 s.
- `choose` (`src/comparison.py:116`): a regra de escolha entre A e B.
- `compare` e `choose_from` (`src/comparison.py:155` e `:169`): fazem tudo de uma vez, nas 3 medidas × 3 métricas.

O coração do bootstrap:

```python
for weights in bootstrap_weights(n, n_boot, seed):
    diff = stat_a(weights) - stat_b(weights)  # a mesma amostra para as duas opções
    if diff > 0:
        a_wins += 1
    elif diff < 0:
        b_wins += 1
...
"conta": max(a_melhor, b_melhor) >= THRESHOLD,  # 0.95
```
Em cada uma das 1 000 amostras, A e B são medidos **nos mesmos contratos** e vê-se quem ganha. A diferença só conta se o mesmo lado ganhar em pelo menos 95% das amostras.

A regra de escolha, passo a passo:

```python
classe_winner = _winner(classe_top5)
eliminated = _other(classe_winner) if classe_winner else None      # 1. quem piora a classe sai
codigo_winner = _winner(codigo_top5)
if codigo_winner is not None:                                       # 2. ganha quem é melhor no código
    if codigo_winner != eliminated:
        return codigo_winner
    return _other(eliminated)
if classe_winner is not None:                                       # 3. empate: desempata a classe
    return classe_winner
return default                                                      # 4. empate em tudo: opção por defeito
```

## Outras mudanças

- `tests/unit/test_evaluation.py` (novo): 37 testes com contratos inventados e o resultado calculado à mão no comentário.
- `tests/unit/test_comparison.py` (novo): 30 testes, incluindo um que confirma que o `f1_macro_stat` dá o mesmo que o `f1_macro`.
- `tests/integration/test_pipeline.py`: 2 testes novos. A avaliação corre, com um oráculo (cada contrato recebe os seus próprios códigos como sugestões), sobre a validação que o pipeline inteiro escreveu a partir de `.xlsx` inventados. Provam que a avaliação lê o formato que o `prepare.py` produz.
- `docs/backlog.md`: as regras de contagem da E4-T1 (código exato, contratos só com divisão, medidas hierárquicas), o critério novo, os 8 critérios marcados, e uma secção "Ideias para depois" (interface por níveis da árvore).
- `docs/qualidade_dados.md`, decisão 8: tirar as divisões do treino passa a estar escrito como suposição de produto, com o custo; a parte da avaliação foi reescrita com a data e o porquê da mudança.

### Ficou igual de propósito

- `prepare.py` e os ficheiros de dados — a avaliação só lê sugestões e códigos; não muda a preparação.
- As etiquetas de treino (decisão 9, `labels.py`) — a avaliação usa sempre os códigos originais, como a decisão 9 manda.
- `requirements.txt` — sem scikit-learn: o F1 e o bootstrap usam só pandas e numpy, para não acrescentar uma dependência antes do E4-H1.
- A regra de comparação — as medidas hierárquicas não entram nela; decide o top-5 no código completo, com a classe como guarda.
- Nenhum número novo no `docs/` — os números da validação deste PR são verificações (oráculo), não resultados de nenhum modelo.

## Validação e provas

### Critérios de aceitação

| Critério (copiado do backlog) | Cumprido? | Prova (comando → resultado) |
|---|---|---|
| Dadas as 5 sugestões de cada contrato e os seus códigos, a avaliação dá o top-1 e o top-5 em cada nível e diz quantos contratos entraram em cada nível. | ✅ | `test_evaluate_contagens_e_taxas` (contagens [3, 2, 4, 1]), `test_contrato_so_com_divisao_nao_conta_no_codigo_completo`; integração: `test_avaliacao_corre_sobre_a_validacao_da_preparacao` (contagens [1, 1, 2, 1]) |
| Dá também o F1 macro em cada medida, como descrito acima. | ✅ | `test_f1_macro_a_mao` (0,65 calculado à mão), `test_f1_pairs_*` |
| Dá a precisão, o recall e o F1 hierárquicos da 1.ª sugestão, em micro e em macro, para todos os contratos e só para os de texto novo. | ✅ | `test_hierarchical_*` (7 testes); `test_hierarchical_report_separa_texto_novo` |
| Dadas as sugestões de duas opções para os mesmos contratos, dá a diferença entre elas em cada medida e diz se essa diferença conta, segundo a regra de comparação acima. | ✅ | `test_compare_a_melhor_no_codigo_e_na_classe_ganha_a`, `test_choose_from_le_o_top5_e_nao_o_top1`, testes de cada ramo de `choose` |
| Com a mesma semente aleatória, o bootstrap dá sempre o mesmo resultado. Duas opções com sugestões iguais dão diferença zero, que não conta. | ✅ | `test_mesma_semente_mesmo_resultado`, `test_compare_usa_a_semente_pedida`, `test_compare_opcoes_iguais_nada_conta` |
| Os resultados aparecem para todos os contratos e, em separado, só para os contratos com texto que não aparece no treino. | ✅ (parcial no `compare`) | `report` e `hierarchical_report` separam os grupos (`test_report_separa_texto_novo`; na integração, o contrato 11 aparece só em "texto novo"). No `compare`, quem chama tem de filtrar os contratos à mão (ver a revisão) |
| Há um teste com contratos inventados, com o resultado calculado à mão, para cada regra de contagem acima (incluindo uma etiqueta de grupo, uma de categoria, e um contrato com um código de divisão e outro de classe). | ✅ | `test_etiqueta_de_grupo_so_acerta_o_codigo_igual`, `test_etiqueta_de_categoria_5_digitos`, `test_etiqueta_de_6_ou_mais_digitos_filho_irmao_e_pai_sao_erro`, `test_divisao_e_classe_no_mesmo_contrato`, `test_contrato_so_com_divisao_nao_conta_na_classe` |
| Cada teste falha quando se estraga de propósito a regra que ele guarda. | ✅ | 53 mutações, todas apanhadas (tabela abaixo e revisão) |

Caixas do `docs/backlog.md` marcadas como `[x]` neste PR: os 8 critérios da E4-T1.

### Testes focados

`.venv/bin/pytest -q tests/unit/test_evaluation.py tests/unit/test_comparison.py` → 67 passed

`.venv/bin/pytest -q tests/integration` → 15 passed (13 antigos + 2 novos)

### O comportamento que já existia continua igual

- Suite inteira: `.venv/bin/pytest -q` → 139 passed (antes do trabalho, no `main` `8b73c17`, numa worktree à parte: 70 passed). Os 70 testes antigos continuam a passar sem alterações.
- O diff não toca em nenhum ficheiro que já existia em `src/`. Em `tests/` só acrescenta 2 testes ao `test_pipeline.py`; os 13 testes que lá estavam ficaram iguais e passam.
- CI: **não corrido**. O trabalho está por guardar (sem commit, por pedido) e o branch remoto `avaliacao/e4-t1` ainda está igual ao `main` antigo. Tem de correr depois do commit e do push, e depois de trazer o `origin/main` (o PR #6 entrou entretanto).

### Os testes novos falham com o código estragado

Seleção. Ao todo: 4 mutações na integração, 17 na avaliação, 10 no bootstrap, 5 na ligação, 10 na regra exata e nas medidas hierárquicas, 7 nos contratos só com divisão, e mais 13 da revisão independente. Uma das 7 sobreviveu à primeira (o top-1 da linha `so_divisao` calculado com k = 5); reforcei o teste e passou a ser apanhada.

| Teste | Estrago introduzido | Resultado com o estrago |
|---|---|---|
| `test_etiqueta_de_grupo_so_acerta_o_codigo_igual` e mais 6 | código completo volta ao prefixo até ao nível da etiqueta (regra antiga) | 7 falharam ✅ |
| `test_contrato_so_com_divisao_nao_conta_no_codigo_completo` e mais 5 | o código completo volta a contar contratos só com divisão | 6 falharam ✅ |
| `test_so_divisao_conta_so_contratos_sem_codigo_abaixo_da_divisao` | `so_divisao` com "algum código é divisão" em vez de "todos" | 2 falharam ✅ |
| `test_avaliacao_corre_sobre_a_validacao_da_preparacao` | `prepare.py` grava `visto_no_treino` ao contrário | falhou ✅ |
| `test_avaliacao_corre_sobre_a_validacao_da_preparacao` | `prepare.py` separa os códigos com vírgula | erro ✅ (apanhado logo pelo `check_sets` da preparação) |
| `test_medidas_hierarquicas_correm_sobre_a_validacao_da_preparacao` | a avaliação conta a raiz nos nós da etiqueta | falhou ✅ |
| `test_f1_pairs_codigo_usa_os_codigos_inteiros` | F1 do código volta a cortar a sugestão | falhou ✅ |
| `test_classe_usa_4_digitos_e_nao_3` | classe com 3 dígitos | falhou ✅ |
| `test_acerta_em_qualquer_codigo_do_contrato_nao_so_na_etiqueta` | acerta só na etiqueta | falhou ✅ |
| `test_hierarchical_scores_codigos_iguais_genericos` | nós em comum sem limite do nível | 4 falharam ✅ |
| `test_hierarchical_macro_faz_media_por_etiqueta` | macro por contrato, não por etiqueta | falhou ✅ |
| `test_emparelhar_deteta_diferenca_que_amostras_independentes_nao_detetam` | amostras independentes para A e B | falhou ✅ |
| `test_exatamente_95_por_cento_conta` | `>` em vez de `>=` no limiar | falhou ✅ |
| `test_ganha_no_codigo_mas_piora_a_classe_e_eliminada` | guarda da classe desligada | falhou ✅ |
| `test_choose_from_le_o_top5_e_nao_o_top1` | `choose_from` lê o top-1 | falhou ✅ |
| `test_compare_usa_a_semente_pedida` | `compare` ignora a semente | falhou ✅ |
| `test_f1_macro_stat_igual_a_f1_macro_da_avaliacao` | F1 rápido com a média sobre os previstos | 3 falharam ✅ |
| `test_report_visto_no_treino_em_falta_da_erro` | sem verificação de valores em falta | falhou ✅ |

### Números que provam que funciona

Ainda não há sugestões de nenhum modelo (a régua é a E4-T2), por isso não há números da validação para mostrar. A prova de que a avaliação funciona com dados no formato real é o teste de integração: o pipeline inteiro corre sobre `.xlsx` inventados e a avaliação lê a validação que ele escreveu, com um **oráculo** (cada contrato recebe os seus próprios códigos como sugestões).

| Medida | Esperado | Obtido | Fonte |
|---|---|---|---|
| Contratos por medida (código, classe, divisão, `so_divisao`) | [1, 1, 2, 1] | [1, 1, 2, 1] | `test_avaliacao_corre_sobre_a_validacao_da_preparacao` |
| Top-1, top-5 e F1 macro, todos os contratos | 1 | 1 | idem |
| Texto novo: só o contrato 11 (só divisão) | [0, 0, 1, 1] contratos | [0, 0, 1, 1] | idem |
| Precisão, recall e F1 hierárquicos | 1 | 1 | `test_medidas_hierarquicas_correm_sobre_a_validacao_da_preparacao` |

Tempo: o `compare` (1 000 amostras, 9 métricas) demorava 300 s em 24 000 contratos sintéticos e passou a cerca de 40 s depois do `f1_macro_stat`. Medido com um script fora do repositório, por isso é só indicativo.

### Revisão de código

Revisão independente sobre `8b73c17..pasta de trabalho`: **pronto para merge, com correções.** Nenhum problema crítico. As 13 mutações do revisor foram todas apanhadas.

- Corrigido: **Importante 1** (a mudança de âmbito e da decisão 8 não estava à vista). Fica explícita no início desta descrição.
- Corrigido: **Importante 2.** Com a regra exata, os contratos que só têm códigos de divisão (`XX000000`) contavam no código completo, mas nenhum modelo os pode acertar: o treino não tem divisões (decisão 8). Escolhida a opção (a): deixam de contar no código completo e mostram-se na linha `so_divisao`. Teste escrito primeiro e visto a falhar.
- Não corrigido, e porquê (menores, ficam para quando forem precisos):
  - o `compare` não recebe `visto_no_treino`; vai ser preciso no E4-S2;
  - `visto_no_treino.astype(bool)` lê a string `"False"` como `True`; a coluna do parquet é booleana, por isso hoje não acontece;
  - não se verificam sugestões repetidas no top-5; garanti-lo é trabalho do E4-H1 ("5 códigos distintos");
  - um grupo vazio dá `NaN` sem aviso;
  - quando A ganha no código mas é eliminada pela guarda, o `choose` escolhe B; isto tem de ficar escrito na regra de comparação.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
