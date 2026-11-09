## E5-T5 · Primeiro treino do Albertina

## Em poucas palavras

Este PR treina pela primeira vez o segundo modelo do épico, o Albertina, para sugerir códigos CPV. Usa exatamente a mesma receita do primeiro modelo, o BERTimbau, para que a única diferença entre os dois seja o modelo. O Albertina passou os dois testes de sanidade e acerta no código completo, entre as 5 sugestões, em 76,98% dos contratos de outubro de 2025: mais 3,6 pontos do que o BERTimbau, e menos 1,4 do que o TF-IDF. Cada treino custa quase o dobro do do BERTimbau (cerca de 1,60 $ contra 0,83 $). Também fica escrita no backlog uma tarefa nova, a perda na validação em cada época, para os próximos treinos.

| Modelo | Top-5 no código completo | Top-5 na classe | Custo de um treino de 3 épocas |
|---|---|---|---|
| Régua "códigos mais comuns" | 26,3% | – | – |
| BERTimbau (E5-T4) | 73,39% | 79,59% | 0,83 $ |
| **Albertina (este PR)** | **76,98%** | **82,38%** | **1,60 $** |
| TF-IDF em vigor | 78,35% | 84,11% | – |

## Porquê

O épico E5 quer saber se afinar um modelo de linguagem português dá sugestões melhores do que o TF-IDF. Afinar (*fine-tuning*) é continuar o treino de um modelo que já sabe português, agora com os nossos contratos e as nossas etiquetas. Há dois modelos candidatos. O E5-T4 treinou o primeiro, e este item treina o segundo nas mesmas condições: verifica que o código funciona com ele, mede quanto custa e dá a configuração de partida para a grelha do E5-S1.

Item: [`E5-T5`](docs/backlog.md) · Desbloqueia: E5-T7, depois E5-S1

## O que mudou no código

Nenhum ficheiro de `src/` mudou: o treino é o mesmo código do E5-T4. Mudaram o ficheiro que regista a experiência e os testes.

### `experiencias/e5_t5.yaml` (novo)

É o pré-registo: a receita escrita **antes** de correr, com a fonte de cada valor. Os artigos do Albertina (Santos et al., 2024; Rodrigues et al., 2023) só publicam a grelha que procuraram, não o valor que escolheram, por isso fica a receita do BERT original (Devlin et al., 2019), dita como tal.

```yaml
treino_completo:
  taxa_aprendizagem: 2.0e-5  # Devlin A.3; dentro da gama do Albertina (1e-5 a 5e-5)
  lote: 32
  epocas: 3                  # Devlin A.3; é o E máximo da grelha do E5-S1 (o Albertina usa 5)
sinal_decorar:
  epocas_max: 100            # 100 e não 30: mudança escrita no épico depois de ver o E5-T4
```
A taxa de aprendizagem (o tamanho de cada passo com que os pesos mudam), o lote e as épocas são os do E5-T4; o sinal de decorar vai até 100 épocas (`experiencias/e5_t5.yaml:45` e `:67`).

### `tests/integration/test_experiencias_bert.py` (era `test_experiencia_e5_t4.py`)

- O teste dos comandos do YAML passa a correr para os dois YAML (`:20` e `:29`):

```python
YAMLS = {"e5_t4": "bertimbau", "e5_t5": "albertina"}  # item -> modelo que esse item treina

@pytest.fixture(params=sorted(YAMLS))
def item(request):
    return request.param
```
Cada `comando` dos dois YAML passa pelo Makefile e pelo `argparse` do script verdadeiro, e tem de chegar ao treino com os valores escritos no YAML. Se o comando dissesse outra taxa, o treino pago corria com uma receita e o registo dizia outra.

- `test_sinal_decorar_do_albertina_vai_ate_100_epocas` (`:76`) e `test_receita_do_albertina_e_a_do_bertimbau` (`:82`) guardam as duas decisões deste item.

### `tests/integration/test_registo_e5_t5.py` (novo)

Guarda os resultados gravados, que já não se podem refazer sem pagar outro treino:

- `test_correu_com_o_yaml_de_hoje_num_commit_limpo` (`:43`): o SHA-256 (a impressão digital de um ficheiro) do YAML gravado em cada execução é o do YAML de hoje, por isso o pré-registo não mudou depois de correr;
- `test_ensaio_e_treino_leram_a_validacao_do_portatil` (`:55`): o treino e a validação no pod eram os do portátil;
- `test_sinal_de_decorar_passou_na_epoca_18_com_100_epocas` (`:77`) e `test_sinal_usou_a_mesma_amostra_do_bertimbau` (`:87`);
- `test_custo_e_memoria_do_pod_info_saem_do_treino_json` (`:96`) e `test_ensaio_mediu_quase_o_dobro_do_tempo_por_passo_do_bertimbau` (`:108`);
- `test_sugestoes_de_exemplo_sao_5_e_nenhuma_e_uma_divisao` (`:117`);
- com dados, `test_curva_e_tau0_da_secao_23` (`:139`): avalia outra vez as sugestões gravadas e tem de dar os números da secção 23.

```python
m = mg.gate_metrics(ev.report(sugestoes, val["codigos"], val["visto_no_treino"]))
assert {k: round(100 * m[k], 2) for k in esperado} == esperado
```
Os números escritos nos docs saem desta avaliação, a mesma que o TF-IDF usa, e não de comandos soltos no terminal.

## Outras mudanças

- `metrics/bert/e5_t5_albertina/`, `e5_t5_ensaio/`, `e5_t5_sinal_decorar/`: os resultados copiados do pod (sugestões por época, `treino.json`, `pod_info.json` com o custo, os registos do pod e as sugestões de 3 descrições de exemplo).
- `docs/qualidade_dados.md`: secção 23 nova, com os sinais, o custo e a tabela por época.
- `docs/backlog.md`: o resultado do E5-T5 com os critérios; o custo medido na medida de sucesso 5; a **tarefa nova E5-T7** (perda na validação em cada época, pedida pelo utilizador), entre o E5-T5 e o E5-S1, com o diagrama e o tamanho do épico atualizados. Também entraram alterações do utilizador ao backlog que já estavam por gravar (a decisão de 7 de novembro sobre a grelha e mudanças de formatação nos épicos E1 a E4).
- `docs/como_funciona_bert.md`: alterações do utilizador, gravadas a pedido dele antes do treino (commit `98dcb33`).
- `Makefile`: o `test-bert` corre o teste renomeado.
- `CLAUDE.md`, `metrics/readme.md`, `tests/integration/readme.md`, `experiencias/readme.md`: as linhas novas para os ficheiros novos e o estado do épico.

### Ficou igual de propósito

- O código de treino (`src/bert.py`, `src/train_bert.py`) — o Albertina treinou no mesmo código do BERTimbau, e qualquer mudança tinha de ser feita antes do commit que treina.
- A perda na validação — fica para o E5-T7, para o código do E5-T5 ser o do commit que treinou.
- A variável `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` não entrou no código: passou-se só no comando do pod, porque muda a arrumação da memória da GPU e não as contas. Fica escrita no `pod_info.json` e no `pod_treino.log`, mas não no `treino.json` nem no MLflow.
- O portão de qualidade (`metrics/validacao.json`) — um modelo tunado só entra pela regra da E5-T6, depois da grelha e das sementes.
- As pontuações de todas as etiquetas por época (cerca de 0,86 GB) — vieram para o portátil, para o E5-T7, mas ficam fora do git.

## Validação e provas

### Critérios de aceitação

Os critérios do E5-T5 são "os mesmos do E5-T4, para o Albertina". Um por linha:

| Critério (copiado do backlog do E5-T4, para o Albertina) | Cumprido? | Prova (comando → resultado) |
|---|---|---|
| Antes do treino completo: um treino curto mede o tempo por lote e a memória da GPU; daí sai o tempo e o custo de uma época, mostrados e aprovados antes de lançar o treino completo | ✅ | `metrics/bert/e5_t5_ensaio/treino.json` → 0,1418 s por passo, época de 2 118,5 s, 21 700 MB reservados; aprovado pelo utilizador antes do treino |
| Janela de 2022, N = 10, o comprimento máximo do E5-T1, e os valores recomendados no artigo do Albertina (ou, na falta deles, os do BERT original, dito como tal), num YAML escrito antes de correr com a fonte de cada um | ✅ | `experiencias/e5_t5.yaml`; `test_experiencias_bert.py` → os comandos chegam ao treino com esses valores |
| Sinal de erro 1, até 100 épocas, com a taxa registada | ✅ | `sinal_decorar.json` → 100 de 100, 95 pela primeira vez na época 18, taxa 1e-4 |
| Sinal de erro 2: top-5 no código completo acima de 26,3% | ✅ | `test_curva_e_tau0_da_secao_23` → 76,98% |
| O comando de sugestões devolve 5 códigos, nenhum ao nível da divisão | ✅ | `sugestoes_exemplos.txt`, `test_sugestoes_de_exemplo_sao_5_e_nenhuma_e_uma_divisao` → 15 códigos, nenhum `XX000000` |
| Treino e validação no pod com o mesmo SHA-256 que no portátil, nenhum caminho para o teste | ✅ | `test_ensaio_e_treino_leram_a_validacao_do_portatil` → passa; `sha256sum` no pod = `shasum` no portátil |
| A página do MLflow mostra o treino, com a curva do top-5 e o commit certo | ✅ | o run `e5_t5_albertina` tem o commit `98dcb33` e o top-5 nas épocas 1, 2 e 3 (lido com a API do MLflow); o utilizador viu a página a 9 de novembro de 2026 |
| Tempo, memória e custo num ficheiro em `metrics/`, e o pod terminado logo a seguir a copiar | ✅ | `pod_info.json` → 6 458,35 s, 1,60 $; `list-pods` depois de apagar → vazio |

Caixas do `docs/backlog.md` marcadas como `[x]` neste PR: as 8 do resultado do E5-T5; as 2 do E5-T3 que esperavam por um treino real (a página do MLflow com a tabela e a curva), confirmadas pelo utilizador; e a do MLflow do E5-T4, que o utilizador já tinha marcado. O E5-T4 e o E5-T5 ficam "feito".

### Testes focados

`.venv/bin/pytest -q tests/integration/test_registo_e5_t5.py tests/integration/test_experiencias_bert.py` → 16 + 10 passed. `make test-bert` → 194 passed (no pod: 192 passed, 2 skipped).

### O comportamento que já existia continua igual

- Suíte inteira: `.venv/bin/pytest -q` → 716 passed, 1 skipped, depois das correções da revisão (antes do trabalho: 188 passed no `make test-bert`).
- O YAML do E5-T4 continua a passar no teste parametrizado (os mesmos 4 testes, agora com `[e5_t4]`), e `test_registo_e5_t4.py` continua verde: os registos do E5-T4 não mudaram.
- CI: `df68409` success, `98dcb33` success, `d163c7e` success; o commit das correções da revisão (`e797418`) estava a correr quando o PR foi aberto.

### Os testes novos falham com o código estragado

| Teste | Estrago introduzido | Resultado com o estrago |
|---|---|---|
| `test_comando_do_sinal_decorar_passa_os_valores_do_yaml[e5_t5]` | `--epocas 100` → `--epocas 30` no comando | falhou ✅ |
| `test_comando_do_treino_completo...[e5_t5]` | taxa `2e-5` → `3e-5` no comando | falhou ✅ |
| `test_comando_do_ensaio...[e5_t5]` | `MODELO=albertina` → `bertimbau` no ensaio | falhou ✅ |
| `test_sinal_decorar_do_albertina_vai_ate_100_epocas` | `epocas_max: 100` → `30` | falhou ✅ |
| `test_correu_com_o_yaml_de_hoje_num_commit_limpo` | uma linha acrescentada ao YAML | falhou ✅ (3 testes) |
| `test_custo_e_memoria_do_pod_info_saem_do_treino_json` | custo 1,6 → 1,5 | falhou ✅ |
| `test_sugestoes_de_exemplo_sao_5_e_nenhuma_e_uma_divisao` | `90919300` → `90000000` | falhou ✅ |
| `test_curva_e_tau0_da_secao_23` | sugestões da época 3 trocadas pelas da 2 | falhou ✅ (2 testes) |
| `test_ensaio_e_treino_leram_a_validacao_do_portatil` | um carácter do SHA da validação mudado | falhou ✅ |

### Números que provam que funciona

Validação de outubro de 2025, 23 945 contratos, τ = 0,25. O τ é a correção das etiquetas frequentes: baixa um pouco a pontuação dos códigos muito comuns.

| Medida | Época 1 | Época 2 | Época 3 | BERTimbau, época 3 | TF-IDF | Fonte |
|---|---|---|---|---|---|---|
| Top-5, código completo | 71,10% | 75,73% | **76,98%** | 73,39% | 78,35% | `test_curva_e_tau0_da_secao_23`, MLflow |
| Top-5, classe | 77,12% | 81,50% | 82,38% | 79,59% | 84,11% | idem |
| Top-5, código, só texto novo | 68,44% | 72,84% | 74,12% | 70,36% | 75,14% | idem |
| Perda no treino | 3,964 | 2,497 | 2,169 | 2,542 | – | `pod_treino.log` |

É um só treino, com uma semente e sem bootstrap. Bootstrap: tira-se 1 000 vezes, à sorte, uma amostra dos contratos, e vê-se se um modelo ganha em pelo menos 95% delas. Por isso nada aqui decide quem ganha: decide a E5-T6, depois da grelha.

| Custo | Valor | Fonte |
|---|---|---|
| Segundos por passo (ensaio) | 0,1418 (BERTimbau 0,073) | `e5_t5_ensaio/treino.json` |
| Treino completo | 6 458,35 s, cerca de 1,60 $ | `pod_info.json` |
| Pod inteiro | cerca de 12 013 s, cerca de 2,97 $ por cálculo (fatura: 2,13 $) | `pod_info.json` |
| Gasto do épico até aqui | cerca de 4,83 $ dos 20 $ | backlog, medida de sucesso 5 |

### Revisão de código

Revisor independente (agente novo, só com o id do item e o diff) sobre `main..d163c7e`: **pronto para merge com correções.** Nenhum problema crítico; todos os testes que estragou falharam.
- Corrigido: "825 MB" passou a "3 × 287 340 128 bytes, cerca de 0,86 GB"; os custos do pod passaram a "cerca de", porque a hora de apagar é aproximada ao minuto; teste novo para o SHA-256 da validação; a frase "o teste não foi para o pod" passou a dizer só o que está provado.
- Importante, corrigido: no E5-T4, a caixa do MLflow estava `[x]` mas o texto dizia que faltava ver a página. O utilizador confirmou que viu a página, por isso o texto, o `CLAUDE.md` e a tabela dos épicos passaram a dizer que está feito.
- Não corrigido, e porquê: a variável de memória não vai para o `treino.json` (o código do treino não muda neste item); as mudanças de formatação nos épicos E1 a E4 são do utilizador.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
