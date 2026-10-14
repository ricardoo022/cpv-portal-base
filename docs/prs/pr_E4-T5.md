## E4-T5 · Portão de qualidade: o melhor modelo até agora, guardado no git e registado no MLflow

## Em poucas palavras

Este PR cria um **portão de qualidade**: um modelo novo só passa se não ficar pior do que o melhor modelo que já temos (o **campeão**). Hoje o campeão é o TF-IDF afinado no E4-S1, e os números dele ficam guardados no git. Se uma mudança futura fizer descer qualquer um de 4 números na validação, o portão recusa o modelo e o campeão fica igual. O CI repete a verificação em cada PR. Além disso, todos os treinos já feitos, e cada avaliação do portão daqui para a frente, ficam registados no **MLflow**, uma página no browser onde se veem os treinos lado a lado e em gráficos.

| Modelo novo (top-5 na validação, código / classe, todos os contratos) | Campeão: 78,35% / 84,11% | Resultado |
|---|---|---|
| O mesmo modelo, avaliado outra vez: 78,35% / 84,11% | empate | **aceite, mas o campeão fica** (o ficheiro não muda) |
| TF-IDF com 12 épocas: 77,52% / 83,13% | desce | **recusado**, o campeão fica |
| Um modelo que sobe no código e desce 0,01 pontos na classe | desce numa | **recusado** |
| Um modelo que sobe pelo menos num número e não desce em nenhum | sobe | **aceite**, passa a campeão |

## Porquê

Sem um portão, uma mudança futura (no código, na preparação ou nos hiperparâmetros) podia piorar o modelo, ou passar a usar o conjunto de teste, sem ninguém reparar. O plano inicial era comparar com a régua ("sugerir sempre os 5 códigos mais comuns"), mas o TF-IDF já a bate por muito (78% contra 26%), por isso essa guarda já não protegia nada. A referência passou a ser o melhor modelo até agora, como no guia de MLOps da Google ("o novo modelo tem de ser melhor do que o atual antes de ser promovido") e no passo *Evaluator* do TFX ("o modelo só é aprovado se a métrica não descer face à referência").

Item: [`E4-T5`](docs/backlog.md) · Desbloqueia: E5-T2 (usa as mesmas verificações que recusam o teste) e o E5 em geral (o BERTimbau e o Albertina passam a ser desafiantes do TF-IDF no mesmo portão).

## O que mudou no código

Termos usados abaixo:
- **campeão / desafiante:** o campeão é o melhor modelo até agora; o desafiante é um modelo novo que tenta tirar-lhe o lugar.
- **top-5:** a percentagem de contratos em que o código certo está entre as 5 sugestões.
- **texto novo:** contratos da validação cujo texto não aparece no treino (mede o modelo sem decorar).
- **MLflow:** biblioteca que guarda cada treino como um *run* (parâmetros + métricas) e mostra-os numa página web (`make mlflow`, em http://localhost:5000). Aqui o registo é um ficheiro local, `mlflow.db`, fora do git.

### `src/model_gate.py` (novo)

- `METRICAS` (`src/model_gate.py:46`): os 4 números do portão, o top-5 no código completo e na classe, em todos os contratos e só no texto novo.
- `check_block` (`src/model_gate.py:60`): erro se o bloco de métricas não vier da validação (outubro de 2025), não disser o período ou lhe faltar uma métrica.

```python
    if block.get("conjunto") != CONJUNTO:
        raise GateError(f"as métricas vêm de {block.get('conjunto')!r}, não da validação")
    inicio, fim = (pd.Timestamp(d) for d in block["periodo"])
    if inicio < VAL_START or fim >= TEST_START:
        raise GateError(...)
```
Um bloco que diga `"teste"`, ou com datas fora de [1 de outubro, 1 de novembro de 2025), não serve para o portão.

- `contracts_fingerprint` (`src/model_gate.py:75`): uma impressão digital (SHA-256, um código de 64 caracteres que muda se mudar um só contrato) da lista de `idcontrato` da validação. Vai para o bloco de métricas como `contratos_sha256`.
- `check_training_dates` (`src/model_gate.py:80`): erro se o `treino.json` do modelo avaliado disser que o treino vai até 1 de outubro de 2025 ou depois, ou que leu o `teste.parquet`. O `train_tfidf.py` já o recusa, mas o portão é por onde vão passar todos os modelos (também os do E5), por isso confirma.

```python
    if PROIBIDO in (treino.get("ficheiro"), val.get("ficheiro")):
        raise GateError(f"{origem}: o treino ou a validação foi o ficheiro de teste")
    datas = treino.get("datas")
    if not datas or pd.Timestamp(datas[1]) >= VAL_START:
        raise GateError(...)
```
Um treino com contratos de outubro teria números inflacionados e ficaria campeão; aqui é recusado.

- `check_window` (`src/model_gate.py`, logo a seguir): erro se o treino avaliado começar noutro dia que não o do `treino.parquet` ao lado da validação. O "texto novo" depende da janela do treino: um treino desde 2019 avaliado com a validação da janela de 2022 contaria como novos textos que o modelo viu. Com os dados reais, `make portao PASTA=metrics/e4_s6/desde_2019` dá `GateError` ("o treino começa em 2019-01-02, mas a validação … é a do treino que começa em 2022-01-01").

- `check_same_validation` (`src/model_gate.py:96`): erro se o desafiante e o campeão foram avaliados em validações diferentes (outro número de contratos, outro período ou outra impressão digital dos contratos). Assim nunca se comparam números medidos em conjuntos diferentes.
- `check_not_worse` (`src/model_gate.py:108`): a regra do portão, feita com o MLflow.

```python
    limiares = {m: MetricThreshold(threshold=campeao[m], greater_is_better=True) for m in METRICAS}
    mlflow.validate_evaluation_results(limiares, EvaluationResult(metrics=novas, artifacts={}))
```
Cada número do campeão é o mínimo que o desafiante tem de atingir: igual passa, qualquer descida (por pequena que seja) levanta `ModelValidationFailedException`. Não se usou `min_absolute_change` porque o MLflow só o aceita maior do que 0, e então um empate (retreinar a mesma configuração) seria recusado.

- `evaluate_run` (`src/model_gate.py:123`): lê o `treino.json` de um treino, avalia na validação o ficheiro de sugestões do modelo por defeito (`ficheiro_do_modelo_por_defeito`, o do τ escolhido, não o de τ = 0) e devolve o bloco de métricas. Recusa o teste pelo nome e pelas datas.
- `training_params` e `training_cost` (`src/model_gate.py:155`, `:170`): o que vai do `treino.json` para o MLflow (N, `alpha`, épocas, τ, datas do treino; tempo, memória, vocabulário, etiquetas).
- `run` (`src/model_gate.py:174`): junta tudo. Avalia, regista o run no MLflow (tag `portao = aceite`, `empate` ou `recusado`) e só grava o `metrics/validacao.json` quando o desafiante sobe.

```python
            try:
                check_not_worse(block["metricas"], campeao["metricas"])
            except ModelValidationFailedException:
                mlflow.set_tag("portao", "recusado")
                raise
            if block["metricas"] == {m: campeao["metricas"][m] for m in block["metricas"]}:
                mlflow.set_tag("portao", "empate")
                return campeao, "empate"
        mlflow.set_tag("portao", "aceite")
```
Se descer, o run fica `recusado` e o erro sobe antes da linha que grava o ficheiro (`src/model_gate.py:212`). Se empatar, é aceite mas o campeão fica: o ficheiro continua a apontar para o registo fixo do campeão, e não para a pasta do desafiante (por exemplo `metrics/tfidf/`, que o próximo treino reescreve).

- `main` (`src/model_gate.py:216`): o comando, `make portao PASTA=metrics/<treino>`.

### `src/train_tfidf.py`

- `check_dates` (`src/train_tfidf.py:70`) e `check_inputs` (`src/train_tfidf.py:83`): recusam o teste **pelas datas**, além do nome. Antes só se recusava um ficheiro chamado `teste.parquet`; uma cópia com outro nome passava.

```python
    check_not_test(train_path)
    check_not_test(val_path)
    check_dates(train_path, VAL_START)
    check_dates(val_path, TEST_START)
```
O treino tem de acabar antes de 1 de outubro de 2025 e a validação antes de 1 de novembro; se não, erro. `run` usa-a logo a seguir a validar as épocas (`src/train_tfidf.py:144`), e o portão e o `scripts/curva_epocas.py` também.

### `scripts/registar_historico.py` (novo)

Regista no MLflow (experiência `historico_e4`) todos os treinos guardados em `metrics/`, sem passar pelo portão e sem mudar ficheiros: um run por ficheiro de sugestões (um por τ), mais a régua e a curva por época do E4-S5.

- `report_metrics` (`scripts/registar_historico.py:59`): todas as métricas da avaliação comum (top-1, top-5 e F1 macro, no código, na classe e na divisão, em todos e no texto novo), cada uma com o seu nome.
- `validation_for` (`scripts/registar_historico.py:81`): o "texto novo" depende do treino, por isso um treino de outra janela (desde 2019 ou desde 2025) avalia-se com a validação dessa janela.

```python
    if inicio == inicio_por_defeito:
        path = dados / "validacao.parquet"
    else:
        path = janelas / inicio[:4] / "validacao.parquet"
```
O treino que começa no mesmo dia que o `treino.parquet` preparado usou a janela por defeito; os outros vão buscar `data/janelas/<ano>/`.

- `run` (`scripts/registar_historico.py:117`): salta os runs que já existem (`:137`), por isso correr outra vez não duplica; a curva grava uma métrica por época com `step=epoca` (`:193`), o que dá um gráfico no MLflow.

### `.github/workflows/testes.yml`

Um passo novo escreve o `metrics/validacao.json` do `main` num ficheiro temporário e põe o caminho em `VALIDACAO_MAIN`, para o teste comparar com o do ramo.

```yaml
          git fetch --depth=1 origin main
          if git cat-file -e origin/main:metrics/validacao.json 2>/dev/null; then
            git show origin/main:metrics/validacao.json > "$RUNNER_TEMP/validacao_main.json"
            echo "VALIDACAO_MAIN=$RUNNER_TEMP/validacao_main.json" >> "$GITHUB_ENV"
          fi
```
Só salta a comparação quando o `main` ainda não tem o ficheiro (até este PR entrar); qualquer outro erro do git faz o passo falhar.

## Outras mudanças

- `metrics/validacao.json` (novo): o primeiro campeão, o TF-IDF de `metrics/e4_s1/n_min_10_ep_24/` (N = 10, `alpha` 1e-7, 24 épocas, τ = 0,25, treino desde 2022), escrito pelo portão.
- `tests/unit/test_model_gate.py` (novo, 15 testes): o campeão guardado vem da validação e não desce face ao do `main`; as regras de `check_block`, `contracts_fingerprint`, `check_same_validation` e `check_not_worse` com blocos inventados.
- `tests/integration/test_model_gate_pipeline.py` (novo, 15 testes): dos `.xlsx` inventados à preparação, ao treino e ao portão, com o campeão e o MLflow em pastas temporárias (empate, subida, descida, outra validação, treino com datas da validação, teste recusado).
- `tests/integration/test_model_gate_dados_reais.py` (novo): as métricas do campeão voltam a dar-se com a validação real (salta no CI).
- `tests/integration/test_registar_historico.py` (novo, 8 testes): o script do histórico com um treino inventado guardado como os de `metrics/`.
- `tests/integration/test_tfidf_pipeline.py`: 3 testes novos, a recusa do teste pelas datas (cópia com outro nome no treino e na validação; a validação passada como treino).
- `tests/unit/test_train_tfidf.py`: 4 testes novos, a fronteira das datas (um contrato no próprio dia 1 de outubro, ou 1 de novembro, dá erro; na véspera passa).
- `requirements.txt`: `mlflow==3.17.1`. Não muda nenhuma versão já fixada (scikit-learn 1.7.2, pandas 2.3.3, numpy 2.2.6, pyarrow 25.0.1).
- `.gitignore`: `mlflow.db` e `mlruns/`.
- `Makefile`: `make portao`, `make historico`, `make mlflow`.
- `scripts/curva_epocas.py`: usa `check_inputs` (recusa o teste também pelas datas).
- `docs/backlog.md`: E4-T5 reescrito (campeão em vez de régua), critérios marcados, decisão 8 do E4, o E4 fica fechado. **Inclui o rascunho do E5**, que estava por commitar; nele mudaram a E5-T3 (MLflow num `mlflow.db`, versão já fixada), o E5-T2 (as verificações do E4-T5) e uma decisão pendente nova (como o portão compara um modelo tunado, que varia entre sementes).
- `docs/workflow.md`: o portão de qualidade fica marcado como feito.
- `CLAUDE.md`, `readme.md`, `src/readme.md`, `scripts/readme.md`, `tests/unit/readme.md`, `tests/integration/readme.md`: descrevem o portão, o histórico, os comandos novos e a recusa pelas datas.
- `docs/prs/pr_E4-T5.md`: esta descrição.

### Ficou igual de propósito

- **Os notebooks dos spikes já fechados** continuam a ler os ficheiros com `pd.read_parquet` — são registos fechados, e reescrevê-los arriscava o Cursor gravar uma versão antiga por cima. A recusa do teste vale para o treino, a curva e o portão.
- **A régua sai do portão** — continua a existir (`src/baselines.py`) e está no MLflow, mas já não é a referência: o TF-IDF bate-a por 52 pontos.
- **O CI não treina nem avalia o modelo** — não tem os dados nem a memória. Mede-se no portátil (o portão) e o CI verifica o ficheiro guardado, como no nível 0 do guia de MLOps da Google.
- **Os números do modelo** — nenhum treino mudou; o campeão é um treino já guardado.
- **O `make portao` continua a ler `metrics/tfidf/` por defeito** — é onde o `make treinar-tfidf` grava; com o empate a manter o campeão, avaliar essa pasta nunca faz o campeão apontar para ela, a não ser que suba.
- **O MLflow não guarda modelos nem dados** — só parâmetros e métricas; o registo fica no portátil, sem servidor na web (decisão 8).
- **Não há "impressão digital" do código** — foi discutida e posta de lado por não ser prática corrente no nível 0; se o código mudar sem retreinar, o portão não o sabe.

## Validação e provas

### Critérios de aceitação

| Critério (copiado do backlog) | Cumprido? | Prova (comando → resultado) |
|---|---|---|
| As métricas da validação do campeão ficam guardadas no git, num ficheiro pequeno (`metrics/validacao.json`), com o conjunto e o período de datas. São as do modelo por defeito: o ficheiro de sugestões a ler é o que o `treino.json` indica em `ficheiro_do_modelo_por_defeito`, e não o de τ = 0. | ✅ | `metrics/validacao.json`: `"conjunto": "validacao"`, `"periodo": ["2025-10-01", "2025-10-31"]`, 23 945 contratos, `sugestoes_validacao_tau0.25.parquet`. `test_avalia_o_ficheiro_do_modelo_por_defeito_e_nao_o_de_tau_0` e `test_metricas_do_campeao_sao_as_da_validacao_atual` (dados reais) → passed |
| O portão (`src/model_gate.py`, `make portao`) avalia um treino novo na validação e recusa-o se **qualquer** top-5 descer face ao campeão (e dá erro se o campeão tiver sido avaliado noutra validação, com outros contratos ou outro período, ou se o treino avaliado não acabar antes da validação ou for de outra janela que não a da validação dada), no código completo ou na classe, em todos os contratos ou só nos de texto novo. Recusado, o ficheiro do campeão não muda. Num empate (os 4 iguais) é aceite, mas o campeão fica, para o ficheiro continuar a apontar para o seu registo fixo; se algum subir e nenhum descer, passa a campeão. | ✅ | `make portao PASTA=metrics/e4_s1/n_min_10_ep_24` → "Empate: aceite, mas o campeão continua a ser metrics/e4_s1/n_min_10_ep_24/…", md5 do `validacao.json` igual antes e depois (`d0148009…`); `make portao PASTA=metrics/e4_s1/n_min_10_ep_12` → `ModelValidationFailedException`, 4 métricas "check failed", md5 igual. Testes `test_qualquer_descida_numa_metrica_falha` (×4), `test_campeao_melhor_recusa_o_modelo_e_nao_muda_o_ficheiro` (×4), `test_empate_vindo_de_outra_pasta_mantem_o_campeao`, `test_subir_numa_metrica_passa_a_campeao`, `test_campeao_avaliado_noutra_validacao_da_erro_e_nao_muda_o_ficheiro`, `test_treino_que_nao_acaba_antes_da_validacao_e_recusado` (×3), `test_treino_de_outra_janela_com_esta_validacao_e_recusado` → passed; `make portao PASTA=metrics/e4_s6/desde_2019` → `GateError` (outra janela) |
| Cada avaliação do portão fica registada como um run no MLflow (parâmetros, métricas, custo do treino e se foi aceite ou recusada), num ficheiro local (`mlflow.db`, fora do git), que se vê com `make mlflow`. | ✅ | `test_run_do_mlflow_leva_as_metricas_os_parametros_e_o_custo_do_treino` → passed. No `mlflow.db` real, a experiência `portao_validacao` tem runs `aceite`, `empate` e `recusado` (FAILED) |
| Todos os treinos já guardados em `metrics/` (E4-H1 a E4-S6, 96 ficheiros de sugestões), a régua e a curva por época do E4-S5 ficam registados no MLflow, numa experiência à parte (`historico_e4`, `make historico`), com todas as métricas da avaliação comum e o custo de cada treino. Não passam pelo portão. | ✅ | `.venv/bin/python scripts/registar_historico.py` → "98 runs novos" (96 + régua + curva), 256 s, pico 0,6 GB; outra vez → "0 runs novos". Conferido com os docs: régua 26,32% / 15,60% (secção 12: 26,3% / 15,6%), E4-H1 43,19% / 37,91% (secção 13: 43,2% / 37,9%), campeão 78,35% / 84,11% (secção 16), desde 2019 78,63% (secção 19); curva 65,1% → 77,8% (secção 15: "nunca passa de 77,8%") |
| O CI falha se o `metrics/validacao.json` do ramo tiver algum top-5 abaixo do do `main`. | ✅ (a partir do merge) | `VALIDACAO_MAIN=<cópia igual> pytest -k nao_desce` → 1 passed; `VALIDACAO_MAIN=<cópia com +0,0001 numa métrica>` → 1 failed. Neste PR salta, porque o `main` ainda não tem o ficheiro |
| O CI falha se o bloco de métricas disser que vem de outro conjunto que não a validação (outubro de 2025). | ✅ | `test_campeao_guardado_vem_da_validacao` (lê o ficheiro guardado), `test_bloco_de_outro_conjunto_falha` (×3), `test_bloco_fora_de_outubro_de_2025_falha` (×3) → passed |
| O treino, a curva por época e o portão recusam o teste com as mesmas funções (`check_inputs` e `check_dates` em `src/train_tfidf.py`): pelo nome (`teste.parquet`) e pelas datas (…), mesmo com outro nome. Testes no CI confirmam as duas recusas. Os notebooks dos spikes já fechados não mudam. | ✅ | `test_caminho_do_teste_da_erro` (×2), `test_copia_do_teste_com_outro_nome_da_erro_pelas_datas` (×2), `test_validacao_no_treino_da_erro_pelas_datas`, `test_teste_e_recusado_e_nada_se_grava`, `test_copia_do_teste_com_outro_nome_e_recusada_pelas_datas`, `test_contrato_no_proprio_dia_do_limite_da_erro` (×2) → passed (correm no CI: dados inventados) |
| Cada verificação falha quando se estraga de propósito o que ela guarda (12 mutações no portão e uma na comparação com o `main`, todas apanhadas). | ✅ | Tabela abaixo: 36 estragos, todos apanhados |

Caixas do `docs/backlog.md` marcadas como `[x]` neste PR: os 8 critérios do E4-T5, e "A versão do MLflow fica fixa no `requirements.txt`" da E5-T3 (feito aqui).

### Testes focados

`.venv/bin/pytest -q tests/unit/test_model_gate.py tests/integration/test_model_gate_pipeline.py tests/integration/test_registar_historico.py tests/integration/test_tfidf_pipeline.py tests/integration/test_curva_epocas.py tests/integration/test_correr_experiencia_pipeline.py` → passed (ver a suíte inteira abaixo; o salto é a comparação com o `main`, que só corre no CI).

### O comportamento que já existia continua igual

- Suíte inteira: `.venv/bin/pytest -q` → **489 passed, 1 skipped**. Depois da 2.ª revisão: 485 passed, 1 skipped; depois da 1.ª revisão: 474 passed, 1 skipped; antes dela: 465 passed, 1 skipped. Antes do trabalho: não medido nesta sessão.
- Os testes já existentes do treino (`test_tfidf_pipeline.py`, `test_curva_epocas.py`, `test_correr_experiencia_pipeline.py`) passam sem mudanças depois da recusa pelas datas: os dados inventados já respeitavam as datas.
- O campeão é um treino já guardado: os números do TF-IDF não mudam (78,35% / 84,11% / 75,14% / 82,18%, iguais aos do `metrics/e4_s1/` avaliados antes deste PR).
- CI: ainda não correu. O ramo só vai para o GitHub (`git push`) depois de aprovada esta descrição; atualizo esta linha com o resultado.

### Os testes novos falham com o código estragado

| Teste | Estrago introduzido | Resultado com o estrago |
|---|---|---|
| `test_bloco_de_outro_conjunto_falha` | `if block.get("conjunto") != CONJUNTO:` → `if False:` | falhou ✅ |
| `test_bloco_fora_de_outubro_de_2025_falha` | período ignorado (`if False:`) | falhou ✅ |
| `test_bloco_fora_de_outubro_de_2025_falha` | fim do período sem limite (`or fim >= TEST_START` → `or False`) | falhou ✅ |
| `test_bloco_sem_uma_metrica_falha` | métricas em falta ignoradas | falhou ✅ |
| `test_qualquer_descida_numa_metrica_falha` | limiar `campeao[m] - 0.01` | falhou ✅ |
| `test_qualquer_descida_numa_metrica_falha` | só a 1.ª métrica entra no portão | falhou ✅ |
| `test_portao_olha_para_o_codigo_e_a_classe_em_todos_e_no_texto_novo` | `MEDIDAS = ("codigo",)` | falhou ✅ |
| `test_teste_e_recusado_e_nada_se_grava` | sem `check_not_test` no portão | falhou ✅ |
| `test_avalia_o_ficheiro_do_modelo_por_defeito_e_nao_o_de_tau_0` | lê sempre o ficheiro de τ = 0 | falhou ✅ |
| `test_sugestoes_por_outra_ordem_dao_erro` | ordem dos contratos não verificada | falhou ✅ |
| `test_campeao_de_outro_conjunto_da_erro` | o campeão guardado não é verificado | falhou ✅ |
| `test_campeao_melhor_recusa_o_modelo_e_nao_muda_o_ficheiro` | grava o campeão mesmo recusado (sem `raise`) | falhou ✅ |
| `test_campeao_guardado_nao_desce_face_ao_main` | o `main` com +0,0001 numa métrica | falhou ✅ |
| `test_campeao_avaliado_noutra_validacao_da_erro_e_nao_muda_o_ficheiro` | sem `check_same_validation` | falhou ✅ |
| `test_validacao_diferente_do_campeao_falha` | só os contratos comparados, não o período | falhou ✅ |
| `test_validacao_no_treino_da_erro_pelas_datas` | sem `check_dates(train_path, VAL_START)` | falhou ✅ |
| `test_copia_do_teste_com_outro_nome_da_erro_pelas_datas` | sem `check_dates(val_path, TEST_START)` no treino | falhou ✅ |
| `test_copia_do_teste_com_outro_nome_e_recusada_pelas_datas` | sem `check_dates` no portão | falhou ✅ |
| `test_outra_janela_fica_de_fora_sem_a_sua_validacao_e_entra_com_ela` | janela ignorada (sempre a validação por defeito) | falhou ✅ |
| `test_correr_outra_vez_nao_duplica` | não salta os runs que já existem | falhou ✅ |
| `test_curva_tem_uma_metrica_por_epoca` | curva sem `step=epoca` | falhou ✅ |
| `test_metricas_sao_as_da_avaliacao_comum` | grupos trocados (todos ↔ texto novo) | falhou ✅ |
| `test_um_run_por_ficheiro_de_sugestoes_mais_a_regua_e_a_curva` | τ errado para o ficheiro sem sufixo | falhou ✅ |
| `test_cada_metrica_do_relatorio_vai_para_o_seu_nome` | top-1 gravado no lugar do top-5 (escapava antes deste teste) | falhou ✅ |
| `test_sugestoes_por_outra_ordem_dao_erro` (histórico) | ordem não verificada (escapava antes deste teste) | falhou ✅ |
| `test_ficheiro_de_sugestoes_sem_tau_no_nome_fica_de_fora` | τ desconhecido não saltado | falhou ✅ |
| `test_treino_que_nao_acaba_antes_da_validacao_e_recusado` | sem `check_training_dates` | falhou ✅ |
| `test_treino_que_nao_acaba_antes_da_validacao_e_recusado` | `teste.parquet` no `treino.json` aceite | falhou ✅ |
| `test_empate_vindo_de_outra_pasta_mantem_o_campeao` | o empate passa a campeão | falhou ✅ |
| `test_validacao_diferente_do_campeao_falha` | a impressão digital dos contratos não é comparada | falhou ✅ |
| `test_contrato_no_proprio_dia_do_limite_da_erro` | `data.max() >= until` → `>` (escapava antes deste teste) | falhou ✅ |
| `test_bloco_sem_periodo_da_erro_do_portao` | período em falta não verificado | falhou ✅ |
| `test_treino_de_outra_janela_com_esta_validacao_e_recusado` | sem `check_window` | falhou ✅ |
| `test_treino_que_nao_acaba_antes_da_validacao_e_recusado` | `>= VAL_START` → `>` (escapava antes do caso de 1 de outubro) | falhou ✅ |
| `test_cada_grupo_e_medida_vai_para_o_seu_nome` | `gate_metrics` lê sempre o grupo "todos" (no CI escapava; só o teste com dados reais o via) | falhou ✅ |
| `test_run_que_falhou_a_meio_volta_a_fazer_se` | os runs FAILED contam como feitos | falhou ✅ |

Em todos, o código foi reposto e a suíte voltou a passar.

### Números que provam que funciona

Com os dados reais (só a validação, outubro de 2025; o teste nunca foi lido):

| Medida | Campeão (`e4_s1/n_min_10_ep_24`, τ = 0,25) | Desafiante `e4_s1/n_min_10_ep_12` | Portão |
|---|---|---|---|
| top-5 código, todos | 78,35% | 77,52% | desce |
| top-5 classe, todos | 84,11% | 83,13% | desce |
| top-5 código, texto novo | 75,14% | 74,15% | desce |
| top-5 classe, texto novo | 82,18% | 81,04% | desce |
| Resultado | — | **recusado**, `validacao.json` igual (md5 `d0148009…`) | |

O mesmo campeão avaliado outra vez dá os mesmos 4 números: empate, aceite, e o ficheiro não muda. Fonte: `make portao PASTA=...`, nesta sessão, e os runs da experiência `portao_validacao` no `mlflow.db`.

### Revisão de código

`/requesting-code-review` (revisor independente, sem contexto da tarefa) sobre `cebba4a..c41bd2b`: **pronto para merge com correções**. 0 críticos, 2 importantes, 7 menores. O revisor não correu comandos de git (não tinha a autorização do utilizador no prompt) e reviu os ficheiros do diretório de trabalho, iguais a `c41bd2b`. Correu a suíte (465 passed, 1 skipped) e confirmou 5 mutações.
- Corrigido (no segundo commit do PR, com esta descrição):
  - **Importante:** o portão não verificava que o campeão e o desafiante foram avaliados na mesma validação → `check_same_validation`, com testes.
  - **Importante:** o teste só se recusava pelo nome, e o E5-T2 contava com uma recusa mais forte → `check_dates`/`check_inputs` (pelas datas) no treino, na curva e no portão, com testes; o texto do E5-T2 diz agora o que existe.
  - Menor: o backlog dizia "os mesmos números da secção 16" e só três o são → reescrito; as provas do empate e da recusa estão neste PR.
  - Menor: `except Exception` marcava como `recusado` qualquer erro → só `ModelValidationFailedException`.
  - Menor: um ficheiro de sugestões com sufixo desconhecido rebentava o histórico → salta-se com aviso, com teste.
  - Depois das correções, a suíte inteira apanhou 5 falhas num teste que já existia ("épocas inválidas dão erro antes de ler os ficheiros"): a verificação pelas datas lia os ficheiros antes de verificar as épocas. A ordem foi trocada e a suíte passou (474 passed, 1 skipped).
- Não corrigido, e porquê:
  - O passo do CI salta a comparação se o `git show` falhar por qualquer razão (não só por o `main` não ter o ficheiro). Depois deste merge o ficheiro existe no `main`; se mudar de nome, o teste unitário que lê o `metrics/validacao.json` também falha. Fica como está, por simplicidade.
  - O teste com dados reais lê o `validacao.json` ao importar; sem o ficheiro dá erro de recolha em vez de falha. É cosmético: o teste unitário já falha nesse caso.
  - "A frase *Planned for phase 4* do `CLAUDE.md` ficou desatualizada": não existe; foi substituída neste PR pela descrição do portão.

2.ª revisão (revisor novo, sem contexto) sobre `cebba4a..cda780b`: **pronto para merge com correções**. 0 críticos, 2 importantes, 5 menores. Também não correu git (mesma razão) e correu a suíte: 474 passed, 1 skipped.
- Corrigido (no terceiro commit do PR):
  - **Importante:** o portão não olhava para as datas do treino que avalia; um treino com contratos de outubro passava com números inflacionados → `check_training_dates`, com testes.
  - **Importante:** num empate, o campeão passava a apontar para a pasta do desafiante (por defeito `metrics/tfidf/`, que o próximo treino reescreve) → num empate o campeão fica; só uma subida o troca (decisão 8 atualizada).
  - Menor: a fronteira `>=` das datas não tinha teste (trocar por `>` passava) → 4 testes unitários de `check_dates`.
  - Menor: só se comparava o número de contratos, não os contratos → `contratos_sha256` no bloco; o `metrics/validacao.json` foi escrito de novo pelo portão (mesmas métricas, agora com a impressão digital).
  - Menor: o passo do CI saltava a comparação em qualquer erro do git → só salta se o `main` não tiver o ficheiro (`git cat-file -e`).
  - Menor: o teste com dados reais lia o ficheiro ao importar → lê dentro do teste.
  - Menor: um bloco sem `periodo` dava `KeyError` → `GateError`, com teste.
- Não corrigido: nada.

3.ª revisão (revisor novo, sem contexto) sobre `cebba4a..7961601`: **pronto para merge com correções pequenas**. 0 críticos, 0 importantes, 8 menores. Correu a suíte numa cópia sem `data/`, como o CI (462 passed, 24 skipped), e confirmou o empate, a fronteira das datas e a comparação com o `main` estragando o código.
- Corrigido (no quarto commit do PR):
  - O portão não confirmava a janela do treino → `check_window`, com teste e com a prova nos dados reais (o treino desde 2019 é recusado com a validação por defeito).
  - A fronteira `>= VAL_START` de `check_training_dates` não tinha teste → caso de 1 de outubro acrescentado.
  - Nenhum teste do CI guardava a correspondência grupo → nome em `gate_metrics` → teste unitário com um relatório inventado.
  - O histórico saltava para sempre um run que falhou a meio → só conta os runs `FINISHED`, com teste.
- Não corrigido, e porquê:
  - A comparação com o `main` no CI só olha para as métricas, não para a impressão digital dos contratos. Se a preparação mudar a validação, o teste com dados reais e o próprio portão (`check_same_validation`) param primeiro; trocar de campeão nesse caso é uma decisão a tomar quando acontecer (avaliar outra vez o campeão sem campeão anterior, num PR próprio).
  - Um `--campeao` mal escrito cria um campeão novo em silêncio. O caminho por defeito é o certo e o ficheiro novo apareceria no `git status` e no diff do PR.
  - A forma do bloco (métricas entre 0 e 1, período invertido) não se valida: o bloco é sempre escrito pelo próprio portão.
  - O histórico registaria também `metrics/tfidf/` se existisse; essa pasta não está no git e só existe no pod.
  - O rascunho do E5 vem no mesmo PR: decisão do utilizador, escrita acima.
- Não se lançou uma 4.ª revisão: as correções são pequenas e cada uma tem um teste que falha com o estrago (tabela acima).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
