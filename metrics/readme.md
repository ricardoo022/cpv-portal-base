# metrics/ · resultados guardados no git

Ficheiros pequenos que os treinos no pod escreveram e que se copiaram para o portátil. São os números que os `docs/` citam, e os notebooks e os testes leem-nos. As pastas `e4_*/` são registos fixos: **nunca se escreve nelas**. Um treino novo do TF-IDF escreve em `metrics/tfidf/` (que o treino seguinte reescreve) e um do BERT em `metrics/bert/<modelo>/`.

Em cada pasta de treino: as sugestões da validação (`sugestoes_validacao*.parquet`, um ficheiro por τ), o `treino.json` (parâmetros, tempo, memória e `ficheiro_do_modelo_por_defeito`) e, quando há, os logs do pod (`pod_*.log`) e o `pod_info.json` (máquina, preço e custo).

| Ficheiro ou pasta | O que tem | Quem o lê |
|---|---|---|
| [`validacao.json`](validacao.json) | O campeão do portão de qualidade (E4-T5): o top-5 do melhor modelo até agora, no código completo e na classe, em todos os contratos e no texto novo, com o conjunto, o período e o SHA-256 da lista de contratos da validação. Só o `make portao` o muda. | `src/model_gate.py`, `tests/unit/test_model_gate.py` (o CI compara-o com o do `main`) |
| `e4_s0/` | Spike E4-S0: tempo e memória de um treino do TF-IDF no portátil e no pod. | resultado do E4-S0 no [`backlog.md`](../docs/backlog.md) |
| `e4_h1/` | História E4-H1: o primeiro TF-IDF e as sugestões para 3 descrições de exemplo. | [`04_tfidf.ipynb`](../notebooks/04_tfidf.ipynb), secção 13 |
| `e4_s4/` | Spike E4-S4: um treino por `alpha` (1e-4 a 1e-7), cada um com vários τ. | [`05_tfidf_afinacao.ipynb`](../notebooks/05_tfidf_afinacao.ipynb), secção 14 |
| `e4_s5/` | Spike E4-S5: `alpha` 1e-7 e 1e-8, épocas fixas (3 a 24) e a curva época a época (`curva/`). | [`06_tfidf_epocas.ipynb`](../notebooks/06_tfidf_epocas.ipynb), secção 15 |
| `e4_s1/` | Spike E4-S1: a grelha N × épocas (18 pastas `n_min_<N>_ep_<épocas>`, cada uma com 3 τ). `n_min_10_ep_24/` é o campeão. | [`07_tfidf_grelha.ipynb`](../notebooks/07_tfidf_grelha.ipynb), secção 16, `validacao.json` |
| `e4_s3/` | Spike E4-S3: o TF-IDF com a entidade compradora no texto (`com_entidade/`). | [`08_entidade.ipynb`](../notebooks/08_entidade.ipynb), secção 17 |
| `e4_s2/` | Spike E4-S2: janelas desde 2019 e 2025, com a regra de paragem por defeito, e os logs e os md5 da preparação no pod. | [`09_janela.ipynb`](../notebooks/09_janela.ipynb), secção 18 |
| `e4_s6/` | Spike E4-S6: a janela desde 2019 com 24 épocas. | [`10_janela_24_epocas.ipynb`](../notebooks/10_janela_24_epocas.ipynb), secção 19 |

Todos os treinos daqui estão também no MLflow, na experiência `historico_e4` (`make historico`, depois `make mlflow`).
