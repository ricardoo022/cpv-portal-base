# tests/unit/ · testes unitários

Testes a funções de [`src/`](../../src/) e [`scripts/`](../../scripts/).

| Ficheiro | Testa | Atalho |
|---|---|---|
| [`test_cpv_data.py`](test_cpv_data.py) | Tirar os códigos do texto do CPV, nível de um código, períodos (também com outro início do treino, E4-T4). | `make test-preparacao` |
| [`test_prepare.py`](test_prepare.py) | Cada passo do `prepare.py`: texto de entrada, cópias repetidas, treino sem divisões, um exemplo por texto, verificações; o início do treino indicado (E4-T4) em `load_all`, `add_columns`, `check_sets` e `run` (data não anterior à validação dá `ValueError`), e a linha de comando com `--inicio-treino` sem `--saida` (erro do argparse, sem escrever nada) e sem argumentos ou com a data por defeito (grava em `data/processed/`, como o `make preparar`). | `make test-preparacao` |
| [`test_labels.py`](test_labels.py) | Códigos raros sobem para a classe ou o grupo, nunca para a divisão. | `make test-preparacao` |
| [`test_evaluation.py`](test_evaluation.py) | As notas: acerto no código, na classe e na divisão, top-1 e top-5, F1 macro, contratos só com divisão, medidas hierárquicas. | `make test-avaliacao` |
| [`test_baselines.py`](test_baselines.py) | A régua: as 5 etiquetas mais frequentes, depois da regra dos códigos raros, empate pelo menor código; a variante que pesa cada exemplo pelo `n_contratos`. | `make test-avaliacao` |
| [`test_comparison.py`](test_comparison.py) | O bootstrap emparelhado e a regra de escolha entre dois modelos. | `make test-avaliacao` |
| [`test_tfidf.py`](test_tfidf.py) | Vetorizador TF-IDF, log-odds prior, modelo SGDClassifier, regularização (`alpha`), épocas fixas (`EPOCAS = 24`), e sugestões com correção τ. | `make test-tfidf` |
| [`test_train_tfidf.py`](test_train_tfidf.py) | Nomes de ficheiros de sugestões por τ (`train_tfidf.py`), parâmetros `sgd_epochs` e validação prévia de argumentos. | `make test-tfidf` |
| [`test_correr_experiencia.py`](test_correr_experiencia.py) | `scripts/correr_experiencia.py` (E4-S5 e E4-S1): os YAML do E4-S5 (modo `fases`) e do E4-S1 (modo `grelha`) têm as opções do backlog (escritas à mão no teste); números escritos como texto (`1e-7`), opções repetidas, listas vazias ou chaves erradas dão erro; os comandos de cada fase (incluindo a grelha N × épocas, com um ficheiro por τ), do mais demorado ao menos. | `.venv/bin/pytest tests/unit/test_correr_experiencia.py` |

