# scripts/ · utilitários fora do pipeline

| Ficheiro | O que faz | Como se corre |
|---|---|---|
| [`download_data.py`](download_data.py) | Descarrega os `.xlsx` de contratos do Portal BASE (dados.gov.pt) para `data/raw/`. Salta os ficheiros que já existem e não mudaram. | `make dados`, ou `.venv/bin/python scripts/download_data.py 2025 2024` para anos escolhidos |
| [`ensaio_treino.py`](ensaio_treino.py) | Spike E4-S0: mede o tempo e o pico de memória de um treino TF-IDF + classificador linear (numa amostra ou no treino completo) e acrescenta uma linha JSON ao ficheiro de saída. Não avalia nem grava o modelo. Os resultados do E4-S0 estão em `metrics/e4_s0/`. | `.venv/bin/python scripts/ensaio_treino.py --n-min 10 --amostra 0 --modelo sgd --saida metrics/e4_s0/pod.jsonl` (`--amostra 0` = treino completo) |
