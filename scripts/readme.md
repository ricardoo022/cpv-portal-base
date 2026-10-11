# scripts/ · utilitários fora do pipeline

| Ficheiro | O que faz | Como se corre |
|---|---|---|
| [`download_data.py`](download_data.py) | Descarrega os `.xlsx` de contratos do Portal BASE (dados.gov.pt) para `data/raw/`. Salta os ficheiros que já existem e não mudaram. | `make dados`, ou `.venv/bin/python scripts/download_data.py 2025 2024` para anos escolhidos |
