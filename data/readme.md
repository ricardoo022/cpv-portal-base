# data/ · dados (fora do git)

Os dados não estão no git (são grandes). Esta pasta enche-se com os comandos abaixo.

| Pasta | O que tem | Como se cria |
|---|---|---|
| `raw/` | Os `.xlsx` originais do Portal BASE, um por ano (2012 a 2026). | `make dados` (descarrega) |
| `processed/raw/` | Os mesmos dados em Parquet, sem nenhuma mudança: `contratosAAAA.parquet`. | `make dados` (converte) |
| `processed/` | `treino.parquet` (jan 2022 a set 2025), `validacao.parquet` (out 2025) e `teste.parquet` (nov e dez 2025). | `make preparar` |

O teste nunca se usa para tomar decisões: as escolhas fazem-se com a validação.
