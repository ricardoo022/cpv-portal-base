# data/ · dados (fora do git)

Os dados não estão no git (são grandes). Esta pasta enche-se com os comandos abaixo.

| Pasta | O que tem | Como se cria |
|---|---|---|
| `raw/` | Os `.xlsx` originais do Portal BASE, um por ano (2012 a 2026). | `make dados` (descarrega) |
| `processed/raw/` | Os mesmos dados em Parquet, sem nenhuma mudança: `contratosAAAA.parquet`. | `make dados` (converte) |
| `processed/` | `treino.parquet` (jan 2022 a set 2025), `validacao.parquet` (out 2025) e `teste.parquet` (nov e dez 2025). | `make preparar` |
| `janelas/AAAA/` | Treino e validação com outra data de início do treino (E4-S2 e E4-S6): `2019/`, copiada do pod, e `2025/`, recriada no portátil (mesmo md5 que no pod). Não há `teste.parquet`: o das janelas preparadas no pod nunca veio, e o que a preparação da de 2025 gravou no portátil foi apagado logo. | `make preparar-janela INICIO=AAAA-01-01 SAIDA=data/janelas/AAAA` (no pod: a de 2019 tem um pico de 4,3 GB) |

O teste nunca se usa para tomar decisões: as escolhas fazem-se com a validação.
