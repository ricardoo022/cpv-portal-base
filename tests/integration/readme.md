# tests/integration/ · teste do pipeline inteiro

| Ficheiro | O que faz |
|---|---|
| [`test_pipeline.py`](test_pipeline.py) | Cria `.xlsx` inventados no formato do Portal BASE, corre a conversão e a preparação, e verifica onde acaba cada contrato. No fim, a régua "códigos mais comuns" conta o treino que a preparação escreveu, e a avaliação corre sobre a validação. Cada contrato inventado põe à prova uma regra do [`qualidade_dados.md`](../../docs/qualidade_dados.md). |

Corre com `make test-integracao`. Quando uma regra do `prepare.py` muda, os contratos inventados daqui mudam também.
