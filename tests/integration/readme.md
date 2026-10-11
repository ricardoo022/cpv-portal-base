# tests/integration/ · teste do pipeline inteiro

| Ficheiro | O que faz |
|---|---|
| [`test_pipeline.py`](test_pipeline.py) | Cria `.xlsx` inventados no formato do Portal BASE, corre a conversão e a preparação, e verifica onde acaba cada contrato. No fim, a régua "códigos mais comuns" conta o treino que a preparação escreveu, e a avaliação corre sobre a validação. Cada contrato inventado põe à prova uma regra do [`qualidade_dados.md`](../../docs/qualidade_dados.md). |
| [`test_regua_dados_reais.py`](test_regua_dados_reais.py) | A régua "códigos mais comuns" sobre o treino e a validação reais: confirma que as 5 etiquetas e o top-5 da secção 12 do `qualidade_dados.md` continuam certos. Salta se `data/processed/` não existir (no CI). Confirma também que o notebook da régua nunca abre o teste. |

Corre com `make test-integracao`. Os testes com dados reais precisam de `make preparar` antes. Quando uma regra do `prepare.py` muda, os contratos inventados daqui mudam também.
