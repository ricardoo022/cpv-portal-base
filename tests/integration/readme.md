# tests/integration/ · teste do pipeline inteiro

| Ficheiro | O que faz |
|---|---|
| [`test_pipeline.py`](test_pipeline.py) | Cria `.xlsx` inventados no formato do Portal BASE, corre a conversão e a preparação, e verifica onde acaba cada contrato. No fim, a régua "códigos mais comuns" conta o treino que a preparação escreveu, e a avaliação corre sobre a validação. Cada contrato inventado põe à prova uma regra do [`qualidade_dados.md`](../../docs/qualidade_dados.md). |
| [`test_regua_dados_reais.py`](test_regua_dados_reais.py) | A régua "códigos mais comuns" sobre o treino e a validação reais: confirma que as 5 etiquetas e o top-5 da secção 12 do `qualidade_dados.md` continuam certos. Salta se `data/processed/` não existir (no CI). Confirma também que o notebook da régua nunca abre o teste. |
| [`test_tfidf_pipeline.py`](test_tfidf_pipeline.py) | O TF-IDF (E4-H1) sobre o treino e a validação que a preparação escreve: as sugestões gravadas, a fuga de dados (nenhum contrato da validação entra no treino nem nas contagens do N), a recusa do teste, o resumo do treino e o comando de sugestões. |
| [`test_tfidf_dados_reais.py`](test_tfidf_dados_reais.py) | As sugestões do TF-IDF guardadas em `metrics/e4_h1/` sobre a validação real: confirma que são da validação atual e que o top-5 da secção 13 do `qualidade_dados.md` continua certo. Salta se `data/processed/` não existir (no CI). Confirma também que o notebook do TF-IDF nunca abre o teste. |
| [`test_tfidf_afinacao_dados_reais.py`](test_tfidf_afinacao_dados_reais.py) | A afinação do TF-IDF (E4-S4) sobre a validação real: os 16 ficheiros de `metrics/e4_s4/` são da validação atual, o `alpha` antigo repete o E4-H1, a regra de escolha dá os vencedores da secção 14 e o vencedor é o valor por defeito em `src/tfidf.py`, e o top-5 do escolhido é o do doc. Salta sem `data/processed/` (no CI); demora 1 a 2 minutos (refaz a escolha com o bootstrap sobre os dados reais). Confirma também que o notebook da afinação nunca abre o teste. |

Os ajudantes partilhados (`run_pipeline`, que corre o pipeline sobre `.xlsx` inventados, e `contratos_regua`) estão em [`tests/conftest.py`](../conftest.py), como fixtures do pytest.

Corre com `make test-integracao`. Os testes com dados reais precisam de `make preparar` antes. Quando uma regra do `prepare.py` muda, os contratos inventados daqui mudam também.
