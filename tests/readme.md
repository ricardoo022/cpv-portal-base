# tests/ · testes (pytest)

Quase todos usam dados **inventados à mão**, com o resultado certo calculado no comentário de cada teste. Não precisam de `data/`, por isso correm no CI do GitHub em cada envio. A exceção são os `tests/integration/test_*_dados_reais.py` (marca `data`): leem o treino e a validação reais de `data/processed/` (nunca o teste) e saltam quando a pasta não existe, como no CI.

| Pasta | O que testa | Como se corre |
|---|---|---|
| [`unit/`](unit/) | Cada função sozinha, com poucos contratos inventados. | `make test-unit` |
| [`integration/`](integration/) | O pipeline inteiro: `.xlsx` inventados → treino, validação e teste → avaliação. | `make test-integracao` |

Os ajudantes e as *fixtures* partilhados (o pipeline sobre `.xlsx` inventados, os contratos da régua, os modelos BERT minúsculos e a troca do `git` real) estão em [`conftest.py`](conftest.py).

Tudo de uma vez: `make test`. Os testes do BERT (`make test-bert`) precisam do `torch` só para processador do `requirements-cpu.txt` (`make setup`).

Cada teste foi visto a falhar quando se estragou de propósito a regra que ele guarda (mutation testing).
