# tests/ · testes (pytest)

Todos usam dados **inventados à mão**, com o resultado certo calculado no comentário de cada teste. Não precisam de `data/`, por isso correm no CI do GitHub em cada envio.

| Pasta | O que testa | Como se corre |
|---|---|---|
| [`unit/`](unit/) | Cada função sozinha, com poucos contratos inventados. | `make test-unit` |
| [`integration/`](integration/) | O pipeline inteiro: `.xlsx` inventados → treino, validação e teste → avaliação. | `make test-integracao` |

Tudo de uma vez: `make test`.

Cada teste foi visto a falhar quando se estragou de propósito a regra que ele guarda (mutation testing).
