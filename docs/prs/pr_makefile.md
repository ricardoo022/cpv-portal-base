## Makefile com atalhos para os comandos do projeto

## Em poucas palavras

Este PR junta um Makefile: atalhos para os comandos do projeto. Em vez de decorar `.venv/bin/pytest tests/unit/test_evaluation.py`, escreve-se `make test-avaliacao`. O comando `make help` (ou só `make`) mostra a lista de atalhos, cada um com uma frase a explicar o que faz. Não muda código nem testes.

| Atalho | O que faz |
|---|---|
| `make help` | mostra a lista |
| `make setup` | cria o ambiente e instala as bibliotecas |
| `make dados` | descarrega os .xlsx e converte para Parquet |
| `make preparar` | cria treino, validação e teste (`prepare.py`) |
| `make test` | todos os testes |
| `make test-unit` / `make test-integracao` | só os unitários / só o pipeline inteiro |
| `make test-preparacao` | testes das regras dos dados |
| `make test-avaliacao` | testes das notas aos modelos e do bootstrap |
| `make notebook` | volta a correr o notebook de exploração |

## O que mudou no código

### `Makefile` (novo)

Cada atalho é uma linha com o nome e um comentário `##`, seguida do comando:

```make
test-avaliacao: ## Testes das notas aos modelos e do bootstrap (evaluation.py, comparison.py)
	$(PYTEST) -q tests/unit/test_evaluation.py tests/unit/test_comparison.py
```

O `make help` lê esses comentários `##` e mostra-os:

```make
help: ## Mostra esta lista
	@grep -E "^[a-z-]+:.*## " $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  make %-18s %s\n", $$1, $$2}'
```
Procura as linhas que têm `nome: ## texto` e escreve "make nome" seguido do texto. Para acrescentar um atalho, basta seguir o mesmo formato.

## Outras mudanças

- `CLAUDE.md`: na secção de comandos, `make help` como primeira linha.
- `readme.md`: uma frase a dizer que `make help` mostra os comandos do projeto.
- `docs/prs/pr_makefile.md`: esta descrição.

### Ficou igual de propósito

- Os comandos são os mesmos do CLAUDE.md: o Makefile só lhes dá nomes curtos.
- O CI continua a correr `pytest` diretamente; não precisa do Makefile.

## Validação e provas

| Atalho | Resultado |
|---|---|
| `make test` | 142 passed |
| `make test-unit` | 127 passed |
| `make test-integracao` | 15 passed |
| `make test-preparacao` | 60 passed |
| `make test-avaliacao` | 67 passed |

As contagens batem: 60 + 67 = 127 unitários; 127 + 15 = 142.

`setup`, `dados`, `preparar` e `notebook` não foram corridos, porque são lentos ou descarregam e gravam dados. Com `make -n` (mostra os comandos sem os executar) confirmou-se que correm exatamente os comandos do CLAUDE.md.

CI do commit de cima: ver os checks deste PR (o merge só se faz com eles verdes).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
