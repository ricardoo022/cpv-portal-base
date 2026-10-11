# Atalhos para os comandos do projeto. `make help` mostra a lista.
# Cada alvo tem um comentário "##" a seguir ao nome: é o texto que o `make help` mostra.

PY := .venv/bin/python
PYTEST := .venv/bin/pytest
JUPYTER := .venv/bin/jupyter

.DEFAULT_GOAL := help
.PHONY: help setup dados preparar test test-unit test-integracao test-preparacao test-avaliacao notebook regua

help: ## Mostra esta lista
	@grep -E "^[a-z-]+:.*## " $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  make %-18s %s\n", $$1, $$2}'

setup: ## Cria o ambiente (.venv) e instala as bibliotecas do requirements.txt
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

dados: ## Descarrega os .xlsx do dados.gov.pt e converte-os para Parquet (demora)
	$(PY) scripts/download_data.py
	$(PY) src/convert_to_parquet.py

preparar: ## Cria treino, validação e teste em data/processed/ (prepare.py, ~1 min)
	$(PY) src/prepare.py

test: ## Todos os testes
	$(PYTEST) -q

test-unit: ## Só os testes unitários: cada função com dados inventados
	$(PYTEST) -q tests/unit

test-integracao: ## Só o teste do pipeline inteiro: .xlsx inventados -> treino, validação e teste -> avaliação
	$(PYTEST) -q tests/integration

test-preparacao: ## Testes das regras dos dados: códigos CPV, códigos raros, prepare.py
	$(PYTEST) -q tests/unit/test_cpv_data.py tests/unit/test_labels.py tests/unit/test_prepare.py

test-avaliacao: ## Testes das notas aos modelos, do bootstrap e da régua (evaluation.py, comparison.py, baselines.py)
	$(PYTEST) -q tests/unit/test_evaluation.py tests/unit/test_comparison.py tests/unit/test_baselines.py

notebook: ## Volta a correr o notebook de exploração (precisa de "make preparar" antes)
	cd notebooks && ../$(JUPYTER) nbconvert --to notebook --execute --inplace 01_exploracao.ipynb

regua: ## Volta a correr o notebook da régua "códigos mais comuns" na validação (precisa de "make preparar" antes)
	cd notebooks && ../$(JUPYTER) nbconvert --to notebook --execute --inplace 03_regua.ipynb
