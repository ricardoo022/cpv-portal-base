# Atalhos para os comandos do projeto. `make help` mostra a lista.
# Cada alvo tem um comentário "##" a seguir ao nome: é o texto que o `make help` mostra.

PY := .venv/bin/python
PYTEST := .venv/bin/pytest
JUPYTER := .venv/bin/jupyter

.DEFAULT_GOAL := help
.PHONY: help setup dados preparar preparar-janela test test-unit test-integracao test-preparacao test-avaliacao test-tfidf notebook regua treinar-tfidf sugerir tfidf afinacao epocas grelha entidade janela janela-24

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

preparar-janela: ## Cria treino, validação e teste com outro início do treino: make preparar-janela INICIO=2019-01-01 SAIDA=data/janelas/2019
	$(PY) src/prepare.py --inicio-treino $(INICIO) --saida $(SAIDA)

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

test-tfidf: ## Testes do modelo TF-IDF, do treino e do comando de sugestões
	$(PYTEST) -q tests/unit/test_tfidf.py tests/integration/test_tfidf_pipeline.py tests/integration/test_tfidf_dados_reais.py

notebook: ## Volta a correr o notebook de exploração (precisa de "make preparar" antes)
	cd notebooks && ../$(JUPYTER) nbconvert --to notebook --execute --inplace 01_exploracao.ipynb

regua: ## Volta a correr o notebook da régua "códigos mais comuns" na validação (precisa de "make preparar" antes)
	cd notebooks && ../$(JUPYTER) nbconvert --to notebook --execute --inplace 03_regua.ipynb

treinar-tfidf: ## Treina o TF-IDF (N = 10, alpha e τ do E4-S4, 24 épocas do E4-S5) e grava as sugestões da validação em metrics/tfidf/ (no pod: ~32 GB)
	$(PY) src/train_tfidf.py --n-min 10

sugerir: ## Mostra 5 códigos para 3 descrições de exemplo (precisa do modelo de "make treinar-tfidf")
	$(PY) src/suggest.py --modelo models/tfidf_n10.joblib

tfidf: ## Volta a correr o notebook do TF-IDF contra a régua (precisa de metrics/e4_h1/ e de "make preparar")
	cd notebooks && ../$(JUPYTER) nbconvert --to notebook --execute --inplace 04_tfidf.ipynb

afinacao: ## Volta a correr o notebook da afinação do TF-IDF, alpha e τ (precisa de metrics/e4_s4/ e de "make preparar")
	cd notebooks && ../$(JUPYTER) nbconvert --to notebook --execute --inplace 05_tfidf_afinacao.ipynb

epocas: ## Volta a correr o notebook das épocas e do alpha mais pequeno do TF-IDF (precisa de metrics/e4_s5/ e de "make preparar")
	cd notebooks && ../$(JUPYTER) nbconvert --to notebook --execute --inplace 06_tfidf_epocas.ipynb

grelha: ## Volta a correr o notebook da grelha N × épocas × τ do TF-IDF (precisa de metrics/e4_s1/ e de "make preparar")
	cd notebooks && ../$(JUPYTER) nbconvert --to notebook --execute --inplace 07_tfidf_grelha.ipynb

entidade: ## Volta a correr o notebook da entidade compradora no TF-IDF (precisa de metrics/e4_s1/, metrics/e4_s3/ e de "make preparar")
	cd notebooks && ../$(JUPYTER) nbconvert --to notebook --execute --inplace 08_entidade.ipynb

janela: ## Volta a correr o notebook da janela de treino (precisa de metrics/e4_s1/, metrics/e4_s2/, data/janelas/ e de "make preparar")
	cd notebooks && ../$(JUPYTER) nbconvert --to notebook --execute --inplace 09_janela.ipynb

janela-24: ## Volta a correr o notebook da janela desde 2019 com 24 épocas (precisa de metrics/e4_s1/, metrics/e4_s6/, data/janelas/2019/ e de "make preparar")
	cd notebooks && ../$(JUPYTER) nbconvert --to notebook --execute --inplace 10_janela_24_epocas.ipynb
