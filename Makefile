# Atalhos para os comandos do projeto. `make help` mostra a lista.
# Cada alvo tem um comentário "##" a seguir ao nome: é o texto que o `make help` mostra.

PY := .venv/bin/python
PYTEST := .venv/bin/pytest
JUPYTER := .venv/bin/jupyter

.DEFAULT_GOAL := help
.PHONY: help setup dados preparar preparar-janela test test-unit test-integracao test-preparacao test-avaliacao test-tfidf notebook regua treinar-tfidf sugerir portao mlflow historico tfidf afinacao epocas grelha entidade janela janela-24 tokens setup-gpu verificar-gpu treinar-bert test-bert

help: ## Mostra esta lista
	@grep -E "^[a-z-]+:.*## " $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  make %-18s %s\n", $$1, $$2}'

setup: ## Cria o ambiente (.venv) no portátil: requirements-cpu.txt (o requirements.txt + torch só para processador)
	python3 -m venv .venv
	.venv/bin/pip install -r requirements-cpu.txt

setup-gpu: ## Cria o ambiente (.venv) no pod com GPU: requirements-gpu.txt (o requirements.txt + torch com CUDA)
	python3 -m venv .venv
	.venv/bin/pip install -r requirements-gpu.txt

verificar-gpu: ## No pod, antes de treinar o BERT: falha se o torch não vir a GPU; se a vir, mostra o nome
	$(PY) -c "import sys; sys.path.insert(0, 'src'); import bert; bert.check_gpu()"

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

test-bert: ## Testes do BERT (E5-T2) com modelos minúsculos de pesos aleatórios, só no processador
	$(PYTEST) -q tests/unit/test_bert.py tests/unit/test_requisitos.py tests/integration/test_bert_pipeline.py tests/integration/test_makefile_bert.py

test-tfidf: ## Testes do modelo TF-IDF, do treino e do comando de sugestões
	$(PYTEST) -q tests/unit/test_tfidf.py tests/integration/test_tfidf_pipeline.py tests/integration/test_tfidf_dados_reais.py

notebook: ## Volta a correr o notebook de exploração (precisa de "make preparar" antes)
	cd notebooks && ../$(JUPYTER) nbconvert --to notebook --execute --inplace 01_exploracao.ipynb

regua: ## Volta a correr o notebook da régua "códigos mais comuns" na validação (precisa de "make preparar" antes)
	cd notebooks && ../$(JUPYTER) nbconvert --to notebook --execute --inplace 03_regua.ipynb

treinar-tfidf: ## Treina o TF-IDF (N = 10, alpha e τ do E4-S4, 24 épocas do E4-S5) e grava as sugestões da validação em metrics/tfidf/ (no pod: ~32 GB)
	$(PY) src/train_tfidf.py --n-min 10

treinar-bert: ## Treina o BERT na GPU (E5-T2): make treinar-bert MODELO=bertimbau|albertina (por defeito bertimbau) SAIDA=... (por defeito metrics/bert/MODELO) ARGS="outras opções"; os ficheiros grandes ficam em models/bert/ no pod
	$(PY) src/train_bert.py --modelo $(or $(MODELO),bertimbau) --saida $(or $(SAIDA),metrics/bert/$(or $(MODELO),bertimbau)) $(ARGS)

portao: ## Portão de qualidade: avalia na validação o treino de PASTA (por defeito metrics/tfidf) e só o grava como campeão (metrics/validacao.json) se nenhum top-5 descer; regista no MLflow
	$(PY) src/model_gate.py --pasta $(or $(PASTA),metrics/tfidf)

historico: ## Regista no MLflow (experiência historico_e4) todos os treinos guardados em metrics/, a régua e a curva do E4-S5 (precisa de "make preparar"; não duplica)
	$(PY) scripts/registar_historico.py

mlflow: ## Abre a página do MLflow (http://localhost:5000) com os runs registados em mlflow.db
	.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

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

tokens: ## Volta a correr o notebook do comprimento dos textos em tokens (E5-T1; descarrega os tokenizadores do Hugging Face, ~3 min; precisa de "make preparar")
	cd notebooks && ../$(JUPYTER) nbconvert --to notebook --execute --inplace 11_comprimento_tokens.ipynb
