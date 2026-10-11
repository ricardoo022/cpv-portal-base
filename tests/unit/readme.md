# tests/unit/ · testes unitários

Um ficheiro de testes por ficheiro de [`src/`](../../src/).

| Ficheiro | Testa | Atalho |
|---|---|---|
| [`test_cpv_data.py`](test_cpv_data.py) | Tirar os códigos do texto do CPV, nível de um código, períodos. | `make test-preparacao` |
| [`test_prepare.py`](test_prepare.py) | Cada passo do `prepare.py`: texto de entrada, cópias repetidas, treino sem divisões, um exemplo por texto, verificações. | `make test-preparacao` |
| [`test_labels.py`](test_labels.py) | Códigos raros sobem para a classe ou o grupo, nunca para a divisão. | `make test-preparacao` |
| [`test_evaluation.py`](test_evaluation.py) | As notas: acerto no código, na classe e na divisão, top-1 e top-5, F1 macro, contratos só com divisão, medidas hierárquicas. | `make test-avaliacao` |
| [`test_baselines.py`](test_baselines.py) | A régua: as 5 etiquetas mais frequentes, depois da regra dos códigos raros, empate pelo menor código; a variante que pesa cada exemplo pelo `n_contratos`. | `make test-avaliacao` |
| [`test_comparison.py`](test_comparison.py) | O bootstrap emparelhado e a regra de escolha entre dois modelos. | `make test-avaliacao` |
