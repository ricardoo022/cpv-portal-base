# src/ · código do pipeline

Só código que transforma dados, treina ou avalia. A exploração fica em [`notebooks/`](../notebooks/).

Ordem em que os dados passam:

```
data/raw/*.xlsx → convert_to_parquet.py → prepare.py → treino / validação / teste → (modelo) → evaluation.py
```

| Ficheiro | O que faz | Como se corre |
|---|---|---|
| [`convert_to_parquet.py`](convert_to_parquet.py) | Converte os `.xlsx` de `data/raw/` para Parquet, sem mudar nada (todas as colunas como texto). | `make dados` |
| [`prepare.py`](prepare.py) | Aplica as decisões do [`qualidade_dados.md`](../docs/qualidade_dados.md) e cria treino, validação e teste em `data/processed/`. Para com erro se uma verificação falhar. | `make preparar` |
| [`cpv_data.py`](cpv_data.py) | Regras partilhadas pelo pipeline e pelos notebooks: tirar os códigos CPV do texto, o nível de um código, as datas que separam treino, validação e teste. | é importado |
| [`labels.py`](labels.py) | Etiqueta de treino: um código raro sobe para a classe ou o grupo (decisão 9). Aplica-se no treino, não no `prepare.py`. | é importado |
| [`evaluation.py`](evaluation.py) | Dá nota às 5 sugestões de um modelo: top-1, top-5 e F1 macro no código completo, na classe e na divisão, e as medidas hierárquicas. | é importado |
| [`comparison.py`](comparison.py) | Decide entre dois modelos: bootstrap emparelhado e a regra de escolha do E4. | é importado |

Testes destes ficheiros: [`tests/`](../tests/).
