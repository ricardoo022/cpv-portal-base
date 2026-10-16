# src/ · código do pipeline

Só código que transforma dados, treina ou avalia. A exploração fica em [`notebooks/`](../notebooks/).

Ordem em que os dados passam:

```
data/raw/*.xlsx → convert_to_parquet.py → prepare.py → treino / validação / teste → (modelo) → evaluation.py
```

| Ficheiro | O que faz | Como se corre |
|---|---|---|
| [`convert_to_parquet.py`](convert_to_parquet.py) | Converte os `.xlsx` de `data/raw/` para Parquet, sem mudar nada (todas as colunas como texto). | `make dados` |
| [`prepare.py`](prepare.py) | Aplica as decisões do [`qualidade_dados.md`](../docs/qualidade_dados.md) e cria treino, validação e teste em `data/processed/`. Para com erro se uma verificação falhar. `--inicio-treino AAAA-MM-DD --saida PASTA` prepara outra janela de treino (E4-T4) sem mexer em `data/processed/`. | `make preparar`, `make preparar-janela INICIO=2019-01-01 SAIDA=...` |
| [`cpv_data.py`](cpv_data.py) | Regras partilhadas pelo pipeline e pelos notebooks: tirar os códigos CPV do texto, o nível de um código, as datas que separam treino, validação e teste. | é importado |
| [`labels.py`](labels.py) | Etiqueta de treino: um código raro sobe para a classe ou o grupo (decisão 9). Aplica-se no treino, não no `prepare.py`. | é importado |
| [`evaluation.py`](evaluation.py) | Dá nota às 5 sugestões de um modelo: top-1, top-5 e F1 macro no código completo, na classe e na divisão, e as medidas hierárquicas. | é importado |
| [`baselines.py`](baselines.py) | Régua "códigos mais comuns": sugere sempre as 5 etiquetas mais frequentes do treino, sem ler o texto. É o mínimo que qualquer modelo tem de bater. | é importado |
| [`comparison.py`](comparison.py) | Decide entre dois modelos: bootstrap emparelhado e a regra de escolha do E4. | é importado |
| [`bert_models.py`](bert_models.py) | Os dois modelos do E5: o nome no Hugging Face, a revisão fixa e o comprimento máximo em *tokens* escolhido no notebook 11 (E5-T1). O treino lê-os daqui. | é importado |
| [`bert.py`](bert.py) | O BERT do E5 (E5-T2): afinar o BERTimbau ou o Albertina como classificador das etiquetas CPV e dar 5 sugestões; `check_gpu()` confirma que o `torch` vê a GPU. | é importado; `make verificar-gpu` |
| [`train_bert.py`](train_bert.py) | Treina o BERT no pod com GPU (E5-T2): recusa o teste (`check_inputs`), grava em `--saida` as sugestões com τ = 0 e τ = 0,25 por época e o `treino.json` (com o SHA-256 do treino e da validação, o commit do git e o YAML de `--experiencia`, E5-T3), e em `models/bert/` (fica no pod) as pontuações de cada época e um *checkpoint* por época. | `make treinar-bert MODELO=bertimbau` |
| [`registo_bert.py`](registo_bert.py) | Regista no MLflow um treino do BERT (E5-T3), no portátil, a partir da pasta `--saida` copiada do pod: um run com os hiperparâmetros, o commit, as impressões digitais e o YAML, e o top-1 e o top-5 de cada época (o passo do MLflow é a época, o que dá a curva). Recusa uma validação diferente da que o pod usou (SHA-256) e não duplica. | `make registar-bert PASTA=...`, `make mlflow` |
| [`model_gate.py`](model_gate.py) | Portão de qualidade (E4-T5): avalia na validação as sugestões de um treino e só o aceita como campeão (`metrics/validacao.json`) se nenhum top-5 descer face ao campeão anterior. Regista cada avaliação no MLflow (`mlflow.db`). | `make portao PASTA=...`, `make mlflow` |

Testes destes ficheiros: [`tests/`](../tests/).
