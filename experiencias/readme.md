# experiencias/ · opções de cada spike, escritas antes de correr

Um YAML por spike com as opções que se testam e a ordem em que se comparam. Entra no git **antes** de o treino correr, e não se muda depois de ver os resultados: o histórico do git mostra que a lista foi escolhida às cegas (na literatura, *pré-registo*). O treino no pod e o notebook que decide leem o mesmo ficheiro, por isso não podem usar listas diferentes.

Os valores **em vigor** do modelo (os que ganharam) não ficam aqui: ficam no código (`ALPHA`, `TAU` e `EPOCAS` em `src/tfidf.py`). O YAML diz o que se experimenta; o código diz o que se usa.

No YAML os números escrevem-se com ponto: `1.0e-7`, não `1e-7`. O PyYAML segue o YAML 1.1, em que `1e-7` é texto; o leitor (`scripts/correr_experiencia.py`, função `ler`) recusa-o com erro.

| Ficheiro | Spike | Quem o lê |
|---|---|---|
| [`e4_s5.yaml`](e4_s5.yaml) | E4-S5: `alpha` 0,00000001, τ (se o `alpha` mudar) e número fixo de épocas | [`scripts/correr_experiencia.py`](../scripts/correr_experiencia.py) (no pod) e [`notebooks/06_tfidf_epocas.ipynb`](../notebooks/06_tfidf_epocas.ipynb) |
