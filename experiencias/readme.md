# experiencias/ · opções de cada spike, escritas antes de correr

Um YAML por spike com as opções que se testam e a ordem em que se comparam. Entra no git **antes** de o treino correr, e não se muda depois de ver os resultados: o histórico do git mostra que a lista foi escolhida às cegas (na literatura, *pré-registo*). O treino no pod e o notebook que decide leem o mesmo ficheiro, por isso não podem usar listas diferentes.

Cada YAML tem um `modo`: `fases` (por defeito; passos, com uma opção em vigor que passa de passo para passo, como o E4-S5), `grelha` (o produto cartesiano dos eixos, como o E4-S1) `entidade` (uma só variante do texto de entrada e a forma de comparar, como o E4-S3) ou `janela` (a mesma configuração em várias datas de início do treino, da mais curta para a mais longa, como o E4-S2 e o E4-S6).

Os valores **em vigor** do modelo (os que ganharam) não ficam aqui: ficam no código (`ALPHA`, `TAU` e `EPOCAS` em `src/tfidf.py`). O YAML diz o que se experimenta; o código diz o que se usa.

No YAML os números escrevem-se com ponto: `1.0e-7`, não `1e-7`. O PyYAML segue o YAML 1.1, em que `1e-7` é texto; o leitor (`scripts/correr_experiencia.py`, função `ler`) recusa-o com erro.

| Ficheiro | Spike | Quem o lê |
|---|---|---|
| [`e4_s5.yaml`](e4_s5.yaml) | E4-S5: `alpha` 0,00000001, τ (se o `alpha` mudar) e número fixo de épocas | [`scripts/correr_experiencia.py`](../scripts/correr_experiencia.py) (no pod) e [`notebooks/06_tfidf_epocas.ipynb`](../notebooks/06_tfidf_epocas.ipynb) |
| [`e4_s1.yaml`](e4_s1.yaml) | E4-S1: grelha N × épocas × τ (modo `grelha`), com o `alpha` fixo | [`scripts/correr_experiencia.py`](../scripts/correr_experiencia.py) (no pod) e [`notebooks/07_tfidf_grelha.ipynb`](../notebooks/07_tfidf_grelha.ipynb) |
| [`e4_s3.yaml`](e4_s3.yaml) | E4-S3: a entidade compradora (`adjudicante`) junta ao texto de entrada (modo `entidade`) | Um só comando `src/train_tfidf.py --com-entidade` (no pod; o script não corre este modo) e [`notebooks/08_entidade.ipynb`](../notebooks/08_entidade.ipynb) |
| [`e4_s2.yaml`](e4_s2.yaml) | E4-S2: janela de treino desde 2025, 2022 ou 2019 (modo `janela`), com a regra de paragem por defeito | `src/prepare.py --inicio-treino` e `src/train_tfidf.py` à mão no pod (o script não corre este modo) e [`notebooks/09_janela.ipynb`](../notebooks/09_janela.ipynb) |
| [`e4_s6.yaml`](e4_s6.yaml) | E4-S6: a janela desde 2019 com 24 épocas contra o modelo em vigor (desde 2022) (modo `janela`) | `src/train_tfidf.py` à mão no pod e [`notebooks/10_janela_24_epocas.ipynb`](../notebooks/10_janela_24_epocas.ipynb) |
| [`e5_t4.yaml`](e5_t4.yaml) | E5-T4: primeiro treino do BERTimbau (sem `modo`: uma só configuração), com a fonte de cada hiperparâmetro, o sinal de erro 1 (decorar 100 contratos), o sinal de erro 2 (bater a régua) e o ensaio de custo | `src/train_bert.py --experiencia` (no pod; só regista o ficheiro no `treino.json`, os valores passam-se nas flags) e [`src/registo_bert.py`](../src/registo_bert.py) (no MLflow) |
