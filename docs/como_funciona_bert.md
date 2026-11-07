# Como funciona o BERT (e o nosso fine-tuning)

O que é o BERTimbau, como aprendeu português sem etiquetas e onde, no nosso código, o afinamos para sugerir códigos CPV. É o par de [`como_funciona_tfidf.md`](como_funciona_tfidf.md) para os modelos do épico E5.

Os números deste documento vêm de [`qualidade_dados.md`](qualidade_dados.md) (secções 11 e 22) e de [`metrics/bert/e5_t4_bertimbau/treino.json`](../metrics/bert/e5_t4_bertimbau/treino.json). As percentagens marcadas como "ilustrativas" são inventadas, só para o exemplo.

## 1. O que é o BERTimbau

O **BERTimbau** é um modelo de linguagem para português, da família BERT (Devlin et al., 2019). No projeto é um dos dois modelos do E5, com o Albertina PT-PT.

Factos de memória, ainda **não verificados** contra a fonte:

- foi feito pela NeuralMind, uma empresa brasileira (Souza, Nogueira e Lotufo, 2020);
- foi pré-treinado no brWaC, um corpus de páginas web brasileiras. É português **do Brasil**, não PT-PT;
- há uma versão *base* (~110 M parâmetros) e uma *large* (~335 M). O projeto usa a *base*;
- é *cased*: distingue maiúsculas de minúsculas ("Lisboa" ≠ "lisboa").

O que está fixado no código, em [`src/bert_models.py`](../src/bert_models.py): o nome `neuralmind/bert-base-portuguese-cased`, uma revisão exata (um commit do Hugging Face, para o resultado ser reprodutível) e o limite de 256 tokens (secção 20).

O Albertina PT-PT foi pré-treinado em português europeu, a língua dos nossos contratos. Comparar os dois diz-nos se a variante da língua pesa nesta tarefa.

## 2. Aprender "sem etiquetas"

Uma **etiqueta** é a resposta certa que um humano escreveu à mão. Nos nossos dados, é o código CPV que a entidade escolheu:

```
texto:    "Fornecimento de refeições escolares"
etiqueta: 55524000   ← alguém teve de escrever isto
```

As etiquetas custam trabalho humano, por isso há poucas. Texto sem etiqueta há quase sem limite: notícias, páginas web, livros.

O truque do **pré-treino** é tirar a resposta do próprio texto. Esconde-se uma palavra e o modelo tem de a adivinhar. A palavra escondida é a resposta, e ninguém precisou de a escrever:

```
frase original:    "O município comprou pão para as escolas."
o que o modelo vê: "O município comprou [MASK] para as escolas."
resposta:          "pão"  ← vem da própria frase
```

Isto repete-se milhares de milhões de vezes. Para acertar, o modelo é obrigado a aprender:

- **gramática**: depois de "as" vem um nome feminino plural;
- **significado**: o que se compra "para as escolas" costuma ser comida, material ou mobiliário;
- **semelhança**: "pão" e "produtos de panificação" aparecem em contextos parecidos.

Por dentro, o BERT transforma cada palavra num vetor de 768 números. Palavras usadas em contextos parecidos ficam com vetores parecidos. É neste sentido que o modelo "percebe" texto. As aspas estão lá de propósito: não é perceber como uma pessoa percebe, são padrões estatísticos muito ricos.

## 3. O fine-tuning, em esquema

O modelo tem duas partes: um **corpo** (o BERT, 12 camadas) e uma **cabeça** (a última camada, que dá a resposta). O **fine-tuning** (afinação) mantém o corpo e troca a cabeça.

```
PRÉ-TREINO (feito pela NeuralMind, não por nós)

  "O município comprou [MASK]..."
              │
              ▼
   ┌──────────────────────┐
   │  CORPO: BERT         │   aprendeu português
   │  12 camadas          │
   └──────────────────────┘
              │
              ▼
   ┌──────────────────────┐
   │ CABEÇA: "que palavra │   uma saída por palavra do vocabulário
   │ está escondida?"     │
   └──────────────────────┘
              │
              ▼
          "pão" 41%   (ilustrativo)


FINE-TUNING (o que fazemos no pod com GPU)

  "Fornecimento de refeições escolares"
              │
              ▼
   ┌──────────────────────┐
   │  CORPO: BERT         │   começa com os pesos do pré-treino
   │  12 camadas          │   (o que já sabe de português)
   └──────────────────────┘
              │  um resumo do texto: 768 números
              ▼
   ┌──────────────────────┐
   │ CABEÇA NOVA:         │   3 000 saídas, uma por etiqueta CPV
   │ "que código CPV?"    │   começa ALEATÓRIA
   └──────────────────────┘
              │
              ▼
   55524000: 0,03%   ← no início, quase ao acaso (ilustrativo)
   ...
```

### O que acontece num passo de treino

1. O modelo lê o texto e dá uma probabilidade a cada código. No início, o código certo recebe muito pouca, porque a cabeça é aleatória.
2. Mede-se o erro, a **perda**. Quanto mais baixa a probabilidade do código certo, maior a perda.
3. Ajustam-se **todos** os pesos um bocadinho, os da cabeça e os do corpo, no sentido que teria subido a probabilidade do código certo.
4. Passa-se ao lote seguinte. Cada **lote** leva 32 contratos, e uma **época** é uma passagem por todo o treino.

O ajuste ao corpo é pequeno de propósito. A **taxa de aprendizagem** (2e-5 = 0,00002) controla o tamanho de cada passo. Com passos grandes, o modelo esqueceria o português do pré-treino.

### O que o modelo devolve

O modelo **não** devolve um código, nem um único número. Há três camadas:

1. **O corpo (BERT) não sabe nada de CPV.** Devolve vetores de 768 números, um por token. A cabeça usa o vetor do primeiro token (`[CLS]`) como resumo do texto.
2. **A cabeça devolve 3 000 números, um por etiqueta.** Chamam-se **pontuações** (ou *logits*). Quanto maior a pontuação, mais o modelo acha que aquele código serve. Não são probabilidades: para isso seria preciso passá-las pelo *softmax*, que as põe entre 0 e 1 e a somar 1. Para ordenar os códigos não é preciso, porque o softmax não muda a ordem.
3. **O nosso código escolhe as 5 maiores.** Antes, aplica o ajuste τ do TF-IDF (pontuação − τ · log da frequência da etiqueta no treino, com τ = 0,25). Como o log de uma frequência é negativo, e mais negativo quanto mais rara a etiqueta, as etiquetas raras sobem um pouco.

Exemplo ilustrativo (números inventados):

```
texto:   "Fornecimento de refeições escolares"
cabeça → 3 000 pontuações:  55524000: 8,1   15894210: 6,3   55520000: 5,9   ...   03111000: −4,2
ajuste τ + ordenar       →  top-5: [55524000, 55520000, 15894210, ...]
```

O TF-IDF faz o mesmo: também dá uma pontuação por etiqueta e as sugestões são as 5 maiores depois do ajuste τ. Os dois modelos só diferem na forma de calcular as pontuações.

## 4. Onde está no nosso código

O fine-tuning está todo na função `train` de [`src/bert.py`](../src/bert.py). O [`src/train_bert.py`](../src/train_bert.py) só a chama (`bert.train(...)`), com os ficheiros e os hiperparâmetros.

### 4.1 As etiquetas: `training_labels`

```python
etiqueta = collapse_rare_codes(codigo, n_min=n_min)
classes = np.sort(etiqueta.dropna().unique()).astype(object)
```

Os códigos com menos de 10 exemplos sobem para a classe ou o grupo, como no TF-IDF ([`src/labels.py`](../src/labels.py)). `classes` é a lista das 3 000 etiquetas, e cada uma vai ser uma saída da cabeça nova.

### 4.2 Corpo pré-treinado + cabeça nova: `load_model`

É aqui que se faz a troca de cabeça do esquema:

```python
model = AutoModelForSequenceClassification.from_pretrained(
    nome, **extra,                      # neuralmind/bert-base-portuguese-cased, revisão fixa
    num_labels=len(classes),            # cabeça nova: uma saída por etiqueta CPV
    id2label={i: str(c) for ...},       # saída 0 → primeiro código, saída 1 → ...
    ...
    ignore_mismatched_sizes=True,
)
```

Esta única chamada faz duas coisas:

- **`from_pretrained`** descarrega o **corpo**, ou seja, os pesos do pré-treino;
- **`ForSequenceClassification`** junta-lhe por cima uma **cabeça** com `num_labels` saídas e pesos **aleatórios**. A cabeça de adivinhar palavras fica de fora.

Antes de `load_model`, `train` chama `set_seed(semente)`. Assim a cabeça aleatória sai igual em todas as execuções.

### 4.3 Os textos passam a números: `encode`, `EncodedTexts`, `Collator`

```python
treino = EncodedTexts(encode(tokenizer, texto[keep], info["max_length"]),
                      etiqueta[keep].map(id_de).to_numpy())
```

- `encode` parte cada texto em tokens e corta-o aos 256. "Fornecimento de refeições" passa a uma lista de números de token.
- `map(id_de)` troca cada código pelo número da sua saída na cabeça.
- O `Collator` junta 32 exemplos num lote e preenche os mais curtos até ficarem todos com o mesmo comprimento.

### 4.4 O tamanho dos passos: `training_arguments`

Com os valores por defeito do início de `src/bert.py` (a receita de Devlin et al. 2019):

```python
learning_rate=taxa_aprendizagem,   # 2e-5: passos pequenos, para não estragar o pré-treino
per_device_train_batch_size=lote,  # 32 contratos por passo
num_train_epochs=epocas,           # 3 passagens pelo treino
warmup_steps=aquecimento,          # 0,1: nos primeiros 10% dos passos a taxa sobe de 0 até 2e-5
lr_scheduler_type="linear",        # ...e depois desce até 0 no fim
weight_decay=decaimento_pesos,     # 0,01: puxa os pesos ligeiramente para zero
```

O **aquecimento** existe porque, no início, a cabeça é aleatória e os erros são enormes. Se a taxa começasse logo cheia, esses primeiros erros podiam estragar o corpo.

### 4.5 O ciclo de treino: `Trainer` e `trainer.train()`

```python
trainer = Trainer(model=model, args=args, train_dataset=treino, ...)
trainer.train(resume_from_checkpoint=retomar_de)
```

O ciclo da secção 3 (ler o texto → probabilidades → perda → ajustar todos os pesos) **não está escrito no nosso código**. Está dentro de `trainer.train()`, no Hugging Face. Em cada passo, por dentro:

1. O lote passa pelo corpo e pela cabeça e sai uma pontuação por etiqueta.
2. Como o lote traz `labels`, o próprio modelo calcula a perda (entropia cruzada, isto é, −log da probabilidade do código certo).
3. Calcula-se em que direção mexer cada peso para baixar a perda (os **gradientes**).
4. O otimizador (AdamW) mexe todos os pesos, os do corpo e os da cabeça, à taxa daquele momento.

No treino do E5-T4 foram 478 234 exemplos ÷ 32 = 14 945 passos por época (`exemplos_usados` e `passos_por_epoca` do `treino.json`).

### 4.6 No fim de cada época: `EpochScores`

O `EpochScores` pontua a validação com o modelo nesse momento. O `train_bert.py` transforma essas pontuações nas 5 sugestões e grava-as em `epoca_<k>/`. O `Trainer` grava ainda um checkpoint por época, que é o que permite o `--retomar`.

### 4.7 Das pontuações às 5 sugestões: `predict_scores` e `suggest_from_scores`

`predict_scores` corre o modelo sobre os textos e guarda a saída da cabeça:

```python
model.eval()                      # sem dropout: a mesma entrada dá sempre a mesma saída
logits = model(**batch).logits    # uma linha por contrato, uma coluna por etiqueta
```

O resultado é uma matriz de contratos × 3 000 pontuações. `suggest_from_scores` aplica o ajuste τ e escolhe as 5 maiores de cada linha:

```python
pontuacoes = pontuacoes - tau * log_prior        # empurrão às etiquetas raras
tfidf.top_k(pontuacoes, classes, k)              # as 5 maiores, empates para o menor código
```

Usa a mesma função `top_k` do TF-IDF. Assim, as sugestões dos dois modelos ficam no mesmo formato e com as mesmas regras de desempate, e avaliam-se com o mesmo código.

### Mapa

| Esquema | Código (`src/bert.py`) |
|---|---|
| etiquetas (códigos CPV) | `training_labels` |
| corpo pré-treinado | `load_model` → `from_pretrained(nome)` |
| cabeça nova aleatória | `load_model` → `num_labels=len(classes)` |
| texto → números | `encode`, `EncodedTexts`, `Collator` |
| tamanho dos passos | `training_arguments` |
| ciclo perda → ajustar pesos | `trainer.train()` (dentro do Hugging Face) |
| avaliar cada época | `EpochScores` |
| 3 000 pontuações por contrato | `predict_scores` |
| ajuste τ + top-5 | `suggest_from_scores` |

## 5. Porque pode ajudar, e porque não é garantido

- O **TF-IDF** não sabe português, só conta palavras. "Pão" e "produtos de panificação" são para ele palavras diferentes, sem relação.
- O **BERT** chega já com vetores parecidos para coisas parecidas. Em princípio, pode acertar num contrato com palavras que nunca viu no treino.

Isto é a hipótese, não um resultado. A primeira afinação do BERTimbau (E5-T4, uma só execução) deu **73,39%** de top-5 por código, contra **78,35%** do TF-IDF (secção 22). A decisão é do E5-T6.

## Lições que valem além do CPV

- **Auto-supervisão**: criar a tarefa de treino a partir dos próprios dados, sem etiquetas humanas.
- **Transferência**: reaproveitar o que se aprendeu numa tarefa barata (adivinhar palavras) para uma tarefa cara (classificar CPV). Os modelos de imagem, de áudio e os grandes modelos de linguagem usam a mesma ideia.
- Com o Hugging Face, quase todo o código de fine-tuning é **preparação**: etiquetas, tokenização, hiperparâmetros e verificações. O ciclo de treino é uma linha. Por isso os erros costumam estar na preparação, por exemplo etiquetas trocadas ou textos cortados, e não no ciclo. É isso que o sinal de decorar (secção 21) procura apanhar.
- Partir de um modelo pré-treinado na língua certa quase sempre compensa face a treinar do zero. Mesmo assim, não garante bater um modelo simples bem afinado: é preciso medir sempre contra a régua.
