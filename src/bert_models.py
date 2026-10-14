"""Os dois modelos tunados do E5: nome no Hugging Face, versão fixa e comprimento máximo.

É a única fonte destes valores: o notebook 11 (E5-T1) mede o comprimento com estes tokenizadores e
confirma que `COMPRIMENTO_MAXIMO` é o que a regra escolheu, e o código de treino (E5-T2) lê daqui o
modelo a tunar. Assim o tokenizador medido é o mesmo que o treino usa.

`revisao` é o commit do repositório no Hugging Face (lido a 14 de outubro de 2026): se os autores
mudarem o tokenizador, o projeto continua a usar a versão medida.

Nota sobre o BERTimbau: o `tokenizer_config.json` não tem `model_max_length`, por isso
`truncation=True` sozinho não corta aos 512. O treino tem de passar `max_length` explicitamente.
"""

MODELOS = {
    "bertimbau": dict(
        nome="neuralmind/bert-base-portuguese-cased",
        revisao="94d69c95f98f7d5b2a8700c420230ae10def0baa",
    ),
    "albertina": dict(
        nome="PORTULAN/albertina-100m-portuguese-ptpt-encoder",
        revisao="d15e086005cdab26d746d9113984cc654439df2b",
    ),
}

# Candidatos e regra do E5-T1 (docs/backlog.md), escritos antes de medir: o menor comprimento em
# que pelo menos 99% dos textos de treino cabem inteiros (com os tokens especiais); senão, 512.
CANDIDATOS = (128, 256, 512)
QUOTA = 0.99

# Escolhido no notebooks/11_comprimento_tokens.ipynb (secção 20 do qualidade_dados.md).
COMPRIMENTO_MAXIMO = {"bertimbau": 256, "albertina": 256}
