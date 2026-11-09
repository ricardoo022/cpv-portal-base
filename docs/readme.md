# docs/ · documentação do projeto

| Ficheiro | O que tem |
|---|---|
| [`workflow.md`](workflow.md) | O plano por fases (CRISP-DM), com caixas de verificação: o que está feito e o que falta. E um glossário dos termos. |
| [`backlog.md`](backlog.md) | Os épicos, as histórias, as tarefas e os spikes, com critérios de aceitação. O que está feito e o que vem a seguir. |
| [`qualidade_dados.md`](qualidade_dados.md) | As medições aos dados e as decisões numeradas que saíram delas (decisão 1 a 10). |
| [`dados.md`](dados.md) | As 39 colunas dos dados: o que é cada uma, se está preenchida e se existe quando o técnico escolhe o código. |
| [`maquina_runpod.md`](maquina_runpod.md) | Como alugar, preparar e apagar a máquina da Runpod onde corre o treino do TF-IDF, e quanto custa (E4-S0); a secção 7 é o guia do pod com GPU para o BERT (E5-T2). |
| [`como_funciona_tfidf.md`](como_funciona_tfidf.md) | O modelo TF-IDF passo a passo, com um exemplo de brincar: como descreve o texto, como treina e de onde vêm as sugestões. |
| [`como_funciona_bert.md`](como_funciona_bert.md) | O que é o BERTimbau, como aprende português sem etiquetas e onde, no `src/bert.py`, se faz o fine-tuning para os códigos CPV. |
| [`fine_tuning_simples.md`](fine_tuning_simples.md) | O fine-tuning do BERTimbau explicado do zero, sem jargão, com diagramas e o sítio no código de cada passo. |
| [`workflow_fine_tuning.md`](workflow_fine_tuning.md) | O fine-tuning do BERT do início ao fim: preparar no laptop, ensaios e treino no pod com GPU, registar e comparar; em cada passo, o comando e o código que corre. |
| [`prs/`](prs/) | A descrição de cada pull request. |
