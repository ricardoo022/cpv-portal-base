# Vídeo 1: o problema

O que vamos dizer sobre o problema, por ordem.

## 1. Porque comecei

- O objetivo era só aprender a fazer *fine-tuning* de um modelo, por causa de uma proposta de emprego.
- Precisava de um problema real com dados reais. Encontrei este.

## 2. O que é o CPV

- É a tabela da UE que classifica o que o Estado compra. Tem cerca de 9 500 códigos em árvore: divisão (2 dígitos) → grupo → classe → categoria → código completo (8 dígitos).
- Quem prepara o procedimento escolhe o código à mão.
- **Ecrã:** uma linha real do Excel do Portal BASE, parada em `objectoContrato` e `CPV`. Depois, a árvore a abrir até um código de 8 dígitos.

## 3. O que dizem os técnicos

Mensagens reais, sem nome e sem entidade. **Ecrã:** cartões que aparecem um a um.

> "A escolha de CPV é uma tarefa que por vezes ocupa demasiado tempo e implica a utilização de várias fontes, nem sempre com a eficácia desejada."

> "Às vezes demoramos algum tempo a encontrar o CPV que esteja mais alinhado ao objeto do contrato e nem sempre é o mais específico para o caso em questão."

> "Problema maior é quando ninguém deteta o erro e só o Tribunal de Contas em fiscalização é que se apercebe do mesmo."

> "Talvez podia começar pelas câmaras municipais… organismos com menos recursos humanos especializados em contratação pública, pelo menos as mais pequenas."

> "O código CPV é fechado, em regra, de forma rápida, desde que o objeto do contrato esteja corretamente definido e de forma concreta e explícita."

> "Considero essencial que as sugestões sejam sempre sujeitas a validação humana."

## 4. Quantos códigos estão errados

- **Estudo Rambøll para a Comissão Europeia:** 23% dos anúncios analisados tinham o código errado. Quase 1 em cada 4.
- **Tribunal de Contas Europeu (2016):** nas compras das próprias instituições da UE, ~10% dos códigos não descreviam a compra. Exemplo: um estudo sobre o mercado automóvel classificado como "transporte público rodoviário".
- **Portal BASE:** 18,8% dos contratos usam só um código de divisão, o nível mais genérico, por exemplo "Equipamento médico, medicamentos e produtos para cuidados pessoais". É genérico, não necessariamente errado.

## 5. Quanto custa ao país

A lógica: com o código errado, menos empresas encontram o concurso. Menos propostas dão um preço mais alto, cerca de +3,9% por cada proposta a menos.

Em 2024 houve ~11 mil milhões € em concursos no Portal BASE.

| Cenário | Códigos errados | Custo extra | Por ano |
|---|---|---|---|
| Prudente | 10% | 3,9% (1 proposta a menos) | ~43 milhões € |
| Relatório europeu | 16,5% | 11,7% (3 propostas a menos) | ~212 milhões € |

Frase: **"Se Portugal errar no código tanto como a média europeia, isso pode custar entre 40 e 200 milhões de euros por ano em compras mais caras."**

- **Ecrã:** a multiplicação à vista: valor dos concursos × % errados × custo extra.
- Dizer que é uma estimativa com percentagens europeias: em Portugal ninguém mediu isto.
- Quem perde: os contribuintes, que pagam mais, e as empresas pequenas, que não veem os concursos.

## 6. O que já se fez lá fora

**Ecrã:** linha do tempo sobre um mapa da Europa, com Portugal no fim.

| Ano | Quem | Resultado |
|---|---|---|
| 2017 | Comissão Europeia | Recomendou *machine learning* para sugerir o código |
| 2022–23 | Câmara de Zwolle (Países Baixos) | Ferramenta com 5 sugestões, ~76% nas 45 divisões; não chegou para uso real |
| 2023 | Estudo italiano | O modelo clássico (77,7%) ficou à frente do BERT (74,1%) nas divisões |
| hoje | Este projeto | Português, medido no código completo |

A mensagem: **todos mediram só as 45 divisões**, que é a parte fácil. O técnico trabalha no código de 8 dígitos, e foi aí que Zwolle falhou. Em português não encontrámos nada.

## 7. A proposta

- Entra a descrição do contrato e saem os 5 códigos mais prováveis.
- O técnico escolhe entre 5 em vez de milhares, e a decisão continua a ser dele.
- **Ecrã:** a pesquisa difícil de antes, agora com as 5 sugestões.

## Antes de gravar

- [ ] Passar a conta do custo para um notebook (`notebooks/12_custo_cpv.ipynb`).
- [ ] Pedir autorização a quem escreveu as mensagens.
- [ ] Confirmar nas fontes os números de Zwolle e do estudo italiano.

## Fontes

- Comissão Europeia, *Revision of CPV* (cosinex, 2017), p. 18–22 e 48–49 (inclui o estudo Rambøll e o Tribunal de Contas Europeu): https://ec.europa.eu/docsroom/documents/27821/attachments/1/translations/en/renditions/pdf
- Estudo italiano: https://ceur-ws.org/Vol-3596/paper45.pdf
- Zwolle (Haskoning): https://www.haskoning.com/en/projects/using-machine-learning-to-predicting-cpv-codes-from-tender-text
- Portal BASE: secção 9 de [`docs/qualidade_dados.md`](../docs/qualidade_dados.md)
