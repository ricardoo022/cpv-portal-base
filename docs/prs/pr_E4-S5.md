## E4-S5 · Spike: épocas e um alpha mais pequeno no TF-IDF

## Em poucas palavras

Depois do E4-S4 afinar o `alpha` e o τ, ficaram duas dúvidas importantes antes do TF-IDF servir de régua ao BERTimbau e à Albertina: será que um `alpha` ainda mais pequeno (0,00000001) melhorava o modelo, dado que o vencedor anterior estava no extremo da lista? E será que parar o treino num número fixo de épocas é melhor do que a paragem automática do scikit-learn, que parava ao fim de 6 a 12 épocas quando o erro de treino deixava de descer?
Testámos as opções no pod da Runpod seguindo o pré-registo escrito antes de correr (`experiencias/e4_s5.yaml`). O `alpha` 0,00000001 perdeu claramente (-7,5 p.p. no top-5), confirmando o 0,0000001 como o ponto ótimo. Em contrapartida, 24 épocas fixas sem paragem automática ganharam na validação, elevando o top-5 no código completo de 76,9% para 78,3% (e de 73,5% para 75,1% em texto novo).

| Na validação (outubro de 2025) | Régua | E4-H1 | E4-S4 (afinado) | E4-S5 (24 épocas) |
|---|---|---|---|---|
| Top-5 no código completo | 26,3% | 43,2% | 76,9% | **78,3%** |
| Top-5 na classe (4 dígitos) | 15,6% | 37,9% | 82,9% | **84,1%** |
| Top-5 código (texto novo) | 23,7% | 38,5% | 73,5% | **75,1%** |
| Top-5 classe (texto novo) | 14,8% | 34,2% | 80,8% | **82,2%** |
| F1 macro no código | 0,0001 | 0,0107 | 0,2565 | **0,2568** |

## Porquê

O TF-IDF vai ser a bitola contra a qual avaliaremos os modelos neurais mais pesados. Uma bitola sub-otimizada daria uma falsa impressão de superioridade aos modelos seguintes.
O scikit-learn avalia a paragem automática somando a perda em todos os exemplos de treino (~478 mil). Em etiquetas desequilibradas, a soma pode estabilizar cedo mesmo que várias classes ainda estejam a aprender separações úteis. Medir um número fixo de épocas na validação permitiu tirar a prova limpa.

> Pergunta do spike: um `alpha` de 0,00000001, ou parar o treino noutra época, melhoram as sugestões na validação?

Item: [`E4-S5`](../backlog.md) · Desbloqueia: E4-S1 (N dos códigos raros) e E4-S2 (janela temporal de treino).

## O que mudou no código

### `src/tfidf.py`
- `EPOCAS = 24`: a nova constante em vigor ao lado de `ALPHA` e `TAU`.
- `sgd_epochs(epocas)`: traduz um número de épocas para os parâmetros do `SGDClassifier` (`max_iter=k`, `tol=None`), ou `{}` se `epocas=None` (para manter a paragem por defeito do scikit-learn).
- `train(..., epocas=EPOCAS)`: aceita o número de épocas. Recusa `max_iter` e `tol` passados em simultâneo com `epocas`.

### `src/train_tfidf.py`
- Suporte a `--epocas K` e `--regra-por-defeito` (mutuamente exclusivos no CLI) e argumento `--n-jobs`.
- O ficheiro `treino.json` regista `epocas`, `tol` e `max_iter`.

### `scripts/correr_experiencia.py` e `scripts/curva_epocas.py`
- `scripts/correr_experiencia.py`: lê o YAML do spike, valida tipos e chaves e lança os treinos em paralelo no pod.
- `scripts/curva_epocas.py`: treina época a época com `partial_fit` e regista as métricas de cada época para permitir traçar a curva de evolução sem re-treinar do zero.

### `experiencias/e4_s5.yaml`
- Pré-registo das opções do spike (passo 1: alpha; passo 2: tau; passo 3: épocas) escrito antes de correr.

## Decisões tomadas pelo caminho

1. **Pré-registo estrito:** As opções foram registadas em `experiencias/e4_s5.yaml` antes de qualquer execução na máquina remota. Um teste unitário garante que o YAML não foi alterado à posteriori.
2. **Rejeição do alpha 1e-8:** O valor 1e-8 piorou claramente o desempenho (-7,5 p.p. no top-5), fechando a busca de regularização nesta gama.
3. **Adoção de 24 épocas fixas:** Venceu em 100% das amostras de bootstrap face à regra por defeito do scikit-learn (+1,4 p.p. no top-5 de código e +1,2 p.p. na classe).
4. **Limitação documentada:** 24 é o extremo testado; a curva mostra que a partir da época 15 o ganho estagna em patamar (77,7% a 77,8%).
5. **Regra para os próximos spikes:** Como 24 épocas foram afinadas especificamente para $N=10$ e janela de 2022, os spikes E4-S1 e E4-S2 usarão `--regra-por-defeito` para não beneficiar nenhuma opção arbitrariamente.

## Outras mudanças

- `notebooks/06_tfidf_epocas.ipynb`: análise e validação passo a passo dos resultados do pod, curvas de treino e comparações com bootstrap.
- `metrics/e4_s5/`: ficheiros brutos de sugestões e logs da execução no pod.
- `docs/qualidade_dados.md`: secção 15 com os dados completos e limitações do spike.
- `docs/backlog.md`: spike E4-S5 marcado como feito e Decisão 7 registada.
- `docs/workflow.md`, `CLAUDE.md`, `Makefile` e READMEs de pastas atualizados.
- Testes unitários e de integração adicionados em `tests/unit/` e `tests/integration/`.

## Validação e provas

- **Reprodutibilidade:** O treino de controlo repetiu rigorosamente 100% das sugestões do E4-S4 nos 4 valores de τ.
- **Suíte de testes:** Todos os 322 testes do repositório passam (`pytest -q`), cobrindo testes unitários, testes de mutação e integração com dados simulados.
- **Custo:** Pod de 32 vCPU na Runpod ligado durante cerca de 44 minutos (~0,95 $).

### Revisão de código

Revisão por um agente juiz, sem o contexto da conversa, sobre as mudanças por fazer commit (12 de outubro de 2026, ~15 min): **aprovado com correções** — sem problemas críticos nem importantes. Dos 6 critérios revistos (fidelidade ao pré-registo, correção dos números, validade das conclusões, código, testes, e documentação e convenções), os 5 primeiros passaram; só a documentação teve problemas, todos menores:

- referências de linhas desatualizadas em `docs/como_funciona_tfidf.md`;
- `tests/unit/readme.md` sem os testes do TF-IDF na tabela, e com uma frase imprecisa sobre que ficheiros os testes cobrem;
- `experiencias/readme.md` sem o `EPOCAS` nos valores em vigor;
- `n_min` validado com o validador errado em `scripts/correr_experiencia.py` (a mensagem de erro falaria em "épocas"; ficou `_inteiro_positivo`);
- os ramos de erro de `--epocas` e `--tau` em `scripts/curva_epocas.py` sem teste;
- ligação no `docs/backlog.md` para este ficheiro, que ainda não existia.

- Corrigido: os 6 problemas, na mesma sessão. O juiz viu 316 testes a passar; as correções acrescentaram testes e a suíte ficou nos 322 acima.
- Não corrigido, e porquê: nada ficou por corrigir. Não houve segunda revisão depois das correções (a sessão terminou por limite de quota), o que fica aqui registado; as correções foram de documentação, mensagens de erro e testes novos.
