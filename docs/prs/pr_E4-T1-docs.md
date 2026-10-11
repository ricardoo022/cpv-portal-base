## E4-T1 · Marcar como feita e atualizar a documentação

## Em poucas palavras

A avaliação (E4-T1) já está no `main` desde o PR #7. Este PR só atualiza os documentos para o dizerem: a tarefa fica marcada como feita, o E4 passa a "em curso", e o readme, o workflow e o CLAUDE.md passam a explicar a avaliação nova. Não muda código nem testes.

## O que mudou

- `docs/backlog.md`: E4-T1 marcada "(M, feito)", com uma nota a apontar para o PR #7 e para `prs/pr_E4-T1.md`. O estado do E4 passa de "próximo" a "em curso (E4-T1 feita)".
- `docs/workflow.md`: na fase 4, uma caixa marcada para a avaliação comum. Na fase 5, as métricas do teste incluem as medidas hierárquicas e usam a mesma avaliação. Nos "Termos", 4 entradas novas com exemplo: top-k, sobre-especialização, medidas hierárquicas e bootstrap emparelhado.
- `readme.md`: a secção de avaliação descreve o código exato, os contratos só com divisão (mostrados à parte), as medidas hierárquicas e o bootstrap. A estrutura passa a mencionar `docs/prs/`.
- `CLAUDE.md`: um parágrafo sobre `src/evaluation.py` e outro sobre `src/comparison.py`, e a pasta `docs/prs/` na lista dos documentos.
- `docs/prs/pr_E4-T1-docs.md`: esta descrição.

### Ficou igual de propósito

- Código e testes: este PR só toca em documentos.
- `docs/qualidade_dados.md`: a decisão 8 já foi atualizada no PR #7.

## Validação e provas

- Suite inteira: `.venv/bin/pytest -q` → 142 passed (igual ao `main`, porque não há código novo).
- CI do commit de cima: ver os checks deste PR (o merge só se faz com eles verdes).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
