# Como são os dados

Descrição das colunas dos contratos do Portal BASE.

- **Uma linha = um contrato.** 39 colunas. As mesmas em todos os ficheiros, de 2012 a 2026.
- No Parquet (`data/processed/raw/`) todas as colunas estão guardadas como texto. Números e datas convertem-se quando são precisos.
- **Os números deste ficheiro vêm só dos contratos de treino** (publicados de janeiro a setembro de 2025, 166 113 contratos), para não espreitar o teste.
- De onde vêm os números: secção "Colunas" do [`notebooks/01_exploracao.ipynb`](../notebooks/01_exploracao.ipynb).

**Aviso sobre o significado das colunas.** A página do dataset no dados.gov.pt só tem os ficheiros, sem dicionário de dados. O "O que é" de cada coluna é a nossa interpretação, a partir do nome e dos valores. Onde não temos a certeza está marcado com **(?)**.

## Um contrato de exemplo

Publicado a 2 de janeiro de 2025 (período de treino). Só as colunas mais importantes:

| Coluna | Valor |
|---|---|
| `objectoContrato` | Serviços de limpeza de bermas, valetas e manutenção de espaços públicos: manter sem vegetação selvagem as valetas... |
| `descContrato` | CPS20241115A - Serviços de limpeza de bermas, valetas e manutenção de espaços públicos |
| `CPV` | 90611000-3 - Serviços de limpeza de ruas |
| `tipoContrato` | Aquisição de serviços |
| `tipoprocedimento` | Consulta Prévia |
| `adjudicante` | 507104676 - Freguesia de Joane |
| `adjudicatarios` | 508725895 - Ambicalendário - Unipessoal, Lda |
| `concorrentes` | 3 empresas, uma por linha, no formato `NIF-Nome` |
| `precoBaseProcedimento` | 19248 |
| `precoContratual` | 19248 |
| `prazoExecucao` | 364 |
| `LocalExecucao` | Portugal, Braga, Vila Nova de Famalicão |
| `NUTs` | PT119 - Ave |
| `dataDecisaoAdjudicacao` | 2024-12-19 |
| `dataCelebracaoContrato` | 2025-01-02 |
| `dataPublicacao` | 2025-01-02 |

## A coluna "Existe na altura?"

Diz se a informação já existe **quando o técnico escolhe o CPV**, ou seja, ao preparar o procedimento, antes de haver propostas e adjudicação. É o nosso raciocínio sobre o processo, não está escrito nos dados.

Só as colunas com **Sim** podem entrar no modelo. Usar uma coluna com **Não** é fuga de dados: o modelo parece bom no teste e falha na realidade, porque essa informação não existe quando ele vai ser usado.

## Texto e etiqueta

| Coluna | O que é | Preenchida | Exemplos | Existe na altura? | Usamos? |
|---|---|---|---|---|---|
| `objectoContrato` | Descrição curta do que se compra | 100% | `Medicamentos` (1 438) | Sim | **Sim, entrada principal** |
| `descContrato` | Descrição do contrato. Em 72% dos casos é igual ao `objectoContrato` | 100% | `Medicamentos` (1 419) | Sim | **Sim, quando é diferente do objeto** |
| `CPV` | Código ou códigos CPV, com a descrição colada. Vários códigos vêm separados por quebra de linha | 100% | `33600000-6 - Produtos farmacêuticos` (18 987) | É o que queremos prever | **Sim, é a etiqueta** |

## Entidades

| Coluna | O que é | Preenchida | Exemplos | Existe na altura? | Usamos? |
|---|---|---|---|---|---|
| `adjudicante` | Entidade que compra, no formato `NIF - Nome`. 5 289 distintas | 100% | `506362299 - Instituto Português de Oncologia do Porto...` (10 947) | Sim | **Experiência** (com e sem) |
| `adjudicatarios` | Empresa ou empresas que ganharam o contrato, `NIF - Nome` | 100% | `506019608 - SMILE - VIAGENS E TURISMO...` (957) | Não | Não |
| `concorrentes` | Empresas que apresentaram proposta, uma por linha, `NIF-Nome` | 43,3% | | Não | Não |
| `adjudicatarioPMEs` | NIF dos adjudicatários que são PME (?). `-` em 15 890 contratos | 86,0% | `506019608` (2 030) | Não | Não |

## Tipo de contrato e de procedimento

| Coluna | O que é | Preenchida | Exemplos | Existe na altura? | Usamos? |
|---|---|---|---|---|---|
| `tipoContrato` | Tipo de contrato. 28 valores distintos | 100% | `Aquisição de bens móveis` (82 747), `Aquisição de serviços` (68 342), `Empreitadas de obras públicas` (11 204) | Sim | Por decidir |
| `tipoprocedimento` | Tipo de procedimento. 17 valores | 100% | `Ajuste Direto Regime Geral` (75 458), `Consulta Prévia` (29 713), `Concurso público` (26 090) | Sim | Por decidir |
| `ProcedimentoCentralizado` | Se o procedimento foi feito por uma central de compras (?) | 100% | `Não` (155 659), `Sim` (10 454) | Sim | Não |
| `Lotes` | Nome do lote, quando o procedimento tem lotes | 23,2% | `Medicamentos` (1 413), `Lote 1` (1 077) | Sim | Não |
| `TipoCriterioAdjudicacao` | Critério para escolher o vencedor | 99,7% | `Monofator` (154 848), `Multifator` (10 666) | Sim | Não |
| `CritMateriais` | Significado não verificado (?) | 100% | `Não` (136 644), `Sim` (29 469) | ? | Não |
| `ContratEcologico` | Se o contrato tem critérios ecológicos (?) | 100% | `Não` (163 685), `Sim` (2 428) | Sim | Não |

## Acordo-quadro

Um acordo-quadro é um contrato geral feito antes, com condições já fixadas, ao abrigo do qual as entidades fazem depois contratos concretos sem um procedimento completo.

| Coluna | O que é | Preenchida | Exemplos | Existe na altura? | Usamos? |
|---|---|---|---|---|---|
| `numAcordoQuadro` | Número do acordo-quadro | 15,5% | `5715196` (3 903) | Sim | Não |
| `DescrAcordoQuadro` | Descrição do acordo-quadro. Diz o que se compra quando o `objectoContrato` não diz | 15,5% (97,7% dos contratos que falam de acordo-quadro) | `Acordo Quadro de Viagens e Alojamento` (3 903) | Sim | **Sim, juntar ao texto quando existe** |

## Preços

Em euros. Não verificámos se incluem IVA.

| Coluna | O que é | Preenchida | Exemplos | Existe na altura? | Usamos? |
|---|---|---|---|---|---|
| `precoBaseProcedimento` | Preço base: o máximo que a entidade aceita pagar | 100% (`0` em 2 537) | `15000` (964) | Sim | Não |
| `precoContratual` | Preço pelo qual o contrato foi feito | 100% | `15000` (769) | Não | Não |
| `PrecoTotalEfetivo` | Preço pago no fim (?). `0` em 131 630 contratos (79%), provavelmente porque ainda não terminaram | 100% | `0` (131 630) | Não | Não |

## Datas

Todas no formato `2025-01-02T00:00:00`.

| Coluna | O que é | Preenchida | Existe na altura? | Usamos? |
|---|---|---|---|---|
| `dataPublicacao` | Data de publicação do contrato no Portal BASE | 100% | — | **Sim, para separar treino, validação e teste** |
| `dataDecisaoAdjudicacao` | Data em que se escolheu o vencedor | 99,9% | Não | Não |
| `dataCelebracaoContrato` | Data em que o contrato foi assinado | 99,9% | Não | Não |
| `dataFechoContrato` | Data em que o contrato terminou | 21,1% | Não | Não |
| `Ano` | Ano do ficheiro. **Não é o ano de publicação**: o ficheiro de 2025 tem contratos publicados em 2024 | 100% | — | Não |

## Local e prazo

| Coluna | O que é | Preenchida | Exemplos | Existe na altura? | Usamos? |
|---|---|---|---|---|---|
| `LocalExecucao` | Onde se executa: país, distrito, concelho | 100% | `Portugal, Lisboa, Lisboa` (22 534), `Portugal` (16 160) | Sim | Não |
| `NUTs` | Região NUTS (divisão estatística do território) | 45,9% | `PT11A - Área Metropolitana do Porto` (28 793) | Sim | Não |
| `prazoExecucao` | Prazo de execução, em dias (?) | 100% | `365` (25 072), `20` (15 009), `30` (11 128) | Sim | Não |

## Enquadramento legal

| Coluna | O que é | Preenchida | Exemplos | Existe na altura? | Usamos? |
|---|---|---|---|---|---|
| `regime` | Lei aplicada. 20 valores | 100% | `Código dos Contratos Públicos (DL111-B/2017) e Lei...` (121 887) | Sim | Não |
| `fundamentacao` | Artigo do Código dos Contratos Públicos que justifica o procedimento. 140 valores | 99,9% | `Artigo 20.º, n.º 1, alínea d)...` (30 665) | Sim | Não |
| `fundamentAjusteDireto` | Razão do ajuste direto | 6,1% | `ausência de recursos próprios` (10 092) | Sim | Não |
| `justifNReducEscrContrato` | Justificação para o contrato não ser escrito (artigo 95.º) | 35,3% | `Artigo 95.º, n.º 1, c)...` (18 507) | Não (?) | Não |
| `tipoFimContrato` | Como terminou o contrato | 21,0% | `O cumprimento, a impossibilidade definitiva...` (34 225) | Não | Não |

## Identificadores e outros

| Coluna | O que é | Preenchida | Usamos? |
|---|---|---|---|
| `idcontrato` | Identificador do contrato no Portal BASE. Há 1 199 repetidos com o mesmo texto e CPV | 100% | **Sim, para tirar repetidos** |
| `idprocedimento` | Identificador do procedimento. Vários contratos podem vir do mesmo (131 499 distintos para 166 113 contratos) | 100% | Não |
| `nAnuncio` | Número do anúncio do procedimento, por exemplo `2108/2025`. Só existe quando houve anúncio | 15,9% | Não |
| `TipoAnuncio` | Tipo de anúncio | 15,9% | Não |
| `idINCM` | Identificador na INCM (Imprensa Nacional-Casa da Moeda, que publica o Diário da República) (?) | 15,9% | Não |
| `linkPecasProc` | Endereço das peças do procedimento | 51,8% | Não |
| `Observacoes` | Observações livres. Muitas vezes `N/A`, `.` ou `-` | 3,6% | Não |

## Resumo: o que entra no modelo

| Coluna | Para quê |
|---|---|
| `objectoContrato` + `descContrato` (quando diferente) + `DescrAcordoQuadro` (quando existe) | Texto de entrada |
| `CPV` (primeiro código) | Etiqueta |
| `adjudicante` | Experiência: com e sem |
| `dataPublicacao` | Separar treino, validação e teste |
| `idcontrato` | Tirar repetidos |

Por decidir: se `tipoContrato` e `tipoprocedimento` entram. Existem na altura e podem ajudar (por exemplo, é de esperar que as empreitadas caiam muito na divisão `45` Construção), mas ainda não medimos.
