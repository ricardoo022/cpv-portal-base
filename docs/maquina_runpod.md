# Alugar, preparar e apagar uma máquina na Runpod

Guia para o treino do TF-IDF num pod alugado (E4-S0 e seguintes); a secção 7 é o pod com GPU para o BERT (E5-T2). O TF-IDF só usa processador e memória; a GPU fica para o BERTimbau e o Albertina, também na Runpod. Tudo o que está aqui foi feito e confirmado a 11 de outubro de 2026, no ensaio do E4-S0 (pod `cpv-e4-s0`), salvo onde diz o contrário.

Termos que aparecem muitas vezes:

- **Pod**: a máquina alugada na Runpod. Na prática é um contentor (um Linux isolado, criado a partir de uma imagem Docker) a correr numa máquina física da Runpod.
- **Template**: a imagem com que o pod arranca, já com sistema e programas instalados. Exemplo: o template oficial "Runpod Ubuntu 22.04" traz Ubuntu, Python 3.10 e o servidor SSH.
- **SSH**: a forma de abrir um terminal numa máquina remota. Usa um par de chaves: a privada fica no portátil, a pública vai para a Runpod.
- **MCP da Runpod**: o plugin `runpod@runpod` do Claude Code liga o Claude à conta da Runpod. Com ele, o Claude cria, lê e apaga pods sem passar pela consola. Instala-se com `claude plugin marketplace add runpod/runpod-plugins-official` e `claude plugin install runpod@runpod`; liga-se à conta com `/mcp` → runpod → Sign in.

## 1. Antes de alugar

1. **Saldo.** A Runpod é pré-paga: carrega-se saldo na consola (https://console.runpod.io → Billing), e cada pod ligado vai descontando ao segundo. Sem saldo, criar um pod falha com `402 Payment Required`. O saldo que sobra fica na conta e não se gasta sozinho.
2. **Chave SSH** (já feita; só para uma máquina nova): uma chave só para a Runpod, sem frase-passe, para os comandos poderem correr sem pedir palavra-passe a meio.
   ```bash
   ssh-keygen -t ed25519 -N "" -C "cpv-runpod" -f ~/.ssh/runpod_cpv
   cat ~/.ssh/runpod_cpv.pub        # esta linha (a chave PÚBLICA) vai para a Runpod
   ```
   A chave pública regista-se na conta (consola → Settings → SSH Public Keys, ou pelo MCP). Só vale para pods criados depois do registo. A privada (`runpod_cpv`, sem `.pub`) nunca sai do portátil. Para lhe pôr uma frase-passe mais tarde: `ssh-keygen -p -f ~/.ssh/runpod_cpv`.

## 2. Criar o pod

O pod do ensaio:

| Campo | Valor | Porquê |
|---|---|---|
| Tipo | `cpu3g` (General Purpose), 8 processadores | Há pods só com processador; neste tipo cada processador traz 4 GB, por isso 8 dão 32 GB |
| Preço | 0,04 $ por processador por hora → **0,32 $/h** | Lido no catálogo de CPU da Runpod e confirmado no campo `cost` do pod criado |
| Imagem | `runpod/base:1.0.2-ubuntu2204` | Template oficial "Runpod Ubuntu 22.04": traz Python 3.10.12, `rsync`, `make` e `tmux` |
| Disco | 20 GB de disco do contentor | O código, o ambiente e os dados cabem com folga. Os pods só com CPU não têm disco de volume |
| Portas | só `22/tcp` | Só é preciso SSH |
| SSH | `startSsh: true` | Põe a chave registada dentro do pod |
| Centro de dados | o que a Runpod escolher | Pedir EU-RO-1 falhou por falta de stock com 8 processadores; sem preferência, calhou US-NC-1. Os dados são só 40 MB, por isso a distância não pesa |

Outros tipos só com CPU, se for preciso mais memória: `cpu3m` (Memory-Optimized, 8 GB por processador, 0,055 $ por processador por hora; 8 processadores dão 64 GB por 0,44 $/h).

**Vários treinos de uma vez** (E4-S5, 12 de outubro de 2026): o mesmo `cpu3g` com 32 processadores dá 128 GB por 1,28 $/h (0,04 $ por processador por hora, lido no catálogo e no campo `cost` do pod). O preço por processador é o mesmo, por isso 4 treinos de 8 processadores ao mesmo tempo custam o mesmo que 4 treinos seguidos num pod de 8, e acabam 4 vezes mais depressa. O `scripts/correr_experiencia.py` lança-os (`--paralelo 4 --n-jobs 8`), com um log por treino. Com 4 treinos ao mesmo tempo, o pico de memória do pod foi 36,0 GB (`metrics/e4_s5/pod_info.json`). Cada treino recebeu um `n_jobs` explícito, para os 4 dividirem os 32 processadores; não se mediu o que faria o `n_jobs=-1` com vários treinos ao mesmo tempo. Em EU-RO-1, o SSH direto respondeu à primeira.

No E4-H1 (11 de outubro de 2026), um pod em EU-SE-1 recebeu IP e porta de SSH direto, mas a porta nunca respondeu (ligação esgotada, *timeout*, durante 4 minutos), enquanto o portátil chegava a outros servidores. Apagou-se e criou-se outro em US-NC-1, que funcionou. Se o SSH direto não responder 2 a 3 minutos depois de aparecer, o mais barato é apagar o pod e criá-lo noutro centro de dados.

Depois de criado, o pod aparece como `RUNNING` logo, mas o **SSH direto** (`ssh.direct`, com IP e porta públicos) só aparece cerca de 1 minuto depois. Até lá só há o SSH por intermediário (`@ssh.runpod.io`), que **não serve**: não deixa correr comandos sem terminal interativo nem copiar ficheiros. O IP e a porta mudam sempre que o pod é recriado ou reiniciado.

## 3. Preparar o pod

No guia, o IP fica como `IP` e a porta como `PORTA` (os valores do campo `ssh.direct` do pod).

```bash
cd ~/projeto_CPV
E="ssh -i ~/.ssh/runpod_cpv -p PORTA"
rsync -a -e "$E" --exclude '.venv/' --exclude 'data/' --exclude '__pycache__/' --exclude '.claude/' ./ root@IP:/root/projeto_CPV/
$E root@IP mkdir -p /root/projeto_CPV/data/processed
rsync -a -e "$E" data/processed/treino.parquet data/processed/validacao.parquet root@IP:/root/projeto_CPV/data/processed/
$E root@IP ls -la /root/projeto_CPV/data/processed/      # só estes 2 ficheiros
```

- O `rsync` copia o código do ramo atual, mesmo o que ainda não está no GitHub, sem pôr nenhuma credencial do GitHub no pod.
- O **`teste.parquet` não vai**: o teste fica fechado até ao fim do projeto, e o que não está no pod não pode ser espreitado por engano.

Instalar e testar:
```bash
$E root@IP 'cd /root/projeto_CPV && make setup && .venv/bin/pytest -q'
```
No ensaio, o `make setup` funcionou à primeira (o `python3` da imagem é o 3.10.12) e os 172 testes passaram. Se o `pytest` falhar aqui e não no portátil, a diferença está no ambiente (versão do Python ou de uma biblioteca), não no modelo: resolve-se antes de gastar horas de máquina.

## 4. Correr o treino e trazer os resultados

**Porquê `tmux`**: se o treino correr diretamente no SSH e a ligação cair, o processo morre. O `tmux` mantém uma sessão viva no pod.
```bash
$E root@IP 'cd /root/projeto_CPV && tmux new -d -s treino "<comando do treino> > ensaio.log 2>&1; echo FIM >> ensaio.log"'
$E root@IP tail -3 /root/projeto_CPV/ensaio.log          # ver como vai; acabou quando aparecer FIM
```

**Memória: não confiar no `free -h`.** Dentro do pod, o `free -h` mostra a memória da máquina física inteira (no ensaio, muito mais do que os 32 GB do pod), não a do pod. O limite do pod e o que ele está a gastar leem-se no cgroup (o mecanismo do Linux que limita a memória do contentor):
```bash
cat /sys/fs/cgroup/memory.max        # limite: 32000000000 bytes = 32 GB
cat /sys/fs/cgroup/memory.current    # quanto está a usar agora
cat /sys/fs/cgroup/memory.peak       # o máximo desde que o pod arrancou
```
Lição que generaliza: dentro de um contentor, as ferramentas clássicas (`free`, `top`, `nproc` às vezes) podem mostrar a máquina física e não o limite do contentor.

**Trazer os resultados** (no portátil):
```bash
rsync -a -e "$E" root@IP:/root/projeto_CPV/metrics/e4_s0/ metrics/e4_s0/
rsync -a -e "$E" root@IP:/root/projeto_CPV/ensaio.log metrics/e4_s0/pod_ensaio.log
```
Confirma que os ficheiros abrem no portátil **antes** de apagar o pod: depois de apagado, o disco desaparece.

## 5. Apagar o pod

**A Runpod cobra enquanto o pod estiver ligado, esteja ele a trabalhar ou sem nada para fazer.** Por isso apaga-se logo que os resultados estão no portátil.

- **Stop** (parar): deixa de cobrar o processador, mas apaga o disco do contentor. Num pod só com CPU não há disco de volume, por isso parar não guarda nada que interesse.
- **Terminate** (apagar): acaba com o pod e com toda a cobrança. É o que se faz no fim.

Como apagar: consola → Pods → o pod → Terminate; ou pelo MCP (`delete-pod` com o id do pod). Depois, confirmar que a lista de pods está vazia (consola, ou `list-pods` no MCP) e que não há Network Volumes (Storage), que são cobrados à parte.

## 6. Custo

Fórmula: **custo = horas com o pod ligado × preço por hora**, cobrado ao segundo. Exemplo: o pod do ensaio (0,32 $/h) ligado 1 h 30 min custa 1,5 × 0,32 = 0,48 $.

O custo real do ensaio e a estimativa para os treinos do épico estão no [`backlog.md`](backlog.md), no resultado do E4-S0.

Fontes (documentação oficial da Runpod, além do que foi lido no próprio pod):
- Preços, cobrança ao segundo, discos: https://docs.runpod.io/pods/pricing
- Pods, templates e tipos de disco: https://docs.runpod.io/pods/overview
- SSH básico vs SSH completo, chaves: https://docs.runpod.io/pods/configuration/use-ssh
- Parar vs apagar: https://docs.runpod.io/pods/manage-pods

## 7. Pod com GPU para o BERT (E5-T2)

Ainda não foi feito: escrito para o E5-T4. Os passos 1 a 5 valem na mesma (saldo, chave SSH, `rsync` do código e só do `treino.parquet` e do `validacao.parquet`, `tmux`, apagar no fim). Mudam o sítio do projeto no pod, a instalação, uma verificação antes de treinar e o que se traz de volta.

**O projeto vai para `/workspace`, não para `/root`.** Segundo a Runpod (https://docs.runpod.io/pods/storage/types, lido a 15 de outubro de 2026), o disco do contentor "is cleared when the Pod stops" (perde-se quando o pod para ou reinicia), e o volume `/workspace` "survives Pod stops and restarts, but is deleted when the Pod is terminated". O treino do TF-IDF cabia numa sessão; o do BERT pode durar horas, e os *checkpoints* (em `models/bert/`, dentro do projeto) só servem para retomar se sobreviverem a um reinício. Por isso, nos passos 3 e 4, o destino do `rsync` é `root@IP:/workspace/projeto_CPV/` em vez de `/root/projeto_CPV/`, e o volume do pod tem de ter espaço para os *checkpoints* e as pontuações (ver a tabela abaixo). O preço por hora não está aqui: lê-se na Runpod no dia e o custo de um treino mede-se no E5-T4.

**Instalar.** No pod usa-se `make setup-gpu` em vez de `make setup`:
```bash
$E root@IP 'cd /workspace/projeto_CPV && make setup-gpu && .venv/bin/pytest -q'
```
- `make setup` (portátil e CI) instala o `requirements-cpu.txt`: o `torch` só para processador, do índice do PyTorch (~200 MB; ordem de grandeza, não medida).
- `make setup-gpu` instala o `requirements-gpu.txt`: o mesmo `torch` 2.14.1, mas do PyPI, que para Linux já traz as bibliotecas CUDA (vários GB). As duas listas incluem o `requirements.txt`, por isso as outras versões são as mesmas.

**O driver do pod tem de servir a CUDA do `torch`.** O *driver* (o programa da NVIDIA que fala com a placa) vem da máquina física, não da imagem nem do `pip`. O `torch` 2.14.1 do PyPI para Linux depende do `cuda-toolkit==13.0.3` e de pacotes `nvidia-*-cu13` (campo `requires_dist` em https://pypi.org/pypi/torch/2.14.1/json, lido a 15 de outubro de 2026), ou seja, CUDA 13.0. Segundo as notas da NVIDIA, um programa CUDA 13.x corre com *driver* >= 580 (https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html, tabela de compatibilidade). Por isso, ao escolher o pod, pede-se uma máquina com CUDA 13.0 ou mais, e no pod confirma-se com:
```bash
$E root@IP nvidia-smi          # no cabeçalho: "Driver Version" (>= 580) e "CUDA Version" (>= 13.0)
```

**Antes de treinar, confirmar que o `torch` vê a GPU:**
```bash
$E root@IP 'cd /workspace/projeto_CPV && make verificar-gpu'
```
Mostra o nome da GPU, ou falha. Se falhar, não se lança o treino: sem GPU, o treino no processador demoraria dias e continuaria a cobrar. O `src/train_bert.py` usa `--dispositivo cuda` por defeito, por isso também para com erro em vez de cair no processador sem avisar.

**Treinar** (dentro do `tmux`, como no passo 4):
```bash
make treinar-bert MODELO=bertimbau SAIDA=metrics/bert/bertimbau ARGS="..."
```

**O que fica no pod e o que vem para o portátil:**

| Onde está | O que é | Para onde vai |
|---|---|---|
| a pasta do `--saida` (`metrics/bert/...`) | sugestões com τ = 0 e τ = 0,25 por época e da última época, `treino.json` | **portátil** (pequenos; `rsync` como no passo 4) |
| `models/bert/...` | pontuações de todas as etiquetas por época (~0,3 GB por época: estimativa, 24 mil contratos × 3 000 etiquetas × 4 bytes; mede-se no E5-T4) e um *checkpoint* por época (o modelo e o estado do otimizador, cerca de 3 vezes o modelo: estimativa, o otimizador AdamW guarda 2 números por peso; mede-se no E5-T4) | **fica no pod** (grandes, fora do git) |

Se um *checkpoint* ou as pontuações forem precisos depois de apagar o pod (por exemplo o da época escolhida no E5-S1), têm de ser guardados antes, num volume de rede ou noutro sítio: o volume `/workspace` apaga-se quando o pod é apagado. O `teste.parquet` nunca vai para o pod.

**Se o treino parar a meio** (o pod parou ou reiniciou, ou o processo caiu): corre-se o mesmo comando com `--retomar`, por exemplo `make treinar-bert MODELO=bertimbau ARGS="--retomar"`, com os mesmos argumentos do treino interrompido (o treino recusa-se a retomar com outros: confirma-os no `argumentos.json` da pasta de saída). Cada *checkpoint* leva o estado do otimizador, da taxa de aprendizagem e dos geradores aleatórios, por isso o treino continua da última época gravada e só se pagam as épocas que faltam. Nos testes, um treino retomado dá as mesmas pontuações que um treino que nunca parou. Isto só funciona porque o projeto está em `/workspace`: no disco do contentor, os *checkpoints* perdiam-se ao reiniciar. Se o pod foi apagado, o volume foi com ele e não há nada para retomar.
