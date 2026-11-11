"""Modelo BERT tunado como classificador: texto -> 5 etiquetas de treino ordenadas (E5-T2).

Serve os dois modelos do E5 (`bert_models.MODELOS`): o BERTimbau (arquitetura BERT) e o
Albertina (arquitetura DeBERTa). O código é o mesmo para os dois porque usa as classes `Auto*` do
Hugging Face: `AutoTokenizer` e `AutoModelForSequenceClassification` (o modelo pré-treinado com
uma camada nova por cima, que dá uma pontuação por etiqueta). O treino usa o `Trainer` do
Hugging Face, como nos exemplos oficiais de classificação de texto da biblioteca.

Etiquetas: as mesmas do TF-IDF (`labels.collapse_rare_codes` com o N, contadas só no treino). A
lista de etiquetas está por ordem crescente, como o `classes_` do scikit-learn, para os empates
se resolverem pelo menor código (`tfidf.top_k`). A tradução número -> código fica dentro do
modelo gravado (`config.id2label`), por isso cada *checkpoint* sabe que código é cada saída.

Correção das etiquetas frequentes (Menon et al., 2021, "logit adjustment"): ordena-se por
pontuação − τ · log(frequência da etiqueta no treino), como no TF-IDF. Aqui as pontuações são os
*logits* de um modelo softmax, que é exatamente o caso do artigo (no TF-IDF era uma adaptação).

Repetível no processador: a mesma semente dá o mesmo modelo e as mesmas sugestões. Em GPU pode
não dar exatamente o mesmo (algumas operações da GPU não são deterministas); isso mede-se no E5-S1.

Este módulo não lê nem grava ficheiros de dados: isso é do `train_bert.py`. Só o `Trainer` grava
os *checkpoints* na pasta pedida.
"""

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainerCallback,
                          TrainingArguments, set_seed)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

import bert_models
import tfidf
from evaluation import TOP_K
from labels import DEFAULT_N_MIN, collapse_rare_codes, to_level

# Hiperparâmetros por defeito: a receita de afinação do artigo do BERT (Devlin et al., 2019,
# secção A.3: taxa de aprendizagem entre 2e-5 e 5e-5, lote de 16 ou 32, 2 a 4 épocas, aquecimento
# em 10% dos passos e descida linear, decaimento dos pesos 0,01). O E5-T4 fixa os valores a usar
# num YAML escrito antes de correr, com a fonte de cada um; estes são só o ponto de partida.
TAXA_APRENDIZAGEM = 2e-5
LOTE = 32
EPOCAS = 3
AQUECIMENTO = 0.1  # fração dos passos de treino com a taxa a subir de 0 até à taxa pedida
DECAIMENTO_PESOS = 0.01
SEMENTE = 0
LOTE_PONTUAR = 64  # contratos por lote ao pontuar a validação (só muda a memória e o tempo)
DISPOSITIVOS = ("cpu", "cuda")


def check_gpu():
    """Erro se o `torch` não vir a GPU; se a vir, devolve o nome da placa.

    Corre no pod antes de ler os dados: sem esta verificação, um `torch` instalado só para
    processador (ou um pod sem GPU) treinava no processador, horas a pagar sem dar erro.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("o torch não vê nenhuma GPU (torch.cuda.is_available() é False): confirme que o pod "
                           "tem GPU e que se instalou o requirements-gpu.txt, ou use dispositivo='cpu'")
    return torch.cuda.get_device_name(0)


def check_dispositivo(dispositivo):
    if dispositivo not in DISPOSITIVOS:
        raise ValueError(f"dispositivo desconhecido: {dispositivo!r} (válidos: {DISPOSITIVOS})")


def training_labels(codigo, n_min=DEFAULT_N_MIN):
    """Etiqueta de treino de cada exemplo (NaN se sai) e a lista de etiquetas, por ordem crescente.

    As mesmas regras do `tfidf.train`: recusa códigos ao nível da divisão (o modelo podia
    sugeri-los) e menos de 5 etiquetas (não chegavam para 5 sugestões).
    """
    if codigo.str.endswith("000000").any():
        # O prepare.py tira-os do treino (decisão 8); se aparecerem, o modelo podia sugeri-los.
        raise ValueError("o treino tem códigos ao nível da divisão (XX000000)")
    etiqueta = collapse_rare_codes(codigo, n_min=n_min)
    classes = np.sort(etiqueta.dropna().unique()).astype(object)
    if len(classes) < TOP_K:
        raise ValueError(f"o treino só tem {len(classes)} etiquetas depois da regra dos códigos raros; "
                         f"são precisas {TOP_K}")
    return etiqueta, classes


def model_max_length(modelo, max_length=None):
    """O comprimento máximo em tokens do `modelo`: o dado, ou, para uma chave do E5, o do E5-T1.

    Com a pasta de um modelo (testes) tem de ser dado. Erro se não for um inteiro >= 2.
    """
    if max_length is None:
        if modelo not in bert_models.MODELOS:
            raise ValueError(f"{modelo}: com a pasta de um modelo é preciso dar o max_length")
        max_length = bert_models.COMPRIMENTO_MAXIMO[modelo]
    if not isinstance(max_length, (int, np.integer)) or isinstance(max_length, bool) or max_length < 2:
        raise ValueError(f"max_length tem de ser um inteiro >= 2: {max_length!r}")
    return int(max_length)


def load_model(modelo, classes, max_length=None):
    """O tokenizador e o modelo pré-treinado com uma camada de classificação nova, uma saída por etiqueta.

    `modelo` é uma chave de `bert_models.MODELOS` ("bertimbau" ou "albertina"; descarrega a
    revisão fixa do Hugging Face) ou a pasta de um modelo gravado com `save_pretrained` (os testes
    usam um modelo minúsculo). Com uma chave, `max_length` por defeito é o `COMPRIMENTO_MAXIMO`
    do E5-T1; com uma pasta tem de ser dado.
    Devolve (modelo, tokenizador, {"nome", "revisao", "max_length"}).
    """
    max_length = model_max_length(modelo, max_length)
    if modelo in bert_models.MODELOS:
        nome, revisao = bert_models.MODELOS[modelo]["nome"], bert_models.MODELOS[modelo]["revisao"]
    else:
        nome, revisao = str(modelo), None
    extra = {} if revisao is None else {"revision": revisao}
    tokenizer = AutoTokenizer.from_pretrained(nome, **extra)
    model = AutoModelForSequenceClassification.from_pretrained(
        nome, **extra,
        num_labels=len(classes),
        # A tradução número -> código fica no config.json de cada checkpoint
        id2label={i: str(c) for i, c in enumerate(classes)},
        label2id={str(c): i for i, c in enumerate(classes)},
        # Com o transformers 5 o tipo por defeito é o do ficheiro: o Albertina está gravado em
        # bfloat16 e treinaria em bfloat16. Os pesos treinam-se em float32.
        dtype=torch.float32,
        # Se a pasta já trouxer uma camada de classificação com outro número de saídas, cria-se
        # uma nova (os modelos do Hugging Face não trazem nenhuma; a camada nova é sempre aleatória)
        ignore_mismatched_sizes=True,
    )
    return model, tokenizer, {"nome": nome, "revisao": revisao, "max_length": max_length}


def encode(tokenizer, textos, max_length, batch_size=1000):
    """Os `input_ids` de cada texto (um array int32 por texto, sem preenchimento), cortados a `max_length`.

    Passa-se sempre `max_length`: o tokenizador do BERTimbau não tem `model_max_length`, e
    `truncation=True` sozinho não cortava aos 512 (bert_models.py).
    """
    textos = list(textos)
    out = []
    for start in range(0, len(textos), batch_size):
        enc = tokenizer(textos[start:start + batch_size], truncation=True, max_length=max_length,
                        return_attention_mask=False, return_token_type_ids=False)
        out.extend(np.asarray(ids, dtype=np.int32) for ids in enc["input_ids"])
    return out


class EncodedTexts(torch.utils.data.Dataset):
    """Os textos já tokenizados (e as etiquetas, como números), para o `Trainer`."""

    def __init__(self, input_ids, labels=None):
        self.input_ids, self.labels = input_ids, labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        item = {"input_ids": self.input_ids[i]}
        if self.labels is not None:
            item["labels"] = int(self.labels[i])
        return item


class Collator:
    """Junta exemplos num lote: preenche com o `pad_token_id` até ao maior texto do lote."""

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        n = max(len(f["input_ids"]) for f in features)
        input_ids = torch.full((len(features), n), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(features), n), dtype=torch.long)
        for i, f in enumerate(features):
            input_ids[i, :len(f["input_ids"])] = torch.as_tensor(f["input_ids"], dtype=torch.long)
            attention_mask[i, :len(f["input_ids"])] = 1
        batch = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "labels" in features[0]:
            batch["labels"] = torch.tensor([f["labels"] for f in features], dtype=torch.long)
        return batch


@torch.inference_mode()
def predict_scores(model, input_ids, pad_token_id, batch_size=LOTE_PONTUAR):
    """Pontuação (logit) de cada etiqueta para cada texto: array float32 (textos x etiquetas).

    As linhas ficam pela ordem de `input_ids`. Põe o modelo em modo de avaliação (sem *dropout*)
    e devolve-o ao modo em que estava.
    """
    collate = Collator(pad_token_id)
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    out = np.empty((len(input_ids), model.config.num_labels), dtype=np.float32)
    try:
        for start in range(0, len(input_ids), batch_size):
            batch = collate([{"input_ids": ids} for ids in input_ids[start:start + batch_size]])
            logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
            out[start:start + len(logits)] = logits.float().cpu().numpy()
    finally:
        model.train(was_training)
    return out


def scores(model, tokenizer, textos, max_length, batch_size=LOTE_PONTUAR):
    """`predict_scores` a partir dos textos (tokeniza-os primeiro)."""
    return predict_scores(model, encode(tokenizer, textos, max_length), tokenizer.pad_token_id, batch_size)


class EpochScores(TrainerCallback):
    """No fim de cada época, pontua os textos da validação e entrega as pontuações a `on_epoch`.

    `on_epoch(epoca, checkpoint, pontuacoes)`: a época (1, 2, ...), o nome da pasta do checkpoint
    que o `Trainer` grava a seguir (`checkpoint-<passo>`) e o array (textos x etiquetas).
    """

    def __init__(self, input_ids, pad_token_id, on_epoch, batch_size=LOTE_PONTUAR):
        self.input_ids, self.pad_token_id = input_ids, pad_token_id
        self.on_epoch, self.batch_size = on_epoch, batch_size

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        pontuacoes = predict_scores(model, self.input_ids, self.pad_token_id, self.batch_size)
        self.on_epoch(int(round(state.epoch)), f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}", pontuacoes)


class StepTimer(TrainerCallback):
    """Mede o tempo dos passos de treino (um passo = um lote: calcular a perda e mudar os pesos).

    Guarda o relógio no fim do 1.º passo e no fim do último desta execução: os segundos por passo
    são o intervalo a dividir pelos passos que o separam. O 1.º passo fica de fora porque inclui
    trabalho de arranque (por exemplo, a GPU a preparar as operações). Na GPU espera-se que ela
    acabe o passo (`synchronize`): as operações da GPU correm em fila, e sem isto o relógio media
    só o tempo de as pôr na fila.
    """

    def __init__(self):
        self.primeiro = self.ultimo = None
        self.passos = 0

    def on_step_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available() and args.device.type == "cuda":
            torch.cuda.synchronize()
        agora = time.perf_counter()
        if self.primeiro is None:
            self.primeiro = agora
        self.ultimo = agora
        self.passos += 1

    def segundos_por_passo(self):
        """Média dos passos depois do 1.º (None com menos de 2 passos)."""
        if self.passos < 2:
            return None
        return (self.ultimo - self.primeiro) / (self.passos - 1)


def training_arguments(checkpoints_dir, taxa_aprendizagem=TAXA_APRENDIZAGEM, lote=LOTE, epocas=EPOCAS,
                       aquecimento=AQUECIMENTO, decaimento_pesos=DECAIMENTO_PESOS, semente=SEMENTE,
                       dispositivo="cuda", max_passos=None, gravar_checkpoints=True):
    """Os `TrainingArguments` do treino (um sítio só, para os testes os poderem ler).

    `max_passos`: pára o treino ao fim desse número de passos (lotes), mesmo a meio de uma época
    (`max_steps` do Trainer); o ensaio de custo do E5-T4. A taxa de aprendizagem sobe e desce ao
    longo desses passos, não das épocas. `gravar_checkpoints=False`: nenhum checkpoint (o ensaio e
    o sinal de decorar, que não se retomam nem se usam depois).
    """
    check_dispositivo(dispositivo)
    if max_passos is not None and (isinstance(max_passos, bool) or not isinstance(max_passos, (int, np.integer))
                                   or max_passos < 1):
        raise ValueError(f"max_passos tem de ser um inteiro >= 1: {max_passos!r}")
    if not 0 <= aquecimento < 1:
        # No transformers 5, warmup_steps < 1 é a fração dos passos; >= 1 seria um número de passos
        raise ValueError(f"aquecimento é uma fração dos passos, em [0, 1): {aquecimento!r}")
    if isinstance(epocas, bool) or not isinstance(epocas, (int, np.integer)) or epocas < 1:
        raise ValueError(f"epocas tem de ser um inteiro >= 1: {epocas!r}")
    return TrainingArguments(
        output_dir=str(checkpoints_dir),
        learning_rate=taxa_aprendizagem,
        per_device_train_batch_size=lote,
        num_train_epochs=epocas,
        max_steps=-1 if max_passos is None else int(max_passos),  # -1: o número de épocas decide
        warmup_steps=aquecimento,
        lr_scheduler_type="linear",
        weight_decay=decaimento_pesos,
        seed=semente,
        data_seed=semente,  # a ordem dos exemplos em cada época
        # Um checkpoint no fim de cada época, todos guardados (sem save_total_limit), para usar o
        # da época escolhida no E5-S1. Cada um leva também o estado do otimizador, do
        # escalonador da taxa e dos geradores aleatórios (save_only_model=False, o normal do
        # Trainer): ocupa cerca de 3 vezes o modelo, mas um treino interrompido (o pod caiu)
        # continua da última época gravada (`retomar`) em vez de se pagar outra vez desde o início.
        save_strategy="epoch" if gravar_checkpoints else "no",
        save_only_model=False,
        eval_strategy="no",  # a validação pontua-se no EpochScores, com todas as etiquetas
        logging_strategy="epoch",  # a perda média de cada época vai para o treino.json
        report_to="none",  # o registo no MLflow é o E5-T3
        disable_tqdm=True,
        dataloader_num_workers=0,
        use_cpu=dispositivo == "cpu",
    )


def last_checkpoint(checkpoints_dir):
    """A pasta do último checkpoint completo em `checkpoints_dir` (None se não houver nenhum).

    O Trainer grava o `trainer_state.json` em último lugar, depois do modelo e do otimizador: um
    checkpoint sem ele ficou a meio (o pod caiu enquanto gravava) e não serve para retomar. O
    `get_last_checkpoint` do transformers escolheria a pasta de número maior, completa ou não.
    """
    pasta = Path(checkpoints_dir)
    if not pasta.is_dir():
        return None
    completos = [p for p in pasta.glob(f"{PREFIX_CHECKPOINT_DIR}-*")
                 if p.is_dir() and p.name.split("-")[-1].isdigit() and (p / "trainer_state.json").is_file()]
    return str(max(completos, key=lambda p: int(p.name.split("-")[-1]))) if completos else None


def train(texto, codigo, modelo, checkpoints_dir, textos_val=None, on_epoch=None, n_min=DEFAULT_N_MIN,
          max_length=None, taxa_aprendizagem=TAXA_APRENDIZAGEM, lote=LOTE, epocas=EPOCAS,
          aquecimento=AQUECIMENTO, decaimento_pesos=DECAIMENTO_PESOS, semente=SEMENTE, dispositivo="cuda",
          retomar=False, classes=None, max_passos=None, gravar_checkpoints=True):
    """Tuna o `modelo` com o treino (`texto` e `codigo`, com o mesmo índice).

    A etiqueta de cada exemplo é `collapse_rare_codes(codigo, n_min)`; os exemplos que saem (NaN)
    não entram no treino.
    `classes`: a lista de etiquetas de outro treino, por ordem crescente (o sinal de erro 1 do E5:
    100 contratos com as ~3 000 etiquetas do treino completo). Então `codigo` já é a etiqueta de
    cada exemplo, contada nesse outro treino: não se aplica a regra dos códigos raros (o `n_min`
    não conta), cada etiqueta tem de estar em `classes`, e o `log_prior` devolvido é None (as
    frequências são as do outro treino).
    `max_passos` e `gravar_checkpoints`: ver `training_arguments` (o ensaio de custo).
    No fim de cada época o `Trainer` grava um checkpoint em `checkpoints_dir` (se
    `gravar_checkpoints`) e, se houver `textos_val`, chama `on_epoch(epoca, checkpoint, pontuacoes)`
    com as pontuações desses textos (ver `EpochScores`).
    `retomar`: continua do último checkpoint de `checkpoints_dir` (pesos, otimizador, taxa e
    geradores aleatórios), com `resume_from_checkpoint` do Trainer; só correm as épocas que
    faltam, e só para essas se chama o `on_epoch`. Os argumentos têm de ser os do treino
    interrompido.
    Devolve um dicionário com "model", "tokenizer", "classes", "log_prior", "info" (nome,
    revisão, max_length), "trainer", "exemplos_usados", "exemplos_saidos", "passos" (os passos
    feitos nesta execução), "segundos_por_passo" (`StepTimer`), "lote_efetivo" (o lote × o n.º de
    GPUs, se o Trainer usar várias) e "passos_por_epoca" (os exemplos usados a dividir pelo lote
    efetivo, arredondado para cima: o último lote de cada época pode ser mais pequeno).
    """
    check_dispositivo(dispositivo)
    if not texto.index.equals(codigo.index):
        raise ValueError("`texto` e `codigo` têm índices diferentes")
    args = training_arguments(checkpoints_dir, taxa_aprendizagem, lote, epocas, aquecimento,
                              decaimento_pesos, semente, dispositivo, max_passos, gravar_checkpoints)
    externas = classes is not None
    if not externas:
        etiqueta, classes = training_labels(codigo, n_min)
    else:
        classes = np.asarray(classes, dtype=object)
        if list(classes) != sorted(set(classes)):
            raise ValueError("as `classes` têm de estar por ordem crescente, sem repetidas")
        fora = sorted(set(codigo) - set(classes))
        if fora:
            raise ValueError(f"etiquetas que não estão nas `classes`: {fora[:3]}")
        etiqueta = codigo
    keep = etiqueta.notna()
    id_de = {c: i for i, c in enumerate(classes)}

    # A semente antes de criar o modelo: a camada de classificação nova começa com pesos aleatórios
    set_seed(semente)
    model, tokenizer, info = load_model(modelo, classes, max_length)
    treino = EncodedTexts(encode(tokenizer, texto[keep], info["max_length"]),
                          etiqueta[keep].map(id_de).to_numpy())
    callbacks = []
    if textos_val is not None:
        if on_epoch is None:
            raise ValueError("com `textos_val` é preciso o `on_epoch`, que recebe as pontuações")
        callbacks.append(EpochScores(encode(tokenizer, textos_val, info["max_length"]),
                                     tokenizer.pad_token_id, on_epoch))
    relogio = StepTimer()
    callbacks.append(relogio)
    trainer = Trainer(model=model, args=args, train_dataset=treino, data_collator=Collator(tokenizer.pad_token_id),
                      processing_class=tokenizer, callbacks=callbacks)
    retomar_de = last_checkpoint(checkpoints_dir) if retomar else None
    if retomar and retomar_de is None:
        raise ValueError(f"{checkpoints_dir}: não há nenhum checkpoint para retomar")
    trainer.train(resume_from_checkpoint=retomar_de)
    lote_efetivo = int(trainer.args.train_batch_size)
    return {"model": trainer.model, "tokenizer": tokenizer, "classes": classes,
            "log_prior": None if externas else tfidf.log_prior(codigo, classes, n_min=n_min),
            "info": info, "trainer": trainer,
            "exemplos_usados": int(keep.sum()), "exemplos_saidos": int((~keep).sum()),
            "passos": relogio.passos, "segundos_por_passo": relogio.segundos_por_passo(),
            "lote_efetivo": lote_efetivo, "passos_por_epoca": math.ceil(int(keep.sum()) / lote_efetivo)}


def validation_labels(codigo, classes):
    """A etiqueta de cada contrato da validação entre as `classes` do treino (NaN se fica de fora).

    `codigo`: o primeiro código de cada contrato (a coluna `codigo` do validacao.parquet). Fica o
    código se for uma etiqueta do treino; senão a sua classe (4 dígitos), se for; senão o seu
    grupo (3 dígitos), se for; senão NaN (o contrato não entra na perda).

    Isto é aplicar a regra dos códigos raros do treino (`collapse_rare_codes`, com as contagens
    só do treino) ao código do contrato. No treino, um código fica etiqueta se tiver pelo menos N
    exemplos; senão sobe para a classe, e a classe para o grupo. Um código da validação que não
    está nas `classes` ou foi raro no treino ou nem apareceu lá (0 exemplos, também menos de N):
    nos dois casos o treino subia-o para a classe, e a classe só é etiqueta se tiver chegado a N;
    o mesmo para o grupo. Exemplo: 33141100 não é etiqueta, mas 33140000 é (dois códigos raros da
    classe 3314 juntaram-se lá) -> 33140000. Um código de divisão (XX000000) fica sempre de fora:
    a classe e o grupo são ele próprio, e o treino não tem divisões. Não se conta nada na
    validação (seria fuga de dados): só se procura nas `classes`.
    """
    codigo = pd.Series(codigo, dtype=object)
    no_treino = set(str(c) for c in classes)
    etiqueta = pd.Series(np.nan, index=codigo.index, dtype=object)
    validos = codigo.notna()
    texto = codigo[validos].astype(str)
    for nivel in (texto, to_level(texto, 4), to_level(texto, 3)):  # código -> classe -> grupo
        falta = etiqueta.loc[texto.index].isna() & nivel.isin(no_treino)
        etiqueta.loc[falta[falta].index] = nivel[falta]
    return etiqueta


def cross_entropy(pontuacoes, etiquetas, classes):
    """A entropia cruzada média das `pontuacoes` (logits) com as `etiquetas`: (perda, n).

    Para cada contrato com etiqueta, −log(softmax(logits)[etiqueta certa]); a média sobre os n
    contratos que entram (os de etiqueta NaN, ver `validation_labels`, ficam de fora). É a mesma
    conta da perda do treino (a `CrossEntropyLoss` do modelo). Faz-se em float64 com o
    log-softmax estável: log softmax(x)_c = x_c − max(x) − log(Σ exp(x − max(x))); sem tirar o
    máximo, exp de um logit grande dava infinito. Sem nenhum contrato, devolve (None, 0).
    """
    classes = np.asarray(classes, dtype=object)
    pontuacoes = np.asarray(pontuacoes)
    if pontuacoes.ndim != 2 or pontuacoes.shape[1] != len(classes):
        raise ValueError(f"as pontuações têm forma {pontuacoes.shape}; são precisas {len(classes)} colunas")
    etiquetas = pd.Series(etiquetas, dtype=object).reset_index(drop=True)
    if len(etiquetas) != len(pontuacoes):
        raise ValueError(f"{len(etiquetas)} etiquetas para {len(pontuacoes)} linhas de pontuações")
    id_de = {str(c): i for i, c in enumerate(classes)}
    entram = etiquetas.notna().to_numpy()
    if not entram.any():
        return None, 0
    fora = sorted(set(etiquetas[entram].astype(str)) - set(id_de))
    if fora:
        raise ValueError(f"etiquetas que não estão nas `classes`: {fora[:3]}")
    colunas = etiquetas[entram].astype(str).map(id_de).to_numpy()
    x = pontuacoes[entram].astype(np.float64)
    maximo = x.max(axis=1, keepdims=True)
    log_soma = np.log(np.exp(x - maximo).sum(axis=1)) + maximo[:, 0]
    perdas = log_soma - x[np.arange(len(x)), colunas]  # −log softmax na etiqueta certa
    return float(perdas.mean()), int(entram.sum())


def suggest_from_scores(pontuacoes, classes, log_prior=None, tau=0.0, k=TOP_K):
    """As k sugestões de cada linha de `pontuacoes`, ordenadas por pontuação − τ · log_prior.

    `classes` por ordem crescente (os empates ficam para o menor código, `tfidf.top_k`). Com
    τ = 0 a ordem é só a das pontuações e o `log_prior` não é preciso.
    """
    classes = np.asarray(classes, dtype=object)
    if pontuacoes.ndim != 2 or pontuacoes.shape[1] != len(classes):
        raise ValueError(f"as pontuações têm forma {pontuacoes.shape}; são precisas {len(classes)} colunas")
    if len(classes) < k:
        raise ValueError(f"só há {len(classes)} etiquetas; são precisas {k}")
    if tau < 0:
        raise ValueError(f"τ tem de ser >= 0: {tau}")
    if tau != 0:
        if log_prior is None:
            raise ValueError("com τ diferente de 0 é preciso o `log_prior`")
        log_prior = np.asarray(log_prior, dtype=float)
        if log_prior.shape != classes.shape:
            raise ValueError("o `log_prior` não tem uma entrada por etiqueta")
        pontuacoes = pontuacoes - tau * log_prior
    return [[str(c) for c in s] for s in tfidf.top_k(pontuacoes, classes, k)]


def suggestions_frame(idcontrato, sugestoes):
    """O ficheiro de sugestões no formato do TF-IDF: `idcontrato` e `sugestoes` (lista de códigos)."""
    return pd.DataFrame({"idcontrato": pd.Series(idcontrato).astype(str).to_numpy(), "sugestoes": sugestoes})
