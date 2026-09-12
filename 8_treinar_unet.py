# =============================================================================
# 8_treinar_unet.py — Treino da U-Net para segmentação de nódulos (v2)
# =============================================================================
#
# Pensado para rodar SEM ALTERAÇÃO tanto localmente (modo debug, CPU,
# amostra pequena) quanto no Google Colab (modo completo, GPU).
#
# NOTA SOBRE NORMALIZAÇÃO: as imagens .npy já saem normalizadas para
# [0,1] do script 4 (clipping de HU + normalização) — não normalizamos
# de novo aqui. Se você gerou os .npy de outra forma, confirme que estão
# nessa escala antes de treinar.
#
# Saídas (em CAMINHO_SAIDA_MODELO):
#   melhor_modelo_unet.pth      — pesos do modelo com melhor Dice em validação
#   historico_treino.csv        — loss/Dice/IoU/recall/precisão por época
#   curvas_treino.png           — gráfico de loss e métricas por época
#   config_treino.json          — hiperparâmetros usados (reprodutibilidade)
#   visualizacoes_treino/       — imagem/GT/predição a cada N épocas, para
#                                  os mesmos exemplos fixos (acompanhar a
#                                  evolução visual do modelo)
#
# =============================================================================

import os
import json
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from monai.networks.nets import UNet
from monai.losses import DiceFocalLoss

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

AMBIENTE = "colab"   # "local" ou "colab"

if AMBIENTE == "colab":
    # IMPORTANTE: monte o Google Drive numa célula do notebook ANTES de
    # rodar este script (drive.mount() só funciona chamado diretamente
    # numa célula — quando este script roda como subprocesso via
    # "!python 8_treinar_unet.py", ele não tem acesso ao kernel do
    # Colab e a chamada quebra). Rode isto numa célula separada primeiro:
    #
    #   from google.colab import drive
    #   drive.mount("/content/drive")
    #
    if not os.path.isdir("/content/drive/MyDrive"):
        raise RuntimeError(
            "Google Drive não está montado em /content/drive. Rode "
            "'from google.colab import drive; drive.mount(\"/content/drive\")' "
            "numa célula do notebook ANTES de rodar este script.")

    # Aponta para o novo slices_2dV3.zip descompactado. CONFIRME depois
    # de descompactar se não ficou uma pasta duplicada (ex:
    # /content/slices_2dV3/slices_2dV3/...) — já aconteceu antes; se
    # acontecer de novo, ajuste esse caminho para incluir o nível extra.
    CAMINHO_SLICES = "/content/slices_2d/slices_2dV3"
    CAMINHO_SAIDA_MODELO = "/content/drive/MyDrive/TG/modelo_unetV3"
else:
    CAMINHO_SLICES = r"D:\TG\slices_2dV3"
    CAMINHO_SAIDA_MODELO = r"D:\TG\modelo_unetV3"

os.makedirs(CAMINHO_SAIDA_MODELO, exist_ok=True)
CAMINHO_VISUALIZACOES = os.path.join(CAMINHO_SAIDA_MODELO, "visualizacoes_treino")
CAMINHO_MANIFESTO = os.path.join(CAMINHO_SLICES, "manifesto.csv")

DEBUG_MODE = False

if DEBUG_MODE:
    N_AMOSTRA_TREINO_DEBUG = 40
    N_AMOSTRA_VAL_DEBUG = 20
    BATCH_SIZE = 2
    N_EPOCAS = 3
    PACIENCIA_EARLY_STOPPING = 3
    NUM_WORKERS = 0
    VISUALIZAR_A_CADA_N_EPOCAS = 1
else:
    N_AMOSTRA_TREINO_DEBUG = None
    N_AMOSTRA_VAL_DEBUG = None
    BATCH_SIZE = 16
    N_EPOCAS = 80
    # Paciência maior: em segmentação de nódulo (classe muito rara por
    # pixel), o Dice pode ficar baixo por 15-20 épocas antes de "decolar"
    # — early stopping curto demais pode interromper o treino antes
    # disso acontecer (foi o que parece ter ocorrido na 1ª tentativa).
    PACIENCIA_EARLY_STOPPING = 20
    NUM_WORKERS = 2
    VISUALIZAR_A_CADA_N_EPOCAS = 5

# LR mais conservador (1e-3 pode ser agressivo demais com AdamW nesse
# problema — risco de o modelo ficar "preso" prevendo algo próximo de
# ruído uniforme, sem conseguir localizar o nódulo).
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
# Paciência do scheduler MAIOR que a do early stopping's primeiros
# estágios — não corta o LR antes do modelo ter uma chance real de sair
# do platô inicial.
PACIENCIA_SCHEDULER = 8
MIN_DELTA = 0.001          # ganho mínimo de Dice para contar como "melhora"
N_EXEMPLOS_VISUALIZACAO = 3

# Tamanho do recorte (patch) usado no treino/validação — ver NoduleDataset.
# Precisa ser múltiplo de 16 (2^4, um para cada nível de downsampling da
# U-Net com strides=(2,2,2,2)), senão as conexões de salto (skip
# connections) não alinham em tamanho e o forward quebra.
TAMANHO_PATCH = 128
assert TAMANHO_PATCH % 16 == 0, "TAMANHO_PATCH precisa ser múltiplo de 16"
JITTER_PATCH_TREINO = 16   # variação aleatória do centro do recorte (só treino)
SEED = 42

# --- Reprodutibilidade total (troca um pouco de velocidade por
# determinismo — se o treino no Colab ficar sensivelmente mais lento por
# causa disso, pode desligar cudnn.deterministic) ---
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USAR_AMP = (DISPOSITIVO.type == "cuda")  # mixed precision só faz sentido em GPU


# =============================================================================
# AUGMENTAÇÃO
# =============================================================================

def aumentar_imagem_mascara(img, mask):
    """
    Augmentação leve, pensada para não distorcer a anatomia pulmonar de
    forma irrealista (por isso NÃO usamos elastic deformation):
      - flip horizontal
      - rotação pequena (±10°)
      - translação pequena (±5 px)
      - variação leve de brilho/contraste
      - ruído gaussiano leve
    """
    if random.random() < 0.5:
        img = np.ascontiguousarray(img[:, ::-1])
        mask = np.ascontiguousarray(mask[:, ::-1])

    if random.random() < 0.5:
        angulo = np.random.uniform(-10, 10)
        img = ndimage.rotate(img, angulo, reshape=False, order=1, mode="nearest")
        mask = ndimage.rotate(mask, angulo, reshape=False, order=0, mode="nearest")

    if random.random() < 0.5:
        dy, dx = np.random.randint(-5, 6, size=2)
        img = ndimage.shift(img, shift=(dy, dx), order=1, mode="nearest")
        mask = ndimage.shift(mask, shift=(dy, dx), order=0, mode="nearest")

    if random.random() < 0.5:
        fator = np.random.uniform(0.9, 1.1)
        offset = np.random.uniform(-0.05, 0.05)
        img = np.clip(img * fator + offset, 0, 1)

    if random.random() < 0.3:
        ruido = np.random.normal(0, 0.02, size=img.shape)
        img = np.clip(img + ruido, 0, 1)

    mask = (mask > 0.5).astype(np.float32)
    return img.astype(np.float32), mask


# =============================================================================
# DATASET
# =============================================================================

def _nome_arquivo(caminho):
    """Extrai o nome do arquivo robusto a caminhos do Windows (\\) mesmo
    rodando em Linux/Colab, onde os.path.basename não trata \\ como separador."""
    return caminho.replace("\\", "/").split("/")[-1]


def corrigir_caminhos(df, caminho_slices):
    """
    O manifesto.csv guarda caminhos ABSOLUTOS de quando o script 4 foi
    rodado (ex: no Windows). Reconstruímos os caminhos a partir de
    CAMINHO_SLICES + split, para funcionar em qualquer ambiente.
    """
    df = df.copy()
    df["arquivo_imagem"] = df.apply(
        lambda r: os.path.join(caminho_slices, r["split"], "imagens",
                                _nome_arquivo(r["arquivo_imagem"])), axis=1)
    df["arquivo_mascara"] = df.apply(
        lambda r: os.path.join(caminho_slices, r["split"], "mascaras",
                                _nome_arquivo(r["arquivo_mascara"])), axis=1)
    return df


def extrair_patch(img, mask, tamanho, jitter_max=0, aleatorio_negativo=True):
    """
    Recorta uma janela (tamanho x tamanho) de img/mask:
      - se mask tem nódulo: centraliza no centróide do nódulo, com
        jitter aleatório de até jitter_max pixels (variação entre
        épocas, funciona como augmentação extra)
      - se mask está vazia (slice negativo): se aleatorio_negativo=True
        (treino), sorteia uma posição com tecido real; se False
        (validação), usa sempre o centro da imagem — determinístico,
        para a métrica de validação não variar de época pra época por
        causa de recortes aleatórios diferentes do mesmo slice.
    Preenche com zero (padding) se a janela ultrapassar a borda da imagem.
    """
    H, W = img.shape
    metade = tamanho // 2

    if mask.sum() > 0:
        ys, xs = np.where(mask > 0)
        cy, cx = int(ys.mean()), int(xs.mean())
        if jitter_max > 0:
            cy += np.random.randint(-jitter_max, jitter_max + 1)
            cx += np.random.randint(-jitter_max, jitter_max + 1)
    else:
        cy, cx = H // 2, W // 2  # fallback / padrão determinístico
        if aleatorio_negativo:
            for _ in range(10):
                cy_cand = np.random.randint(metade, max(H - metade, metade + 1))
                cx_cand = np.random.randint(metade, max(W - metade, metade + 1))
                if img[cy_cand, cx_cand] > 0.05:  # proxy simples de "tem tecido"
                    cy, cx = cy_cand, cx_cand
                    break

    y0, x0 = cy - metade, cx - metade
    y1, x1 = y0 + tamanho, x0 + tamanho

    y0c, x0c = max(0, y0), max(0, x0)
    y1c, x1c = min(H, y1), min(W, x1)

    patch_img = np.zeros((tamanho, tamanho), dtype=np.float32)
    patch_mask = np.zeros((tamanho, tamanho), dtype=np.float32)

    py0, px0 = y0c - y0, x0c - x0
    patch_img[py0:py0 + (y1c - y0c), px0:px0 + (x1c - x0c)] = img[y0c:y1c, x0c:x1c]
    patch_mask[py0:py0 + (y1c - y0c), px0:px0 + (x1c - x0c)] = mask[y0c:y1c, x0c:x1c]

    return patch_img, patch_mask


class NoduleDataset(Dataset):
    """
    Retorna (imagem, máscara, paciente, slice) — paciente/slice não
    entram no treino, mas facilitam gerar figuras específicas depois
    (ex: capítulo de resultados do TG).

    Retorna um RECORTE (patch) de TAMANHO_PATCH x TAMANHO_PATCH em vez
    do slice inteiro (512x512) — o nódulo ocupa uma fração minúscula do
    slice completo (~0.02-0.1%), desbalanceamento severo demais para o
    treino aprender a localizar. Recortando uma janela menor centrada
    no nódulo (ou em tecido, para negativos), essa proporção sobe para
    ~1%, uma melhora de ~15-20x.
    """
    def __init__(self, df, augmentar=False, tamanho_patch=128, jitter_max=16):
        self.df = df.reset_index(drop=True)
        self.augmentar = augmentar
        self.tamanho_patch = tamanho_patch
        self.jitter_max = jitter_max if augmentar else 0
        self.aleatorio_negativo = augmentar  # só varia a posição no treino

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        linha = self.df.iloc[idx]
        img = np.load(linha["arquivo_imagem"]).astype(np.float32)
        mask = np.load(linha["arquivo_mascara"]).astype(np.float32)

        img, mask = extrair_patch(img, mask, self.tamanho_patch, self.jitter_max,
                                   aleatorio_negativo=self.aleatorio_negativo)

        if self.augmentar:
            img, mask = aumentar_imagem_mascara(img, mask)

        img_t = torch.from_numpy(img).unsqueeze(0)
        mask_t = torch.from_numpy(mask).unsqueeze(0)
        return img_t, mask_t, str(linha["paciente"]), int(linha["z"])


def carregar_manifesto_split(manifesto, split, n_amostra=None, seed=SEED):
    sub = manifesto[manifesto["split"] == split].copy()
    if n_amostra is not None and len(sub) > n_amostra:
        sub = sub.sample(n=n_amostra, random_state=seed)
    sub = corrigir_caminhos(sub, CAMINHO_SLICES)
    return sub


# =============================================================================
# MÉTRICAS
# =============================================================================

def somar_confusao_batch(logits, targets, limiar=0.5):
    """
    Soma bruta de TP/FP/FN/TN de um batch — SEM calcular razões aqui.
    As razões (Dice/IoU/recall/precisão) só são calculadas depois de
    somar TP/FP/FN/TN de TODOS os batches da época (ver _metricas_de_confusao).

    Por quê: calcular Dice/recall por batch e depois tirar a média (como
    a v1 deste script fazia) sofre do mesmo viés que já identificamos e
    corrigimos no método clássico — um slice sem nódulo, quando o modelo
    não prediz nada ali, ganha recall/Dice=1.0 "de graça" por definição
    da fórmula. Num dataset com boa fração de slices negativos, isso
    infla a média e mascara o desempenho real de localização. Somando a
    confusão da época inteira e calculando a razão UMA vez no final,
    esse viés desaparece.
    """
    probs = torch.sigmoid(logits)
    preds = (probs > limiar).float()
    tp = (preds * targets).sum().item()
    fp = (preds * (1 - targets)).sum().item()
    fn = ((1 - preds) * targets).sum().item()
    tn = ((1 - preds) * (1 - targets)).sum().item()
    return tp, fp, fn, tn


def _metricas_de_confusao(tp, fp, fn, tn, eps=1e-6):
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    precisao = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    # Fração de pixels previstos como nódulo vs. fração real — útil para
    # diagnosticar rapidamente se o modelo "colapsou" (prevendo quase
    # tudo ou quase nada como nódulo, independente de onde o nódulo está).
    total = tp + fp + fn + tn
    fracao_prevista = (tp + fp) / total if total > 0 else 0.0
    fracao_real = (tp + fn) / total if total > 0 else 0.0
    return dict(dice=dice, iou=iou, recall=recall, precisao=precisao,
                fracao_prevista_positiva=fracao_prevista,
                fracao_real_positiva=fracao_real)


# =============================================================================
# TREINO
# =============================================================================

def treinar_uma_epoca(modelo, loader, otimizador, funcao_perda, scaler):
    modelo.train()
    perda_total, n_batches = 0.0, 0
    tp_t = fp_t = fn_t = tn_t = 0.0

    for imgs, masks, _, _ in loader:
        imgs, masks = imgs.to(DISPOSITIVO), masks.to(DISPOSITIVO)
        otimizador.zero_grad()

        with torch.autocast(device_type=DISPOSITIVO.type, enabled=USAR_AMP):
            logits = modelo(imgs)
            perda = funcao_perda(logits, masks)

        if USAR_AMP:
            scaler.scale(perda).backward()
            scaler.step(otimizador)
            scaler.update()
        else:
            perda.backward()
            otimizador.step()

        perda_total += perda.item()
        tp, fp, fn, tn = somar_confusao_batch(logits.detach(), masks)
        tp_t += tp; fp_t += fp; fn_t += fn; tn_t += tn
        n_batches += 1

    metricas = _metricas_de_confusao(tp_t, fp_t, fn_t, tn_t)
    metricas["perda"] = perda_total / n_batches
    return metricas


def validar(modelo, loader, funcao_perda):
    modelo.eval()
    perda_total, n_batches = 0.0, 0
    tp_t = fp_t = fn_t = tn_t = 0.0

    with torch.no_grad():
        for imgs, masks, _, _ in loader:
            imgs, masks = imgs.to(DISPOSITIVO), masks.to(DISPOSITIVO)
            logits = modelo(imgs)
            perda = funcao_perda(logits, masks)
            perda_total += perda.item()
            tp, fp, fn, tn = somar_confusao_batch(logits, masks)
            tp_t += tp; fp_t += fp; fn_t += fn; tn_t += tn
            n_batches += 1

    metricas = _metricas_de_confusao(tp_t, fp_t, fn_t, tn_t)
    metricas["perda"] = perda_total / n_batches
    return metricas


def salvar_visualizacao_epoca(modelo, exemplos, epoca, pasta_saida):
    modelo.eval()
    os.makedirs(pasta_saida, exist_ok=True)
    with torch.no_grad():
        for img_t, mask_t, paciente, z in exemplos:
            logits = modelo(img_t.unsqueeze(0).to(DISPOSITIVO))
            pred = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()[0, 0]
            img_np = img_t.numpy()[0]
            mask_np = mask_t.numpy()[0]

            fig, eixos = plt.subplots(1, 3, figsize=(9, 3.3))
            eixos[0].imshow(img_np, cmap="gray")
            eixos[0].set_title("Imagem"); eixos[0].axis("off")

            eixos[1].imshow(img_np, cmap="gray")
            if mask_np.sum() > 0:
                eixos[1].contour(mask_np, levels=[0.5], colors=["lime"], linewidths=1.5)
            eixos[1].set_title("Ground truth"); eixos[1].axis("off")

            eixos[2].imshow(img_np, cmap="gray")
            if pred.sum() > 0:
                eixos[2].contour(pred, levels=[0.5], colors=["red"], linewidths=1.5)
            eixos[2].set_title("Predição"); eixos[2].axis("off")

            fig.suptitle("%s — slice %d — época %d" % (paciente, z, epoca), fontsize=10)
            plt.tight_layout()
            nome = "epoca%03d_%s_z%03d.png" % (epoca, paciente, z)
            plt.savefig(os.path.join(pasta_saida, nome), dpi=120)
            plt.close(fig)


def salvar_curvas(historico, caminho_saida):
    fig, eixos = plt.subplots(1, 3, figsize=(15, 4.5))

    eixos[0].plot(historico["epoca"], historico["perda_treino"], label="Treino")
    eixos[0].plot(historico["epoca"], historico["perda_val"], label="Validação")
    eixos[0].set_xlabel("Época"); eixos[0].set_ylabel("Perda (DiceFocal)")
    eixos[0].set_title("Perda por época"); eixos[0].legend(); eixos[0].grid(alpha=0.3)

    eixos[1].plot(historico["epoca"], historico["dice_treino"], label="Dice treino")
    eixos[1].plot(historico["epoca"], historico["dice_val"], label="Dice validação")
    eixos[1].plot(historico["epoca"], historico["iou_val"], "--", label="IoU validação")
    eixos[1].set_xlabel("Época"); eixos[1].set_ylabel("Score")
    eixos[1].set_title("Dice / IoU por época"); eixos[1].legend(); eixos[1].grid(alpha=0.3)

    eixos[2].plot(historico["epoca"], historico["recall_val"], label="Recall validação")
    eixos[2].plot(historico["epoca"], historico["precisao_val"], label="Precisão validação")
    eixos[2].set_xlabel("Época"); eixos[2].set_ylabel("Score")
    eixos[2].set_title("Recall / Precisão (validação)"); eixos[2].legend(); eixos[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=150)
    plt.close(fig)
    print("Curvas salvas em: %s" % caminho_saida)


def main():
    print("Dispositivo: %s | Mixed precision: %s" % (DISPOSITIVO, USAR_AMP))
    if DEBUG_MODE:
        print("*** MODO DEBUG: amostra pequena, poucas épocas — só para "
              "validar o pipeline, não para resultados reais ***")

    manifesto = pd.read_csv(CAMINHO_MANIFESTO)
    df_treino = carregar_manifesto_split(manifesto, "train", N_AMOSTRA_TREINO_DEBUG)
    df_val = carregar_manifesto_split(manifesto, "val", N_AMOSTRA_VAL_DEBUG)
    print("Treino: %d slices | Validação: %d slices" % (len(df_treino), len(df_val)))

    ds_treino = NoduleDataset(df_treino, augmentar=True,
                               tamanho_patch=TAMANHO_PATCH,
                               jitter_max=JITTER_PATCH_TREINO)
    ds_val = NoduleDataset(df_val, augmentar=False,
                            tamanho_patch=TAMANHO_PATCH, jitter_max=0)

    # Superamostra slices POSITIVOS no treino (proporção alvo ~50/50 em
    # vez da proporção natural ~40/60) — dá ao modelo mais oportunidades
    # de ver e aprender a localizar nódulos, em vez do gradiente ser
    # dominado pelos slices negativos (mais numerosos).
    tem_nodulo = df_treino["tem_nodulo"].values
    n_pos, n_neg = tem_nodulo.sum(), (~tem_nodulo.astype(bool)).sum()
    peso_pos = 1.0 / max(n_pos, 1)
    peso_neg = 1.0 / max(n_neg, 1)
    pesos_amostra = np.where(tem_nodulo.astype(bool), peso_pos, peso_neg)
    sampler_treino = WeightedRandomSampler(
        weights=pesos_amostra, num_samples=len(ds_treino), replacement=True)
    print("Superamostragem: %d slices positivos, %d negativos no treino "
          "(sorteio agora ~50/50 por época)" % (n_pos, n_neg))

    loader_treino = DataLoader(
        ds_treino, batch_size=BATCH_SIZE, sampler=sampler_treino,
        num_workers=NUM_WORKERS, pin_memory=(DISPOSITIVO.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0))
    loader_val = DataLoader(
        ds_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=(DISPOSITIVO.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0))

    # Exemplos fixos (sem augmentação) para acompanhar a evolução visual
    n_exemplos = min(N_EXEMPLOS_VISUALIZACAO, len(ds_val))
    exemplos_visualizacao = [ds_val[i] for i in range(n_exemplos)]

    modelo = UNet(
        spatial_dims=2, in_channels=1, out_channels=1,
        channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(DISPOSITIVO)

    # DiceFocalLoss em vez de DiceCELoss: a Focal Loss reduz o peso de
    # pixels "fáceis" (a esmagadora maioria de fundo) e concentra o
    # gradiente nos pixels raros/difíceis (o nódulo) — mais adequada
    # para o desbalanceamento severo observado (nódulo é <0.1% dos
    # pixels de um slice 512x512). A DiceCE simples estava levando o
    # modelo a "aprender a proporção geral" em vez de localizar.
    funcao_perda = DiceFocalLoss(sigmoid=True, gamma=2.0)
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=LEARNING_RATE,
                                    weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        otimizador, mode="max", factor=0.5, patience=PACIENCIA_SCHEDULER)
    scaler = torch.amp.GradScaler(enabled=USAR_AMP)

    historico = {k: [] for k in [
        "epoca", "perda_treino", "dice_treino", "iou_treino", "recall_treino",
        "precisao_treino", "perda_val", "dice_val", "iou_val", "recall_val",
        "precisao_val", "lr"]}

    melhor_dice_val = -1.0
    epocas_sem_melhora = 0
    caminho_melhor_modelo = os.path.join(CAMINHO_SAIDA_MODELO, "melhor_modelo_unet.pth")

    print("\nIniciando treino (%d época(s) máx., paciência=%d)..." % (
        N_EPOCAS, PACIENCIA_EARLY_STOPPING))
    t0 = time.time()

    for epoca in range(1, N_EPOCAS + 1):
        t_epoca = time.time()
        m_treino = treinar_uma_epoca(modelo, loader_treino, otimizador, funcao_perda, scaler)
        m_val = validar(modelo, loader_val, funcao_perda)
        scheduler.step(m_val["dice"])
        lr_atual = otimizador.param_groups[0]["lr"]

        historico["epoca"].append(epoca)
        historico["perda_treino"].append(m_treino["perda"])
        historico["dice_treino"].append(m_treino["dice"])
        historico["iou_treino"].append(m_treino["iou"])
        historico["recall_treino"].append(m_treino["recall"])
        historico["precisao_treino"].append(m_treino["precisao"])
        historico["perda_val"].append(m_val["perda"])
        historico["dice_val"].append(m_val["dice"])
        historico["iou_val"].append(m_val["iou"])
        historico["recall_val"].append(m_val["recall"])
        historico["precisao_val"].append(m_val["precisao"])
        historico["lr"].append(lr_atual)

        marcador = ""
        if m_val["dice"] > melhor_dice_val + MIN_DELTA:
            melhor_dice_val = m_val["dice"]
            epocas_sem_melhora = 0
            torch.save(modelo.state_dict(), caminho_melhor_modelo)
            marcador = "  <- melhor até agora, salvo"
        else:
            epocas_sem_melhora += 1

        print("Época %2d/%d | treino: perda=%.4f dice=%.4f | "
              "val: perda=%.4f dice=%.4f iou=%.4f recall=%.4f precisao=%.4f | "
              "lr=%.2e | %.1fs%s" % (
                  epoca, N_EPOCAS, m_treino["perda"], m_treino["dice"],
                  m_val["perda"], m_val["dice"], m_val["iou"],
                  m_val["recall"], m_val["precisao"], lr_atual,
                  time.time() - t_epoca, marcador))
        print("        diagnóstico: %% pixels previstos como nódulo=%.4f%% "
              "(real=%.4f%%) — se previsto >> real, o modelo pode estar "
              "'colapsando' (marcando região demais)" % (
                  100 * m_val["fracao_prevista_positiva"],
                  100 * m_val["fracao_real_positiva"]))

        if epoca % VISUALIZAR_A_CADA_N_EPOCAS == 0:
            salvar_visualizacao_epoca(modelo, exemplos_visualizacao, epoca,
                                       CAMINHO_VISUALIZACOES)

        if epocas_sem_melhora >= PACIENCIA_EARLY_STOPPING:
            print("Early stopping: sem melhora (>%.3f) em %d épocas." % (
                MIN_DELTA, PACIENCIA_EARLY_STOPPING))
            break

    tempo_total = time.time() - t0
    print("\nTreino concluído em %.1f min. Melhor dice_val=%.4f" % (
        tempo_total / 60, melhor_dice_val))

    pd.DataFrame(historico).to_csv(
        os.path.join(CAMINHO_SAIDA_MODELO, "historico_treino.csv"), index=False)
    salvar_curvas(historico, os.path.join(CAMINHO_SAIDA_MODELO, "curvas_treino.png"))

    config = dict(
        AMBIENTE=AMBIENTE, DEBUG_MODE=DEBUG_MODE, BATCH_SIZE=BATCH_SIZE,
        N_EPOCAS_MAX=N_EPOCAS, PACIENCIA_EARLY_STOPPING=PACIENCIA_EARLY_STOPPING,
        MIN_DELTA=MIN_DELTA, LEARNING_RATE=LEARNING_RATE,
        WEIGHT_DECAY=WEIGHT_DECAY, SEED=SEED,
        n_slices_treino=len(df_treino), n_slices_val=len(df_val),
        melhor_dice_val=melhor_dice_val, tempo_treino_min=tempo_total / 60,
        dispositivo=str(DISPOSITIVO), mixed_precision=USAR_AMP,
    )
    with open(os.path.join(CAMINHO_SAIDA_MODELO, "config_treino.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("\nModelo salvo em: %s" % caminho_melhor_modelo)
    print("Visualizações salvas em: %s" % CAMINHO_VISUALIZACOES)
    if DEBUG_MODE:
        print("\n*** Pipeline validado com sucesso. Para o treino de "
              "verdade: mude AMBIENTE='colab' e DEBUG_MODE=False no Colab. ***")


if __name__ == "__main__":
    main()
