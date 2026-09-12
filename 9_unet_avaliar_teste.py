# =============================================================================
# 9_unet_avaliar_teste.py — Avaliação da U-Net nos volumes COMPLETOS de teste
# =============================================================================
#
# Espelha o script 6 (método clássico), mas usando a U-Net treinada com
# SLIDING WINDOW INFERENCE — o modelo foi treinado em patches 128x128
# centrados no nódulo (usando a máscara real), mas na avaliação ele NÃO
# tem acesso a essa informação: a varredura percorre o slice inteiro
# (512x512) com janelas sobrepostas, exatamente como aconteceria em uso
# real. Isso garante uma comparação justa com o método clássico (scripts
# 6 e 7), que também foi avaliado "às cegas" nos volumes completos.
#
# Métricas calculadas por SLICE (pixel a pixel), agregadas por paciente
# e no geral — mesmo formato do script 6.
#
# Saída: metricas_unet_teste.csv + resumo no console.
#
# =============================================================================

import os
import json
import time
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from scipy import ndimage

from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference

# --- Configurações ---

CAMINHO_SAIDA = r"D:\TG\processadoV3"          # volumes + máscaras de consenso
CAMINHO_MODELO = r"D:\TG\modelo_unetV3\melhor_modelo_unet.pth"
CAMINHO_RESULTADOS = r"D:\TG\modelo_unetV3\metricas_unet_teste.csv"
CAMINHO_PARAMS_POS = r"D:\TG\modelo_unetV3\pos_processamento.json"

CAMINHO_SPLIT = os.path.join(CAMINHO_SAIDA, "split_pacientes.csv")

# Precisam bater exatamente com o script 8 (treino), senão os pesos não
# carregam corretamente ou a inferência fica incoerente com o treino.
HU_MIN, HU_MAX = -1000, 400
TAMANHO_PATCH = 128
CANAIS_UNET = (16, 32, 64, 128, 256)
STRIDES_UNET = (2, 2, 2, 2)
NUM_RES_UNITS = 2

# Parâmetros específicos da inferência por janela deslizante
SOBREPOSICAO_JANELAS = 0.5   # fração de sobreposição entre janelas vizinhas
SW_BATCH_SIZE = 16           # quantas janelas processar de uma vez (GPU/CPU)

# --- Pós-processamento (threshold + componentes conexos + área mínima) ---
# Mesma ideia usada no método clássico (script 5): remove candidatos
# pequenos demais para ser um nódulo real. Valores default (sem
# calibração) — o script 11 calibra esses dois valores em VALIDAÇÃO
# (nunca no teste) e sobrescreve automaticamente se o JSON existir.
LIMIAR_PREDICAO = 0.5
AREA_MIN_PX_POS = 0   # 0 = desligado (compatível com versões antigas)

if os.path.exists(CAMINHO_PARAMS_POS):
    with open(CAMINHO_PARAMS_POS) as _f:
        _params_pos = json.load(_f)
    LIMIAR_PREDICAO = _params_pos.get("limiar_predicao", LIMIAR_PREDICAO)
    AREA_MIN_PX_POS = _params_pos.get("area_min_px", AREA_MIN_PX_POS)
    print("Pós-processamento calibrado carregado de %s: limiar=%.2f, "
          "area_min_px=%d" % (CAMINHO_PARAMS_POS, LIMIAR_PREDICAO, AREA_MIN_PX_POS))

DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalizar_hu(volume_hu):
    clip = np.clip(volume_hu, HU_MIN, HU_MAX)
    return ((clip - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float32)


def carregar_modelo():
    modelo = UNet(
        spatial_dims=2, in_channels=1, out_channels=1,
        channels=CANAIS_UNET, strides=STRIDES_UNET, num_res_units=NUM_RES_UNITS,
    ).to(DISPOSITIVO)
    modelo.load_state_dict(torch.load(CAMINHO_MODELO, map_location=DISPOSITIVO))
    modelo.eval()
    return modelo


def prever_slice_probs(modelo, img_norm):
    """
    Roda sliding window inference num slice 2D (H,W) já normalizado, e
    retorna o mapa de PROBABILIDADE bruto (sem threshold) — usado tanto
    pela avaliação final quanto pela calibração (script 11), que precisa
    do mapa bruto para testar vários thresholds sem repetir a inferência.
    """
    tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(DISPOSITIVO)
    with torch.no_grad():
        logits = sliding_window_inference(
            inputs=tensor, roi_size=(TAMANHO_PATCH, TAMANHO_PATCH),
            sw_batch_size=SW_BATCH_SIZE, predictor=modelo,
            overlap=SOBREPOSICAO_JANELAS, mode="gaussian")
        probs = torch.sigmoid(logits)
    return probs.cpu().numpy()[0, 0].astype(np.float32)


def pos_processar(mapa_prob, limiar=None, area_min_px=None):
    """
    Threshold + remoção de componentes conexos menores que area_min_px —
    mesma ideia do método clássico (script 5): nódulo real tem um
    tamanho mínimo plausível, candidatos menores que isso tendem a ser
    ruído da rede. area_min_px=0 desliga o filtro (comportamento antigo).
    """
    limiar = LIMIAR_PREDICAO if limiar is None else limiar
    area_min_px = AREA_MIN_PX_POS if area_min_px is None else area_min_px

    pred = (mapa_prob > limiar).astype(np.uint8)
    if area_min_px > 0 and pred.sum() > 0:
        rotulado, n = ndimage.label(pred)
        if n > 0:
            tamanhos = ndimage.sum(pred, rotulado, range(1, n + 1))
            manter = np.zeros(n + 1, dtype=bool)
            manter[1:] = tamanhos >= area_min_px
            pred = manter[rotulado].astype(np.uint8)
    return pred


def prever_slice(modelo, img_norm):
    """
    Roda sliding window inference num slice 2D (H,W) já normalizado.
    O modelo NÃO recebe nenhuma informação de onde o nódulo está — a
    varredura cobre a imagem inteira com janelas de TAMANHO_PATCH,
    sobrepostas, misturando as previsões nas regiões de sobreposição
    (mode="gaussian" pondera mais o centro de cada janela). Depois,
    aplica o pós-processamento (threshold calibrado + filtro de área).
    """
    mapa_prob = prever_slice_probs(modelo, img_norm)
    return pos_processar(mapa_prob)


def calcular_metricas_slice(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()

    soma = pred.sum() + gt.sum()
    dice = 1.0 if soma == 0 else 2.0 * tp / soma
    uniao = tp + fp + fn
    iou = 1.0 if uniao == 0 else tp / uniao
    precisao = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if gt.sum() == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    especificidade = tn / (tn + fp) if (tn + fp) > 0 else 1.0
    f1 = (2 * precisao * recall / (precisao + recall)
          if (precisao + recall) > 0 else 0.0)

    return {"dice": dice, "iou": iou, "precisao": precisao, "recall": recall,
            "especificidade": especificidade, "f1": f1,
            "tem_nodulo_gt": bool(gt.sum() > 0)}


def carregar_mascara_gt_teste(pasta_paciente, cluster_ids_validos, shape,
                               metodo_preferido="staple"):
    uniao = np.zeros(shape, dtype=bool)
    for cid in cluster_ids_validos:
        ordem = [metodo_preferido,
                 "votacao" if metodo_preferido == "staple" else "staple"]
        for metodo in ordem:
            cam = os.path.join(
                pasta_paciente,
                "mascara_consenso_nodulo%02d_%s.nii.gz" % (cid, metodo))
            if os.path.exists(cam):
                m = sitk.GetArrayFromImage(sitk.ReadImage(cam)).astype(bool)
                uniao |= m
                break
    return uniao


def avaliar_paciente(modelo, paciente, cluster_ids_validos):
    pasta_paciente = os.path.join(CAMINHO_SAIDA, paciente)
    cam_volume = os.path.join(pasta_paciente, "volume_tc.nii.gz")
    if not os.path.exists(cam_volume):
        print("  [ERRO] volume não encontrado para %s — pulando" % paciente)
        return []

    volume_hu = sitk.GetArrayFromImage(sitk.ReadImage(cam_volume))
    volume_norm = normalizar_hu(volume_hu)
    shape = volume_hu.shape

    mascara_gt = carregar_mascara_gt_teste(pasta_paciente, cluster_ids_validos, shape)

    linhas = []
    for z in range(shape[0]):
        t0 = time.perf_counter()
        pred = prever_slice(modelo, volume_norm[z])
        tempo_s = time.perf_counter() - t0

        metricas = calcular_metricas_slice(pred, mascara_gt[z])
        linhas.append({"paciente": paciente, "z": z, "tempo_s": tempo_s, **metricas})

    return linhas


def main():
    print("Dispositivo: %s" % DISPOSITIVO)
    print("Carregando modelo de: %s" % CAMINHO_MODELO)
    modelo = carregar_modelo()

    split_df = pd.read_csv(CAMINHO_SPLIT)
    pacientes_teste = split_df[split_df["split"] == "test"]["paciente"].tolist()
    print("Pacientes de teste: %d" % len(pacientes_teste))

    nodulos_df = pd.read_csv(os.path.join(CAMINHO_SAIDA, "nodulos_validos.csv"))
    clusters_por_paciente = (
        nodulos_df.groupby("paciente")["cluster_id"].unique().apply(list).to_dict()
    )

    todas_linhas = []
    for paciente in pacientes_teste:
        cluster_ids = clusters_por_paciente.get(paciente, [])
        t0 = time.time()
        linhas = avaliar_paciente(modelo, paciente, cluster_ids)
        todas_linhas.extend(linhas)
        if linhas:
            df_p = pd.DataFrame(linhas)
            print("  %s: %d slices | Dice médio=%.3f | tempo total=%.1fs" % (
                paciente, len(df_p), df_p["dice"].mean(), time.time() - t0))

    df_resultado = pd.DataFrame(todas_linhas)
    os.makedirs(os.path.dirname(CAMINHO_RESULTADOS), exist_ok=True)
    df_resultado.to_csv(CAMINHO_RESULTADOS, index=False)

    print("\n" + "=" * 60)
    print("Resumo geral — TODOS os slices dos pacientes de teste (U-Net)")
    print("=" * 60)
    print("  Total de slices avaliados: %d" % len(df_resultado))
    print("  Slices com nódulo (GT): %d (%.1f%%)" % (
        df_resultado["tem_nodulo_gt"].sum(),
        100 * df_resultado["tem_nodulo_gt"].mean()))
    for col in ["dice", "iou", "precisao", "recall", "especificidade", "f1"]:
        print("  %-15s: média=%.4f" % (col, df_resultado[col].mean()))
    print("  %-15s: média=%.4f s/slice" % ("tempo", df_resultado["tempo_s"].mean()))

    df_pos = df_resultado[df_resultado["tem_nodulo_gt"]]
    if len(df_pos) > 0:
        print("\n  --- Apenas slices COM nódulo (GT), n=%d ---" % len(df_pos))
        for col in ["dice", "iou", "recall"]:
            print("  %-15s: média=%.4f" % (col, df_pos[col].mean()))

    print("\nResultados salvos em: %s" % CAMINHO_RESULTADOS)


if __name__ == "__main__":
    main()
