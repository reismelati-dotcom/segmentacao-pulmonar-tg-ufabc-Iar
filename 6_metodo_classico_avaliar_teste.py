# =============================================================================
# 6_metodo_classico_avaliar_teste.py — Avaliação final ("às cegas") no teste
# =============================================================================
#
# Aplica o método clássico (com os parâmetros calibrados pelo script 5) em
# TODOS os slices dos volumes COMPLETOS dos pacientes de teste — não apenas
# no subconjunto balanceado gerado pelo script 4.
#
# Isso é intencional: um sistema de detecção de nódulos precisa funcionar
# "às cegas" em um exame inteiro, com a proporção real de slices sem
# nódulo. Avaliar só no subconjunto balanceado infla artificialmente
# métricas como precisão e especificidade.
#
# Métricas calculadas por SLICE e agregadas por PACIENTE e no geral:
#   Dice, IoU, Precisão, Recall (Sensibilidade), Especificidade, F1,
#   tempo de processamento por slice.
#
# Saída: metricas_metodo_classico_teste.csv (1 linha por slice) +
#        resumo impresso no console (médias por paciente e geral).
#
# =============================================================================

import os
import json
import time
import numpy as np
import pandas as pd
import SimpleITK as sitk

from importlib.machinery import SourceFileLoader

# --- Configurações ---

CAMINHO_SAIDA = r"D:\TG\processadoV3"       # saída do script 1 (volumes completos)
CAMINHO_SLICES = r"D:\TG\slices_2dV3"         # onde está melhores_parametros.json
CAMINHO_SPLIT = os.path.join(CAMINHO_SAIDA, "split_pacientes.csv")
CAMINHO_PARAMS = os.path.join(CAMINHO_SLICES, "melhores_parametros.json")
CAMINHO_RESULTADOS = os.path.join(CAMINHO_SLICES, "metricas_metodo_classico_teste.csv")

# Reaproveita as funções de segmentação do script 5 (mesmo código,
# garantindo que calibração e avaliação usem exatamente o mesmo pipeline).
CAMINHO_SCRIPT_5 = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "5_metodo_classico_calibrar.py")
_mod5 = SourceFileLoader("metodo_classico", CAMINHO_SCRIPT_5).load_module()

HU_MIN, HU_MAX = _mod5.HU_MIN, _mod5.HU_MAX


def normalizar_hu(volume_hu):
    clip = np.clip(volume_hu, HU_MIN, HU_MAX)
    return ((clip - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float32)


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

    return {
        "dice": dice, "iou": iou, "precisao": precisao, "recall": recall,
        "especificidade": especificidade, "f1": f1,
        "tem_nodulo_gt": bool(gt.sum() > 0),
    }


def carregar_mascara_gt_teste(pasta_paciente, cluster_ids_validos, shape,
                               metodo_preferido="staple"):
    """
    Máscara-alvo (ground truth) do paciente: união dos consensos dos
    clusters VÁLIDOS. Mesma lógica usada no script 4 para os slices 2D,
    aqui aplicada ao volume inteiro do paciente de teste.
    """
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


def avaliar_paciente(paciente, cluster_ids_validos, params):
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
        pulmao = _mod5.segmentar_pulmao(volume_norm[z])
        pred = _mod5.segmentar_nodulos_classico(volume_norm[z], pulmao, **params)
        tempo_s = time.perf_counter() - t0

        metricas = calcular_metricas_slice(pred, mascara_gt[z])
        linhas.append({
            "paciente": paciente, "z": z, "tempo_s": tempo_s, **metricas,
        })

    return linhas


def main():
    with open(CAMINHO_PARAMS) as f:
        params = json.load(f)
    print("Parâmetros carregados do script 5:", params)

    split_df = pd.read_csv(CAMINHO_SPLIT)
    pacientes_teste = split_df[split_df["split"] == "test"]["paciente"].tolist()
    print("Pacientes de teste: %d" % len(pacientes_teste))

    nodulos_df = pd.read_csv(
        os.path.join(CAMINHO_SAIDA, "nodulos_validos.csv"))
    clusters_por_paciente = (
        nodulos_df.groupby("paciente")["cluster_id"].unique().apply(list).to_dict()
    )

    todas_linhas = []
    for paciente in pacientes_teste:
        cluster_ids = clusters_por_paciente.get(paciente, [])
        t0 = time.time()
        linhas = avaliar_paciente(paciente, cluster_ids, params)
        todas_linhas.extend(linhas)
        if linhas:
            df_p = pd.DataFrame(linhas)
            print("  %s: %d slices | Dice médio=%.3f | tempo total=%.1fs" % (
                paciente, len(df_p), df_p["dice"].mean(), time.time() - t0))

    df_resultado = pd.DataFrame(todas_linhas)
    df_resultado.to_csv(CAMINHO_RESULTADOS, index=False)

    print("\n" + "=" * 60)
    print("Resumo geral — TODOS os slices dos pacientes de teste")
    print("=" * 60)
    print("  Total de slices avaliados: %d" % len(df_resultado))
    print("  Slices com nódulo (GT): %d (%.1f%%)" % (
        df_resultado["tem_nodulo_gt"].sum(),
        100 * df_resultado["tem_nodulo_gt"].mean()))
    for col in ["dice", "iou", "precisao", "recall", "especificidade", "f1"]:
        print("  %-15s: média=%.4f" % (col, df_resultado[col].mean()))
    print("  %-15s: média=%.4f s/slice" % ("tempo", df_resultado["tempo_s"].mean()))

    # Métricas calculadas SÓ nos slices que de fato têm nódulo (mais
    # informativo para Dice/IoU, já que slices vazios inflam a média)
    df_pos = df_resultado[df_resultado["tem_nodulo_gt"]]
    if len(df_pos) > 0:
        print("\n  --- Apenas slices COM nódulo (GT), n=%d ---" % len(df_pos))
        for col in ["dice", "iou", "recall"]:
            print("  %-15s: média=%.4f" % (col, df_pos[col].mean()))

    print("\nResultados salvos em: %s" % CAMINHO_RESULTADOS)


if __name__ == "__main__":
    main()
