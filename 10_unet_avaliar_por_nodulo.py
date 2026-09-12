# =============================================================================
# 10_unet_avaliar_por_nodulo.py — Avaliação da U-Net por NÓDULO
# =============================================================================
#
# Espelha o script 7 (método clássico), mas usando a U-Net treinada com
# sliding window inference (mesma lógica do script 9). Mesma definição
# de "nódulo encontrado" / falso positivo / Dice-só-dos-encontrados.
#
# Saída: metricas_unet_por_nodulo_teste.csv + resumo no console.
#
# =============================================================================

import os
import time
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage
from importlib.machinery import SourceFileLoader

CAMINHO_SAIDA = r"D:\TG\processadoV3"
CAMINHO_MODELO = r"D:\TG\modelo_unetV3\melhor_modelo_unet.pth"
CAMINHO_RESULTADOS = r"D:\TG\modelo_unetV3\metricas_unet_por_nodulo_teste.csv"
CAMINHO_SPLIT = os.path.join(CAMINHO_SAIDA, "split_pacientes.csv")

# Reaproveita as funções do script 9 (mesmo modelo, mesma inferência),
# garantindo que os dois scripts usem exatamente o mesmo pipeline.
CAMINHO_SCRIPT_9 = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "9_unet_avaliar_teste.py")
_mod9 = SourceFileLoader("unet_avaliar", CAMINHO_SCRIPT_9).load_module()

# Conectividade 3D para agrupar voxels previstos em candidatos (26-conectividade)
ESTRUTURA_3D = ndimage.generate_binary_structure(3, 3)


def dice(a, b):
    a = a.astype(bool); b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    soma = a.sum() + b.sum()
    return 1.0 if soma == 0 else 2.0 * inter / soma


def carregar_mascara_nodulo(pasta_paciente, cluster_id, shape,
                             metodo_preferido="staple"):
    ordem = [metodo_preferido,
             "votacao" if metodo_preferido == "staple" else "staple"]
    for metodo in ordem:
        cam = os.path.join(
            pasta_paciente, "mascara_consenso_nodulo%02d_%s.nii.gz" % (
                cluster_id, metodo))
        if os.path.exists(cam):
            return sitk.GetArrayFromImage(sitk.ReadImage(cam)).astype(bool)
    return np.zeros(shape, dtype=bool)


def avaliar_paciente(modelo, paciente, cluster_ids_validos):
    pasta_paciente = os.path.join(CAMINHO_SAIDA, paciente)
    cam_volume = os.path.join(pasta_paciente, "volume_tc.nii.gz")
    if not os.path.exists(cam_volume):
        print("  [ERRO] volume não encontrado para %s — pulando" % paciente)
        return [], 0

    volume_hu = sitk.GetArrayFromImage(sitk.ReadImage(cam_volume))
    volume_norm = _mod9.normalizar_hu(volume_hu)
    shape = volume_hu.shape

    # Roda a U-Net (sliding window) em TODO slice do volume
    pred_3d = np.zeros(shape, dtype=bool)
    for z in range(shape[0]):
        pred_3d[z] = _mod9.prever_slice(modelo, volume_norm[z]).astype(bool)

    rotulado_pred, n_candidatos = ndimage.label(pred_3d, structure=ESTRUTURA_3D)

    linhas = []
    rotulos_correspondidos = set()

    for cluster_id in cluster_ids_validos:
        gt_nodulo = carregar_mascara_nodulo(pasta_paciente, cluster_id, shape)
        if gt_nodulo.sum() == 0:
            continue

        rotulos_sobrepostos = set(
            np.unique(rotulado_pred[gt_nodulo & (rotulado_pred > 0)]))
        rotulos_sobrepostos.discard(0)

        encontrado = len(rotulos_sobrepostos) > 0
        if encontrado:
            rotulos_correspondidos |= rotulos_sobrepostos
            pred_correspondente = np.isin(rotulado_pred, list(rotulos_sobrepostos))
            dice_nodulo = dice(pred_correspondente, gt_nodulo)
        else:
            dice_nodulo = np.nan

        linhas.append({
            "paciente": paciente, "cluster_id": cluster_id,
            "encontrado": encontrado, "dice": dice_nodulo,
            "volume_gt_voxels": int(gt_nodulo.sum()),
        })

    todos_rotulos = set(range(1, n_candidatos + 1))
    falsos_positivos = len(todos_rotulos - rotulos_correspondidos)

    return linhas, falsos_positivos


def main():
    print("Dispositivo: %s" % _mod9.DISPOSITIVO)
    print("Carregando modelo de: %s" % CAMINHO_MODELO)
    modelo = _mod9.carregar_modelo()

    split_df = pd.read_csv(CAMINHO_SPLIT)
    pacientes_teste = split_df[split_df["split"] == "test"]["paciente"].tolist()
    print("Pacientes de teste: %d" % len(pacientes_teste))

    nodulos_df = pd.read_csv(os.path.join(CAMINHO_SAIDA, "nodulos_validos.csv"))
    clusters_por_paciente = (
        nodulos_df.groupby("paciente")["cluster_id"].unique().apply(list).to_dict()
    )

    todas_linhas = []
    total_fp = 0
    for paciente in pacientes_teste:
        cluster_ids = clusters_por_paciente.get(paciente, [])
        t0 = time.time()
        linhas, fp = avaliar_paciente(modelo, paciente, cluster_ids)
        todas_linhas.extend(linhas)
        total_fp += fp
        n_encontrados = sum(1 for l in linhas if l["encontrado"])
        print("  %s: %d nódulo(s) real(is) | %d encontrado(s) | "
              "%d falso(s) positivo(s) | %.1fs" % (
                  paciente, len(linhas), n_encontrados, fp, time.time() - t0))

    df = pd.DataFrame(todas_linhas)
    os.makedirs(os.path.dirname(CAMINHO_RESULTADOS), exist_ok=True)
    df.to_csv(CAMINHO_RESULTADOS, index=False)

    n_total = len(df)
    n_encontrados = int(df["encontrado"].sum())
    n_perdidos = n_total - n_encontrados
    dice_medio = df.loc[df["encontrado"], "dice"].mean()

    print("\n" + "=" * 60)
    print("Resumo — avaliação por NÓDULO (U-Net, todos os pacientes de teste)")
    print("=" * 60)
    print("  Nódulos reais no teste (válidos):  %d" % n_total)
    print("  Nódulos ENCONTRADOS:               %d (%.1f%%)" % (
        n_encontrados, 100 * n_encontrados / n_total if n_total else 0))
    print("  Nódulos PERDIDOS (não detectados): %d (%.1f%%)" % (
        n_perdidos, 100 * n_perdidos / n_total if n_total else 0))
    print("  Falsos positivos (candidatos sem nódulo real correspondente): %d" %
          total_fp)
    print("  Dice médio — SÓ dos nódulos encontrados (sem contar FP): %.4f" %
          dice_medio)
    print("\nResultados salvos em: %s" % CAMINHO_RESULTADOS)


if __name__ == "__main__":
    main()
