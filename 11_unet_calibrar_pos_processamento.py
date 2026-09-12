# =============================================================================
# 11_unet_calibrar_pos_processamento.py — Calibra threshold + filtro de área
# =============================================================================
#
# A U-Net foi treinada/avaliada até aqui com threshold fixo em 0.5 e sem
# nenhum pós-processamento — e o script 10 mostrou uma quantidade grande
# de falsos positivos. Este script calibra dois parâmetros nos volumes
# COMPLETOS dos pacientes de VALIDAÇÃO (nunca no teste):
#
#   1) LIMIAR_PREDICAO — threshold aplicado à probabilidade (sigmoid) da
#      rede. Testa uma grade de valores em vez de usar 0.5 fixo.
#   2) AREA_MIN_PX — filtro de componentes conexos por área mínima,
#      igual ao usado no método clássico (script 5) — remove candidatos
#      pequenos demais para ser um nódulo plausível.
#
# Mesma filosofia de justiça metodológica usada no método clássico: os
# dois métodos usam pós-processamento por componentes conexos + área
# mínima, cada um calibrado nos seus próprios dados de treino/validação.
#
# A inferência por sliding window é o passo caro — por isso é rodada
# UMA VEZ por slice (mapa de probabilidade bruto), e a grade de
# threshold/área é testada de forma barata em cima desse mapa já
# calculado (só operações numpy, sem repetir a rede neural).
#
# Métrica de seleção: Dice médio nos slices que têm nódulo (mesma lógica
# honesta usada no script 5 — não conta os negativos "de graça"), com
# especificidade mínima como filtro de sanidade.
#
# Saída: pos_processamento.json (lido automaticamente pelos scripts 9 e 10)
#
# =============================================================================

import os
import json
import time
import itertools
import numpy as np
import pandas as pd
import SimpleITK as sitk
from importlib.machinery import SourceFileLoader

CAMINHO_SAIDA = r"D:\TG\processadoV3"
CAMINHO_MODELO = r"D:\TG\modelo_unetV3\melhor_modelo_unet.pth"
CAMINHO_SAIDA_PARAMS = r"D:\TG\modelo_unetV3\pos_processamento.json"
CAMINHO_SPLIT = os.path.join(CAMINHO_SAIDA, "split_pacientes.csv")

# Reaproveita as funções do script 9 (mesmo modelo, mesma inferência).
CAMINHO_SCRIPT_9 = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "9_unet_avaliar_teste.py")
_mod9 = SourceFileLoader("unet_avaliar", CAMINHO_SCRIPT_9).load_module()

# --- Grade de busca ---
GRADE_LIMIAR = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
GRADE_AREA_MIN_PX = [0, 10, 20, 40]
# Total: 6*4 = 24 combinações

ESPECIFICIDADE_MINIMA = 0.90


def carregar_mascara_gt_paciente(pasta_paciente, cluster_ids_validos, shape):
    return _mod9.carregar_mascara_gt_teste(pasta_paciente, cluster_ids_validos, shape)


def coletar_mapas_probabilidade(modelo, pacientes, clusters_por_paciente):
    """
    Roda a inferência (sliding window) UMA VEZ por slice de cada paciente
    de validação, retornando os mapas de probabilidade e o ground truth —
    para depois testar a grade de threshold/área sem repetir a rede.

    Retorna uma lista de (mapa_prob, mascara_gt) por PACIENTE (não por
    slice, para não estourar RAM guardando tudo junto) — o chamador
    processa e descarta paciente a paciente.
    """
    for paciente in pacientes:
        pasta_paciente = os.path.join(CAMINHO_SAIDA, paciente)
        cam_volume = os.path.join(pasta_paciente, "volume_tc.nii.gz")
        if not os.path.exists(cam_volume):
            print("  [ERRO] volume não encontrado para %s — pulando" % paciente)
            continue

        volume_hu = sitk.GetArrayFromImage(sitk.ReadImage(cam_volume))
        volume_norm = _mod9.normalizar_hu(volume_hu)
        shape = volume_hu.shape

        cluster_ids = clusters_por_paciente.get(paciente, [])
        mascara_gt = carregar_mascara_gt_paciente(pasta_paciente, cluster_ids, shape)

        t0 = time.time()
        mapas_prob = np.stack([
            _mod9.prever_slice_probs(modelo, volume_norm[z])
            for z in range(shape[0])
        ])
        print("  %s: %d slices inferidos em %.1fs" % (
            paciente, shape[0], time.time() - t0))

        yield paciente, mapas_prob, mascara_gt


def main():
    print("Dispositivo: %s" % _mod9.DISPOSITIVO)
    print("Carregando modelo de: %s" % CAMINHO_MODELO)
    modelo = _mod9.carregar_modelo()

    split_df = pd.read_csv(CAMINHO_SPLIT)
    pacientes_val = split_df[split_df["split"] == "val"]["paciente"].tolist()
    print("Pacientes de VALIDAÇÃO: %d" % len(pacientes_val))

    nodulos_df = pd.read_csv(os.path.join(CAMINHO_SAIDA, "nodulos_validos.csv"))
    clusters_por_paciente = (
        nodulos_df.groupby("paciente")["cluster_id"].unique().apply(list).to_dict()
    )

    combinacoes = list(itertools.product(GRADE_LIMIAR, GRADE_AREA_MIN_PX))
    print("Testando %d combinação(ões) de (limiar, área mínima)..." % len(combinacoes))

    # Acumuladores GLOBAIS (soma de todos os pacientes de validação) por combinação
    acumulado = {c: [0, 0, 0, 0] for c in combinacoes}  # tp, fp, fn, tn

    # Só usamos slices POSITIVOS (com nódulo) para o Dice de seleção —
    # mas precisamos separar, dentro do acumulado, os totais vindos de
    # slices positivos dos vindos de slices negativos (para a
    # especificidade). Fazemos isso separando dois acumuladores.
    acumulado_pos = {c: [0, 0, 0, 0] for c in combinacoes}   # só slices com nódulo
    acumulado_neg = {c: [0, 0, 0, 0] for c in combinacoes}   # só slices sem nódulo

    for paciente, mapas_prob, mascara_gt in coletar_mapas_probabilidade(
            modelo, pacientes_val, clusters_por_paciente):
        for limiar, area_min in combinacoes:
            for z in range(mapas_prob.shape[0]):
                pred = _mod9.pos_processar(mapas_prob[z], limiar, area_min)
                gt = mascara_gt[z]
                tp = np.logical_and(pred, gt).sum()
                fp = np.logical_and(pred, np.logical_not(gt)).sum()
                fn = np.logical_and(np.logical_not(pred), gt).sum()
                tn = np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum()

                alvo = acumulado_pos if gt.sum() > 0 else acumulado_neg
                alvo[(limiar, area_min)][0] += tp
                alvo[(limiar, area_min)][1] += fp
                alvo[(limiar, area_min)][2] += fn
                alvo[(limiar, area_min)][3] += tn

    print("\nCalculando Dice/especificidade por combinação...")
    resultados = []
    for (limiar, area_min) in combinacoes:
        tp_p, fp_p, fn_p, _ = acumulado_pos[(limiar, area_min)]
        _, fp_n, _, tn_n = acumulado_neg[(limiar, area_min)]

        dice_pos = (2 * tp_p / (2 * tp_p + fp_p + fn_p)
                    if (2 * tp_p + fp_p + fn_p) > 0 else 1.0)
        especificidade = tn_n / (tn_n + fp_n) if (tn_n + fp_n) > 0 else 1.0

        resultados.append({
            "limiar_predicao": limiar, "area_min_px": area_min,
            "dice_positivos": dice_pos, "especificidade": especificidade,
        })

    df = pd.DataFrame(resultados)
    aptos = df[df["especificidade"] >= ESPECIFICIDADE_MINIMA]
    if len(aptos) == 0:
        print("  [AVISO] Nenhuma combinação atingiu especificidade >= %.2f — "
              "usando todas mesmo assim." % ESPECIFICIDADE_MINIMA)
        aptos = df
    aptos = aptos.sort_values("dice_positivos", ascending=False)

    melhor = aptos.iloc[0].to_dict()

    print("\n" + "=" * 60)
    print("Top 5 combinações (por Dice em slices positivos):")
    print("=" * 60)
    print(aptos.head(5).to_string(index=False))

    print("\nMelhor combinação escolhida:")
    print("  limiar_predicao = %.2f" % melhor["limiar_predicao"])
    print("  area_min_px     = %d" % melhor["area_min_px"])
    print("  dice_positivos  = %.4f" % melhor["dice_positivos"])
    print("  especificidade  = %.4f" % melhor["especificidade"])

    params_finais = {
        "limiar_predicao": float(melhor["limiar_predicao"]),
        "area_min_px": int(melhor["area_min_px"]),
    }
    os.makedirs(os.path.dirname(CAMINHO_SAIDA_PARAMS), exist_ok=True)
    with open(CAMINHO_SAIDA_PARAMS, "w") as f:
        json.dump(params_finais, f, indent=2)
    print("\nParâmetros salvos em: %s" % CAMINHO_SAIDA_PARAMS)
    print("(o teste NÃO foi usado nesta calibração — reservado para os "
          "scripts 9 e 10, que já vão carregar esse JSON automaticamente)")


if __name__ == "__main__":
    main()
