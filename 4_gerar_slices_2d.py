# =============================================================================
# 4_gerar_slices_2d.py — Pré-processamento e geração de slices 2D
# =============================================================================
#
# Lê as saídas dos scripts 1 e 3 (volumes, máscaras de consenso, split) e
# gera, para cada paciente, os slices axiais 2D prontos para treino/avaliação:
#
#   1) Janela de Hounsfield Units (clipping) + normalização para [0, 1]
#   2) Segmentação pulmonar simples (thresholding + morfologia), para focar
#      a análise no parênquima e reduzir falsos positivos
#   3) Slices POSITIVOS: onde há nódulo VÁLIDO (consenso de clusters com
#      >= LIMIAR_ANOTACOES anotações, filtrado no script 3)
#   4) Slices NEGATIVOS: amostrados aleatoriamente de regiões pulmonares
#      SEM nenhuma anotação (válida OU inválida) — evita rotular como
#      "negativo" um slice que na verdade tem um nódulo ambíguo descartado
#
# Saída (em CAMINHO_SLICES/<split>/):
#   imagens/<paciente>_z###.npy   — slice CT pré-processado, float32 [0,1]
#   mascaras/<paciente>_z###.npy  — máscara binária do nódulo (uint8)
#   manifesto.csv                 — 1 linha por slice gerado, com metadados
#
# Observação importante: a segmentação pulmonar aqui é uma heurística
# simples (threshold + maior(es) componente(s) conexo(s)), suficiente
# para o pipeline inicial. Vale revisar visualmente numa amostra antes de
# rodar no dataset completo — casos com patologia pulmonar extensa ou
# pouca área de pulmão no slice podem exigir ajuste dos parâmetros.
#
# =============================================================================

import os
import re
import glob
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage

# --- Configurações ---

CAMINHO_SAIDA = r"D:\TG\processadoV3"          # saída do script 1 (+ CSVs do script 3)
CAMINHO_SLICES = r"D:\TG\slices_2dV3"            # onde os slices 2D serão salvos

CAMINHO_SPLIT = os.path.join(CAMINHO_SAIDA, "split_pacientes.csv")
CAMINHO_NODULOS_VALIDOS = os.path.join(CAMINHO_SAIDA, "nodulos_validos.csv")

# Janela de HU para pulmão/nódulo (padrão da literatura)
HU_MIN, HU_MAX = -1000, 400

# Método de consenso preferido para a máscara-alvo (cai para o outro se
# o arquivo não existir para aquele nódulo)
METODO_CONSENSO_PREFERIDO = "staple"   # "staple" ou "votacao"

# Segmentação pulmonar (heurística thresholding + componentes conexos)
APLICAR_MASCARA_PULMAO = True
LIMIAR_HU_PULMAO = -320      # abaixo disso é considerado "ar" (pulmão/fundo)
AREA_MINIMA_PULMAO_PX = 800  # área mínima (em pixels) p/ considerar slice válido p/ negativos

# Slices negativos (sem nódulo) amostrados por paciente, para dar ao
# treino exemplos de background
# Slices negativos (sem nódulo) amostrados por paciente, para dar ao
# treino exemplos de background.
#
# Em vez de um número FIXO (que gerava datasets desbalanceados — pacientes
# com muitos nódulos grandes acabavam dominados por positivos), o número
# de negativos é PROPORCIONAL ao número de positivos daquele paciente:
#   n_negativos = FATOR_NEGATIVOS_POR_POSITIVO * n_positivos_do_paciente
#
# FATOR_NEGATIVOS_POR_POSITIVO = 1.0 -> dataset ~50/50
# FATOR_NEGATIVOS_POR_POSITIVO = 1.5 -> um pouco mais de negativos que positivos
FATOR_NEGATIVOS_POR_POSITIVO = 1.5

# Piso mínimo de negativos por paciente (garante pelo menos alguns
# exemplos de background mesmo em pacientes com poucos positivos).
MIN_SLICES_NEGATIVOS_POR_PACIENTE = 3
SEED = 42

rng = np.random.default_rng(SEED)


# --- Pré-processamento ---

def normalizar_hu(volume_hu):
    clip = np.clip(volume_hu, HU_MIN, HU_MAX)
    return ((clip - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float32)


def segmentar_pulmao_slice(slice_hu):
    """
    Segmentação pulmonar simples por slice 2D:
      1) threshold (ar/pulmão vs. tecido)
      2) remove componentes conectados à borda da imagem (ar externo)
      3) preenche buracos (nódulos/vasos aparecem como "buracos" no pulmão)
      4) mantém só os 2 maiores componentes (pulmão E/D)
    Heurística — não substitui uma segmentação pulmonar validada.
    """
    binario = slice_hu < LIMIAR_HU_PULMAO

    rotulado, n = ndimage.label(binario)
    if n == 0:
        return np.zeros_like(binario, dtype=np.uint8)

    # remove componentes que tocam a borda (ar externo ao paciente)
    rotulos_borda = set(rotulado[0, :]) | set(rotulado[-1, :]) | \
                    set(rotulado[:, 0]) | set(rotulado[:, -1])
    rotulos_borda.discard(0)
    for r in rotulos_borda:
        binario[rotulado == r] = False

    binario = ndimage.binary_fill_holes(binario)

    rotulado, n = ndimage.label(binario)
    if n == 0:
        return np.zeros_like(binario, dtype=np.uint8)

    tamanhos = ndimage.sum(binario, rotulado, range(1, n + 1))
    maiores = np.argsort(tamanhos)[::-1][:2] + 1  # até 2 maiores componentes
    mascara = np.isin(rotulado, maiores)

    return mascara.astype(np.uint8)


# --- Localização dos arquivos gerados pelo script 1 ---

def carregar_mascara_uniao_qualquer_anotacao(pasta_paciente, shape):
    """
    União de TODAS as máscaras individuais (válidas ou não) do paciente.
    Usada para excluir da amostragem de negativos qualquer slice que
    tenha alguma anotação, mesmo que tenha sido filtrada por baixo
    consenso — evita rotular como "negativo" um nódulo ambíguo descartado.
    """
    uniao = np.zeros(shape, dtype=bool)
    arquivos = glob.glob(os.path.join(pasta_paciente, "mascara_nodulo*_rad*_ann*.nii.gz"))
    for f in arquivos:
        m = sitk.GetArrayFromImage(sitk.ReadImage(f)).astype(bool)
        if m.shape == shape:
            uniao |= m
    return uniao


def carregar_mascara_consenso_validos(pasta_paciente, cluster_ids_validos, shape):
    """
    União das máscaras de CONSENSO dos clusters válidos do paciente —
    é a máscara-alvo (ground truth) usada como supervisão positiva.
    """
    uniao = np.zeros(shape, dtype=bool)
    encontrados, faltando = [], []

    for cid in cluster_ids_validos:
        ordem = [METODO_CONSENSO_PREFERIDO,
                 "votacao" if METODO_CONSENSO_PREFERIDO == "staple" else "staple"]
        achou = False
        for metodo in ordem:
            cam = os.path.join(
                pasta_paciente, "mascara_consenso_nodulo%02d_%s.nii.gz" % (cid, metodo))
            if os.path.exists(cam):
                m = sitk.GetArrayFromImage(sitk.ReadImage(cam)).astype(bool)
                uniao |= m
                encontrados.append((cid, metodo))
                achou = True
                break
        if not achou:
            faltando.append(cid)

    return uniao, encontrados, faltando


# --- Geração dos slices ---

def gerar_slices_paciente(paciente, split, cluster_ids_validos, pasta_slices_split):
    pasta_paciente = os.path.join(CAMINHO_SAIDA, paciente)
    cam_volume = os.path.join(pasta_paciente, "volume_tc.nii.gz")
    if not os.path.exists(cam_volume):
        print("  [ERRO] volume não encontrado para %s — pulando" % paciente)
        return []

    volume_hu = sitk.GetArrayFromImage(sitk.ReadImage(cam_volume))
    shape = volume_hu.shape
    volume_norm = normalizar_hu(volume_hu)

    mascara_positiva, encontrados, faltando = carregar_mascara_consenso_validos(
        pasta_paciente, cluster_ids_validos, shape)
    if faltando:
        print("  [AVISO] %s: consenso não encontrado p/ cluster(s) %s "
              "(verifique se o script 1 foi rodado com METODO_CONSENSO "
              "compatível)" % (paciente, faltando))

    mascara_qualquer_anotacao = carregar_mascara_uniao_qualquer_anotacao(
        pasta_paciente, shape)

    if APLICAR_MASCARA_PULMAO:
        mascara_pulmao = np.stack(
            [segmentar_pulmao_slice(volume_hu[z]) for z in range(shape[0])])
    else:
        mascara_pulmao = np.ones(shape, dtype=np.uint8)

    linhas_manifesto = []
    pasta_img = os.path.join(pasta_slices_split, "imagens")
    pasta_msk = os.path.join(pasta_slices_split, "mascaras")
    os.makedirs(pasta_img, exist_ok=True)
    os.makedirs(pasta_msk, exist_ok=True)

    def salvar_slice(z, tem_nodulo):
        nome = "%s_z%03d" % (paciente, z)
        cam_img = os.path.join(pasta_img, nome + "_img.npy")
        cam_msk = os.path.join(pasta_msk, nome + "_mask.npy")
        np.save(cam_img, volume_norm[z])
        np.save(cam_msk, mascara_positiva[z].astype(np.uint8))
        linhas_manifesto.append({
            "paciente": paciente, "z": z, "split": split,
            "tem_nodulo": int(tem_nodulo),
            "arquivo_imagem": cam_img, "arquivo_mascara": cam_msk,
        })

    # Slices positivos: onde há máscara de consenso válida
    slices_positivos = np.where(mascara_positiva.sum(axis=(1, 2)) > 0)[0]
    for z in slices_positivos:
        salvar_slice(z, tem_nodulo=True)

    # Slices negativos: sem NENHUMA anotação (válida ou não), com área
    # pulmonar mínima no slice
    area_pulmao_por_slice = mascara_pulmao.sum(axis=(1, 2))
    sem_anotacao = mascara_qualquer_anotacao.sum(axis=(1, 2)) == 0
    candidatos_negativos = np.where(
        sem_anotacao & (area_pulmao_por_slice >= AREA_MINIMA_PULMAO_PX))[0]

    if len(candidatos_negativos) > 0:
        n_positivos = len(slices_positivos)
        n_alvo = max(
            MIN_SLICES_NEGATIVOS_POR_PACIENTE,
            round(FATOR_NEGATIVOS_POR_POSITIVO * n_positivos),
        )
        n_amostrar = min(n_alvo, len(candidatos_negativos))
        if n_amostrar < n_alvo:
            print("  [AVISO] %s: só %d slice(s) negativo(s) elegível(is) "
                  "disponível(is) (gostaria de %d)" % (
                      paciente, n_amostrar, n_alvo))
        escolhidos = rng.choice(candidatos_negativos, size=n_amostrar, replace=False)
        for z in escolhidos:
            salvar_slice(int(z), tem_nodulo=False)
    else:
        n_amostrar = 0
        print("  [AVISO] %s: nenhum slice negativo elegível encontrado" % paciente)

    print("  %s (%s): %d slices positivos, %d negativos salvos" % (
        paciente, split, len(slices_positivos), n_amostrar))

    return linhas_manifesto


def main():
    split_df = pd.read_csv(CAMINHO_SPLIT)
    nodulos_df = pd.read_csv(CAMINHO_NODULOS_VALIDOS)

    clusters_validos_por_paciente = (
        nodulos_df.groupby("paciente")["cluster_id"].unique().apply(list).to_dict()
    )

    manifesto = []
    for _, row in split_df.iterrows():
        paciente, split = row["paciente"], row["split"]
        cluster_ids = clusters_validos_por_paciente.get(paciente, [])
        pasta_slices_split = os.path.join(CAMINHO_SLICES, split)
        linhas = gerar_slices_paciente(
            paciente, split, cluster_ids, pasta_slices_split)
        manifesto.extend(linhas)

    df_manifesto = pd.DataFrame(manifesto)
    os.makedirs(CAMINHO_SLICES, exist_ok=True)
    cam_manifesto = os.path.join(CAMINHO_SLICES, "manifesto.csv")
    df_manifesto.to_csv(cam_manifesto, index=False)

    print("\n" + "=" * 60)
    print("Resumo final")
    print("=" * 60)
    for split in ["train", "val", "test"]:
        sub = df_manifesto[df_manifesto["split"] == split]
        print("  %-5s: %4d slices (%4d positivos, %4d negativos)" % (
            split, len(sub), sub["tem_nodulo"].sum(),
            (sub["tem_nodulo"] == 0).sum()))
    print("\nManifesto salvo em: %s" % cam_manifesto)


if __name__ == "__main__":
    main()
