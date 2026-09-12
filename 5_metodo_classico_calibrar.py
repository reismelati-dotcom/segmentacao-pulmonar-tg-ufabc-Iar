# =============================================================================
# 5_metodo_classico_calibrar.py — Calibração do método clássico (busca em ESTÁGIOS)
# =============================================================================
#
# Pipeline clássico de segmentação de nódulos (por slice 2D):
#   1) Segmentação pulmonar (threshold + componentes conexos)
#   2) Suavização Gaussiana leve (reduz ruído do CT antes do threshold)
#   3) Threshold de intensidade DENTRO do pulmão (candidatos a nódulo)
#   4) Abertura/fechamento morfológico (remove ruído fino)
#   5) Filtros por região (regionprops), aplicados em cascata:
#        área, circularidade, excentricidade, solidez, extent,
#        intensidade média
#
# ESTRATÉGIA DE CALIBRAÇÃO — busca em ESTÁGIOS (greedy), não uma grade
# única gigante:
#   Estágio 0: threshold + área + circularidade + sigma do Gaussiano
#              (busca conjunta, pois interagem fortemente)
#   Estágio 1: + excentricidade (fixando o estágio 0 no melhor valor)
#   Estágio 2: + solidez        (fixando os estágios 0-1)
#   Estágio 3: + extent         (fixando os estágios 0-2)
#   Estágio 4: + intensidade média (fixando os estágios 0-3)
#
# Cada estágio só otimiza a(s) dimensão(ões) nova(s), reaproveitando o
# melhor resultado do estágio anterior. Isso é MUITO mais barato que uma
# grade conjunta de 7 dimensões (que seria inviável em tempo), e tem um
# benefício extra: gera uma tabela de "ganho de Dice por filtro
# adicionado", útil para discutir a contribuição de cada atributo no TG.
#
# O TESTE não é usado aqui — reservado para os scripts 6 e 7.
#
# =============================================================================
# HISTÓRICO (para a metodologia do TG)
# =============================================================================
# Antes desta versão em estágios, foram testadas: grade única grande
# (800 combinações, pior resultado), transformada Top-Hat (pior e mais
# lenta), seleção por "Dice sem FP" (piora o Dice real), filtro de
# precisão mínima (nenhuma combinação passou). O melhor resultado até
# então era Dice=0.180 (limiar=-400, área_min=40, circularidade=0.15,
# sem os filtros de forma adicionais abaixo).
# =============================================================================

import os
import json
import time
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure

CAMINHO_SLICES = r"D:\TG\slices_2dV3"
CAMINHO_MANIFESTO = os.path.join(CAMINHO_SLICES, "manifesto.csv")
CAMINHO_SAIDA_PARAMS = os.path.join(CAMINHO_SLICES, "melhores_parametros.json")
CAMINHO_GRAFICO_DICE = os.path.join(CAMINHO_SLICES, "dice_normal_vs_semfp.png")
CAMINHO_GRAFICO_ESTAGIOS = os.path.join(CAMINHO_SLICES, "dice_por_estagio.png")

HU_MIN, HU_MAX = -1000, 400
LIMIAR_HU_PULMAO = -320

ESPECIFICIDADE_MINIMA = 0.90

USAR_SUBAMOSTRA_TREINO = False
N_SUBAMOSTRA_TREINO = 400
SEED = 42

# --- Grades de cada estágio ---
# Estágio 0: não congelamos o limiar (pode mudar depois que os filtros
# de forma entrarem) — testamos um intervalo em torno do valor
# historicamente bom, mais permissivo, já que os filtros de forma
# novos podem compensar um threshold mais solto.
GRADE_LIMIAR_HU_NODULO = [-500, -400, -300]
GRADE_AREA_MIN_PX = [20, 40]
GRADE_AREA_MAX_PX = [3000]
GRADE_CIRCULARIDADE_MIN = [0.0, 0.15]
GRADE_SIGMA_GAUSSIANO = [0.0, 0.8, 1.2]   # 0.0 = sem suavização

GRADE_EXCENTRICIDADE_MAX = [1.0, 0.9, 0.8, 0.7, 0.6]   # 1.0 = desligado
GRADE_SOLIDEZ_MIN = [0.0, 0.6, 0.7, 0.8, 0.85, 0.9]     # 0.0 = desligado
GRADE_EXTENT_MIN = [0.0, 0.2, 0.3, 0.4, 0.5]            # 0.0 = desligado
# Grade de intensidade média é construída dinamicamente no Estágio 4,
# relativa ao limiar escolhido no Estágio 0 (ver função main()).


def normalizar_para_escala(hu_valor):
    return (hu_valor - HU_MIN) / (HU_MAX - HU_MIN)


def segmentar_pulmao(img_norm, limiar_hu_pulmao=LIMIAR_HU_PULMAO):
    limiar_norm = normalizar_para_escala(limiar_hu_pulmao)
    binario = img_norm < limiar_norm

    rotulado, n = ndimage.label(binario)
    if n == 0:
        return np.zeros_like(binario, dtype=bool)

    rotulos_borda = set(rotulado[0, :]) | set(rotulado[-1, :]) | \
                    set(rotulado[:, 0]) | set(rotulado[:, -1])
    rotulos_borda.discard(0)
    for r in rotulos_borda:
        binario[rotulado == r] = False

    binario = ndimage.binary_fill_holes(binario)
    rotulado, n = ndimage.label(binario)
    if n == 0:
        return np.zeros_like(binario, dtype=bool)

    tamanhos = ndimage.sum(binario, rotulado, range(1, n + 1))
    maiores = np.argsort(tamanhos)[::-1][:2] + 1
    return np.isin(rotulado, maiores)


def segmentar_nodulos_classico(img_norm, mascara_pulmao, limiar_hu_nodulo,
                                area_min_px, area_max_px, circularidade_min,
                                sigma_gaussiano=0.0, excentricidade_max=1.0,
                                solidez_min=0.0, extent_min=0.0,
                                intensidade_media_min=0.0):
    """
    Todos os filtros novos têm valor default "desligado", então continua
    compatível com um melhores_parametros.json salvo por uma versão
    anterior (chaves ausentes = usa o default = sem filtrar por aquilo).
    """
    if sigma_gaussiano > 0:
        img_proc = ndimage.gaussian_filter(img_norm, sigma=sigma_gaussiano)
    else:
        img_proc = img_norm

    limiar_norm = normalizar_para_escala(limiar_hu_nodulo)
    candidato = (img_proc > limiar_norm) & mascara_pulmao

    estrutura = np.ones((3, 3), dtype=bool)
    candidato = ndimage.binary_opening(candidato, structure=estrutura)
    candidato = ndimage.binary_closing(candidato, structure=estrutura)

    rotulado = measure.label(candidato)
    saida = np.zeros_like(candidato, dtype=np.uint8)

    for regiao in measure.regionprops(rotulado, intensity_image=img_proc):
        area = regiao.area
        perimetro = regiao.perimeter if regiao.perimeter > 0 else 1e-6
        circularidade = 4 * np.pi * area / (perimetro ** 2)

        if not (area_min_px <= area <= area_max_px):
            continue
        if circularidade < circularidade_min:
            continue
        if regiao.eccentricity > excentricidade_max:
            continue
        if regiao.solidity < solidez_min:
            continue
        if regiao.extent < extent_min:
            continue
        intensidade_media = getattr(regiao, "intensity_mean", None)
        if intensidade_media is None:
            intensidade_media = regiao.mean_intensity
        if intensidade_media < intensidade_media_min:
            continue

        saida[rotulado == regiao.label] = 1

    return saida


def dice(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    soma = pred.sum() + gt.sum()
    return 1.0 if soma == 0 else 2.0 * inter / soma


def dice_sem_fp(pred, gt):
    tp = np.logical_and(pred, gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    denom = 2 * tp + fn
    return 1.0 if denom == 0 else 2.0 * tp / denom


def carregar_subset(manifesto, split, n_amostra=None, seed=SEED):
    sub = manifesto[manifesto["split"] == split].copy()
    if n_amostra is not None and len(sub) > n_amostra:
        sub = sub.sample(n=n_amostra, random_state=seed)
    imgs, gts, pulmoes = [], [], []
    for _, row in sub.iterrows():
        img = np.load(row["arquivo_imagem"])
        gt = np.load(row["arquivo_mascara"]).astype(bool)
        pulmao = segmentar_pulmao(img)
        imgs.append(img)
        gts.append(gt)
        pulmoes.append(pulmao)
    return imgs, gts, pulmoes


def avaliar_parametros(imgs, gts, pulmoes, params):
    dices_normais, dices_semfp = [], []
    tn_total, fp_total = 0, 0
    for img, gt, pulmao in zip(imgs, gts, pulmoes):
        pred = segmentar_nodulos_classico(img, pulmao, **params)
        if gt.sum() > 0:
            dices_normais.append(dice(pred, gt))
            dices_semfp.append(dice_sem_fp(pred, gt))
        else:
            tn_total += np.logical_and(~pred, ~gt).sum()
            fp_total += pred.sum()

    dice_normal_medio = float(np.mean(dices_normais)) if dices_normais else 0.0
    dice_semfp_medio = float(np.mean(dices_semfp)) if dices_semfp else 0.0
    especificidade = (tn_total / (tn_total + fp_total)
                       if (tn_total + fp_total) > 0 else 1.0)
    return dice_normal_medio, dice_semfp_medio, especificidade


def buscar_estagio(nome_estagio, chaves_novas, grades_novas, params_fixos,
                    imgs_treino, gts_treino, pulmoes_treino,
                    imgs_val, gts_val, pulmoes_val, top_n=5):
    combinacoes = list(itertools.product(*grades_novas))
    print("\n" + "-" * 60)
    print("Estágio: %s (%d combinação(ões) novas, params fixos: %s)" % (
        nome_estagio, len(combinacoes), params_fixos))
    print("-" * 60)

    t0 = time.time()
    resultados = []
    for valores in combinacoes:
        params = {**params_fixos, **dict(zip(chaves_novas, valores))}
        dice_normal_t, dice_semfp_t, espec_t = avaliar_parametros(
            imgs_treino, gts_treino, pulmoes_treino, params)
        resultados.append({
            **dict(zip(chaves_novas, valores)),
            "dice_normal_treino": dice_normal_t,
            "dice_semfp_treino": dice_semfp_t,
            "especificidade_treino": espec_t,
        })
    print("  %d combinações testadas em %.1fs" % (len(combinacoes), time.time() - t0))

    df = pd.DataFrame(resultados)
    aptos = df[df["especificidade_treino"] >= ESPECIFICIDADE_MINIMA]
    if len(aptos) == 0:
        aptos = df
    aptos = aptos.sort_values("dice_normal_treino", ascending=False)

    candidatos = aptos.head(min(top_n, len(aptos))).to_dict("records")
    for cand in candidatos:
        params = {**params_fixos,
                   **{k: cand[k] for k in chaves_novas}}
        dice_normal_v, dice_semfp_v, espec_v = avaliar_parametros(
            imgs_val, gts_val, pulmoes_val, params)
        cand["dice_normal_val"] = dice_normal_v
        cand["dice_semfp_val"] = dice_semfp_v
        cand["especificidade_val"] = espec_v

    aptos_val = [c for c in candidatos if c["especificidade_val"] >= ESPECIFICIDADE_MINIMA]
    if not aptos_val:
        aptos_val = candidatos
    aptos_val.sort(key=lambda c: c["dice_normal_val"], reverse=True)
    melhor = aptos_val[0]

    print("  Melhor deste estágio: %s" % {k: melhor[k] for k in chaves_novas})
    print("  dice_normal_val=%.4f | dice_semfp_val=%.4f | especificidade_val=%.4f" % (
        melhor["dice_normal_val"], melhor["dice_semfp_val"], melhor["especificidade_val"]))

    novos_params_fixos = {**params_fixos, **{k: melhor[k] for k in chaves_novas}}
    return novos_params_fixos, melhor


def salvar_grafico_dice(valores, caminho_saida):
    labels = ["Treino", "Validação"]
    normais = [valores["dice_normal_treino"], valores["dice_normal_val"]]
    semfp = [valores["dice_semfp_treino"], valores["dice_semfp_val"]]

    x = np.arange(len(labels))
    largura = 0.35
    fig, ax = plt.subplots(figsize=(6, 4.5))
    b1 = ax.bar(x - largura / 2, normais, largura, label="Dice normal (penaliza FP)", color="#4363d8")
    b2 = ax.bar(x + largura / 2, semfp, largura, label="Dice sem FP", color="#f58231")
    for barras in (b1, b2):
        for b in barras:
            altura = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, altura + 0.015, "%.3f" % altura,
                    ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Dice"); ax.set_ylim(0, 1.0)
    ax.set_title("Método clássico final — Dice normal vs Dice sem FP")
    ax.legend(loc="upper right"); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=150)
    plt.close(fig)
    print("Gráfico salvo em: %s" % caminho_saida)


def salvar_grafico_estagios(progressao, caminho_saida):
    """Gráfico de barras: Dice normal (validação) após cada estágio —
    mostra a contribuição individual de cada filtro adicionado."""
    nomes = [p[0] for p in progressao]
    valores = [p[1] for p in progressao]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    cores = plt.cm.viridis(np.linspace(0.2, 0.8, len(nomes)))
    barras = ax.bar(nomes, valores, color=cores)
    for b, v in zip(barras, valores):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, "%.3f" % v,
                ha="center", fontsize=9)
    ax.set_ylabel("Dice normal (validação)")
    ax.set_title("Ganho de Dice por filtro adicionado (busca em estágios)")
    ax.set_ylim(0, max(valores) * 1.25)
    plt.xticks(rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=150)
    plt.close(fig)
    print("Gráfico salvo em: %s" % caminho_saida)


def main():
    manifesto = pd.read_csv(CAMINHO_MANIFESTO)

    print("Carregando slices de treino...")
    n_treino = N_SUBAMOSTRA_TREINO if USAR_SUBAMOSTRA_TREINO else None
    imgs_treino, gts_treino, pulmoes_treino = carregar_subset(manifesto, "train", n_amostra=n_treino)
    print("  %d slices de treino carregados" % len(imgs_treino))

    print("Carregando slices de validação...")
    imgs_val, gts_val, pulmoes_val = carregar_subset(manifesto, "val")
    print("  %d slices de validação carregados" % len(imgs_val))

    progressao = []  # (nome_estagio, dice_normal_val, dice_semfp_val)

    # Estágio 0: base (threshold + área + circularidade + gaussiano)
    params, melhor0 = buscar_estagio(
        "0: base (threshold+área+circ+gaussiano)",
        ["limiar_hu_nodulo", "area_min_px", "area_max_px",
         "circularidade_min", "sigma_gaussiano"],
        [GRADE_LIMIAR_HU_NODULO, GRADE_AREA_MIN_PX, GRADE_AREA_MAX_PX,
         GRADE_CIRCULARIDADE_MIN, GRADE_SIGMA_GAUSSIANO],
        {}, imgs_treino, gts_treino, pulmoes_treino, imgs_val, gts_val, pulmoes_val,
        top_n=10)
    progressao.append(("Base", melhor0["dice_normal_val"], melhor0["dice_semfp_val"]))

    # Estágio 1: + excentricidade
    params, melhor1 = buscar_estagio(
        "1: + excentricidade", ["excentricidade_max"], [GRADE_EXCENTRICIDADE_MAX],
        params, imgs_treino, gts_treino, pulmoes_treino, imgs_val, gts_val, pulmoes_val)
    progressao.append(("+ Excentricidade", melhor1["dice_normal_val"], melhor1["dice_semfp_val"]))

    # Estágio 2: + solidez
    params, melhor2 = buscar_estagio(
        "2: + solidez", ["solidez_min"], [GRADE_SOLIDEZ_MIN],
        params, imgs_treino, gts_treino, pulmoes_treino, imgs_val, gts_val, pulmoes_val)
    progressao.append(("+ Solidez", melhor2["dice_normal_val"], melhor2["dice_semfp_val"]))

    # Estágio 3: + extent
    params, melhor3 = buscar_estagio(
        "3: + extent", ["extent_min"], [GRADE_EXTENT_MIN],
        params, imgs_treino, gts_treino, pulmoes_treino, imgs_val, gts_val, pulmoes_val)
    progressao.append(("+ Extent", melhor3["dice_normal_val"], melhor3["dice_semfp_val"]))

    # Estágio 4: + intensidade média (grade relativa ao limiar escolhido)
    limiar_norm_atual = normalizar_para_escala(params["limiar_hu_nodulo"])
    grade_intensidade = [0.0, limiar_norm_atual + 0.05,
                          limiar_norm_atual + 0.10, limiar_norm_atual + 0.15]
    params, melhor4 = buscar_estagio(
        "4: + intensidade média", ["intensidade_media_min"], [grade_intensidade],
        params, imgs_treino, gts_treino, pulmoes_treino, imgs_val, gts_val, pulmoes_val)
    progressao.append(("+ Intensidade média", melhor4["dice_normal_val"], melhor4["dice_semfp_val"]))

    print("\n" + "=" * 60)
    print("PROGRESSÃO DO DICE POR ESTÁGIO (validação)")
    print("=" * 60)
    for nome, dn, ds in progressao:
        print("  %-22s: dice_normal=%.4f | dice_sem_FP=%.4f" % (nome, dn, ds))

    print("\nParâmetros finais:", params)

    with open(CAMINHO_SAIDA_PARAMS, "w") as f:
        json.dump(params, f, indent=2)
    print("\nParâmetros salvos em: %s" % CAMINHO_SAIDA_PARAMS)

    salvar_grafico_dice(melhor4, CAMINHO_GRAFICO_DICE)
    salvar_grafico_estagios(progressao, CAMINHO_GRAFICO_ESTAGIOS)

    print("(o teste NÃO foi usado nesta etapa — reservado para os scripts 6 e 7)")


if __name__ == "__main__":
    main()
