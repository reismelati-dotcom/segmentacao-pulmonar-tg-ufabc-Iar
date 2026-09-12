# =============================================================================
# 3_filtrar_e_dividir.py — Filtragem de consenso + divisão treino/val/teste
# =============================================================================
#
# O que este script faz:
#
#   1) Lê o indice_nodulos.csv gerado pelo script 1 (uma linha por anotação
#      individual, já com cluster_id e n_anotacoes_cluster).
#   2) Filtra os NÓDULOS FÍSICOS (clusters) que têm poucas anotações
#      concordando — por padrão, mantém só clusters com >= LIMIAR_ANOTACOES
#      radiologistas/anotações (reduz ruído de nódulos ambíguos/pequenos,
#      como o caso do LIDC-IDRI-0701 nódulo 05 discutido antes).
#   3) Divide os PACIENTES (não os nódulos, nem os slices) em
#      treino / validação / teste, de forma ESTRATIFICADA pela quantidade
#      de nódulos válidos por paciente, para os três grupos ficarem
#      comparáveis em "dificuldade".
#
# Por que treino/validação/teste (e não só treino/teste):
#   O conjunto de validação é usado para early stopping e ajuste de
#   hiperparâmetros da U-Net DURANTE o desenvolvimento. O conjunto de
#   teste só deve ser tocado UMA VEZ, no final, para comparar o método
#   clássico com a U-Net de forma justa. Se você usar o "teste" para
#   decidir quando parar o treino, ele deixa de ser um teste cego.
#
# Saídas (em CAMINHO_SAIDA, a mesma pasta usada no script 1):
#   - split_pacientes.csv  : 1 linha por paciente, com a coluna 'split'
#                            (train/val/test) e contagens de nódulos
#   - nodulos_validos.csv  : indice_nodulos.csv filtrado (só clusters
#                            válidos) + coluna 'split' — use este arquivo
#                            para montar o treino/avaliação dos dois métodos
#
# =============================================================================

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# --- Configurações ---

CAMINHO_SAIDA = r"D:\TG\processadoV3"
CAMINHO_CSV_ENTRADA = os.path.join(CAMINHO_SAIDA, "indice_nodulos.csv")

# Nº mínimo de anotações (radiologistas) concordando para o cluster ser
# considerado um nódulo válido para treino/avaliação.
LIMIAR_ANOTACOES = 3

# Tamanho do conjunto de teste (mantido intocado até a comparação final).
# Com 100 pacientes: 70 treino / 10 validação / 20 teste.
N_TESTE = 20

# Tamanho do conjunto de validação, tirado do restante (usado para
# early stopping / ajuste de hiperparâmetros da U-Net).
N_VALIDACAO = 10
# O que sobrar (100 - N_TESTE - N_VALIDACAO = 70) vai para treino.

SEED = 42

# --- Modo incremental ---
# Se True e já existir um split_pacientes.csv de uma rodada anterior,
# os pacientes que já tinham split definido MANTÊM o mesmo split
# (train/val/test) — o teste e a validação ficam intocados, protegendo
# a validade de tudo que já foi calibrado/avaliado com eles. Pacientes
# NOVOS (que não estavam no split anterior) são todos adicionados ao
# TREINO. Se False, ignora qualquer split anterior e sorteia do zero
# (só use False se quiser descartar a divisão anterior de propósito).
#
# Desligado agora de propósito: você optou por uma divisão nova do zero
# com os 100 pacientes (70/10/20), não por preservar o split anterior.
MODO_INCREMENTAL = False


def dividir_pacientes_incremental(resumo, cam_split_existente):
    split_existente = pd.read_csv(cam_split_existente)
    mapa_split = dict(zip(split_existente["paciente"], split_existente["split"]))

    resumo = resumo.copy()
    resumo["split"] = resumo["paciente"].map(mapa_split)

    novos = resumo["split"].isna()
    n_novos = int(novos.sum())
    n_existentes = len(resumo) - n_novos
    resumo.loc[novos, "split"] = "train"

    print("Modo incremental: %d paciente(s) mantiveram o split anterior, "
          "%d paciente(s) novo(s) adicionado(s) ao TREINO" % (n_existentes, n_novos))
    if n_novos > 0:
        print("  Novos pacientes: %s" % sorted(resumo.loc[novos, "paciente"].tolist()))

    pacientes_atuais = set(resumo["paciente"])
    pacientes_sumidos = set(mapa_split.keys()) - pacientes_atuais
    if pacientes_sumidos:
        print("  [AVISO] %d paciente(s) do split anterior não aparecem mais "
              "no indice_nodulos.csv atual (removidos ou não reprocessados): %s"
              % (len(pacientes_sumidos), sorted(pacientes_sumidos)))

    return resumo


def carregar_e_filtrar(caminho_csv, limiar_anotacoes):
    df = pd.read_csv(caminho_csv)

    obrigatorias = {"paciente", "cluster_id", "n_anotacoes_cluster"}
    faltando = obrigatorias - set(df.columns)
    if faltando:
        raise ValueError(
            "CSV não tem as colunas esperadas (%s). Confira se foi gerado "
            "pela versão mais recente do script 1." % faltando)

    # 1 linha por cluster (nódulo físico), independente do formato
    # XML ou DICOM-SEG — n_anotacoes_cluster já conta quantas anotações
    # formam aquele cluster, então serve como critério único para os dois.
    clusters = (
        df.groupby(["paciente", "cluster_id"], as_index=False)
        .agg(
            n_anotacoes_cluster=("n_anotacoes_cluster", "first"),
            malignidade_media=("malignidade", "mean"),
            volume_mm3_medio=("volume_mm3", "mean"),
        )
    )

    clusters["valido"] = clusters["n_anotacoes_cluster"] >= limiar_anotacoes

    n_total = len(clusters)
    n_validos = clusters["valido"].sum()
    print("Nódulos físicos (clusters) no total: %d" % n_total)
    print("Nódulos válidos (>= %d anotações): %d (%.1f%%)" % (
        limiar_anotacoes, n_validos, 100 * n_validos / n_total))

    return df, clusters


def resumo_por_paciente(clusters):
    resumo = (
        clusters.groupby("paciente")
        .agg(
            n_nodulos_total=("cluster_id", "count"),
            n_nodulos_validos=("valido", "sum"),
        )
        .reset_index()
    )

    sem_nodulo_valido = resumo[resumo["n_nodulos_validos"] == 0]
    if len(sem_nodulo_valido) > 0:
        print("\n[AVISO] %d paciente(s) ficaram SEM nenhum nódulo válido "
              "após o filtro (ainda são mantidos na divisão, mas não "
              "contribuem com nódulos positivos — úteis só como fonte de "
              "slices negativos, se for esse o seu desenho de treino):" %
              len(sem_nodulo_valido))
        for p in sem_nodulo_valido["paciente"]:
            print("    - %s" % p)

    return resumo


ORDEM_FAIXAS = ["0", "1", "2", "3+"]


def _mesclar_estratos_raros(estratos, ordem=ORDEM_FAIXAS, minimo=2):
    """
    train_test_split(stratify=...) exige pelo menos 2 membros por classe.
    Com poucos pacientes numa faixa (ex.: só 1 paciente sem nenhum
    nódulo válido), a estratificação quebra. Aqui mesclamos
    iterativamente qualquer classe com menos de `minimo` membros na
    classe vizinha (na ordem definida), até todas as classes restantes
    terem membros suficientes.
    """
    estratos = np.array(estratos, dtype=object)
    while True:
        valores, contagens = np.unique(estratos, return_counts=True)
        mapa_contagem = dict(zip(valores, contagens))
        pequenas = [v for v in ordem if v in mapa_contagem and mapa_contagem[v] < minimo]
        if not pequenas or len(mapa_contagem) <= 1:
            break
        classe = pequenas[0]
        idx = ordem.index(classe)
        alvo = None
        for j in range(idx + 1, len(ordem)):
            if ordem[j] in mapa_contagem:
                alvo = ordem[j]
                break
        if alvo is None:
            for j in range(idx - 1, -1, -1):
                if ordem[j] in mapa_contagem:
                    alvo = ordem[j]
                    break
        if alvo is None:
            break
        print("  [Estratificação] mesclando classe '%s' (%d paciente(s)) "
              "com '%s' — poucos membros para estratificar sozinha" % (
                  classe, mapa_contagem[classe], alvo))
        estratos[estratos == classe] = alvo
    return estratos


def dividir_pacientes(resumo, n_teste, n_validacao, seed):
    n_total = len(resumo)
    n_treino = n_total - n_teste - n_validacao
    if n_treino <= 0:
        raise ValueError(
            "N_TESTE + N_VALIDACAO (%d) >= número de pacientes (%d)." %
            (n_teste + n_validacao, n_total))

    # Estratifica pela quantidade de nódulos válidos por paciente (em
    # faixas), para treino/val/teste ficarem comparáveis em dificuldade.
    faixas = pd.cut(
        resumo["n_nodulos_validos"],
        bins=[-1, 0, 1, 2, np.inf],
        labels=ORDEM_FAIXAS,
    )

    pacientes = resumo["paciente"].values
    y_estrato = _mesclar_estratos_raros(faixas.astype(str).values)

    # Passo 1: separar teste do restante
    idx_resto, idx_teste = train_test_split(
        np.arange(n_total),
        test_size=n_teste,
        random_state=seed,
        stratify=y_estrato,
    )

    # Passo 2: dentro do resto, separar validação do treino — remescla
    # as classes raras de novo, já que a distribuição mudou depois de
    # remover os pacientes de teste.
    y_resto = _mesclar_estratos_raros(y_estrato[idx_resto])
    idx_treino_rel, idx_val_rel = train_test_split(
        np.arange(len(idx_resto)),
        test_size=n_validacao,
        random_state=seed,
        stratify=y_resto,
    )
    idx_treino = idx_resto[idx_treino_rel]
    idx_val = idx_resto[idx_val_rel]

    split = pd.Series(index=np.arange(n_total), dtype=object)
    split.iloc[idx_treino] = "train"
    split.iloc[idx_val] = "val"
    split.iloc[idx_teste] = "test"

    resumo = resumo.copy()
    resumo["split"] = split.values
    return resumo


def imprimir_resumo_split(resumo):
    print("\n" + "=" * 60)
    print("Divisão final (%d pacientes)" % len(resumo))
    print("=" * 60)
    for grupo in ["train", "val", "test"]:
        sub = resumo[resumo["split"] == grupo]
        print("  %-5s: %2d pacientes | nódulos válidos: total=%3d, "
              "média=%.2f/paciente" % (
                  grupo, len(sub), sub["n_nodulos_validos"].sum(),
                  sub["n_nodulos_validos"].mean()))


def main():
    df, clusters = carregar_e_filtrar(CAMINHO_CSV_ENTRADA, LIMIAR_ANOTACOES)
    resumo = resumo_por_paciente(clusters)

    cam_split = os.path.join(CAMINHO_SAIDA, "split_pacientes.csv")
    if MODO_INCREMENTAL and os.path.exists(cam_split):
        resumo = dividir_pacientes_incremental(resumo, cam_split)
    else:
        if MODO_INCREMENTAL:
            print("MODO_INCREMENTAL=True, mas não há split_pacientes.csv "
                  "anterior — fazendo a divisão do zero.")
        resumo = dividir_pacientes(resumo, N_TESTE, N_VALIDACAO, SEED)

    imprimir_resumo_split(resumo)

    # Salva o resumo por paciente (com o split)
    cam_split = os.path.join(CAMINHO_SAIDA, "split_pacientes.csv")
    resumo.to_csv(cam_split, index=False)
    print("\nSalvo: %s" % cam_split)

    # Salva o CSV de nódulos, filtrado (só clusters válidos) e já com
    # a coluna 'split' — este é o arquivo que deve alimentar o treino
    # e a avaliação dos dois métodos de segmentação.
    clusters_validos = clusters[clusters["valido"]][["paciente", "cluster_id"]]
    df_validos = df.merge(clusters_validos, on=["paciente", "cluster_id"], how="inner")
    df_validos = df_validos.merge(
        resumo[["paciente", "split"]], on="paciente", how="left")

    cam_nodulos = os.path.join(CAMINHO_SAIDA, "nodulos_validos.csv")
    df_validos.to_csv(cam_nodulos, index=False)
    print("Salvo: %s (%d linhas, de %d originais)" % (
        cam_nodulos, len(df_validos), len(df)))


if __name__ == "__main__":
    main()
