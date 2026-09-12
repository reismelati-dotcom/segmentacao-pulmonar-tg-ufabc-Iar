# ============================================================
# SCRIPT 0 - SELEÇÃO DOS 50 PACIENTES DO LIDC-IDRI
# ============================================================
# Objetivo:
# Filtrar os pacientes mais adequados para o estudo piloto
# de segmentação + radiômica.
#
# Critérios utilizados:
# 1. Espessura de corte <= 2.0 mm
# 2. Pelo menos um nódulo com diâmetro >= 3 mm
# 3. Pelo menos 2 radiologistas anotando o nódulo
#
# Saídas:
# - lista_pacientes.txt
# - pacientes_selecionados.csv
# ============================================================

import pylidc as pl
import pandas as pd
import numpy as np

# -----------------------------
# PARÂMETROS DE FILTRAGEM
# -----------------------------
MAX_ESPESSURA_MM = 2.0
MIN_DIAMETRO_MM = 3.0
MIN_ANOTADORES = 2
N_PACIENTES = 100


def avaliar_scan(scan):
    """
    Avalia se um scan/paciente atende aos critérios do estudo.

    Retorna:
        dict com métricas do paciente, se ele for válido
        None, caso contrário
    """
    try:
        # Espessura de corte
        espessura = scan.slice_thickness
        if espessura is None or espessura > MAX_ESPESSURA_MM:
            return None

        # Agrupa anotações que pertencem ao mesmo nódulo
        nodulos = scan.cluster_annotations()

        nodulos_validos = []

        for nodulo in nodulos:
            # Exigir mínimo de radiologistas
            if len(nodulo) < MIN_ANOTADORES:
                continue

            diametros = []
            for ann in nodulo:
                try:
                    diametros.append(float(ann.diameter))
                except Exception:
                    pass

            if len(diametros) == 0:
                continue

            diametro_medio = np.mean(diametros)

            # Exigir diâmetro mínimo
            if diametro_medio < MIN_DIAMETRO_MM:
                continue

            nodulos_validos.append({
                "n_anotadores": len(nodulo),
                "diametro_medio": diametro_medio
            })

        if len(nodulos_validos) == 0:
            return None

        # Score simples para ranquear pacientes
        # Ideia: priorizar menor espessura, mais concordância e mais nódulos válidos
        score = 0
        score += (MAX_ESPESSURA_MM - espessura) * 10
        score += sum(n["n_anotadores"] for n in nodulos_validos)
        score += len(nodulos_validos) * 5
        score += np.mean([n["diametro_medio"] for n in nodulos_validos]) * 0.5

        return {
            "patient_id": scan.patient_id,
            "series_instance_uid": scan.series_instance_uid,
            "slice_thickness": espessura,
            "n_nodulos_validos": len(nodulos_validos),
            "diametro_medio_nodulos": np.mean([n["diametro_medio"] for n in nodulos_validos]),
            "max_anotadores_nodulo": max([n["n_anotadores"] for n in nodulos_validos]),
            "score": score
        }

    except Exception as e:
        print(f"Erro ao avaliar {scan.patient_id}: {e}")
        return None


def main():
    print("Consultando scans do LIDC-IDRI...")
    scans = pl.query(pl.Scan).all()
    print(f"Total de scans encontrados: {len(scans)}")

    resultados = []

    for i, scan in enumerate(scans):
        if (i + 1) % 100 == 0:
            print(f"Processados {i+1}/{len(scans)} scans")

        info = avaliar_scan(scan)
        if info is not None:
            resultados.append(info)

    if len(resultados) == 0:
        print("Nenhum paciente atendeu aos critérios.")
        return

    df = pd.DataFrame(resultados)

    # Em alguns casos pode haver mais de uma série por paciente.
    # Mantemos apenas a de maior score.
    df = df.sort_values("score", ascending=False)
    df = df.drop_duplicates(subset=["patient_id"], keep="first")

    selecionados = df.head(N_PACIENTES).copy()

    print("\n=== RESUMO DOS PACIENTES SELECIONADOS ===")
    print(selecionados[[
        "patient_id",
        "slice_thickness",
        "n_nodulos_validos",
        "diametro_medio_nodulos",
        "score"
    ]])

    # Salvar lista simples
    with open("lista_pacientes.txt", "w", encoding="utf-8") as f:
        for pid in selecionados["patient_id"]:
            f.write(pid + "\n")

    # Salvar tabela detalhada
    selecionados.to_csv("pacientes_selecionados.csv", index=False)

    print("\nArquivos gerados com sucesso:")
    print("- lista_pacientes.txt")
    print("- pacientes_selecionados.csv")


if __name__ == "__main__":
    main()
