# =============================================================================
# 1_ler_dados.py — Lê os DICOMs e as anotações dos radiologistas
# =============================================================================
#
# Suporta dois formatos de anotação do LIDC-IDRI:
#   - XML (41 pacientes): formato original com contornos slice-a-slice
#   - DICOM-SEG + SR (9 pacientes): formato mais recente sem XML
#
# Saídas por paciente (em CAMINHO_SAIDA/LIDC-IDRI-XXXX/):
#   - volume_tc.nii.gz                          : volume 3D em Hounsfield Units
#   - mascara_rad1_noduloXXX.nii.gz              : máscara por radiologista/nódulo
#   - mascara_consenso_noduloNN_votacao.nii.gz   : consenso por votação, por
#                                                   nódulo físico (cluster NN)
#   - mascara_consenso_noduloNN_staple.nii.gz    : consenso via STAPLE, por
#                                                   nódulo físico (cluster NN)
#   - indice_nodulos.csv                         : índice geral com métricas
#                                                   por anotação (com cluster_id)
#
# As anotações de diferentes radiologistas são primeiro agrupadas por
# proximidade espacial (agrupar_nodulos_por_proximidade) para estimar quais
# se referem ao mesmo nódulo físico, antes de gerar qualquer consenso —
# ver TOL_CLUSTER_MM abaixo.
#
# =============================================================================

import os
import glob
import numpy as np
import pydicom
import SimpleITK as sitk
import xml.etree.ElementTree as ET
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import threading

# =============================================================================
# CONFIGURAÇÕES DO PROJETO
# =============================================================================

CAMINHO_BASE  = r"D:\TG\manifest-1784043382254\LIDC-IDRI"
CAMINHO_SAIDA = r"D:\TG\processadoV3"
os.makedirs(CAMINHO_SAIDA, exist_ok=True)

# Salvar volume em NIfTI
SALVAR_VOLUME_NIFTI = True

# Salvar máscaras individuais de cada radiologista
SALVAR_MASCARAS = True

# Gerar máscara de consenso
GERAR_CONSENSO = True

# --- Agrupamento de anotações por nódulo físico (clustering) ---
#
# O LIDC-IDRI não garante que "nódulo 1 do radiologista A" seja o mesmo
# nódulo físico que "nódulo 1 do radiologista B". Antes de gerar o
# consenso, agrupamos anotações pela proximidade espacial entre seus
# contornos (mesmo algoritmo usado pelo pylidc: distância mínima entre
# voxels de borda + componentes conexos). Duas anotações são consideradas
# do mesmo nódulo se a distância mínima entre elas for <= TOL_CLUSTER_MM.
TOL_CLUSTER_MM = 5.0

# Estratégia de consenso aplicada DENTRO de cada cluster (nódulo físico):
#   "votacao" = fração mínima de anotações que precisam concordar num voxel
#   "staple"  = SimpleITK STAPLE (pondera confiabilidade de cada anotador)
#   "ambos"   = gera as duas versões (recomendado para comparar no TG)
METODO_CONSENSO = "ambos"

# Fração mínima de votos para o método de votação (0.5 = maioria simples).
# Só é usada quando METODO_CONSENSO inclui "votacao".
FRACAO_VOTOS_CONSENSO = 0.5

# Limiar de probabilidade aplicado ao mapa do STAPLE para binarizar.
LIMIAR_STAPLE = 0.5

# Salvar CSV com índice geral ao final
SALVAR_CSV = True

# Namespace XML do LIDC-IDRI
NS = "http://www.nih.gov"


# =============================================================================
# FUNÇÕES AUXILIARES — LEITURA
# =============================================================================

def encontrar_arquivos_serie(pasta_paciente):
    """
    Navega pelas subpastas e retorna:
      (lista_dcm_CT, xml_path, arquivos_SEG, arquivos_SR)

    O LIDC-IDRI mistura na mesma pasta:
      - Série CT (centenas de .dcm com ImagePositionPatient)
      - Arquivos SEG (1 nódulo por arquivo, sem ImagePositionPatient)
      - Arquivos SR  (relatório com malignidade, sem ImagePositionPatient)
      - Eventual radiografia (ignorada)
    """
    todos_dcms = glob.glob(
        os.path.join(pasta_paciente, "**", "*.dcm"), recursive=True)
    xmls = glob.glob(
        os.path.join(pasta_paciente, "**", "*.xml"), recursive=True)
    xml_path = xmls[0] if xmls else None

    if not todos_dcms:
        return [], xml_path, [], []

    series_ct  = {}
    arqs_seg   = []
    arqs_sr    = []

    for f in todos_dcms:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            mod = getattr(ds, "Modality", "")
            if mod == "SEG":
                arqs_seg.append(f); continue
            if mod == "SR":
                arqs_sr.append(f);  continue
            if mod != "CT":
                continue
            if not hasattr(ds, "ImagePositionPatient"):
                continue
            uid = getattr(ds, "SeriesInstanceUID", "SEM_UID")
            series_ct.setdefault(uid, []).append(f)
        except Exception:
            continue

    if not series_ct:
        print("  [AVISO] Nenhuma série CT válida encontrada")
        return [], xml_path, arqs_seg, arqs_sr

    melhor_uid  = max(series_ct, key=lambda u: len(series_ct[u]))
    lista_dcm   = series_ct[melhor_uid]
    if len(series_ct) > 1:
        print("  [INFO] %d séries CT — usando a maior (%d arqs)" % (
            len(series_ct), len(lista_dcm)))

    return sorted(lista_dcm), xml_path, arqs_seg, arqs_sr


def carregar_volume_dicom(lista_dcm):
    """
    Monta o volume 3D e constrói mapa z_pos → índice a partir de
    ImagePositionPatient (mais confiável que TransformPhysicalPointToIndex).
    """
    pares_z = []
    for f in lista_dcm:
        ds = pydicom.dcmread(f, stop_before_pixels=True)
        z  = float(ds.ImagePositionPatient[2])
        pares_z.append((z, f))

    pares_z.sort(key=lambda x: x[0])
    lista_ord = [f for _, f in pares_z]
    mapa_z    = {round(z, 4): idx for idx, (z, _) in enumerate(pares_z)}

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(lista_ord)
    img_sitk = reader.Execute()
    volume   = sitk.GetArrayFromImage(img_sitk)

    ds0       = pydicom.dcmread(lista_ord[0])
    slope     = float(getattr(ds0, "RescaleSlope",     1))
    intercept = float(getattr(ds0, "RescaleIntercept", 0))
    if volume.min() > -500:
        volume = volume * slope + intercept

    espacamento = img_sitk.GetSpacing()
    print("  Volume: %s | HU [%d, %d] | "
          "Espaç. %.2fx%.2fx%.2f mm | Z: %.1f→%.1f mm" % (
        volume.shape,
        volume.min(), volume.max(),
        espacamento[0], espacamento[1], espacamento[2],
        pares_z[0][0], pares_z[-1][0]))

    return volume, espacamento, mapa_z, img_sitk


def parsear_xml_lidc(xml_path, shape_volume, mapa_z):
    """
    Extrai máscaras binárias 3D do XML do LIDC-IDRI (41 pacientes).
    Retorna lista de dicts com radiologista, nodulo_id, mascara, malignidade.
    """
    tree    = ET.parse(xml_path)
    root    = tree.getroot()
    tag_r   = root.tag
    ns_real = tag_r.split("}")[0].strip("{") if "}" in tag_r else NS

    nodulos          = []
    num_radiologista = 0

    for sessao in root.iter(f"{{{ns_real}}}readingSession"):
        num_radiologista += 1
        for nodulo in sessao.iter(f"{{{ns_real}}}unblindedReadNodule"):
            nodulo_id   = nodulo.findtext(
                f"{{{ns_real}}}noduleID", default="?").strip()
            carac       = nodulo.find(f"{{{ns_real}}}characteristics")
            malignidade = -1
            if carac is not None:
                mal = carac.find(f"{{{ns_real}}}malignancy")
                if mal is not None and mal.text:
                    malignidade = int(mal.text)

            mascara = np.zeros(shape_volume, dtype=np.uint8)

            for roi in nodulo.iter(f"{{{ns_real}}}roi"):
                z_el = roi.find(f"{{{ns_real}}}imageZposition")
                if z_el is None or z_el.text is None:
                    continue
                z_pos = float(z_el.text.strip())
                z_key = round(z_pos, 4)

                if z_key in mapa_z:
                    z_idx = mapa_z[z_key]
                else:
                    zs   = list(mapa_z.keys())
                    zprx = min(zs, key=lambda z: abs(z - z_pos))
                    if abs(zprx - z_pos) > 3.0:
                        continue
                    z_idx = mapa_z[zprx]

                if not (0 <= z_idx < shape_volume[0]):
                    continue

                px, py = [], []
                for edge in roi.findall(f"{{{ns_real}}}edgeMap"):
                    xel = edge.find(f"{{{ns_real}}}xCoord")
                    yel = edge.find(f"{{{ns_real}}}yCoord")
                    if xel is not None and yel is not None:
                        try:
                            px.append(int(float(xel.text.strip())))
                            py.append(int(float(yel.text.strip())))
                        except (ValueError, AttributeError):
                            pass

                if len(px) < 3:
                    continue

                from skimage.draw import polygon
                rows, cols = polygon(py, px, shape=shape_volume[1:])
                mascara[z_idx, rows, cols] = 1

            if mascara.sum() > 0:
                nodulos.append({
                    "radiologista": num_radiologista,
                    "nodulo_id":    nodulo_id,
                    "mascara":      mascara,
                    "malignidade":  malignidade,
                })

    return nodulos


def parsear_dicom_seg(arquivos_seg, shape_volume, mapa_z):
    """
    Extrai máscaras de arquivos DICOM-SEG (9 pacientes sem XML).
    Cada arquivo SEG = 1 nódulo de 1 radiologista.
    """
    nodulos = []

    for arq in arquivos_seg:
        try:
            ds = pydicom.dcmread(arq)
        except Exception as e:
            print("  [AVISO] Falha ao ler SEG %s: %s" % (
                os.path.basename(arq), e))
            continue

        if getattr(ds, "Modality", "") != "SEG":
            continue

        descricao = getattr(ds, "SeriesDescription", os.path.basename(arq))
        nodulo_id = descricao.replace("Segmentation of ", "").strip()

        try:
            pixel_array = ds.pixel_array
        except Exception as e:
            print("  [AVISO] pixel_array falhou em %s: %s" % (
                os.path.basename(arq), e))
            continue

        if not hasattr(ds, "PerFrameFunctionalGroupsSequence"):
            continue

        mascara = np.zeros(shape_volume, dtype=np.uint8)
        for i, frame in enumerate(ds.PerFrameFunctionalGroupsSequence):
            if not hasattr(frame, "PlanePositionSequence"):
                continue
            z_pos = float(frame.PlanePositionSequence[0].ImagePositionPatient[2])
            z_key = round(z_pos, 4)

            if z_key in mapa_z:
                z_idx = mapa_z[z_key]
            else:
                zs   = list(mapa_z.keys())
                zprx = min(zs, key=lambda z: abs(z - z_pos))
                if abs(zprx - z_pos) > 3.0:
                    continue
                z_idx = mapa_z[zprx]

            if not (0 <= z_idx < shape_volume[0]):
                continue

            fm = (pixel_array[i] > 0).astype(np.uint8)
            if fm.shape != shape_volume[1:]:
                print("  [AVISO] Shape frame SEG diferente (arq=%s frame=%d): %s vs %s" % (
                    os.path.basename(arq), i, fm.shape, shape_volume[1:]))
                continue

            mascara[z_idx] = np.maximum(mascara[z_idx], fm)

        if mascara.sum() > 0:
            nodulos.append({
                "radiologista": "seg",
                "nodulo_id":    nodulo_id,
                "mascara":      mascara,
                "malignidade":  -1,
            })

    return nodulos


def extrair_caracteristicas_sr(arquivos_sr):
    """
    Lê arquivos DICOM-SR e extrai malignidade e outras características
    clínicas (subtlety, margem, textura, etc.).
    Retorna dict {nodulo_id: {campo: valor}}.
    """
    campos_code = [
        "Malignancy", "Subtlety score", "Internal structure",
        "Calcification", "Sphericity", "Margin",
        "Lobular Pattern", "Spiculation", "Texture",
    ]
    campos_num = ["Volume", "Diameter", "Surface area of mesh"]
    resultado  = {}

    for arq in arquivos_sr:
        try:
            ds = pydicom.dcmread(arq)
        except Exception as e:
            print("  [AVISO] Falha ao ler SR %s: %s" % (
                os.path.basename(arq), e))
            continue

        if getattr(ds, "Modality", "") != "SR":
            continue

        descricao = getattr(ds, "SeriesDescription", os.path.basename(arq))
        nodulo_id = descricao.replace(" evaluations", "").strip()
        carac     = {}

        def percorrer(seq):
            if not seq:
                return
            for item in seq:
                vt      = getattr(item, "ValueType", "")
                concept = getattr(item, "ConceptNameCodeSequence", None)
                nome    = concept[0].CodeMeaning if concept else None
                if nome in campos_code and vt == "CODE":
                    cc = getattr(item, "ConceptCodeSequence", None)
                    if cc:
                        carac[nome] = cc[0].CodeMeaning
                elif nome in campos_num and vt == "NUM":
                    mv = getattr(item, "MeasuredValueSequence", None)
                    if mv:
                        carac[nome] = float(mv[0].NumericValue)
                if hasattr(item, "ContentSequence"):
                    percorrer(item.ContentSequence)

        if hasattr(ds, "ContentSequence"):
            percorrer(ds.ContentSequence)

        resultado[nodulo_id] = carac

    return resultado


def parsear_malignidade(texto):
    """Converte '1 out of 5 (...)' → int 1-5, ou -1 se inválido."""
    if not texto:
        return -1
    try:
        return int(texto.split(" ")[0])
    except (ValueError, IndexError):
        return -1


# =============================================================================
# FUNÇÕES AUXILIARES — PROCESSAMENTO
# =============================================================================

def validar_mascara(mascara, shape_volume):
    """
    Verifica se a máscara é válida (shape, dtype, valores, não-vazia).
    Retorna a máscara corrigida.
    """
    if mascara.shape != shape_volume:
        raise ValueError("Shape incorreto: %s != %s" % (
            mascara.shape, shape_volume))

    if mascara.dtype != np.uint8:
        print("  [AVISO] Convertendo máscara para uint8")
        mascara = mascara.astype(np.uint8)

    valores = np.unique(mascara)
    if not np.all(np.isin(valores, [0, 1])):
        print("  [AVISO] Máscara com valores fora de {0,1}: %s" % valores)

    if mascara.sum() == 0:
        print("  [AVISO] Máscara vazia")

    return mascara


def _distancia_min_mm(mascara_a, mascara_b, espacamento_mm):
    """
    Distância mínima (mm) entre os voxels de borda de duas máscaras 3D.
    Retorna np.inf se alguma máscara for vazia.
    """
    dx, dy, dz = espacamento_mm  # sitk retorna (x, y, z); volume é (z, y, x)
    escala = np.array([dz, dy, dx])  # reordenado para bater com eixos (z,y,x)

    coords_a = np.argwhere(mascara_a)
    coords_b = np.argwhere(mascara_b)
    if len(coords_a) == 0 or len(coords_b) == 0:
        return np.inf

    pts_a = coords_a * escala
    pts_b = coords_b * escala

    arvore = cKDTree(pts_b)
    dists, _ = arvore.query(pts_a, k=1)
    return float(dists.min())


def agrupar_nodulos_por_proximidade(lista_nodulos, espacamento_mm,
                                     tol_mm=5.0):
    """
    Agrupa anotações que provavelmente se referem ao mesmo nódulo físico,
    usando a distância mínima em mm entre os contornos (mesma ideia do
    pylidc.Scan.cluster_annotations): duas anotações são "adjacentes" se
    d(a,b) <= tol_mm; os clusters são as componentes conexas do grafo
    de adjacência resultante.

    Retorna lista de clusters, cada um sendo uma lista de dicts (o mesmo
    formato de 'info' usado no restante do script).
    """
    n = len(lista_nodulos)
    if n == 0:
        return []
    if n == 1:
        return [[lista_nodulos[0]]]

    linhas, colunas = [], []
    for i in range(n):
        for j in range(i + 1, n):
            d = _distancia_min_mm(
                lista_nodulos[i]["mascara"],
                lista_nodulos[j]["mascara"],
                espacamento_mm,
            )
            if d <= tol_mm:
                linhas.append(i)
                colunas.append(j)

    grafo = csr_matrix(
        (np.ones(len(linhas)), (linhas, colunas)), shape=(n, n)
    )
    n_clusters, rotulos = connected_components(grafo, directed=False)

    clusters = [[] for _ in range(n_clusters)]
    for idx, rot in enumerate(rotulos):
        clusters[rot].append(lista_nodulos[idx])

    return clusters


def gerar_consenso_votacao(cluster, shape_volume, fracao=0.5):
    """
    Consenso por votação dentro de um cluster (mesmo nódulo físico).
    fracao=0.5 -> pelo menos metade das anotações do cluster concordam.
    """
    contagem = np.zeros(shape_volume, dtype=np.uint8)
    for info in cluster:
        contagem += info["mascara"]
    minimo_votos = max(1, int(np.ceil(fracao * len(cluster))))
    return (contagem >= minimo_votos).astype(np.uint8)


TIMEOUT_STAPLE_SEGUNDOS = 60


def gerar_consenso_staple(cluster, shape_volume, limiar=0.5):
    """
    Consenso via SimpleITK STAPLE dentro de um cluster.
    Cai de volta para votação simples se o STAPLE falhar (ex.: cluster
    com uma única anotação, máscaras idênticas/degeneradas) OU DEMORAR
    DEMAIS — o STAPLE é um algoritmo iterativo (EM) que pode não
    convergir (travar de verdade, sem lançar erro) em entradas
    degeneradas, como clusters com 2+ anotações do MESMO radiologista
    (sinal de que o agrupamento por proximidade uniu anotações demais —
    vale revisar TOL_CLUSTER_MM depois se isso acontecer com frequência).

    Roda o STAPLE numa thread DAEMON separada com timeout: um try/except
    sozinho não pega travamento (só pega erro), e usar
    ThreadPoolExecutor sem daemon=True trava o processo inteiro no
    final mesmo depois do timeout ser capturado (a thread não-daemon
    impede o processo de encerrar). daemon=True resolve isso — se a
    thread nunca terminar, ela é simplesmente descartada quando o
    processo principal encerra, sem bloquear nada.
    """
    if len(cluster) == 1:
        return cluster[0]["mascara"].copy()

    try:
        imgs_sitk = [
            sitk.GetImageFromArray(info["mascara"].astype(np.int16))
            for info in cluster
        ]

        resultado_container = {}

        def _alvo():
            try:
                staple = sitk.STAPLEImageFilter()
                resultado_container["saida"] = staple.Execute(imgs_sitk)
            except Exception as e:
                resultado_container["erro"] = e

        thread = threading.Thread(target=_alvo, daemon=True)
        thread.start()
        thread.join(timeout=TIMEOUT_STAPLE_SEGUNDOS)

        if thread.is_alive():
            print("  [AVISO] STAPLE excedeu %ds (provável não-convergência "
                  "— cluster com anotações degeneradas/duplicadas do mesmo "
                  "radiologista?) — usando votação como fallback."
                  % TIMEOUT_STAPLE_SEGUNDOS)
            return gerar_consenso_votacao(cluster, shape_volume, fracao=0.5)

        if "erro" in resultado_container:
            raise resultado_container["erro"]

        resultado = resultado_container["saida"]
        prob = sitk.GetArrayFromImage(resultado)
        return (prob >= limiar).astype(np.uint8)
    except Exception as e:
        print("  [AVISO] STAPLE falhou (%s) — usando votação como fallback" % e)
        return gerar_consenso_votacao(cluster, shape_volume, fracao=0.5)


def calcular_metricas_mascara(mascara, espacamento_mm):
    """
    Calcula métricas geométricas de uma máscara 3D:
    voxels, volume_mm3, centroide (z,y,x), bounding box,
    slice_inicial, slice_final, altura, largura, profundidade.
    """
    dx, dy, dz = espacamento_mm
    voxels     = int(mascara.sum())
    volume_mm3 = round(voxels * dx * dy * dz, 2)

    if voxels == 0:
        return {
            "voxels": 0, "volume_mm3": 0.0,
            "centroide_z": None, "centroide_y": None, "centroide_x": None,
            "slice_inicial": None, "slice_final": None,
            "altura_mm": None, "largura_mm": None, "profundidade_mm": None,
        }

    # Centroide
    cm = ndimage.center_of_mass(mascara)
    cz, cy, cx = float(cm[0]), float(cm[1]), float(cm[2])

    # Bounding box via slice_objects
    slices = ndimage.find_objects(mascara)[0] if mascara.max() > 0 else None

    # Slices com nódulo
    presenca_z = np.where(mascara.sum(axis=(1, 2)) > 0)[0]
    sl_ini = int(presenca_z[0])
    sl_fin = int(presenca_z[-1])

    # Dimensões em mm
    if slices:
        alt_mm  = (slices[1].stop - slices[1].start) * dy
        larg_mm = (slices[2].stop - slices[2].start) * dx
        prof_mm = (slices[0].stop - slices[0].start) * dz
    else:
        alt_mm = larg_mm = prof_mm = 0.0

    return {
        "voxels":         voxels,
        "volume_mm3":     volume_mm3,
        "centroide_z":    round(cz, 2),
        "centroide_y":    round(cy, 2),
        "centroide_x":    round(cx, 2),
        "slice_inicial":  sl_ini,
        "slice_final":    sl_fin,
        "altura_mm":      round(alt_mm, 2),
        "largura_mm":     round(larg_mm, 2),
        "profundidade_mm":round(prof_mm, 2),
    }


def salvar_nifti(array_3d, img_ref, caminho):
    img = sitk.GetImageFromArray(array_3d.astype(np.float32))
    img.CopyInformation(img_ref)
    sitk.WriteImage(img, caminho)


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

def processar_paciente(pasta_paciente, caminho_saida):
    nome = os.path.basename(pasta_paciente)
    print("\n%s\nProcessando: %s\n%s" % ("="*60, nome, "="*60))

    pasta_saida = os.path.join(caminho_saida, nome)
    os.makedirs(caminho_saida, exist_ok=True)
    os.makedirs(pasta_saida,   exist_ok=True)

    # 1. Encontrar arquivos
    lista_dcm, xml_path, arqs_seg, arqs_sr = encontrar_arquivos_serie(
        pasta_paciente)

    if not lista_dcm:
        print("  [AVISO] Nenhum .dcm CT encontrado")
        return None
    print("  %d arquivos CT encontrados" % len(lista_dcm))

    if xml_path:
        print("  XML: %s" % os.path.basename(xml_path))
    elif arqs_seg:
        print("  Sem XML — fallback DICOM-SEG (%d arqs) + SR (%d arqs)" % (
            len(arqs_seg), len(arqs_sr)))
    else:
        print("  [AVISO] Sem anotações (sem XML e sem SEG)")

    # 2. Carregar volume
    volume_hu, espacamento, mapa_z, img_sitk = carregar_volume_dicom(lista_dcm)

    # 3. Salvar volume
    if SALVAR_VOLUME_NIFTI:
        cam_vol = os.path.join(pasta_saida, "volume_tc.nii.gz")
        salvar_nifti(volume_hu, img_sitk, cam_vol)
        print("  Volume salvo: %s" % cam_vol)

    # 4. Parsear anotações
    nodulos = []
    if xml_path:
        nodulos = parsear_xml_lidc(xml_path, volume_hu.shape, mapa_z)
        print("  %d anotações (XML)" % len(nodulos))
    elif arqs_seg:
        nodulos = parsear_dicom_seg(arqs_seg, volume_hu.shape, mapa_z)
        print("  %d anotações (DICOM-SEG)" % len(nodulos))
        # Cruzar malignidade do SR
        if arqs_sr:
            carac_sr = extrair_caracteristicas_sr(arqs_sr)
            for info in nodulos:
                carac = carac_sr.get(info["nodulo_id"], {})
                info["malignidade"] = parsear_malignidade(
                    carac.get("Malignancy"))
                info["caracteristicas_extra"] = carac

    # 5. Agrupar anotações por nódulo físico (clustering espacial)
    clusters = agrupar_nodulos_por_proximidade(
        nodulos, espacamento, tol_mm=TOL_CLUSTER_MM)
    print("  %d anotações agrupadas em %d nódulo(s) físico(s) "
          "(tol=%.1fmm)" % (len(nodulos), len(clusters), TOL_CLUSTER_MM))

    # 6. Salvar máscaras individuais + calcular métricas
    resultados = []
    for cluster_id, cluster in enumerate(clusters, start=1):
        for info in cluster:
            mascara = validar_mascara(info["mascara"], volume_hu.shape)
            metricas = calcular_metricas_mascara(mascara, espacamento)

            if SALVAR_MASCARAS:
                sufixo  = "nodulo%02d_rad%s_ann%s" % (
                    cluster_id, info["radiologista"], info["nodulo_id"])
                cam_msk = os.path.join(
                    pasta_saida, "mascara_%s.nii.gz" % sufixo)
                salvar_nifti(mascara, img_sitk, cam_msk)
            else:
                cam_msk = ""

            print("  Salvo: cluster=%02d rad=%s nod=%s | vox=%d | "
                  "vol=%.1f mm³ | slices=%s→%s | mal=%d" % (
                cluster_id, info["radiologista"], info["nodulo_id"],
                metricas["voxels"], metricas["volume_mm3"],
                metricas["slice_inicial"], metricas["slice_final"],
                info["malignidade"]))

            resultados.append({
                "paciente":        nome,
                "cluster_id":      cluster_id,
                "n_anotacoes_cluster": len(cluster),
                "radiologista":    info["radiologista"],
                "nodulo_id":       info["nodulo_id"],
                "malignidade":     info["malignidade"],
                **metricas,
                "arquivo_mascara": cam_msk,
                "arquivo_volume":  os.path.join(
                    pasta_saida, "volume_tc.nii.gz"),
            })

    # 7. Gerar e salvar máscara de consenso — uma por nódulo físico (cluster)
    if GERAR_CONSENSO:
        for cluster_id, cluster in enumerate(clusters, start=1):
            if METODO_CONSENSO in ("votacao", "ambos"):
                cons_v = gerar_consenso_votacao(
                    cluster, volume_hu.shape, fracao=FRACAO_VOTOS_CONSENSO)
                met_v = calcular_metricas_mascara(cons_v, espacamento)
                cam_v = os.path.join(
                    pasta_saida,
                    "mascara_consenso_nodulo%02d_votacao.nii.gz" % cluster_id)
                salvar_nifti(cons_v, img_sitk, cam_v)
                print("  Consenso nódulo %02d (votação, frac>=%.2f, "
                      "n=%d anotações): %d voxels | %.1f mm³" % (
                    cluster_id, FRACAO_VOTOS_CONSENSO, len(cluster),
                    met_v["voxels"], met_v["volume_mm3"]))

            if METODO_CONSENSO in ("staple", "ambos"):
                cons_s = gerar_consenso_staple(
                    cluster, volume_hu.shape, limiar=LIMIAR_STAPLE)
                met_s = calcular_metricas_mascara(cons_s, espacamento)
                cam_s = os.path.join(
                    pasta_saida,
                    "mascara_consenso_nodulo%02d_staple.nii.gz" % cluster_id)
                salvar_nifti(cons_s, img_sitk, cam_s)
                print("  Consenso nódulo %02d (STAPLE, limiar=%.2f, "
                      "n=%d anotações): %d voxels | %.1f mm³" % (
                    cluster_id, LIMIAR_STAPLE, len(cluster),
                    met_s["voxels"], met_s["volume_mm3"]))

    return resultados


if __name__ == "__main__":
    import pandas as pd

    pastas = sorted([
        os.path.join(CAMINHO_BASE, d)
        for d in os.listdir(CAMINHO_BASE)
        if os.path.isdir(os.path.join(CAMINHO_BASE, d))
        and d.startswith("LIDC-IDRI")
    ])

    print("Pacientes encontrados: %d" % len(pastas))

    # Pula pacientes já processados (evita reprocessar os antigos quando
    # CAMINHO_BASE aponta para a pasta com TODOS os pacientes, antigos +
    # novos — identifica "já processado" pela presença de volume_tc.nii.gz
    # na pasta de saída daquele paciente).
    pastas_pendentes = []
    for pasta in pastas:
        nome = os.path.basename(pasta)
        cam_volume_existente = os.path.join(CAMINHO_SAIDA, nome, "volume_tc.nii.gz")
        if os.path.exists(cam_volume_existente):
            print("  [PULANDO] %s já processado" % nome)
        else:
            pastas_pendentes.append(pasta)
    print("Pacientes pendentes de processamento: %d" % len(pastas_pendentes))

    todos = []
    for pasta in pastas_pendentes:
        resultado = processar_paciente(pasta, CAMINHO_SAIDA)
        if resultado:
            todos.extend(resultado)

    if SALVAR_CSV and todos:
        df_novos = pd.DataFrame(todos)
        cam_csv = os.path.join(CAMINHO_SAIDA, "indice_nodulos.csv")

        # Mescla com o CSV existente em vez de sobrescrever — preserva
        # as anotações dos pacientes já processados em rodadas anteriores.
        if os.path.exists(cam_csv):
            df_existente = pd.read_csv(cam_csv)
            pacientes_novos = set(df_novos["paciente"].unique())
            df_existente = df_existente[~df_existente["paciente"].isin(pacientes_novos)]
            df = pd.concat([df_existente, df_novos], ignore_index=True)
            print("\nCSV mesclado: %d anotações já existentes + %d novas" % (
                len(df_existente), len(df_novos)))
        else:
            df = df_novos

        df.to_csv(cam_csv, index=False)
        print("CSV salvo: %s" % cam_csv)
        print("Total de anotações no CSV: %d (%d pacientes)" % (
            len(df), df["paciente"].nunique()))
        colunas = ["paciente","cluster_id","radiologista","nodulo_id",
                   "malignidade","volume_mm3","slice_inicial","slice_final"]
        print(df_novos[colunas].to_string())
    else:
        print("\nNenhum paciente novo processado (todos já existiam, ou "
              "nenhum resultado gerado).")
