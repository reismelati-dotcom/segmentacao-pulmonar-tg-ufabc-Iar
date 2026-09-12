 # Comparação entre Método Clássico de Processamento Digital de Imagens e Rede Neural para Segmentação de Nódulos Pulmonares em Tomografias Computadorizadas

Repositório oficial do Trabalho de Graduação desenvolvido na Fundação Universidade Federal do ABC (UFABC).

* **Autores:** Daniel Augusto de Carvalho e Leonardo Reis dos Santos Melati
* **Orientador:** Prof. Dr. Luiz Antonio Celiberto Junior
* **Curso:** Engenharia de Instrumentação, Automação e Robótica
* **Ano:** 2026

## Resumo do Projeto
Este repositório armazena os códigos e scripts computacionais utilizados para comparar o desempenho de um método clássico de Processamento Digital de Imagens (baseado em limiarização de Otsu, operações morfológicas, componentes conectados e filtros geométricos) e de uma Rede Neural Convolucional baseada na arquitetura U-Net na segmentação de nódulos pulmonares em exames de Tomografia Computadorizada (TC). O estudo utilizou o banco de dados público LIDC-IDRI, empregando máscaras de referência obtidas por consenso entre radiologistas (votação e método STAPLE).

## Principais Resultados
* **Qualidade de Segmentação:** A U-Net apresentou desempenho superior nos cortes contendo nódulos (Dice de 0,4537 e IoU de 0,3856), enquanto o método clássico obteve Dice de 0,1759 e IoU de 0,1319.
* **Capacidade de Detecção:** Em nível de nódulo, a U-Net detectou 75,9% das lesões (60/79), comparado a 49,4% (39/79) do método clássico, gerando proporcionalmente menos falsos positivos.
* **Complexidade Computacional:** O método clássico destacou-se pela velocidade, apresentando tempo médio de processamento por corte aproximadamente 8 vezes menor do que a U-Net.

## Estrutura dos Scripts
* `selecionar_pacientes.py`: Script automatizado com a biblioteca `pylidc` para filtragem e ranqueamento dos exames do LIDC-IDRI com base na espessura de corte (<= 2,0 mm), diâmetro mínimo do nódulo (>= 3,0 mm) e consenso de múltiplos especialistas.
* Pipelines de pré-processamento em Unidades Hounsfield (HU), calibração de limiares clássicos e treinamento da U-Net 2D utilizando a biblioteca MONAI.

## Como Acessar a Monografia Completa
O documento textual detalhado com a fundamentação teórica, metodologia, tabelas completas de resultados e discussões encontra-se anexado no repositório (`MONOGRAFIA_TG_UFABC_V1.pdf`).