# Monografia Ninapro DB1

## Lista de notebooks

A continuación se listan cada uno de los notebooks:

1. **Combinación del dataset**: [00_combinacion_dataset.ipynb](00_combinacion_dataset.ipynb)
2. **Selección de posturas**: [01_seleccion_posturas_dataset.ipynb](01_seleccion_posturas_dataset.ipynb)
3. **Analisis estadistico de las señales**: 
   * **EDA señales crudas**: [02_posturas_dataset_EDA.ipynb](02_posturas_dataset_EDA.ipynb)
   * **EDA señales filtradas**: [02_filtered_posturas_dataset_EDA.ipynb](02_filtered_posturas_dataset_EDA.ipynb)
4. **Extracción de caracteristias**: [03_features_extraction.ipynb](03_features_extraction.ipynb)
5. **Analisis estadistico de las caracteristicas extraidas**: [04_features_EDA.ipynb](04_features_EDA.ipynb)
6. **Modelos por caracteristica**: 
   * **Modelos RMS**: [05_models_RMS.ipynb](05_models_RMS.ipynb)
   * **Modelos WL**: [05_models_WL.ipynb](05_models_WL.ipynb)
   * **Modelos IAV**: [05_models_IAV.ipynb](05_models_IAV.ipynb)
   * **Modelo de clustering para el RMS**: [05_models_cluster.ipynb](05_models_cluster.ipynb)
   * **Modelo CNN para el RMS**: [05_models_CNN.ipynb](05_models_CNN.ipynb)

## Para trabajo futuro

Ensayar con otros otros movimientos teniendo en cuenta la siguiente información:

| **Grupo Funcional**                     | **Números de Movimientos** | **Movimientos Seleccionados** |
| --------------------------------------- | -------------------------- | - |
| **1. Movimientos bilaterales**          | 1, 2                       |   |
| **2. Interacción con objetos**          | 3, 4, 5, 6                 |   |
| **3. Uso de utensilios**                | 7, 8, 9, 10                |   |
| **4. Higiene personal**                 | 11, 12, 13, 14             |   |
| **5. Actividades cotidianas simples**   | 15, 16, 17, 18             |   |
| **6. Movimientos funcionales diversos** | 19, 20, 21, 22, 23         |   |

Mejorar el pipeline de trabajo emploando MLOPS.
* https://prsdm.github.io/mlops-project/
