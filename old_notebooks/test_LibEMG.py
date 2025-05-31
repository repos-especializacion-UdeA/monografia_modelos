import os
import sys
import scipy.io  # Asegúrate de importar scipy.io para cargar archivos .mat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Agregar el directorio raíz del proyecto al PYTHONPATH

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from globals import RAW_DATA_DIR


if __name__ == '__main__':
    signal_data = load_mat_file(RAW_DATA_DIR, 'S1_A1_E1.mat')
 





    
