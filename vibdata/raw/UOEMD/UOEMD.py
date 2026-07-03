import os
import glob
import numpy as np
import pandas as pd
import scipy.io as sio
from vibdata.raw.base import RawVibrationDataset

# Mapeamentos baseados na descrição do Mendeley
SPEED_MAP = {
    '1': '15Hz', '2': '30Hz', '3': '45Hz', '4': '60Hz',
    '5': 'Inc_15_to_45Hz', '6': 'Inc_30_to_60Hz',
    '7': 'Dec_45_to_15Hz', '8': 'Dec_60_to_30Hz'
}

LOAD_MAP = {
    '0': 'No_Load', 
    '1': 'Loaded'
}

class UOEMD_raw:    
    def __init__(self, root_dir, download=False):
        super().__init__(root_dir=root_dir)
        self.dataset_dir = os.path.join(root_dir, "UOEMD_raw")
        
        # Encontra todos os arquivos .mat de forma recursiva
        self.files = glob.glob(os.path.join(self.dataset_dir, "**/*.mat"), recursive=True)
        
        if len(self.files) == 0:
            print(f"Aviso: Nenhum arquivo encontrado em {self.dataset_dir}.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        file_name = os.path.basename(file_path).split('.')[0] # Exemplo: 'H-H-1-0'
        
        # 1. Extração do Sinal
        mat_data = sio.loadmat(file_path)
        raw_signal = None
        
        # Procura a variável principal dentro do .mat ignorando metadados do matlab
        for key in mat_data.keys():
            if not key.startswith('__'):
                # Extrai todas as linhas e apenas a Coluna 0 (Acelerômetro 1)
                raw_signal = mat_data[key][:, 0] 
                break
                
        # 2. Extração Precisa de Metadados via Nome do Arquivo
        # Quebra o 'H-H-1-0' em ['H', 'H', '1', '0']
        parts = file_name.split('-')
        
        if len(parts) >= 4:
            class_code = f"{parts[0]}-{parts[1]}" # Ex: 'H-H' ou 'S-W'
            speed_code = parts[2]
            load_code = parts[3]
            
            fault_class = class_code
            speed_val = SPEED_MAP.get(speed_code, 'Unknown')
            load_val = LOAD_MAP.get(load_code, 'Unknown')
        else:
            fault_class, speed_val, load_val = "Unknown", "Unknown", "Unknown"
        
        # O dicionário padronizado da vibdata
        metainfo = {
            'dataset': 'UOEMD',
            'file_name': file_name + '.mat',
            'label': fault_class,       # Usado para classificação de defeito
            'speed': speed_val,         # Usado para Cross-Validation (Leave-One-Speed-Out)
            'load': load_val,           # Usado para Cross-Validation (Leave-One-Load-Out)
            'sample_rate': 42000        # Dado oficial do Mendeley
        }

        return {"signal": raw_signal, "metainfo": metainfo}
