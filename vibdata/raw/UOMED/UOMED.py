import os
import glob
import numpy as np
import pandas as pd
import scipy.io as sio
from vibdata.raw.base import RawVibrationDataset

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
        
        # 1. Carrega o sinal bruto do Matlab
        mat_data = sio.loadmat(file_path)
        signal = None
        for key in mat_data.keys():
            if not key.startswith('__'):
                signal = mat_data[key]
                break
                
        # 2. Lógica de extração de Metadados (Ajustaremos quando virmos os nomes reais dos arquivos)
        folder_name = os.path.basename(os.path.dirname(file_path))
        file_name = os.path.basename(file_path).lower()
        
        speed = "unknown" 
        load = "unknown"  
        label = "Class_X" 
        
        if 'normal' in file_name or 'healthy' in file_name or '0' in file_name:
            label = "Class_Normal"

        meta = {
            'dataset': 'UOEMD',
            'sample_rate': 50000, # Valor temporário (vamos auditar em breve)
            'speed': speed,
            'load': load,
            'label': label
        }

        return {'signal': signal, 'metainfo': pd.DataFrame([meta])}
