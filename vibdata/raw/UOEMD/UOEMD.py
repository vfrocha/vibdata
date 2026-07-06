import os
import glob
import numpy as np
import scipy.io as sio
from vibdata.raw.base import RawVibrationDataset

# --- MAPEAMENTOS MENDELEY ---
SPEED_MAP = {
    '1': '15Hz', '2': '30Hz', '3': '45Hz', '4': '60Hz',
    '5': 'Inc_15_to_45Hz', '6': 'Inc_30_to_60Hz',
    '7': 'Dec_45_to_15Hz', '8': 'Dec_60_to_30Hz'
}

LOAD_MAP = {
    '0': 'No_Load', 
    '1': 'Loaded'
}

FAULT_MAP = {
    'H-H': 'Normal',
    'R-U': 'Unbalance',
    'R-M': 'Misalignment',
    'S-W': 'Stator Winding',
    'V-V': 'Voltage Imbalance',
    'K-A': 'Broken Rotor Bar',   
    'F-B': 'Faulty Bearing',     
    'B-R': 'Bowed Rotor'         
}

class UOEMD_raw(RawVibrationDataset):
    """
    Carregador Nativo para o University of Ottawa Electric Motor Dataset (UOEMD).
    """
    def __init__(self, root_dir, download=False):
        # Chama o super vazio para evitar o erro do object.__init__
        super().__init__()
        self.root_dir = root_dir
        self.dataset_dir = os.path.join(root_dir, "UOEMD_raw")
        
        # Busca recursiva para encontrar os .mat onde quer que estejam
        self.files = glob.glob(os.path.join(self.dataset_dir, "**/*.mat"), recursive=True)
        
        if len(self.files) == 0:
            print(f"Aviso: Nenhum arquivo encontrado em {self.dataset_dir}.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        file_name_full = os.path.basename(file_path)
        file_name = file_name_full.split('.')[0] # Ex: 'H-H-1-0'
        
        # 1. Carrega o sinal bruto do Matlab (Acelerômetro 1)
        mat_data = sio.loadmat(file_path)
        raw_signal = None
        for key in mat_data.keys():
            if not key.startswith('__'):
                # Extrai todas as linhas e apenas a Coluna 0 (Acelerômetro 1)
                raw_signal = mat_data[key][:, 0] 
                break
                
        # 2. Extração Precisa de Metadados via Nome do Arquivo
        parts = file_name.split('-')
        
        if len(parts) >= 4:
            class_code = f"{parts[0]}-{parts[1]}"
            speed_code = parts[2]
            load_code = parts[3]
            
            # Converte a sigla para o nome exato cadastrado no labels.csv
            fault_class = FAULT_MAP.get(class_code, class_code) 
            speed_val = SPEED_MAP.get(speed_code, 'Unknown')
            load_val = LOAD_MAP.get(load_code, 'Unknown')
        else:
            fault_class, speed_val, load_val = "Unknown", "Unknown", "Unknown"
        
        meta = {
            'dataset': 'UOEMD',
            'file_name': file_name_full,
            'label': fault_class,
            'speed': speed_val,
            'load': load_val,
            'sample_rate': 42000
        }

        return {"signal": raw_signal, "metainfo": meta}
