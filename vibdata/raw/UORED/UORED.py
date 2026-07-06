import os
import glob
import scipy.io as sio
from vibdata.raw.base import RawVibrationDataset

FAULT_MAP = {
    '1_healthy': 'Normal',
    '2_inner_race_faults': 'Inner Race',
    '3_outer_race_faults': 'Outer Race',
    '4_ball_faults': 'Ball',
    '5_cage_faults': 'Cage'
}

# Mapeamento do estágio de falha baseado no último número do arquivo
STAGE_MAP = {
    '0': 'healthy',
    '1': 'development',
    '2': 'fault'
}

class UORED_raw(RawVibrationDataset):
    """
    Carregador Moderno e Flexível para o UORED.
    Extrai metadados (Bearing ID e Stage) diretamente dos nomes dos arquivos
    para garantir suporte à validação cruzada (Leave-One-Bearing-Out).
    """
    def __init__(self, root_dir: str, download: bool = False) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.dataset_dir = os.path.join(root_dir, "UORED_raw")
        self.files = glob.glob(os.path.join(self.dataset_dir, "**/*.mat"), recursive=True)
        
        if len(self.files) == 0:
            print(f"[AVISO] Nenhum ficheiro .mat encontrado em {self.dataset_dir}.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        file_path = self.files[idx]
        file_name_full = os.path.basename(file_path)
        file_name_no_ext = file_name_full.replace('.mat', '')
        
        # 1. Carrega o sinal bruto do Matlab
        try:
            mat_data = sio.loadmat(file_path)
        except Exception as e:
            print(f"Erro ao carregar o ficheiro {file_path}: {e}")
            return {"signal": None, "metainfo": None}
            
        raw_signal = None
        for key in mat_data.keys():
            if not key.startswith('__'):
                try:
                    raw_signal = mat_data[key][:, 0] 
                except IndexError:
                    raw_signal = mat_data[key].flatten()
                break
                
        # 2. Extração da Classe Base (via nome da pasta)
        path_lower = file_path.lower()
        fault_class = "Unknown"
        for key_pattern, label in FAULT_MAP.items():
            if key_pattern in path_lower:
                fault_class = label
                break
                
        # 3. Extração Inteligente de Metadados (Bearing ID e Stage) via nome do arquivo
        # Exemplo: 'C_18_1' -> parts = ['C', '18', '1']
        parts = file_name_no_ext.split('_')
        
        bearing_id = "Unknown"
        stage_val = "unknown"
        
        if len(parts) >= 3:
            bearing_id = parts[1]
            stage_code = parts[2]
            stage_val = STAGE_MAP.get(stage_code, "unknown")
        
        meta = {
            'dataset': 'UORED',
            'file_name': file_name_full,
            'label': fault_class,
            
            # --- Informações Vitais para o make_dataset.py ---
            'bearing_id': bearing_id,
            'stage': stage_val,
            
            # Informações fixas baseadas no UORED.csv
            'load_N': 400,
            'rotation_hz': 29.166667,
            'sample_rate': 42000 
        }

        return {"signal": raw_signal, "metainfo": meta}
