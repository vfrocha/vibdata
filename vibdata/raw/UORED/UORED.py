import os
import glob
import scipy.io as sio
from vibdata.raw.base import RawVibrationDataset, DownloadableDataset

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

# 2. IMPORTANTE: Herança dupla
class UORED_raw(RawVibrationDataset, DownloadableDataset):
    """
    Carregador Moderno e Flexível para o UORED.
    Extrai metadados (Bearing ID e Stage) diretamente dos nomes dos arquivos
    para garantir suporte à validação cruzada (Leave-One-Bearing-Out).
    """
    
    # 3. LISTAS DE MÚLTIPLOS DOWNLOADS
    # Colocamos os IDs na ordem correta (parte 1 e parte 2)
    urls = [
        "1SkPpD_BcF6YCnvGoYv7Z__gJzs8SnOza", # ID da parte 001
        "16Zaojba8PyRTENFMfCrHK41kgS_4mViY"  # ID da parte 002
    ]
    
    resources = [
        ("UORED_raw-20260526T191215Z-3-001.zip", None),
        ("UORED_raw-20260526T191215Z-3-002.zip", None)
    ]

    def __init__(self, root_dir: str, download: bool = False) -> None:
        
        # 4. LÓGICA DE DOWNLOAD
        if download:
            super().__init__(
                root_dir=root_dir,
                download_resources=UORED_raw.resources,
                download_urls=UORED_raw.urls,
                extract_files=True,
            )
        else:
            super().__init__(root_dir=root_dir, download_resources=UORED_raw.resources)
            
        self.root_dir = root_dir
        
        # 5. CORREÇÃO DE DIRETÓRIO (A mesma aplicada na base UOEMD)
        self.dataset_dir = self.raw_folder
        
        self.files = glob.glob(os.path.join(self.dataset_dir, "**/*.mat"), recursive=True)
        
        if len(self.files) == 0:
            print(f"[AVISO] Nenhum ficheiro .mat encontrado em {self.dataset_dir}. Certifique-se de usar download=True.")

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
