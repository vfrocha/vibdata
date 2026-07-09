import os
import glob
import pandas as pd
from vibdata.raw.base import RawVibrationDataset

class Electric_Motor_raw(RawVibrationDataset):
    """
    Carregador Nativo para o Electric Motor Vibrations Dataset (Zenodo).
    Lê os arquivos CSV contendo os eixos AccX, AccY e AccZ, classificando
    as falhas dinamicamente através do nome do arquivo (01 a 30).
    """
    def __init__(self, root_dir, download=False):
        super().__init__()
        self.root_dir = root_dir
        # Com base no seu log de terminal, os dados estão na pasta EMP_raw
        self.dataset_dir = os.path.join(root_dir, "EMP_raw")
        
        self.files = glob.glob(os.path.join(self.dataset_dir, "**/*.csv"), recursive=True)
        
        if len(self.files) == 0:
            print(f"[AVISO] Nenhum arquivo .csv encontrado em {self.dataset_dir}.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        file_name_full = os.path.basename(file_path)
        file_name_no_ext = file_name_full.replace('.csv', '')
        
        # 1. Carrega o sinal bruto do .csv
        try:
            # O Pandas lê automaticamente a 1ª linha como cabeçalho
            df = pd.read_csv(file_path)
            
            # Para o pipeline 1D, vamos focar no Eixo X (AccX).
            # Para testar os outros, basta mudar para 'AccY' ou 'AccZ'.
            if 'AccX' in df.columns:
                raw_signal = df['AccX'].dropna().values 
            else:
                # Fallback caso a coluna mude de nome: pega a coluna de índice 1
                raw_signal = df.iloc[:, 1].dropna().values
                
        except Exception as e:
            print(f"Erro ao carregar o arquivo {file_path}: {e}")
            return {"signal": None, "metainfo": None}
            
        # 2. Extração Inteligente da Rótulo (Label) via Palavras-Chave
        name_lower = file_name_no_ext.lower()
        
        is_mechanical = 'umbalanced' in name_lower or 'imbalanced' in name_lower or 'misalignment' in name_lower
        is_electrical = 'electrical' in name_lower or 'ohm' in name_lower
        
        if is_mechanical and is_electrical:
            fault_class = 'Combined_Fault'
        elif is_mechanical:
            fault_class = 'Mechanical_Fault'
        elif is_electrical:
            fault_class = 'Electrical_Fault'
        else:
            fault_class = 'Healthy'
        
        # A condição exata do experimento é o próprio nome do arquivo (ex: "01 - m1_half_shaft_speed...")
        # Limpamos espaços para evitar problemas na criação das pastas
        condition_safe = file_name_no_ext.replace(' ', '_').replace('-', '_')
        
        meta = {
            'dataset': 'Electric_Motor',
            'file_name': file_name_full,
            'label': fault_class,
            'condition': condition_safe,
            # Documentação Oficial: Frequência de 50.000 Hz (50 kHz)
            'sample_rate': 50000 
        }

        return {"signal": raw_signal, "metainfo": meta}
