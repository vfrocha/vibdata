import os
import glob
import pandas as pd
from vibdata.raw.base import RawVibrationDataset

# H = Healthy, B = Broken Tooth, M = Missing/Chipped Tooth
FAULT_MAP = {
    'H': 'Healthy',
    'B': 'Broken',
    'M': 'Missing'
}

class HUST_Gearbox_raw(RawVibrationDataset):
    """
    Carregador Nativo para o HUST Gearbox Dataset.
    Lê arquivos .txt separados por espaços/tabs dinamicamente.
    """
    def __init__(self, root_dir, download=False):
        super().__init__()
        self.root_dir = root_dir
        # A pasta base que configuramos no seu download_all.py
        self.dataset_dir = os.path.join(root_dir, "HUST_Gearbox_raw")
        
        # Procura por arquivos .txt em qualquer subpasta
        self.files = glob.glob(os.path.join(self.dataset_dir, "**/*.txt"), recursive=True)
        
        if len(self.files) == 0:
            print(f"[AVISO] Nenhum arquivo .txt encontrado em {self.dataset_dir}.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        file_name_full = os.path.basename(file_path)
        file_name_no_ext = file_name_full.replace('.txt', '')
        
        # 1. Carrega o sinal bruto do .txt
        try:
            # O truque 'names' força o Pandas a aceitar linhas irregulares sem quebrar (Error tokenizing data)
            # 'sep='\s+'' é o método mais robusto para separar por qualquer quantidade de espaços/tabs
            df = pd.read_csv(file_path, header=None, sep='\s+', names=[0, 1, 2, 3, 4])
            
            # Pegamos apenas a 1ª coluna (índice 0) que contém a vibração e ignoramos falhas (dropna)
            raw_signal = df[0].dropna().values 
            
            # Segurança extra: Garante que pegamos os 262144 pontos descritos no artigo
            if len(raw_signal) > 262144:
                raw_signal = raw_signal[:262144]
                
        except Exception as e:
            print(f"Erro ao carregar o arquivo {file_path}: {e}")
            return {"signal": None, "metainfo": None}
            
        # 2. Extração Inteligente da Condição e Rótulo
        # O nome do arquivo é o Rótulo_Condição (ex: "B_20_1" -> Broken, Condição "20_1")
        parts = file_name_no_ext.split('_')
        
        fault_code = parts[0]
        fault_class = FAULT_MAP.get(fault_code, "Unknown")
        
        # A condição é todo o resto do nome do arquivo após o código da classe
        # Isso garante que "20_1" (20Hz, carga 1) seja uma pasta diferente de "20_2"
        if len(parts) > 1:
            condition = "_".join(parts[1:]) 
        else:
            condition = "Unknown"
        
        meta = {
            'dataset': 'HUST_Gearbox',
            'file_name': file_name_full,
            'label': fault_class,
            'condition': condition,
            # Documentação Oficial: Frequência de 25.6 kHz
            'sample_rate': 25600 
        }

        return {"signal": raw_signal, "metainfo": meta}
