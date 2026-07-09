import os
import glob
import pandas as pd
from vibdata.raw.base import RawVibrationDataset

class Mechanical_Gear_raw(RawVibrationDataset):
    """
    Carregador Nativo para o Mechanical Gear Vibration Dataset (Kaggle).
    Separa automaticamente os sinais baseando-se nas colunas de Velocidade e Carga
    embutidas no próprio arquivo CSV.
    """
    def __init__(self, root_dir, download=False):
        super().__init__()
        self.root_dir = root_dir
        self.dataset_dir = os.path.join(root_dir, "Gearbox_raw")
        
        self.files = glob.glob(os.path.join(self.dataset_dir, "**/*.csv"), recursive=True)
        
        if len(self.files) == 0:
            print(f"[AVISO] Nenhum arquivo .csv encontrado em {self.dataset_dir}.")
            
        self.samples = []
        
        if len(self.files) > 0:
            print(f"-> Mapeando {len(self.files)} arquivos do Mechanical Gear...")
            for file_path in self.files:
                file_name_full = os.path.basename(file_path)
                file_name_no_ext = file_name_full.replace('.csv', '')
                
                # O nome do arquivo define a falha
                fault_class = file_name_no_ext.replace('_', ' ').title()
                if fault_class.lower() == 'no fault':
                    fault_class = 'Healthy'
                    
                try:
                    df = pd.read_csv(file_path, header=None, low_memory=False)
                    
                    # ---------------------------------------------------------
                    # CORREÇÃO DE ÍNDICES: O Pandas conta a partir do 0!
                    # Coluna 2 (Sensor X)   -> Índice 1
                    # Coluna 4 (Velocidade) -> Índice 3
                    # Coluna 5 (Carga)      -> Índice 4
                    # ---------------------------------------------------------
                    df[1] = pd.to_numeric(df[1], errors='coerce') 
                    df[3] = pd.to_numeric(df[3], errors='coerce') 
                    df[4] = pd.to_numeric(df[4], errors='coerce') 
                    
                    # Removemos qualquer linha corrompida
                    df = df.dropna(subset=[1, 3, 4])
                    
                    # Agrupa instantaneamente pela velocidade e carga
                    for (speed, load), group in df.groupby([3, 4]):
                        
                        # Extraímos apenas a Coluna 2 (Sensor X)
                        signal_chunk = group[1].values
                        
                        print(f"      -> Chunk {fault_class} ({speed}Hz, {load}Nm): {len(signal_chunk)} pontos")
                        
                        if len(signal_chunk) < 500:
                            continue
                            
                        self.samples.append({
                            'signal': signal_chunk,
                            'speed': speed,
                            'load': load,
                            'class': fault_class,
                            'file_name': file_name_no_ext
                        })
                except Exception as e:
                    print(f"Erro ao processar {file_name_full}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        speed = item['speed']
        load = item['load']
        
        meta = {
            'dataset': 'Mechanical_Gear',
            'file_name': f"{item['file_name']}_Spd_{speed}_Load_{load}",
            'label': item['class'],
            'condition': f"{speed}Hz_{load}Nm",
            'sample_rate': 5000 
        }

        return {"signal": item['signal'], "metainfo": meta}
