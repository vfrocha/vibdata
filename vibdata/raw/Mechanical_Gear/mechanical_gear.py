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
            
        # Lista onde guardaremos os "pedaços" separados por condição
        self.samples = []
        
        # Mapeamento Dinâmico no momento da inicialização
        if len(self.files) > 0:
            print(f"-> Mapeando {len(self.files)} arquivos gigantes do Mechanical Gear. Isso pode levar alguns segundos...")
            for file_path in self.files:
                file_name_full = os.path.basename(file_path)
                file_name_no_ext = file_name_full.replace('.csv', '')
                
                # O nome do arquivo define a falha (ex: 'no_fault', 'root_crack')
                fault_class = file_name_no_ext.replace('_', ' ').title()
                if fault_class.lower() == 'no fault':
                    fault_class = 'Healthy'
                    
                try:
                    # Lê o CSV gigante (low_memory=False previne o DtypeWarning)
                    df = pd.read_csv(file_path, header=None, low_memory=False)
                    
                    # ---------------------------------------------------------
                    # LIMPEZA BLINDADA: Força as colunas alvo a serem números.
                    # Se houver texto (cabeçalhos perdidos, erros de sensor),
                    # o 'coerce' transforma em NaN.
                    # ---------------------------------------------------------
                    df[2] = pd.to_numeric(df[2], errors='coerce') # Sinal (X)
                    df[4] = pd.to_numeric(df[4], errors='coerce') # Velocidade
                    df[5] = pd.to_numeric(df[5], errors='coerce') # Carga
                    
                    # Removemos qualquer linha corrompida que virou NaN
                    df = df.dropna(subset=[2, 4, 5])
                    
                    # Coluna 4 = Speed, Coluna 5 = Load
                    # O groupby agrupa instantaneamente todas as linhas que têm a mesma velocidade e carga!
                    for (speed, load), group in df.groupby([4, 5]):
                        
                        # Extraímos apenas a Coluna 2 (Sensor 1 - Eixo X)
                        signal_chunk = group[2].values
                        
                        # Guardamos este pedaço específico como uma amostra independente
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
        # Em vez de ler do disco, já devolvemos o pedaço de memória pré-carregado!
        item = self.samples[idx]
        
        speed = item['speed']
        load = item['load']
        
        meta = {
            'dataset': 'Mechanical_Gear',
            'file_name': f"{item['file_name']}_Spd_{speed}_Load_{load}",
            'label': item['class'],
            'condition': f"{speed}Hz_{load}Nm",
            # Documentação Oficial: Frequência de 5.000 Hz (0.0002s)
            'sample_rate': 5000 
        }

        return {"signal": item['signal'], "metainfo": meta}
