import os
import requests
import pandas as pd
from datetime import datetime
import time

class ExternalDataManager:
    """
    Gerencia o download e carregamento de dados do Football-Data.co.uk.
    Suporta múltiplas ligas e temporadas históricas.
    """
    
    BASE_URL = "https://www.football-data.co.uk"
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'raw')
    
    # Configuração das Ligas Suportadas
    LEAGUES = {
        'BRA': {
            'name': 'Brasileirão Série A',
            'type': 'single_file', # Arquivo único contendo histórico
            'url': 'https://www.football-data.co.uk/new/BRA.csv',
            'code': 'BRA'
        },
        'E0': {
            'name': 'Premier League',
            'type': 'season_split', # Arquivos divididos por temporada
            'code': 'E0',
            'seasons': 10 # Últimas X temporadas
        },
        'SP1': {
            'name': 'La Liga',
            'type': 'season_split',
            'code': 'SP1',
            'seasons': 10
        },
        'D1': {
            'name': 'Bundesliga',
            'type': 'season_split',
            'code': 'D1',
            'seasons': 10
        },
        'I1': {
            'name': 'Serie A',
            'type': 'season_split',
            'code': 'I1',
            'seasons': 10
        },
        'F1': {
            'name': 'Ligue 1',
            'type': 'season_split',
            'code': 'F1',
            'seasons': 10
        }
    }
    
    def __init__(self):
        os.makedirs(self.DATA_DIR, exist_ok=True)
        
    def _get_season_string(self, year_start):
        """
        Converte ano inicial (ex: 2023) para formato URL (ex: '2324').
        """
        y1 = year_start % 100
        y2 = (year_start + 1) % 100
        return f"{y1:02d}{y2:02d}"

    def download_data(self, league_key):
        """
        Baixa os dados para a liga especificada.
        """
        config = self.LEAGUES.get(league_key)
        if not config:
            raise ValueError(f"Liga {league_key} não suportada.")
            
        print(f"📥 Baixando dados para: {config['name']}...")
        
        if config['type'] == 'single_file':
            self._download_file(config['url'], f"{league_key}.csv")
            
        elif config['type'] == 'season_split':
            current_year = datetime.now().year
            # Se estamos no meio/fim do ano (Agosto+), a temporada atual começou este ano.
            # Se Janeiro-Julho, começou ano passado.
            # Simples: baixar ultimos X anos a partir de "agora" regressivamente
            
            # Ajuste ano base: temporada 24/25 começa em 2024.
            base_year = current_year if datetime.now().month >= 7 else current_year - 1
            
            for i in range(config['seasons']):
                start_year = base_year - i
                season_str = self._get_season_string(start_year)
                url = f"{self.BASE_URL}/mmz4281/{season_str}/{config['code']}.csv"
                success = self._download_file(url, f"{league_key}_{season_str}.csv")
                
                if not success:
                    print(f"   ⚠️ Temporada {season_str} não encontrada ou falha.")
                
                time.sleep(0.5) # Politeness
                
    def _download_file(self, url, filename):
        path = os.path.join(self.DATA_DIR, filename)
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                with open(path, 'wb') as f:
                    f.write(r.content)
                print(f"   ✅ Arquivo salvo: {filename}")
                return True
            else:
                print(f"   ❌ Erro {r.status_code} ao baixar {url}")
                return False
        except Exception as e:
            print(f"   ❌ Exceção: {e}")
            return False

    def load_combined_data(self, league_key):
        """
        Carrega e combina todos os CSVs disponíveis para uma liga.
        """
        all_dfs = []
        files = [f for f in os.listdir(self.DATA_DIR) if f.startswith(league_key)]
        
        if not files:
            print(f"⚠️ Nenhum arquivo local encontrado para {league_key}. Execute download_data primeiro.")
            return pd.DataFrame()
            
        print(f"📂 Carregando {len(files)} arquivos para {league_key}...")
        
        for f in files:
            p = os.path.join(self.DATA_DIR, f)
            try:
                # encoding='latin1' é comum no Football-Data antigo, 'utf-8' ou 'cp1252' pode variar
                # Vamos tentar ler, tratar erros de encoding se necessário
                try:
                    df = pd.read_csv(p, encoding='cp1252') # Padrão Excel Windows/Europa
                except UnicodeDecodeError:
                     df = pd.read_csv(p, encoding='utf-8')
                
                # Normaliza colunas se necessário (Date format varia)
                all_dfs.append(df)
            except Exception as e:
                print(f"   ❌ Erro ao ler {f}: {e}")
                
        if not all_dfs:
            return pd.DataFrame()
            
        combined = pd.concat(all_dfs, ignore_index=True)
        print(f"   📊 Total de linhas carregadas: {len(combined)}")
        return combined

if __name__ == "__main__":
    # Teste rápido
    manager = ExternalDataManager()
    # manager.download_data('BRA')
    # df = manager.load_combined_data('BRA')
    # print(df.head())
