import sys
import os
import pandas as pd
import sqlite3
from datetime import datetime
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Fix Windows Unicode Output
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from src.database.db_manager import DBManager
from src.data.external.manager import ExternalDataManager
from src.data.external.mapper import TeamNameMapper

class OddsImporter:
    
    # Mapping Football-Data codes to Internal DB Tournament Names (Partial Match)
    TOURNAMENT_MAP = {
        'BRA': ['Brasileirão Série A', 'Brasileirão Betano'],
        'E0': ['Premier League'],
        'SP1': ['LaLiga', 'La Liga'],
        'D1': ['Bundesliga'],
        'I1': ['Serie A', 'Italy Serie A'],
        'F1': ['Ligue 1']
    }
    
    def __init__(self):
        self.db = DBManager()
        self.manager = ExternalDataManager()
        self.mapper = TeamNameMapper()
        
    def run(self):
        print("="*60)
        print("💰 IMPORTADOR DE ODDS HISTÓRICAS (Multi-League)")
        print("="*60)
        
        # Ensure DB has columns
        self.db.create_tables() 
        
        for code, names in self.TOURNAMENT_MAP.items():
            self._process_league(code, names)
            
    def _process_league(self, league_code, db_names):
        print(f"\n🔄 Processando Liga: {league_code} ({db_names[0]})...")
        
        # 1. Carrega CSVs
        df_ext = self.manager.load_combined_data(league_code)
        if df_ext.empty:
            print("   ⚠️ Sem dados externos disponíveis.")
            return

        # Normaliza colunas de data e times
        # Rename Home/Away to HomeTeam/AwayTeam if necessary (BRA.csv uses mismatched names)
        df_ext = df_ext.rename(columns={'Home': 'HomeTeam', 'Away': 'AwayTeam'})

        try:
            # Football-Data usa Date (dd/mm/yy ou dd/mm/yyyy)
            df_ext['Date'] = pd.to_datetime(df_ext['Date'], dayfirst=True, errors='coerce')
            df_ext = df_ext.dropna(subset=['Date'])
        except Exception as e:
            print(f"   ❌ Erro convertendo datas: {e}")
            return

        # 2. Identifica Torneio no Banco
        conn = self.db.connect()
        # Busca ID do torneio (tenta match por nome)
        t_id = None
        for name in db_names:
            query = f"SELECT tournament_id, tournament_name FROM matches WHERE tournament_name LIKE '%{name}%' LIMIT 1"
            res = pd.read_sql_query(query, conn)
            if not res.empty:
                t_id = res.iloc[0]['tournament_id']
                real_name = res.iloc[0]['tournament_name']
                print(f"   ✅ Encontrado no DB: {real_name} (ID: {t_id})")
                break
        
        if not t_id:
            print(f"   ⚠️ Liga '{db_names}' não encontrada no banco de dados local. Pulando.")
            return

        # 3. Carrega Jogos do Banco
        query_matches = f"""
            SELECT match_id, start_timestamp, home_team_name, away_team_name, home_score, away_score
            FROM matches 
            WHERE tournament_id = {t_id} AND status = 'finished'
        """
        df_int = pd.read_sql_query(query_matches, conn)
        
        if df_int.empty:
            print("   ⚠️ Nenhum jogo finalizado encontrado no banco para esta liga.")
            return

        print(f"   📊 Comparando {len(df_ext)} registros externos com {len(df_int)} jogos internos...")

        # 4. Mapeamento de Times (Entity Resolution)
        # Coleta listas únicas
        ext_teams = pd.concat([df_ext['HomeTeam'], df_ext['AwayTeam']]).unique()
        int_teams = pd.concat([df_int['home_team_name'], df_int['away_team_name']]).unique()
        
        self.mapper.auto_map_league(ext_teams, int_teams, league_code)
        
        # 5. Matching e Importação
        updated_count = 0
        updates = []
        
        # Otimização: Criar lookup table por data para o banco interno
        # Convert timestamp to date string YYYY-MM-DD
        df_int['date_str'] = pd.to_datetime(df_int['start_timestamp'], unit='s').dt.strftime('%Y-%m-%d')
        df_ext['date_str'] = df_ext['Date'].dt.strftime('%Y-%m-%d')
        
        # Itera sobre dados EXTERNOS (pois costumam ser a fonte da verdade para odds)
        # e tenta encontrar no banco interno
        
        for idx, row in df_ext.iterrows():
            ext_date = row['date_str']
            ext_home = row['HomeTeam']
            ext_away = row['AwayTeam']
            
            int_home = self.mapper.get_internal_name(ext_home, int_teams) # Implement method wrapper
            int_away = self.mapper.get_internal_name(ext_away, int_teams)
            
            # Debug only Brazil (limit log volume)
            if league_code == 'BRA' and idx > len(df_ext) - 50: # Check recent games 
                print(f"   🔎 Debug Row: {ext_date} | {ext_home}->{int_home} vs {ext_away}->{int_away}")

            if not int_home or not int_away:
                continue
                
            # Busca match no DB com tolerância de +/- 1 dia
            match = pd.DataFrame()
            
            # Check current date, next day, prev day
            for delta in [0, 1, -1]:
                 target_date = (row['Date'] + pd.Timedelta(days=delta)).strftime('%Y-%m-%d')
                 candidates = df_int[
                    (df_int['date_str'] == target_date) & 
                    (df_int['home_team_name'] == int_home) & 
                    (df_int['away_team_name'] == int_away)
                 ]
                 if not candidates.empty:
                     match = candidates
                     break
            
            if len(match) == 1:
                match_id = match.iloc[0]['match_id']
                
                # Extract Odds (Priority: Bet365 > Pinnacle > Avg)
                o_h, o_d, o_a = None, None, None
                prov = None
                
                if 'B365H' in row and pd.notna(row['B365H']):
                    o_h, o_d, o_a = row['B365H'], row['B365D'], row['B365A']
                    prov = 'Bet365'
                elif 'PSH' in row and pd.notna(row['PSH']):
                    o_h, o_d, o_a = row['PSH'], row['PSD'], row['PSA']
                    prov = 'Pinnacle'
                elif 'AvgH' in row and pd.notna(row['AvgH']):
                    o_h, o_d, o_a = row['AvgH'], row['AvgD'], row['AvgA']
                    prov = 'Average'
                
                if prov:
                    updates.append((float(o_h), float(o_d), float(o_a), prov, int(match_id)))
                    updated_count += 1
        # Batch Update
        if updates:
            cursor = conn.cursor()
            cursor.executemany("""
                UPDATE matches 
                SET odds_home = ?, odds_draw = ?, odds_away = ?, odds_provider = ?
                WHERE match_id = ?
            """, updates)
            conn.commit()
            print(f"   💾 Atualizados {updated_count} jogos com Odds ({prov}).")
        else:
            print("   ⚠️ Nenhuma correspondência de odds encontrada.")
            
        # conn.close() - Do not close here, let manager handle it or close at end of run

    # Wrapper helper for mapper that uses the list already loaded
    def _get_internal_name_wrapper(self, ext_name, candidates):
        # We assume mapper has finding logic
        return self.mapper.find_match(ext_name, candidates)

if __name__ == "__main__":
    # Monkey patch wrapper because I was lazy in class def
    TeamNameMapper.get_internal_name = TeamNameMapper.find_match 
    
    importer = OddsImporter()
    importer.run()
    importer.db.close()
