"""
Módulo de Gerenciamento de Banco de Dados.

Este módulo é o "Caderno" do sistema. É aqui que guardamos tudo o que aprendemos:
jogos passados, estatísticas detalhadas e nossas próprias previsões.
"""

import sqlite3
import pandas as pd
from datetime import datetime


class DBManager:
    """
    Gerenciador de banco de dados SQLite para o sistema de previsão de escanteios.
    """
    
    def __init__(self, db_path: str = "data/football_data.db"):
        self.db_path = db_path
        self.conn = None
        self.create_tables()

    def connect(self) -> sqlite3.Connection:
        if self.conn is None:
            # timeout=30.0 aumenta a tolerância para "database is locked"
            self.conn = sqlite3.connect(self.db_path, timeout=30.0)
        return self.conn

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None

    def create_tables(self) -> None:
        conn = self.connect()
        cursor = conn.cursor()

        # Tabela de Jogos
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS matches (
                match_id INTEGER PRIMARY KEY,
                tournament_name TEXT,
                season_id INTEGER,
                round INTEGER,
                status TEXT,
                start_timestamp INTEGER,
                home_team_id INTEGER,
                home_team_name TEXT,
                away_team_id INTEGER,
                away_team_name TEXT,
                home_score INTEGER,
                away_score INTEGER
            )
        ''')

        # Tabela de Estatísticas
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS match_stats (
                match_id INTEGER PRIMARY KEY,
                
                -- Escanteios
                corners_home_ft INTEGER,
                corners_away_ft INTEGER,
                corners_home_ht INTEGER,
                corners_away_ht INTEGER,
                
                -- Chutes
                shots_ot_home_ft INTEGER,
                shots_ot_away_ft INTEGER,
                shots_ot_home_ht INTEGER,
                shots_ot_away_ht INTEGER,
                
                -- Novas Estatísticas (Profissional)
                possession_home INTEGER,
                possession_away INTEGER,
                total_shots_home INTEGER,
                total_shots_away INTEGER,
                fouls_home INTEGER,
                fouls_away INTEGER,
                yellow_cards_home INTEGER,
                yellow_cards_away INTEGER,
                red_cards_home INTEGER,
                red_cards_away INTEGER,
                big_chances_home INTEGER,
                big_chances_away INTEGER,
                expected_goals_home REAL,
                expected_goals_away REAL,
                
                FOREIGN KEY(match_id) REFERENCES matches(match_id)
            )
        ''')

        # Tabela de Previsões (Feedback Loop)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id INTEGER,
                prediction_type TEXT, -- 'ML', 'Statistical'
                predicted_value REAL, -- ex: 9.5 escanteios
                market TEXT, -- ex: 'Over 9.5'
                probability REAL,
                odds REAL, -- Odd Justa
                category TEXT, -- 'Top7', 'Easy', 'Medium', 'Hard'
                market_group TEXT,
                status TEXT DEFAULT 'PENDING', -- 'PENDING', 'GREEN', 'RED'
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(match_id) REFERENCES matches(match_id)
            )
        ''')
        
        self._run_migrations(cursor)
        conn.commit()

    def _run_migrations(self, cursor):
        """Executa migrações de banco de dados seguras."""
        migrations = [
            "ALTER TABLE predictions ADD COLUMN odds REAL",
            "ALTER TABLE predictions ADD COLUMN category TEXT",
            "ALTER TABLE predictions ADD COLUMN market_group TEXT",
            "ALTER TABLE match_stats ADD COLUMN possession_home INTEGER",
            "ALTER TABLE match_stats ADD COLUMN possession_away INTEGER",
            "ALTER TABLE match_stats ADD COLUMN total_shots_home INTEGER",
            "ALTER TABLE match_stats ADD COLUMN total_shots_away INTEGER",
            "ALTER TABLE match_stats ADD COLUMN fouls_home INTEGER",
            "ALTER TABLE match_stats ADD COLUMN fouls_away INTEGER",
            "ALTER TABLE match_stats ADD COLUMN yellow_cards_home INTEGER",
            "ALTER TABLE match_stats ADD COLUMN yellow_cards_away INTEGER",
            "ALTER TABLE match_stats ADD COLUMN red_cards_home INTEGER",
            "ALTER TABLE match_stats ADD COLUMN red_cards_away INTEGER",
            "ALTER TABLE match_stats ADD COLUMN big_chances_home INTEGER",
            "ALTER TABLE match_stats ADD COLUMN big_chances_away INTEGER",
            "ALTER TABLE match_stats ADD COLUMN expected_goals_home REAL",
            "ALTER TABLE match_stats ADD COLUMN expected_goals_away REAL"
        ]
        
        for sql in migrations:
            try:
                cursor.execute(sql)
            except:
                pass 

    # --- NOVOS MÉTODOS PARA OTIMIZAÇÃO DO SCRAPER ---
    
    def get_season_stats(self, season_id: int) -> dict:
        """
        Retorna estatísticas da temporada para otimização do scraper.
        
        Útil para saber se a temporada já está completa ou onde parar.
        
        Args:
            season_id: ID da temporada.
            
        Returns:
            dict: {
                'total_matches': int (total de jogos salvos),
                'last_round': int (última rodada salva)
            }
        """
        conn = self.connect()
        cursor = conn.cursor()
        try:
            # Conta total de jogos e pega a maior rodada registrada
            cursor.execute('''
                SELECT COUNT(*), MAX(round) 
                FROM matches 
                WHERE season_id = ?
            ''', (season_id,))
            
            row = cursor.fetchone()
            total = row[0] if row[0] else 0
            last_round = row[1] if row[1] else 0
            
            return {'total_matches': total, 'last_round': last_round}
        except Exception as e:
            print(f"Erro ao buscar stats da temporada: {e}")
            return {'total_matches': 0, 'last_round': 0}

    # ------------------------------------------------

    def save_prediction(self, match_id: int, pred_type: str, value: float, 
                       market: str, prob: float, odds: float = 0.0, 
                       category: str = None, market_group: str = None, 
                       verbose: bool = False) -> None:
        conn = self.connect()
        cursor = conn.cursor()
        try:
            # Verifica se já existe previsão idêntica (proteção contra duplicatas)
            cursor.execute('''
                SELECT id FROM predictions 
                WHERE match_id = ? AND prediction_type = ? AND category = ? AND market = ?
            ''', (match_id, pred_type, category, market))
            
            existing = cursor.fetchone()
            if existing:
                if verbose:
                    print(f"⚠️ Previsão duplicada ignorada para jogo {match_id} ({category}/{market})")
                return  # Não salva duplicata

            cursor.execute('''
                INSERT INTO predictions (match_id, prediction_type, predicted_value, market, probability, odds, category, market_group)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (match_id, pred_type, value, market, prob, odds, category, market_group))
            conn.commit()
            if verbose:
                print(f"Previsão salva para o jogo {match_id}!")
        except Exception as e:
            print(f"Erro ao salvar previsão: {e}")

    def check_predictions(self) -> None:
        conn = self.connect()
        cursor = conn.cursor()
        
        query = '''
            SELECT p.id, p.match_id, p.market, p.market_group, p.predicted_value, 
                   s.corners_home_ft, s.corners_away_ft,
                   s.corners_home_ht, s.corners_away_ht,
                   m.home_team_name, m.away_team_name
            FROM predictions p
            JOIN matches m ON p.match_id = m.match_id
            JOIN match_stats s ON m.match_id = s.match_id
            WHERE p.status = 'PENDING' AND m.status = 'finished'
        '''
        
        pending = pd.read_sql_query(query, conn)
        
        if pending.empty:
            print("Nenhuma previsão pendente para verificar.")
            return
            
        print(f"Verificando {len(pending)} previsões pendentes...")
        
        for _, row in pending.iterrows():
            corners_home_ft = row['corners_home_ft']
            corners_away_ft = row['corners_away_ft']
            corners_home_ht = row['corners_home_ht']
            corners_away_ht = row['corners_away_ht']
            
            total_ft = corners_home_ft + corners_away_ft
            total_ht = corners_home_ht + corners_away_ht
            total_2t = total_ft - total_ht
            
            corners_home_2t = corners_home_ft - corners_home_ht
            corners_away_2t = corners_away_ft - corners_away_ht
            
            actual_value = 0
            market_group = row['market_group']
            
            if market_group == "JOGO COMPLETO":
                actual_value = total_ft
            elif market_group == "TOTAL MANDANTE":
                actual_value = corners_home_ft
            elif market_group == "TOTAL VISITANTE":
                actual_value = corners_away_ft
            elif market_group == "1º TEMPO (HT)":
                actual_value = total_ht
            elif market_group == "2º TEMPO":
                actual_value = total_2t
            elif market_group == "MANDANTE 1º TEMPO":
                actual_value = corners_home_ht
            elif market_group == "VISITANTE 1º TEMPO":
                actual_value = corners_away_ht
            elif market_group == "MANDANTE 2º TEMPO":
                actual_value = corners_home_2t
            elif market_group == "VISITANTE 2º TEMPO":
                actual_value = corners_away_2t
            else:
                actual_value = total_ft

            status = 'RED'
            
            try:
                if 'Over' in row['market']:
                    line = float(row['market'].split(' ')[1])
                    if actual_value > line:
                        status = 'GREEN'
                elif 'Under' in row['market']:
                    line = float(row['market'].split(' ')[1])
                    if actual_value < line:
                        status = 'GREEN'
            except Exception as e:
                print(f"Erro ao parsear mercado '{row['market']}': {e}")
                continue
            
            cursor.execute("UPDATE predictions SET status = ? WHERE id = ?", (status, row['id']))
            print(f"[{status}] Jogo {row['match_id']} ({row['home_team_name']} vs {row['away_team_name']}): {row['market']} (Real: {actual_value})")
            
        conn.commit()

    def delete_predictions(self, match_id: int) -> None:
        conn = self.connect()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM predictions WHERE match_id = ?", (match_id,))
            conn.commit()
            print(f"Previsões antigas removidas para o jogo {match_id}.")
        except Exception as e:
            print(f"Erro ao remover previsões antigas: {e}")

    def save_match(self, match_data: dict) -> None:
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO matches (
                    match_id, tournament_name, season_id, round, status, 
                    start_timestamp, home_team_id, home_team_name, 
                    away_team_id, away_team_name, home_score, away_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                match_data['id'], match_data['tournament'], match_data['season_id'],
                match_data.get('round'), match_data['status'], match_data['timestamp'],
                match_data['home_id'], match_data['home_name'],
                match_data['away_id'], match_data['away_name'],
                match_data['home_score'], match_data['away_score']
            ))
            conn.commit()
        except Exception as e:
            print(f"Erro ao salvar jogo {match_data['id']}: {e}")

    def save_stats(self, match_id: int, stats_data: dict) -> None:
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO match_stats (
                    match_id, 
                    corners_home_ft, corners_away_ft, 
                    corners_home_ht, corners_away_ht,
                    shots_ot_home_ft, shots_ot_away_ft,
                    shots_ot_home_ht, shots_ot_away_ht,
                    possession_home, possession_away,
                    total_shots_home, total_shots_away,
                    fouls_home, fouls_away,
                    yellow_cards_home, yellow_cards_away,
                    red_cards_home, red_cards_away,
                    big_chances_home, big_chances_away,
                    expected_goals_home, expected_goals_away
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                match_id,
                stats_data.get('corners_home_ft', 0), stats_data.get('corners_away_ft', 0),
                stats_data.get('corners_home_ht', 0), stats_data.get('corners_away_ht', 0),
                stats_data.get('shots_ot_home_ft', 0), stats_data.get('shots_ot_away_ft', 0),
                stats_data.get('shots_ot_home_ht', 0), stats_data.get('shots_ot_away_ht', 0),
                stats_data.get('possession_home', 0), stats_data.get('possession_away', 0),
                stats_data.get('total_shots_home', 0), stats_data.get('total_shots_away', 0),
                stats_data.get('fouls_home', 0), stats_data.get('fouls_away', 0),
                stats_data.get('yellow_cards_home', 0), stats_data.get('yellow_cards_away', 0),
                stats_data.get('red_cards_home', 0), stats_data.get('red_cards_away', 0),
                stats_data.get('big_chances_home', 0), stats_data.get('big_chances_away', 0),
                stats_data.get('expected_goals_home', 0.0), stats_data.get('expected_goals_away', 0.0)
            ))
            conn.commit()
        except Exception as e:
            print(f"Erro ao salvar stats do jogo {match_id}: {e}")

    def get_historical_data(self) -> pd.DataFrame:
        conn = self.connect()
        query = '''
            SELECT 
                m.*, 
                s.corners_home_ft, s.corners_away_ft, 
                s.corners_home_ht, s.corners_away_ht,
                s.shots_ot_home_ft, s.shots_ot_away_ft,
                s.shots_ot_home_ht, s.shots_ot_away_ht,
                s.big_chances_home, s.big_chances_away
            FROM matches m
            JOIN match_stats s ON m.match_id = s.match_id
            WHERE m.status = 'finished'
            ORDER BY m.start_timestamp ASC
        '''
        return pd.read_sql_query(query, conn)

    def get_pending_matches(self) -> list:
        """Retorna jogos pendentes (agendados no passado ou em andamento)."""
        conn = self.connect()
        cursor = conn.cursor()
        
        import time
        now = int(time.time())
        
        # Jogos 'scheduled' que já deveriam ter começado OU jogos 'inprogress'
        query = '''
            SELECT match_id, home_team_name, away_team_name, status, start_timestamp
            FROM matches 
            WHERE (status = 'scheduled' AND start_timestamp < ?)
               OR (status = 'inprogress')
            ORDER BY start_timestamp ASC
        '''
        
        cursor.execute(query, (now,))
        rows = cursor.fetchall()
        
        matches = []
        for row in rows:
            matches.append({
                'match_id': row[0],
                'home_team': row[1],
                'away_team': row[2],
                'status': row[3],
                'start_timestamp': row[4]
            })
            
        return matches