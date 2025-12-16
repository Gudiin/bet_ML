"""
Módulo de Gerenciamento de Banco de Dados.
"""

import sqlite3
import pandas as pd
from datetime import datetime
import json

class DBManager:
    """Gerenciador de banco de dados SQLite."""
    
    def __init__(self, db_path: str = "data/football_data.db"):
        self.db_path = db_path
        self.conn = None
        self.create_tables()

    def connect(self) -> sqlite3.Connection:
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path, timeout=30.0)
        return self.conn

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None

    def create_tables(self) -> None:
        """
        Cria as tabelas necessárias no banco de dados se não existirem.
        
        Regra de Negócio:
            Garante a estrutura do banco para armazenar partidas, estatísticas e predições.
            Executa migrações automáticas para manter compatibilidade com versões anteriores.
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS matches (
                match_id INTEGER PRIMARY KEY,
                tournament_name TEXT,
                tournament_id INTEGER,
                season_id INTEGER,
                round INTEGER,
                status TEXT,
                start_timestamp INTEGER,
                home_team_id INTEGER,
                home_team_name TEXT,
                away_team_id INTEGER,
                away_team_name TEXT,
                home_score INTEGER,
                away_score INTEGER,
                odds_home REAL,
                odds_draw REAL,
                odds_away REAL
            )
        ''')
        
        # Migração: Adiciona colunas de odds se não existirem
        try:
             cursor.execute("ALTER TABLE matches ADD COLUMN odds_home REAL")
             cursor.execute("ALTER TABLE matches ADD COLUMN odds_draw REAL")
             cursor.execute("ALTER TABLE matches ADD COLUMN odds_away REAL")
             print("✅ Migração de schema: Colunas de odds adicionadas.")
        except sqlite3.OperationalError:
             pass # Colunas já existem
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS match_stats (
                match_id INTEGER PRIMARY KEY,
                corners_home_ft INTEGER,
                corners_away_ft INTEGER,
                corners_home_ht INTEGER,
                corners_away_ht INTEGER,
                shots_ot_home_ft INTEGER,
                shots_ot_away_ft INTEGER,
                shots_ot_home_ht INTEGER,
                shots_ot_away_ht INTEGER,
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
                dangerous_attacks_home INTEGER,
                dangerous_attacks_away INTEGER,
                expected_goals_home REAL,
                expected_goals_away REAL,
                FOREIGN KEY (match_id) REFERENCES matches (match_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id INTEGER,
                model_version TEXT,
                prediction_value REAL,
                prediction_label TEXT,
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_correct BOOLEAN,
                category TEXT,
                market_group TEXT,
                odds REAL,
                status TEXT DEFAULT 'PENDING',
                FOREIGN KEY (match_id) REFERENCES matches (match_id)
            )
        ''')
        
        # Migrations
        try:
            cursor.execute("SELECT tournament_id FROM matches LIMIT 1")
        except sqlite3.OperationalError:
            cursor.execute("ALTER TABLE matches ADD COLUMN tournament_id INTEGER")
            conn.commit()

        try:
            cursor.execute("SELECT predicted_value FROM predictions LIMIT 1")
            cursor.execute("ALTER TABLE predictions RENAME COLUMN predicted_value TO prediction_value")
            cursor.execute("ALTER TABLE predictions RENAME COLUMN prediction_type TO model_version")
            cursor.execute("ALTER TABLE predictions RENAME COLUMN market TO prediction_label")
            cursor.execute("ALTER TABLE predictions RENAME COLUMN probability TO confidence")
            cursor.execute("ALTER TABLE predictions ADD COLUMN is_correct BOOLEAN")
            conn.commit()
        except sqlite3.OperationalError:
            pass

        try:
            cursor.execute("SELECT is_correct FROM predictions LIMIT 1")
        except sqlite3.OperationalError:
            cursor.execute("ALTER TABLE predictions ADD COLUMN is_correct BOOLEAN")
            conn.commit()

        try:
            cursor.execute("SELECT status FROM predictions LIMIT 1")
        except sqlite3.OperationalError:
            cursor.execute("ALTER TABLE predictions ADD COLUMN status TEXT DEFAULT 'PENDING'")
            conn.commit()
        except sqlite3.OperationalError:
            pass

        try:
            cursor.execute("SELECT feedback_text FROM predictions LIMIT 1")
        except sqlite3.OperationalError:
            cursor.execute("ALTER TABLE predictions ADD COLUMN feedback_text TEXT")
            conn.commit()

        try:
            cursor.execute("SELECT fair_odds FROM predictions LIMIT 1")
        except sqlite3.OperationalError:
            cursor.execute("ALTER TABLE predictions ADD COLUMN fair_odds REAL")
            conn.commit()

        try:
            cursor.execute("SELECT home_league_position FROM matches LIMIT 1")
        except sqlite3.OperationalError:
            cursor.execute("ALTER TABLE matches ADD COLUMN home_league_position INTEGER")
            cursor.execute("ALTER TABLE matches ADD COLUMN away_league_position INTEGER")
            conn.commit()

        # Dangerous Attacks Migration
        try:
            cursor.execute("SELECT dangerous_attacks_home FROM match_stats LIMIT 1")
        except sqlite3.OperationalError:
            cursor.execute("ALTER TABLE match_stats ADD COLUMN dangerous_attacks_home INTEGER DEFAULT 0")
            cursor.execute("ALTER TABLE match_stats ADD COLUMN dangerous_attacks_away INTEGER DEFAULT 0")
            print("[OK] Migracao de schema: Colunas de Dangerous Attacks adicionadas.")
            conn.commit()

        # Odds Migration
        try:
            cursor.execute("SELECT odds_home FROM matches LIMIT 1")
        except sqlite3.OperationalError:
            cursor.execute("ALTER TABLE matches ADD COLUMN odds_home REAL")
            cursor.execute("ALTER TABLE matches ADD COLUMN odds_draw REAL")
            cursor.execute("ALTER TABLE matches ADD COLUMN odds_away REAL")
            cursor.execute("ALTER TABLE matches ADD COLUMN odds_provider TEXT")
            conn.commit()

        # xG Migration
        try:
            cursor.execute("SELECT expected_goals_home FROM match_stats LIMIT 1")
        except sqlite3.OperationalError:
            cursor.execute("ALTER TABLE match_stats ADD COLUMN expected_goals_home REAL")
            cursor.execute("ALTER TABLE match_stats ADD COLUMN expected_goals_away REAL")
            print("✅ Migração: Colunas xG adicionadas.")
            conn.commit()

        # Match Minute Migration (Live Data)
        try:
            cursor.execute("SELECT match_minute FROM matches LIMIT 1")
        except sqlite3.OperationalError:
            cursor.execute("ALTER TABLE matches ADD COLUMN match_minute TEXT")
            print("✅ Migração: Coluna match_minute adicionada.")
            conn.commit()

        # Tactical Metrics Migration (Gap Analysis)
        tactical_cols = [
            'blocked_shots', 'crosses', 'tackles', 'interceptions', 'clearances', 'recoveries'
        ]
        try:
            cursor.execute("SELECT blocked_shots_home FROM match_stats LIMIT 1")
        except sqlite3.OperationalError:
            for col in tactical_cols:
                cursor.execute(f"ALTER TABLE match_stats ADD COLUMN {col}_home INTEGER DEFAULT 0")
                cursor.execute(f"ALTER TABLE match_stats ADD COLUMN {col}_away INTEGER DEFAULT 0")
            print("✅ Migração: Métricas Táticas (Blocked Shots, Crosses, etc) adicionadas.")
            conn.commit()

        conn.commit()

    def save_match(self, match_data: dict) -> None:
        """
        Salva ou atualiza os dados básicos de uma partida.
        
        Args:
            match_data (dict): Dicionário com dados da partida (id, times, placar, etc).
            
        Regra de Negócio:
            Centraliza a persistência de dados brutos das partidas para histórico e feature engineering.
        """
        conn = self.connect()
        cursor = conn.cursor()

        # --- AUTO-MIGRATE IDS (FIREWALL) ---
        # Garante que IDs legados do SofaScore sejam convertidos para o ID Unificado do nosso banco.
        unified_ids = {
            1: 17,   # Premier League
            42: 35,  # Bundesliga
            36: 8,   # LaLiga
            33: 23,  # Serie A
            4: 34    # Ligue 1
        }
        
        original_id = match_data.get('tournament_id')
        if original_id in unified_ids:
            # print(f"🔄 Auto-Corrigindo Liga: ID {original_id} -> {unified_ids[original_id]}")
            match_data['tournament_id'] = unified_ids[original_id]
        # -----------------------------------

        try:
            cursor.execute('''
                INSERT OR REPLACE INTO matches (
                    match_id, tournament_name, tournament_id, season_id, round, status, 
                    start_timestamp, home_team_id, home_team_name, 
                    away_team_id, away_team_name, home_score, away_score, match_minute
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                match_data['id'], match_data['tournament'], match_data.get('tournament_id'),
                match_data['season_id'], match_data.get('round'), match_data['status'], 
                match_data['timestamp'], match_data['home_id'], match_data['home_name'],
                match_data['away_id'], match_data['away_name'],
                match_data['home_score'], match_data['away_score'],
                match_data.get('match_minute')
            ))
            conn.commit()
        except Exception as e:
            print(f"Erro ao salvar jogo {match_data.get('id')}: {e}")

    def save_stats(self, match_id: int, stats_data: dict) -> None:
        """
        Salva as estatísticas detalhadas de uma partida.
        
        Args:
            match_id (int): ID da partida.
            stats_data (dict): Dicionário com estatísticas (escanteios, chutes, etc).
            
        Regra de Negócio:
            Armazena métricas profundas usadas para calcular médias e tendências dos times.
        """
        conn = self.connect()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO match_stats (
                    match_id, corners_home_ft, corners_away_ft, corners_home_ht, corners_away_ht,
                    shots_ot_home_ft, shots_ot_away_ft, shots_ot_home_ht, shots_ot_away_ht,
                    possession_home, possession_away, total_shots_home, total_shots_away,
                    fouls_home, fouls_away, yellow_cards_home, yellow_cards_away,
                    red_cards_home, red_cards_away, big_chances_home, big_chances_away,
                    dangerous_attacks_home, dangerous_attacks_away,
                    expected_goals_home, expected_goals_away,
                    blocked_shots_home, blocked_shots_away,
                    crosses_home, crosses_away,
                    tackles_home, tackles_away,
                    interceptions_home, interceptions_away,
                    clearances_home, clearances_away,
                    recoveries_home, recoveries_away
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                stats_data.get('dangerous_attacks_home', 0), stats_data.get('dangerous_attacks_away', 0),
                stats_data.get('expected_goals_home', 0.0), stats_data.get('expected_goals_away', 0.0),
                stats_data.get('blocked_shots_home', 0), stats_data.get('blocked_shots_away', 0),
                stats_data.get('crosses_home', 0), stats_data.get('crosses_away', 0),
                stats_data.get('tackles_home', 0), stats_data.get('tackles_away', 0),
                stats_data.get('interceptions_home', 0), stats_data.get('interceptions_away', 0),
                stats_data.get('clearances_home', 0), stats_data.get('clearances_away', 0),
                stats_data.get('recoveries_home', 0), stats_data.get('recoveries_away', 0)
            ))
            conn.commit()
            # print(f"DEBUG: Stats saved for match {match_id}") # Uncomment for deeper debug if needed
        except Exception as e:
            print(f"Erro ao salvar stats do jogo {match_id}: {e}")

    def get_historical_data(self) -> pd.DataFrame:
        """
        Recupera todo o histórico de partidas finalizadas com estatísticas.
        
        Returns:
            pd.DataFrame: DataFrame contendo dados de partidas e estatísticas unificadas.
            
        Regra de Negócio:
            Fornece a base de dados completa para o treinamento do modelo de Machine Learning.
        """
        conn = self.connect()
        query = '''
            SELECT m.*, s.corners_home_ft, s.corners_away_ft, s.corners_home_ht, s.corners_away_ht,
                   s.shots_ot_home_ft, s.shots_ot_away_ft, s.shots_ot_home_ht, s.shots_ot_away_ht,
                   s.big_chances_home, s.big_chances_away,
                   s.dangerous_attacks_home, s.dangerous_attacks_away,
                   s.blocked_shots_home, s.blocked_shots_away,
                   s.crosses_home, s.crosses_away,
                   s.tackles_home, s.tackles_away,
                   s.interceptions_home, s.interceptions_away,
                   s.clearances_home, s.clearances_away,
                   s.recoveries_home, s.recoveries_away
            FROM matches m
            JOIN match_stats s ON m.match_id = s.match_id
            WHERE m.status = 'finished'
            ORDER BY m.start_timestamp ASC
        '''
        return pd.read_sql_query(query, conn)

    def get_season_stats(self, season_id: int) -> dict:
        """
        Retorna estatísticas resumidas de uma temporada.
        
        Args:
            season_id (int): ID da temporada.
            
        Returns:
            dict: {'total_matches': int, 'last_round': int}
            
        Regra de Negócio:
            Permite controle incremental de atualizações, evitando re-processar temporadas completas.
        """
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT COUNT(*), MAX(round) 
            FROM matches 
            WHERE season_id = ? AND status = 'finished'
        ''', (season_id,))
        row = cursor.fetchone()
        return {
            'total_matches': row[0] if row else 0,
            'last_round': row[1] if row and row[1] else 0
        }

    def save_prediction(self, match_id: int, model_version: str, value: float, label: str, confidence: float, category: str = None, market_group: str = None, odds: float = 0.0, feedback_text: str = None, fair_odds: float = 0.0, verbose: bool = False) -> None:
        """
        Salva uma predição gerada pelo modelo ou análise estatística.
        
        Args:
            match_id (int): ID da partida.
            model_version (str): Identificador do modelo (ex: 'Professional V2', 'Statistical').
            value (float): Valor numérico da predição (ex: 9.5 escanteios).
            label (str): Rótulo legível (ex: 'Over 9.5').
            confidence (float): Grau de confiança ou probabilidade (0.0 a 1.0).
            category (str, optional): Categoria de risco (ex: 'Top7', 'Suggestion_Easy').
            market_group (str, optional): Grupo de mercado (ex: 'Escanteios Totais').
            odds (float, optional): Odd no momento da análise.
            verbose (bool): Se deve imprimir confirmação no console.
            
        Regra de Negócio:
            Registra as previsões para posterior validação (backtesting) e exibição ao usuário.
            Evita duplicatas verificando se já existe previsão igual.
        """
        conn = self.connect()
        cursor = conn.cursor()
        try:
            # Verifica se já existe previsão idêntica
            cursor.execute('''
                SELECT id FROM predictions 
                WHERE match_id = ? AND prediction_label = ? AND category = ? AND market_group = ?
            ''', (match_id, label, category, market_group))
            
            existing = cursor.fetchone()
            if existing:
                # Atualiza em vez de duplicar
                cursor.execute('''
                    UPDATE predictions 
                    SET prediction_value = ?, confidence = ?, odds = ?, model_version = ?, feedback_text = ?, fair_odds = ?
                    WHERE id = ?
                ''', (value, confidence, odds, model_version, feedback_text, fair_odds, existing[0]))
            else:
                # Insere nova previsão
                cursor.execute('''
                    INSERT INTO predictions (
                        match_id, model_version, prediction_value, prediction_label, 
                        confidence, category, market_group, odds, feedback_text, fair_odds
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (match_id, model_version, value, label, confidence, category, market_group, odds, feedback_text, fair_odds))
            
            conn.commit()
            if verbose:
                print(f"✅ Predição salva: {label} ({category})")
        except Exception as e:
            print(f"Erro ao salvar predição: {e}")

    def delete_predictions(self, match_id: int) -> None:
        """
        Remove predições existentes para uma partida.
        
        Args:
            match_id (int): ID da partida.
            
        Regra de Negócio:
            Garante que ao re-analisar um jogo, não fiquem predições duplicadas ou obsoletas.
        """
        conn = self.connect()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM predictions WHERE match_id = ?", (match_id,))
            conn.commit()
        except Exception as e:
            print(f"Erro ao limpar predições antigas: {e}")

    def check_predictions(self) -> None:
        """
        Verifica se predições passadas acertaram (Feedback Loop).
        
        Regra de Negócio:
            - Total Mandante → usa corners_home_ft
            - Total Visitante → usa corners_away_ft  
            - Jogo Completo / outros → usa soma total
            - 1º Tempo → usa corners_*_ht
            - 2º Tempo → usa corners_*_ft - corners_*_ht
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        # Busca predictions com market_group para determinar qual valor usar
        query = '''
            SELECT p.id, p.match_id, p.prediction_value, p.prediction_label, p.market_group,
                   s.corners_home_ft, s.corners_away_ft, s.corners_home_ht, s.corners_away_ht
            FROM predictions p
            JOIN matches m ON p.match_id = m.match_id
            LEFT JOIN match_stats s ON m.match_id = s.match_id
            WHERE (p.is_correct IS NULL OR p.status = 'PENDING')
              AND m.status = 'finished'
              AND s.corners_home_ft IS NOT NULL
        '''
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        if not rows:
            return

        print(f"Verificando {len(rows)} predições pendentes...")
        
        for row in rows:
            pred_id, match_id, pred_val, pred_label, market_group, h_corners_ft, a_corners_ft, h_corners_ht, a_corners_ht = row
            
            # Garante valores numéricos (fallback para 0 se None)
            h_corners_ft = h_corners_ft or 0
            a_corners_ft = a_corners_ft or 0
            h_corners_ht = h_corners_ht or 0
            a_corners_ht = a_corners_ht or 0
            
            # Determina qual valor usar baseado no market_group
            market_group_lower = (market_group or '').lower()
            
            if 'mandante' in market_group_lower or 'home' in market_group_lower:
                # Total Mandante: usa apenas escanteios do time da casa
                corners_value = h_corners_ft
            elif 'visitante' in market_group_lower or 'away' in market_group_lower:
                # Total Visitante: usa apenas escanteios do visitante
                corners_value = a_corners_ft
            elif '1' in market_group_lower or 'ht' in market_group_lower or 'primeiro' in market_group_lower:
                # 1º Tempo: usa soma dos escanteios do 1º tempo
                corners_value = h_corners_ht + a_corners_ht
            elif '2' in market_group_lower or 'segundo' in market_group_lower:
                # 2º Tempo: usa diferença (FT - HT)
                corners_value = (h_corners_ft - h_corners_ht) + (a_corners_ft - a_corners_ht)
            else:
                # Jogo Completo ou outros: usa soma total
                corners_value = h_corners_ft + a_corners_ft
            
            is_over = 'over' in pred_label.lower() if pred_label else False
            is_under = 'under' in pred_label.lower() if pred_label else False
            
            is_correct = False
            if pred_val is not None and pred_val > 0:
                line = pred_val
                if is_over:
                    is_correct = corners_value > line
                elif is_under:
                    is_correct = corners_value < line
            
            status = 'GREEN' if is_correct else 'RED'
            cursor.execute("UPDATE predictions SET is_correct = ?, status = ? WHERE id = ?", (is_correct, status, pred_id))
            
        conn.commit()
        print("✅ Verificação de predições concluída.")
    
    def fix_existing_predictions_values(self) -> int:
        """
        Corrige previsões antigas que foram salvas com prediction_value=0.
        Extrai o valor da linha do prediction_label (ex: 'Over 3.5' -> 3.5).
        
        Returns:
            int: Número de previsões corrigidas.
        """
        import re
        
        conn = self.connect()
        cursor = conn.cursor()
        
        # Busca predictions com valor 0 mas que têm label
        cursor.execute('''
            SELECT id, prediction_label 
            FROM predictions 
            WHERE (prediction_value IS NULL OR prediction_value = 0) 
              AND prediction_label IS NOT NULL
        ''')
        
        rows = cursor.fetchall()
        
        if not rows:
            print("✅ Nenhuma previsão precisa de correção.")
            return 0
        
        fixed_count = 0
        for pred_id, label in rows:
            # Extrai número do label (ex: "Over 3.5" -> 3.5)
            match = re.search(r'(\d+\.?\d*)', label or '')
            if match:
                line_value = float(match.group(1))
                cursor.execute("UPDATE predictions SET prediction_value = ? WHERE id = ?", (line_value, pred_id))
                fixed_count += 1
        
        conn.commit()
        print(f"✅ {fixed_count} previsões corrigidas.")
        return fixed_count

    def get_pending_matches(self) -> list:
        """
        Retorna lista de jogos pendentes (agendados ou em andamento).
        
        Returns:
            list: Lista de dicionários com dados dos jogos.
            
        Regra de Negócio:
            Identifica jogos que precisam de monitoramento ou atualização de status.
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        import time
        now = int(time.time())
        
        query = '''
            SELECT match_id, home_team_name, away_team_name, status, start_timestamp
            FROM matches 
            WHERE (status = 'scheduled' AND start_timestamp < ?)
               OR (status = 'inprogress')
               OR (status = 'notstarted' AND start_timestamp < ?)
               OR (status = 'finished' AND start_timestamp > ? - 10800)
            ORDER BY start_timestamp ASC
        '''
        
        cursor.execute(query, (now, now, now))
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