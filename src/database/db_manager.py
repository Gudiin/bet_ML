"""
Módulo de Gerenciamento de Banco de Dados.

Este módulo é o "Caderno" do sistema. É aqui que guardamos tudo o que aprendemos:
jogos passados, estatísticas detalhadas e nossas próprias previsões.

Conceitos Principais:
---------------------
1. **SQLite**:
   Um banco de dados leve que fica em um único arquivo (.db). Não precisa instalar
   servidores complexos. É como uma planilha Excel superpoderosa.

2. **Persistência**:
   A garantia de que os dados não somem quando desligamos o computador.

3. **Feedback Loop (Ciclo de Feedback)**:
   O sistema lembra do que previu ontem. Hoje, ele confere se acertou ou errou.
   Isso é crucial para sabermos se a IA está ficando inteligente ou burra.

Regras de Negócio:
------------------
- Usamos "INSERT OR REPLACE" para evitar duplicatas (se o jogo já existe, atualizamos).
- O banco é criado automaticamente na primeira vez que rodamos o sistema.
"""

import sqlite3
import pandas as pd
from datetime import datetime


class DBManager:
    """
    Gerenciador de banco de dados SQLite para o sistema de previsão de escanteios.
    
    Esta classe encapsula todas as operações de banco de dados, incluindo
    criação de tabelas, CRUD de partidas/estatísticas e gerenciamento de previsões.
    
    Regras de Negócio:
        - Banco de dados é criado automaticamente se não existir
        - Utiliza INSERT OR REPLACE para evitar duplicatas
        - Previsões têm sistema de feedback (GREEN/RED) para validação
        - Migrações são aplicadas automaticamente para compatibilidade
    
    Attributes:
        db_path (str): Caminho para o arquivo do banco de dados SQLite.
        conn: Conexão ativa com o banco de dados.
    
    Example:
        >>> db = DBManager()
        >>> db.save_match(match_data)
        >>> df = db.get_historical_data()
        >>> db.close()
    """
    
    def __init__(self, db_path: str = "data/football_data.db"):
        """
        Inicializa o gerenciador e cria tabelas se necessário.
        
        Args:
            db_path: Caminho para o arquivo SQLite.
                    Default: "data/football_data.db"
        
        Lógica:
            1. Armazena caminho do banco
            2. Inicializa conexão como None (lazy loading)
            3. Chama create_tables() para garantir estrutura
        
        Regras de Negócio:
            - Diretório 'data/' deve existir ou será erro
            - Banco é criado automaticamente se não existir
        """
        self.db_path = db_path
        self.conn = None
        self.create_tables()

    def connect(self) -> sqlite3.Connection:
        """
        Estabelece conexão com o banco de dados SQLite.
        
        Implementa padrão Singleton para reutilizar conexão existente.
        
        Returns:
            sqlite3.Connection: Conexão ativa com o banco.
        
        Lógica:
            1. Verifica se já existe conexão ativa
            2. Se não, cria nova conexão
            3. Retorna conexão (nova ou existente)
        
        Regras de Negócio:
            - Conexão é reutilizada enquanto não for fechada
            - sqlite3 suporta múltiplos cursores na mesma conexão
        """
        if self.conn is None:
            # timeout=30.0 aumenta a tolerância para "database is locked"
            # Útil quando rodamos múltiplos terminais simultaneamente
            self.conn = sqlite3.connect(self.db_path, timeout=30.0)
        return self.conn

    def close(self) -> None:
        """
        Fecha a conexão com o banco de dados.
        
        Deve ser chamado ao final das operações para liberar recursos.
        
        Lógica:
            1. Verifica se existe conexão ativa
            2. Fecha conexão e define como None
        
        Regras de Negócio:
            - Seguro para chamar múltiplas vezes
            - Sempre usar em bloco finally ou context manager
        """
        if self.conn:
            self.conn.close()
            self.conn = None

    def create_tables(self) -> None:
        """
        Cria as tabelas do banco de dados se não existirem.
        
        Define a estrutura do banco com três tabelas principais:
        matches, match_stats e predictions.
        
        Lógica:
            1. Cria tabela 'matches' para informações básicas das partidas
            2. Cria tabela 'match_stats' para estatísticas detalhadas
            3. Cria tabela 'predictions' para previsões e feedback loop
            4. Aplica migrações para colunas novas (retrocompatibilidade)
        
        Regras de Negócio:
            - Tabelas: matches (partidas), match_stats (estatísticas), predictions (previsões)
            - match_stats tem FK para matches (match_id)
            - predictions tem status: PENDING, GREEN, RED para feedback loop
            - Migrações são silenciosas (ignora se coluna já existe)
        
        Estrutura das Tabelas:
            matches:
                - match_id (PK): ID único da partida no SofaScore
                - tournament_name: Nome do torneio
                - season_id: ID da temporada
                - round: Número da rodada
                - status: 'finished', 'inprogress', 'notstarted'
                - home_team_id/name, away_team_id/name: Dados dos times
                - home_score, away_score: Placar final
            
            match_stats:
                - corners_home/away_ft/ht: Escanteios por time e período
                - shots_ot_home/away_ft/ht: Chutes no gol por time e período
            
            predictions:
                - prediction_type: 'ML' ou 'Statistical'
                - market: 'Over 9.5', 'Under 10.5', etc.
                - probability, odds: Probabilidade e odd justa calculadas
                - status: 'PENDING', 'GREEN', 'RED' (feedback loop)
        """
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
                status TEXT DEFAULT 'PENDING', -- 'PENDING', 'GREEN', 'RED'
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(match_id) REFERENCES matches(match_id)
            )
        ''')
        
        # Migrações para colunas novas (retrocompatibilidade)
        self._run_migrations(cursor)

        conn.commit()

    def _run_migrations(self, cursor):
        """Executa migrações de banco de dados seguras."""
        migrations = [
            "ALTER TABLE predictions ADD COLUMN odds REAL",
            "ALTER TABLE predictions ADD COLUMN category TEXT",
            "ALTER TABLE predictions ADD COLUMN market_group TEXT",
            # Novas colunas de estatísticas
            "ALTER TABLE match_stats ADD COLUMN possession_home INTEGER",
            "ALTER TABLE match_stats ADD COLUMN possession_away INTEGER",
            "ALTER TABLE match_stats ADD COLUMN total_shots_home INTEGER",
            "ALTER TABLE match_stats ADD COLUMN total_shots_away INTEGER",
            "ALTER TABLE match_stats ADD COLUMN fouls_home INTEGER",
            "ALTER TABLE match_stats ADD COLUMN fouls_away INTEGER",
            "ALTER TABLE match_stats ADD COLUMN yellow_cards_home INTEGER",
            "ALTER TABLE match_stats ADD COLUMN yellow_cards_away INTEGER"
        ]
        
        for sql in migrations:
            try:
                cursor.execute(sql)
            except:
                pass # Coluna já existe ou erro ignorável



    def save_prediction(self, match_id: int, pred_type: str, value: float, 
                       market: str, prob: float, odds: float = 0.0, 
                       category: str = None, market_group: str = None, 
                       verbose: bool = False) -> None:
        """
        Salva uma previsão no banco de dados para posterior validação.
        
        As previsões são armazenadas com status 'PENDING' e posteriormente
        validadas quando o jogo terminar (feedback loop).
        
        Args:
            match_id: ID da partida no SofaScore.
            pred_type: Tipo da previsão - 'ML' ou 'Statistical'.
            value: Valor numérico previsto (ex: 9.5 escanteios).
            market: Mercado apostado (ex: 'Over 9.5', 'Under 10.5').
            prob: Probabilidade calculada (0.0 a 1.0).
            odds: Odd justa calculada (ex: 1.85).
            category: Categoria da previsão - 'Top7', 'Easy', 'Medium', 'Hard'.
            market_group: Grupo do mercado - 'JOGO COMPLETO', '1º TEMPO', etc.
            verbose: Se True, imprime confirmação no console.
        
        Lógica:
            1. Insere nova linha na tabela predictions
            2. Commit automático após inserção
            3. Imprime confirmação se verbose=True
        
        Regras de Negócio:
            - Status inicial sempre é 'PENDING'
            - Previsões são validadas por check_predictions()
            - Múltiplas previsões podem existir para mesma partida
        """
        conn = self.connect()
        cursor = conn.cursor()
        try:
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
        """
        Valida previsões pendentes de jogos já finalizados (Feedback Loop).
        
        Compara as previsões feitas com os resultados reais das partidas,
        atualizando o status para 'GREEN' (acerto) ou 'RED' (erro).
        
        Lógica:
            1. Busca previsões com status='PENDING' de jogos finalizados
            2. Para cada previsão, calcula o valor real correspondente ao mercado
            3. Compara com a linha apostada (Over/Under)
            4. Atualiza status: GREEN se acertou, RED se errou
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        # Busca previsões pendentes de jogos que já terminaram
        # Traz todas as colunas de stats necessárias
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
            # Calcula estatísticas derivadas
            corners_home_ft = row['corners_home_ft']
            corners_away_ft = row['corners_away_ft']
            corners_home_ht = row['corners_home_ht']
            corners_away_ht = row['corners_away_ht']
            
            # Totais
            total_ft = corners_home_ft + corners_away_ft
            total_ht = corners_home_ht + corners_away_ht
            total_2t = total_ft - total_ht
            
            # Times 2º Tempo
            corners_home_2t = corners_home_ft - corners_home_ht
            corners_away_2t = corners_away_ft - corners_away_ht
            
            # Determina valor real a ser comparado baseado no grupo de mercado
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
            elif market_group == "2º TEMPO": # Corrigido label
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
                # Fallback para total FT se não reconhecer (ou para ML prediction que é sempre FT)
                actual_value = total_ft

            status = 'RED'
            
            # Lógica para Over/Under
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
            
            # Atualiza status
            cursor.execute("UPDATE predictions SET status = ? WHERE id = ?", (status, row['id']))
            print(f"[{status}] Jogo {row['match_id']} ({row['home_team_name']} vs {row['away_team_name']}): {row['market']} (Real: {actual_value})")
            
        conn.commit()

    def delete_predictions(self, match_id: int) -> None:
        """
        Remove todas as previsões de uma partida específica.
        
        Usado para limpar previsões antigas antes de fazer novas,
        evitando duplicatas quando o mesmo jogo é analisado novamente.
        
        Args:
            match_id: ID da partida cujas previsões serão removidas.
        
        Lógica:
            1. Deleta todas as linhas com match_id correspondente
            2. Commit automático
            3. Imprime confirmação
        
        Regras de Negócio:
            - Remove TODAS as previsões da partida (ML e Statistical)
            - Chamado antes de nova análise para evitar duplicatas
            - Operação irreversível (sem soft delete)
        """
        conn = self.connect()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM predictions WHERE match_id = ?", (match_id,))
            conn.commit()
            print(f"Previsões antigas removidas para o jogo {match_id}.")
        except Exception as e:
            print(f"Erro ao remover previsões antigas: {e}")

    def save_match(self, match_data: dict) -> None:
        """
        Salva ou atualiza informações de uma partida no banco.
        
        Utiliza INSERT OR REPLACE para upsert (inserir ou atualizar).
        
        Args:
            match_data: Dicionário com dados da partida:
                - id: ID único da partida
                - tournament: Nome do torneio
                - season_id: ID da temporada
                - round: Número da rodada
                - status: Status da partida
                - timestamp: Timestamp Unix do início
                - home_id/home_name: ID e nome do mandante
                - away_id/away_name: ID e nome do visitante
                - home_score/away_score: Placar final
        
        Lógica:
            1. Extrai campos do dicionário
            2. Executa INSERT OR REPLACE (upsert)
            3. Commit automático
        
        Regras de Negócio:
            - INSERT OR REPLACE atualiza se match_id já existe
            - Campos ausentes podem ser None
            - Placar pode ser 0 para jogos não finalizados
        """
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
        """
        Salva estatísticas detalhadas de uma partida.
        
        Armazena escanteios e chutes por time e período (HT/FT).
        
        Args:
            match_id: ID da partida.
            stats_data: Dicionário com estatísticas:
                - corners_home/away_ft/ht: Escanteios
                - shots_ot_home/away_ft/ht: Chutes no gol
        
        Lógica:
            1. Extrai estatísticas do dicionário
            2. Executa INSERT OR REPLACE
            3. Commit automático
        
        Regras de Negócio:
            - Estatísticas de 2º tempo: calculadas como FT - HT
            - Valores 0 são válidos (time sem escanteios)
            - INSERT OR REPLACE para upsert
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO match_stats (
                    match_id, 
                    corners_home_ft, corners_away_ft, 
                    corners_home_ht, corners_away_ht,
                    shots_ot_home_ft, shots_ot_away_ft,
                    shots_ot_home_ht, shots_ot_away_ht
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                match_id,
                stats_data['corners_home_ft'], stats_data['corners_away_ft'],
                stats_data['corners_home_ht'], stats_data['corners_away_ht'],
                stats_data['shots_ot_home_ft'], stats_data['shots_ot_away_ft'],
                stats_data['shots_ot_home_ht'], stats_data['shots_ot_away_ht']
            ))
            conn.commit()
        except Exception as e:
            print(f"Erro ao salvar stats do jogo {match_id}: {e}")

    def get_historical_data(self) -> pd.DataFrame:
        """
        Recupera todos os dados históricos para treinamento do modelo ML.
        
        Faz JOIN entre matches e match_stats para obter dataset completo
        de jogos finalizados com todas as estatísticas.
        
        Returns:
            pd.DataFrame: DataFrame com colunas de partidas e estatísticas,
                         ordenado por timestamp (mais antigo primeiro).
        
        Lógica:
            1. JOIN entre matches e match_stats
            2. Filtra apenas status='finished'
            3. Ordena por timestamp ascendente
            4. Retorna como DataFrame pandas
        
        Regras de Negócio:
            - Apenas jogos finalizados são retornados
            - Ordenação cronológica é importante para features temporais
            - Usado como input para prepare_training_data()
        
        Colunas Retornadas:
            - match_id, tournament_name, season_id, round
            - home_team_id, home_team_name, away_team_id, away_team_name
            - home_score, away_score, start_timestamp
            - corners_home/away_ft/ht, shots_ot_home/away_ft/ht
        """
        conn = self.connect()
        query = '''
            SELECT 
                m.*, 
                s.corners_home_ft, s.corners_away_ft, 
                s.corners_home_ht, s.corners_away_ht,
                s.shots_ot_home_ft, s.shots_ot_away_ft,
                s.shots_ot_home_ht, s.shots_ot_away_ht
            FROM matches m
            JOIN match_stats s ON m.match_id = s.match_id
            WHERE m.status = 'finished'
            ORDER BY m.start_timestamp ASC
        '''
        return pd.read_sql_query(query, conn)
