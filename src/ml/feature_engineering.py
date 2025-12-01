"""
Módulo de Engenharia de Features para Machine Learning.

Este módulo é responsável por transformar dados brutos de partidas em
features significativas para o modelo de ML, incluindo médias móveis
e estatísticas rolling por time.

Regras de Negócio:
    - Features são calculadas como médias dos últimos N jogos
    - Dados são transformados para perspectiva de cada time
    - Valores NaN (início de temporada) são removidos
"""

import pandas as pd


def calculate_rolling_stats(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Calcula estatísticas rolling (média móvel) por time.
    
    Transforma os dados de partidas em features de média móvel dos últimos
    N jogos para cada time, considerando tanto jogos em casa quanto fora.
    
    Args:
        df: DataFrame com dados históricos de partidas.
            Colunas necessárias:
            - match_id, start_timestamp
            - home_team_id, away_team_id
            - corners_home_ft, corners_away_ft
            - shots_ot_home_ft, shots_ot_away_ft
            - home_score, away_score
        window: Janela para cálculo da média móvel (default: 5 jogos).
    
    Returns:
        pd.DataFrame: DataFrame original acrescido de colunas:
            - home_avg_corners: Média de escanteios do mandante nos últimos 5 jogos
            - home_avg_shots: Média de chutes no gol do mandante
            - home_avg_goals: Média de gols do mandante
            - away_avg_corners: Média de escanteios do visitante nos últimos 5 jogos
            - away_avg_shots: Média de chutes no gol do visitante
            - away_avg_goals: Média de gols do visitante
    
    Lógica:
        1. Ordena dados por timestamp (cronológico)
        2. Reestrutura: cada linha vira 2 (perspectiva home e away)
        3. Agrupa por time e calcula rolling mean com shift(1)
        4. Faz merge de volta ao DataFrame original
        5. Remove linhas com NaN (primeiros jogos sem histórico)
    
    Regras de Negócio:
        - shift(1) garante que usamos apenas dados passados (sem data leakage)
        - min_periods=1 permite cálculo mesmo com menos que 5 jogos
        - Times no início da temporada terão médias baseadas em menos jogos
    
    Cálculo do Rolling Mean:
        ```
        Para cada time T no jogo N:
        avg_corners = média(corners do time T nos jogos N-1, N-2, ..., N-5)
        
        Exemplo para Flamengo no jogo 10:
        avg_corners = média(corners nos jogos 9, 8, 7, 6, 5)
        ```
    
    Note:
        O shift(1) é crucial para evitar data leakage - usamos apenas
        dados de jogos anteriores, nunca do jogo atual.
    """
    # Garante ordenação cronológica
    df = df.sort_values('start_timestamp')
    
    # Reestrutura: separa dados de casa e fora para cada time
    matches_home = df[['match_id', 'start_timestamp', 'home_team_id', 'corners_home_ft', 'shots_ot_home_ft', 'home_score']].copy()
    matches_home.columns = ['match_id', 'timestamp', 'team_id', 'corners', 'shots', 'goals']
    matches_home['is_home'] = 1
    
    matches_away = df[['match_id', 'start_timestamp', 'away_team_id', 'corners_away_ft', 'shots_ot_away_ft', 'away_score']].copy()
    matches_away.columns = ['match_id', 'timestamp', 'team_id', 'corners', 'shots', 'goals']
    matches_away['is_home'] = 0
    
    # Concatena e ordena por time + tempo
    team_stats = pd.concat([matches_home, matches_away]).sort_values(['team_id', 'timestamp'])
    
    # Calcula médias rolling com shift para evitar data leakage
    team_stats['avg_corners_last_5'] = team_stats.groupby('team_id')['corners'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    )
    team_stats['avg_shots_last_5'] = team_stats.groupby('team_id')['shots'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    )
    team_stats['avg_goals_last_5'] = team_stats.groupby('team_id')['goals'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    )
    
    # Merge de volta ao DataFrame original
    df_features = df.copy()
    
    # Join para estatísticas do mandante
    home_stats = team_stats[team_stats['is_home'] == 1][['match_id', 'avg_corners_last_5', 'avg_shots_last_5', 'avg_goals_last_5']]
    home_stats.columns = ['match_id', 'home_avg_corners', 'home_avg_shots', 'home_avg_goals']
    df_features = df_features.merge(home_stats, on='match_id', how='left')
    
    # Join para estatísticas do visitante
    away_stats = team_stats[team_stats['is_home'] == 0][['match_id', 'avg_corners_last_5', 'avg_shots_last_5', 'avg_goals_last_5']]
    away_stats.columns = ['match_id', 'away_avg_corners', 'away_avg_shots', 'away_avg_goals']
    df_features = df_features.merge(away_stats, on='match_id', how='left')
    
    # Remove linhas com NaN (primeiros jogos da temporada)
    df_features = df_features.dropna()
    
    return df_features


def prepare_training_data(df: pd.DataFrame) -> tuple:
    """
    Prepara dados para treinamento do modelo de ML.
    
    Transforma o DataFrame histórico em features (X) e target (y)
    prontos para alimentar o modelo de Machine Learning.
    
    Args:
        df: DataFrame com dados históricos de partidas.
    
    Returns:
        tuple: (X, y, df_processed)
            - X: DataFrame com features de entrada (6 colunas)
            - y: Series com target (total de escanteios)
            - df_processed: DataFrame completo processado
    
    Lógica:
        1. Aplica calculate_rolling_stats() para criar features
        2. Seleciona colunas de features
        3. Calcula target como soma de escanteios (home + away)
    
    Features (X):
        - home_avg_corners: Média de escanteios do mandante (últimos 5)
        - home_avg_shots: Média de chutes no gol do mandante
        - home_avg_goals: Média de gols do mandante
        - away_avg_corners: Média de escanteios do visitante (últimos 5)
        - away_avg_shots: Média de chutes no gol do visitante
        - away_avg_goals: Média de gols do visitante
    
    Target (y):
        Total de escanteios da partida = corners_home_ft + corners_away_ft
    
    Regras de Negócio:
        - Features representam "o que sabemos antes do jogo"
        - Target representa "o que queremos prever"
        - Médias rolling evitam data leakage
    
    Example:
        >>> X, y, df_full = prepare_training_data(df_historico)
        >>> print(f"Features: {X.shape}, Target: {y.shape}")
    """
    df_processed = calculate_rolling_stats(df)
    
    # Features de entrada (X)
    features = [
        'home_avg_corners', 'home_avg_shots', 'home_avg_goals',
        'away_avg_corners', 'away_avg_shots', 'away_avg_goals'
    ]
    
    X = df_processed[features]
    
    # Target (y) - Total de Escanteios da Partida
    y_corners = df_processed['corners_home_ft'] + df_processed['corners_away_ft']
    
    return X, y_corners, df_processed
