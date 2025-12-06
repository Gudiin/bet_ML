import pandas as pd
import numpy as np

def create_advanced_features(df: pd.DataFrame, window_short: int = 3, window_long: int = 5) -> tuple:
    """
    Pipeline unificado de Feature Engineering (Vetorizado e Anti-Leakage) - V3 (Advanced).
    
    Gera:
    1. Médias móveis Gerais (Momentum)
    2. Médias móveis Específicas (Home/Away)
    3. Médias de Concessão (Defesa)
    4. H2H (Confronto Direto)
    5. Trend (Curto vs Longo Prazo) - NOVO
    6. Volatilidade (Desvio Padrão) - NOVO
    7. Rest Days (Cansaço) - NOVO
    8. EMA (Exponential Moving Average) - NOVO
    9. Força Relativa (Interações) - NOVO
    
    Returns:
        X (pd.DataFrame): Features
        y (pd.Series): Target (Total Corners)
        timestamps (pd.Series): Data do jogo (para validação temporal)
    """
    # 1. Ordenação Temporal Obrigatória
    df = df.sort_values('start_timestamp').copy()
    
    # --- CORREÇÃO DE COMPATIBILIDADE (Tournament ID) ---
    if 'tournament_id' not in df.columns:
        if 'tournament_name' in df.columns:
            df['tournament_id'] = df['tournament_name']
        else:
            df['tournament_id'] = 'Unknown'
    # ---------------------------------------------------
    
    # 2. Estratégia "Team-Centric" (Transforma Partida em Linhas de Time)
    cols_metrics = ['corners_ft', 'shots_ot_ft', 'goals_ft', 'corners_ht']
    
    # Home Stats
    df_home = df[['match_id', 'start_timestamp', 'home_team_id', 'away_team_id'] + [f'corners_home_ft', 'shots_ot_home_ft', 'home_score', 'corners_home_ht', 'corners_away_ft']].copy()
    df_home.columns = ['match_id', 'start_timestamp', 'team_id', 'opponent_id', 'corners', 'shots', 'goals', 'corners_ht', 'corners_conceded']
    df_home['is_home'] = 1
    
    # Away Stats
    df_away = df[['match_id', 'start_timestamp', 'away_team_id', 'home_team_id'] + [f'corners_away_ft', 'shots_ot_away_ft', 'away_score', 'corners_away_ht', 'corners_home_ft']].copy()
    df_away.columns = ['match_id', 'start_timestamp', 'team_id', 'opponent_id', 'corners', 'shots', 'goals', 'corners_ht', 'corners_conceded']
    df_away['is_home'] = 0
    
    # Stack de todos os jogos na visão do time
    team_stats = pd.concat([df_home, df_away]).sort_values(['team_id', 'start_timestamp'])
    
    # 3. Engenharia Vetorizada
    grouped = team_stats.groupby('team_id')
    
    feature_cols = ['corners', 'shots', 'goals', 'corners_conceded']
    
    # --- A. Features Temporais (Rest Days) ---
    team_stats['prev_timestamp'] = grouped['start_timestamp'].shift(1)
    team_stats['rest_days'] = (team_stats['start_timestamp'] - team_stats['prev_timestamp']) / 86400
    team_stats['rest_days'] = team_stats['rest_days'].fillna(7) # Fallback: 7 dias de descanso
    # Clip para evitar outliers (ex: 100 dias de pausa)
    team_stats['rest_days'] = team_stats['rest_days'].clip(0, 14)
    
    for col in feature_cols:
        # --- B. Médias GERAIS (Momentum) & EMA ---
        # Rolling Mean Long
        team_stats[f'avg_{col}_general'] = grouped[col].transform(
            lambda x: x.shift(1).rolling(window=window_long, min_periods=1).mean()
        )
        # Rolling Mean Short (para Trend)
        team_stats[f'avg_{col}_short'] = grouped[col].transform(
            lambda x: x.shift(1).rolling(window=window_short, min_periods=1).mean()
        )
        # EMA (Exponential Moving Average) - Dá mais peso ao recente
        team_stats[f'ema_{col}_general'] = grouped[col].transform(
            lambda x: x.shift(1).ewm(span=window_long, min_periods=1).mean()
        )
        # Volatilidade (Std Dev)
        team_stats[f'std_{col}_general'] = grouped[col].transform(
            lambda x: x.shift(1).rolling(window=window_long, min_periods=2).std()
        ).fillna(0)
        
        # Trend (Curto - Longo)
        team_stats[f'trend_{col}'] = team_stats[f'avg_{col}_short'] - team_stats[f'avg_{col}_general']

    # --- C. Médias ESPECÍFICAS (Home/Away) ---
    home_games = team_stats[team_stats['is_home'] == 1].sort_values(['team_id', 'start_timestamp'])
    away_games = team_stats[team_stats['is_home'] == 0].sort_values(['team_id', 'start_timestamp'])
    
    for col in feature_cols:
        # Média EM CASA
        home_games[f'avg_{col}_home'] = home_games.groupby('team_id')[col].transform(
            lambda x: x.shift(1).rolling(window=window_long, min_periods=1).mean()
        )
        # Média FORA
        away_games[f'avg_{col}_away'] = away_games.groupby('team_id')[col].transform(
            lambda x: x.shift(1).rolling(window=window_long, min_periods=1).mean()
        )

    # Merge de volta
    team_stats = team_stats.merge(
        home_games[['match_id', 'team_id'] + [f'avg_{col}_home' for col in feature_cols]], 
        on=['match_id', 'team_id'], how='left'
    )
    team_stats = team_stats.merge(
        away_games[['match_id', 'team_id'] + [f'avg_{col}_away' for col in feature_cols]], 
        on=['match_id', 'team_id'], how='left'
    )
    
    # Fillna com média geral
    for col in feature_cols:
        team_stats[f'avg_{col}_home'] = team_stats[f'avg_{col}_home'].fillna(team_stats[f'avg_{col}_general'])
        team_stats[f'avg_{col}_away'] = team_stats[f'avg_{col}_away'].fillna(team_stats[f'avg_{col}_general'])

    # --- D. H2H (Confronto Direto) ---
    team_stats = team_stats.sort_values(['team_id', 'opponent_id', 'start_timestamp'])
    h2h_grouped = team_stats.groupby(['team_id', 'opponent_id'])
    
    for col in ['corners', 'corners_conceded']:
        team_stats[f'avg_{col}_h2h'] = h2h_grouped[col].transform(
            lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
        )
    
    team_stats['avg_corners_h2h'] = team_stats['avg_corners_h2h'].fillna(team_stats['avg_corners_general'])
    team_stats['avg_corners_conceded_h2h'] = team_stats['avg_corners_conceded_h2h'].fillna(team_stats['avg_corners_conceded_general'])

    # 4. Reconstrução do Dataset de Partidas
    stats_home = team_stats[team_stats['is_home'] == 1].add_prefix('home_')
    stats_away = team_stats[team_stats['is_home'] == 0].add_prefix('away_')
    
    stats_home = stats_home.rename(columns={'home_match_id': 'match_id'})
    stats_away = stats_away.rename(columns={'away_match_id': 'match_id'})
    
    df_features = df[['match_id', 'start_timestamp', 'tournament_id', 'corners_home_ft', 'corners_away_ft']].merge(
        stats_home, on='match_id', how='inner'
    ).merge(
        stats_away, on='match_id', how='inner'
    )
    
    # 5. Features de Interação (Força Relativa) - NOVO
    # Ataque Casa vs Defesa Visitante
    df_features['home_attack_adv'] = df_features['home_avg_corners_home'] - df_features['away_avg_corners_conceded_away']
    # Ataque Visitante vs Defesa Casa
    df_features['away_attack_adv'] = df_features['away_avg_corners_away'] - df_features['home_avg_corners_conceded_home']
    
    # H2H Dominance (Quem domina o confronto direto comparado à média geral)
    df_features['home_h2h_dominance'] = df_features['home_avg_corners_h2h'] - df_features['home_avg_corners_general']
    df_features['away_h2h_dominance'] = df_features['away_avg_corners_h2h'] - df_features['away_avg_corners_general']
    
    # Diferença de Momentum (Quem está em melhor fase?)
    df_features['momentum_diff'] = df_features['home_trend_corners'] - df_features['away_trend_corners']
    
    # Diferença de Cansaço
    df_features['rest_diff'] = df_features['home_rest_days'] - df_features['away_rest_days']
    
    df_features['tournament_id'] = df_features['tournament_id'].astype('category')
    
    # Limpeza
    df_features = df_features.dropna()
    
    # Definição de X e y
    feature_columns = [
        # Gerais (Momentum) & EMA
        'home_avg_corners_general', 'away_avg_corners_general',
        'home_ema_corners_general', 'away_ema_corners_general',
        'home_avg_corners_conceded_general', 'away_avg_corners_conceded_general',
        
        # Trend & Volatilidade
        'home_trend_corners', 'away_trend_corners',
        'home_std_corners_general', 'away_std_corners_general',
        
        # Específicas (Home/Away)
        'home_avg_corners_home', 'away_avg_corners_away',
        'home_avg_corners_conceded_home', 'away_avg_corners_conceded_away',
        
        # H2H
        'home_avg_corners_h2h', 'away_avg_corners_h2h',
        
        # Interações (Força Relativa)
        'home_attack_adv', 'away_attack_adv',
        'home_h2h_dominance', 'away_h2h_dominance',
        'momentum_diff', 'rest_diff',
        
        # Contexto
        'home_rest_days', 'away_rest_days',
        'home_avg_shots_general', 'away_avg_shots_general',
        'home_avg_goals_general', 'away_avg_goals_general',
        'tournament_id'
    ]
    
    # Garante que todas as colunas existem
    for col in feature_columns:
        if col not in df_features.columns:
            df_features[col] = 0
            
    X = df_features[feature_columns]
    y = df_features['corners_home_ft'] + df_features['corners_away_ft']
    
    return X, y, df_features['start_timestamp']

def prepare_features_for_prediction(home_id, away_id, db_manager, window_long=5):
    """
    Prepara features para um único jogo (Predição) - V3.
    """
    df = db_manager.get_historical_data()
    
    # Filtra jogos relevantes
    relevant_games = df[
        (df['home_team_id'] == home_id) | (df['away_team_id'] == home_id) |
        (df['home_team_id'] == away_id) | (df['away_team_id'] == away_id)
    ].copy()
    
    # Adiciona dummy row para o futuro
    import time
    future_timestamp = int(time.time()) + 86400
    last_tourn = relevant_games['tournament_id'].iloc[-1] if not relevant_games.empty else 'Unknown'
    
    dummy_row = pd.DataFrame([{
        'match_id': 999999999,
        'start_timestamp': future_timestamp,
        'home_team_id': home_id,
        'away_team_id': away_id,
        'corners_home_ft': 0, 'corners_away_ft': 0,
        'shots_ot_home_ft': 0, 'shots_ot_away_ft': 0,
        'home_score': 0, 'away_score': 0,
        'corners_home_ht': 0, 'corners_away_ht': 0,
        'tournament_id': last_tourn,
        'tournament_name': 'Prediction'
    }])
    
    df_combined = pd.concat([relevant_games, dummy_row], ignore_index=True)
    
    # Gera features V3
    X, _, _ = create_advanced_features(df_combined, window_long=window_long)
    
    features = X.iloc[[-1]]
    
    # Verificação de Histórico
    if features['home_avg_corners_general'].iloc[0] == 0 and len(relevant_games) < 5:
         home_games_count = len(relevant_games[(relevant_games['home_team_id'] == home_id) | (relevant_games['away_team_id'] == home_id)])
         away_games_count = len(relevant_games[(relevant_games['home_team_id'] == away_id) | (relevant_games['away_team_id'] == away_id)])
         
         msg = "Histórico insuficiente."
         if home_games_count < window_long:
             msg += f" Mandante tem apenas {home_games_count} jogos."
         if away_games_count < window_long:
             msg += f" Visitante tem apenas {away_games_count} jogos."
             
         raise ValueError(f"{msg} (Mínimo {window_long}).")

    return features