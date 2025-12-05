import pandas as pd
import numpy as np

def create_advanced_features(df: pd.DataFrame, window_short: int = 3, window_long: int = 5) -> tuple:
    """
    Pipeline unificado de Feature Engineering (Vetorizado e Anti-Leakage).
    
    Gera:
    1. Médias móveis Gerais (Momentum)
    2. Médias móveis Específicas (Home/Away) - NOVO
    3. Médias de Concessão (Defesa) - NOVO
    4. H2H (Confronto Direto) - NOVO
    
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
    
    for col in feature_cols:
        # A. Médias GERAIS (Momentum) - Todos os jogos
        team_stats[f'avg_{col}_general'] = grouped[col].transform(
            lambda x: x.shift(1).rolling(window=window_long, min_periods=1).mean()
        )
        
        # B. Médias ESPECÍFICAS (Home/Away)
        # Se is_home=1, pega média dos jogos anteriores onde is_home=1
        # Se is_home=0, pega média dos jogos anteriores onde is_home=0
        
        # Truque vetorizado: Separar em dois grupos, calcular rolling, e juntar
        # Mas precisamos manter o alinhamento temporal.
        # Abordagem: Calcular rolling apenas sobre as linhas filtradas e depois dar merge/join
        
    # Implementação Otimizada de Home/Away Specifics
    # Separa dataframes filtrados
    home_games = team_stats[team_stats['is_home'] == 1].sort_values(['team_id', 'start_timestamp'])
    away_games = team_stats[team_stats['is_home'] == 0].sort_values(['team_id', 'start_timestamp'])
    
    # Calcula rolling específico
    for col in feature_cols:
        # Média EM CASA (para quando joga em casa)
        home_games[f'avg_{col}_home'] = home_games.groupby('team_id')[col].transform(
            lambda x: x.shift(1).rolling(window=window_long, min_periods=1).mean()
        )
        # Média FORA (para quando joga fora)
        away_games[f'avg_{col}_away'] = away_games.groupby('team_id')[col].transform(
            lambda x: x.shift(1).rolling(window=window_long, min_periods=1).mean()
        )

    # Merge de volta para team_stats
    team_stats = team_stats.merge(
        home_games[['match_id', 'team_id'] + [f'avg_{col}_home' for col in feature_cols]], 
        on=['match_id', 'team_id'], how='left'
    )
    team_stats = team_stats.merge(
        away_games[['match_id', 'team_id'] + [f'avg_{col}_away' for col in feature_cols]], 
        on=['match_id', 'team_id'], how='left'
    )
    
    # Fillna: Se não tem histórico específico (ex: 1º jogo em casa), usa a média geral como fallback
    for col in feature_cols:
        team_stats[f'avg_{col}_home'] = team_stats[f'avg_{col}_home'].fillna(team_stats[f'avg_{col}_general'])
        team_stats[f'avg_{col}_away'] = team_stats[f'avg_{col}_away'].fillna(team_stats[f'avg_{col}_general'])

    # 4. H2H (Confronto Direto)
    # Agrupa por par (team_id, opponent_id) e calcula média histórica
    # Nota: H2H é simétrico em termos de "aconteceu", mas as stats são do time
    team_stats = team_stats.sort_values(['team_id', 'opponent_id', 'start_timestamp'])
    h2h_grouped = team_stats.groupby(['team_id', 'opponent_id'])
    
    for col in ['corners', 'corners_conceded']:
        team_stats[f'avg_{col}_h2h'] = h2h_grouped[col].transform(
            lambda x: x.shift(1).rolling(window=3, min_periods=1).mean() # Janela menor para H2H
        )
    
    # Fallback para H2H: Se não tem confronto direto, usa média geral
    team_stats['avg_corners_h2h'] = team_stats['avg_corners_h2h'].fillna(team_stats['avg_corners_general'])
    team_stats['avg_corners_conceded_h2h'] = team_stats['avg_corners_conceded_h2h'].fillna(team_stats['avg_corners_conceded_general'])

    # 5. Reconstrução do Dataset de Partidas
    stats_home = team_stats[team_stats['is_home'] == 1].add_prefix('home_')
    stats_away = team_stats[team_stats['is_home'] == 0].add_prefix('away_')
    
    stats_home = stats_home.rename(columns={'home_match_id': 'match_id'})
    stats_away = stats_away.rename(columns={'away_match_id': 'match_id'})
    
    df_features = df[['match_id', 'start_timestamp', 'tournament_id', 'corners_home_ft', 'corners_away_ft']].merge(
        stats_home, on='match_id', how='inner'
    ).merge(
        stats_away, on='match_id', how='inner'
    )
    
    # 6. Seleção e Renomeação Final das Features
    # O modelo espera nomes específicos? Vamos criar as features finais baseadas no pedido.
    
    # Média a favor do time da casa (jogando em casa) -> home_avg_corners_home
    # Média a favor do time visitante (jogando fora) -> away_avg_corners_away
    # Média de escanteios cedidos pelo mandante em casa -> home_avg_corners_conceded_home
    # Média de escanteios cedidos pelo visitante fora -> away_avg_corners_conceded_away
    # H2H -> home_avg_corners_h2h (do mandante contra esse visitante)
    
    # Features de Interação
    df_features['expected_corners_h2h'] = (df_features['home_avg_corners_h2h'] + df_features['away_avg_corners_h2h']) / 2
    
    df_features['tournament_id'] = df_features['tournament_id'].astype('category')
    
    # Limpeza de NaNs (Primeiros jogos da história)
    df_features = df_features.dropna()
    
    # Definição de X e y
    feature_columns = [
        # Gerais (Momentum)
        'home_avg_corners_general', 'away_avg_corners_general',
        'home_avg_corners_conceded_general', 'away_avg_corners_conceded_general',
        
        # Específicas (Home/Away)
        'home_avg_corners_home', 'away_avg_corners_away',
        'home_avg_corners_conceded_home', 'away_avg_corners_conceded_away',
        
        # H2H
        'home_avg_corners_h2h', 'away_avg_corners_h2h',
        
        # Outros
        'home_avg_shots_general', 'away_avg_shots_general',
        'home_avg_goals_general', 'away_avg_goals_general',
        'tournament_id'
    ]
    
    # Garante que todas as colunas existem (fill com 0 se algo falhou no merge)
    for col in feature_columns:
        if col not in df_features.columns:
            df_features[col] = 0
            
    X = df_features[feature_columns]
    y = df_features['corners_home_ft'] + df_features['corners_away_ft']
    
    return X, y, df_features['start_timestamp']

def prepare_features_for_prediction(home_id, away_id, db_manager, window_long=5):
    """
    Prepara features para um único jogo (Predição).
    """
    df = db_manager.get_historical_data()
    
    # Filtra jogos relevantes para os times (para performance)
    # Mas precisamos de histórico suficiente
    # Pega todos os jogos onde home_id ou away_id jogaram
    relevant_games = df[
        (df['home_team_id'] == home_id) | (df['away_team_id'] == home_id) |
        (df['home_team_id'] == away_id) | (df['away_team_id'] == away_id)
    ].copy()
    
    if len(relevant_games) < window_long * 2:
         # Tenta pegar mais dados se possível, ou avisa
         pass

    # Adiciona uma linha "fictícia" para o jogo atual no futuro
    # para que o shift(1) funcione e pegue as médias até agora
    import time
    future_timestamp = int(time.time()) + 86400
    
    # Precisamos do tournament_id do último jogo desses times ou da liga
    last_tourn = relevant_games['tournament_id'].iloc[-1] if not relevant_games.empty else 'Unknown'
    
    dummy_row = pd.DataFrame([{
        'match_id': 999999999, # Dummy ID
        'start_timestamp': future_timestamp,
        'home_team_id': home_id,
        'away_team_id': away_id,
        'corners_home_ft': 0, 'corners_away_ft': 0, # Dummy
        'shots_ot_home_ft': 0, 'shots_ot_away_ft': 0,
        'home_score': 0, 'away_score': 0,
        'corners_home_ht': 0, 'corners_away_ht': 0,
        'tournament_id': last_tourn,
        'tournament_name': 'Prediction'
    }])
    
    # Concatena e processa
    df_combined = pd.concat([relevant_games, dummy_row], ignore_index=True)
    
    # Gera features
    X, _, _ = create_advanced_features(df_combined, window_long=window_long)
    
    # Pega a última linha (que corresponde ao jogo dummy)
    features = X.iloc[[-1]]
    
    # Verificação de Histórico (Segurança)
    # Se as médias forem 0 ou NaN, significa falta de dados
    if features['home_avg_corners_general'].iloc[0] == 0 and len(relevant_games) < 5:
         # Verifica especificamente quem tem pouco jogo
         home_games_count = len(relevant_games[(relevant_games['home_team_id'] == home_id) | (relevant_games['away_team_id'] == home_id)])
         away_games_count = len(relevant_games[(relevant_games['home_team_id'] == away_id) | (relevant_games['away_team_id'] == away_id)])
         
         msg = "Histórico insuficiente."
         if home_games_count < window_long:
             msg += f" Mandante tem apenas {home_games_count} jogos."
         if away_games_count < window_long:
             msg += f" Visitante tem apenas {away_games_count} jogos."
             
         raise ValueError(f"{msg} (Mínimo {window_long}).")

    return features