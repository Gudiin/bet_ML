import pandas as pd
import numpy as np

# ============================================================================
# FUNÇÕES AUXILIARES - MELHORIAS V5 (Auditoria ML)
# ============================================================================

def exponential_decay_weight(days_ago: float, half_life: float = 14.0) -> float:
    """
    Peso exponencial baseado em física de decaimento radioativo.
    
    Jogos recentes têm mais peso que jogos antigos.
    
    Args:
        days_ago: Dias desde o jogo
        half_life: Tempo para peso reduzir à metade (default: 14 dias)
    
    Returns:
        Peso entre 0 e 1
    
    Fórmula:
        w(t) = exp(-λ * t)
        onde λ = ln(2) / half_life
    """
    decay_constant = np.log(2) / half_life
    return np.exp(-decay_constant * days_ago)


def calculate_entropy(values: pd.Series) -> float:
    """
    Calcula entropia de Shannon normalizada para série de resultados.
    
    Alta entropia = time imprevisível (resultados variam muito)
    Baixa entropia = time consistente (resultados similares)
    
    Args:
        values: Série de escanteios
    
    Returns:
        Entropia normalizada entre 0 e 1
    
    Fórmula:
        H = -Σ p(x) * log2(p(x))
        Normalizada: H / log2(n_bins)
    """
    if len(values) < 3:
        return 0.5  # Neutro sem dados
    
    # Discretiza em bins (0-5, 5-10, 10-15, 15+)
    bins = [0, 5, 10, 15, 100]
    hist, _ = np.histogram(values, bins=bins)
    
    # Remove zeros
    hist = hist[hist > 0]
    if len(hist) == 0:
        return 0.5
        
    probs = hist / hist.sum()
    
    # Entropia de Shannon
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(len(bins) - 1)
    
    return entropy / max_entropy if max_entropy > 0 else 0

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
    
    # --- A2. Resultado Anterior (Win=1, Draw=0.5, Loss=0) ---
    # Primeiro calcula se ganhou, empatou ou perdeu
    team_stats['goals_scored'] = team_stats.apply(
        lambda row: df.loc[df['match_id'] == row['match_id'], 'home_score' if row['is_home'] == 1 else 'away_score'].values[0] if row['match_id'] in df['match_id'].values else 0, axis=1
    ) if 'goals' in team_stats.columns else team_stats['goals']
    
    team_stats['goals_conceded'] = team_stats.apply(
        lambda row: df.loc[df['match_id'] == row['match_id'], 'away_score' if row['is_home'] == 1 else 'home_score'].values[0] if row['match_id'] in df['match_id'].values else 0, axis=1
    ) if 'goals' in team_stats.columns else 0
    
    # Fallback simples: usa 'goals' que já existe
    team_stats['last_result'] = grouped['goals'].transform(
        lambda x: x.shift(1).apply(lambda g: 1 if g > 1 else (0.5 if g == 1 else 0))
    ).fillna(0.5)  # Neutro se não há histórico
    
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
    
    # --- E. FEATURES V5 (Auditoria ML - CORRIGIDO) ---
    # CORREÇÃO: Decaimento exponencial SEM LEAKAGE
    # Em vez de usar max_timestamp global (que inclui futuro), calculamos
    # o decaimento relativo ao timestamp do PRÓPRIO jogo atual
    
    def calculate_decay_weighted_avg(group):
        """
        Calcula média ponderada por decaimento SEM leakage.
        Para cada jogo, usa apenas jogos ANTERIORES.
        """
        group = group.sort_values('start_timestamp').copy()
        result = []
        
        for idx in range(len(group)):
            current_ts = group.iloc[idx]['start_timestamp']
            
            # Pega apenas jogos ANTERIORES (strict temporal order)
            if idx == 0:
                result.append(np.nan)
                continue
            
            prev_games = group.iloc[:idx].copy()
            
            # Calcula dias desde cada jogo anterior até o jogo atual
            prev_games['days_ago'] = (current_ts - prev_games['start_timestamp']) / 86400
            
            # Aplica peso de decaimento
            prev_games['weight'] = prev_games['days_ago'].apply(
                lambda d: exponential_decay_weight(d, half_life=14.0)
            )
            
            # Média ponderada (últimos 5 jogos para eficiência)
            recent = prev_games.tail(5)
            if len(recent) > 0 and recent['weight'].sum() > 0:
                weighted_avg = (recent['corners'] * recent['weight']).sum() / recent['weight'].sum()
                result.append(weighted_avg)
            else:
                result.append(np.nan)
        
        return pd.Series(result, index=group.index)
    
    team_stats['decay_weighted_corners'] = grouped.apply(calculate_decay_weighted_avg).reset_index(level=0, drop=True)
    team_stats['decay_weighted_corners'] = team_stats['decay_weighted_corners'].fillna(team_stats['avg_corners_general'])
    
    # Entropia (imprevisibilidade) - alta = time instável
    team_stats['entropy_corners'] = grouped['corners'].transform(
        lambda x: x.shift(1).rolling(window=10, min_periods=3).apply(calculate_entropy, raw=False)
    ).fillna(0.5)

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
    
    # --- F. STRENGTH OF SCHEDULE (SoS) - Auditoria V6 ---
    # Mede a força média dos adversários enfrentados
    # Um time que faz 10 escanteios contra o lanterna != 10 contra o líder
    
    # 1. Força defensiva de cada time (proxy: escanteios que cede)
    team_stats = team_stats.sort_values(['team_id', 'start_timestamp'])
    grouped = team_stats.groupby('team_id')  # Regrouping after sort
    
    team_stats['own_defense_strength'] = grouped['corners_conceded'].transform(
        lambda x: x.shift(1).rolling(window=10, min_periods=3).mean()
    ).fillna(5.0)  # Média global como fallback
    
    # 2. Para cada jogo, traz a força defensiva do oponente
    # Cria mapeamento: match_id + team_id -> own_defense_strength
    defense_map = team_stats[['match_id', 'team_id', 'own_defense_strength']].copy()
    defense_map.columns = ['match_id', 'opponent_id', 'opponent_defense_strength']
    
    team_stats = team_stats.merge(
        defense_map,
        on=['match_id', 'opponent_id'],
        how='left'
    )
    team_stats['opponent_defense_strength'] = team_stats['opponent_defense_strength'].fillna(5.0)
    
    # 3. SoS Rolling = Média da força dos oponentes enfrentados recentemente
    grouped = team_stats.groupby('team_id')  # Regrouping
    team_stats['sos_rolling'] = grouped['opponent_defense_strength'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    ).fillna(5.0)
    
    # --- G. GAME STATE (Comportamento Histórico) - Auditoria V7 ---
    # Mede como o time se comporta quando está ganhando vs perdendo
    # Usa resultado PASSADO para prever comportamento futuro
    
    # 1. Determina resultado de cada jogo (gol do time vs gol do oponente)
    team_stats['game_result'] = team_stats.apply(
        lambda row: 'win' if row['goals'] > row.get('goals_conceded', 0) 
                    else ('loss' if row['goals'] < row.get('goals_conceded', 0) else 'draw'),
        axis=1
    )
    
    # 2. Média de escanteios quando PERDEU (histórico)
    def avg_corners_when_result(group, result_type):
        """Calcula média de escanteios em jogos com resultado específico."""
        result = []
        for idx in range(len(group)):
            if idx == 0:
                result.append(np.nan)
                continue
            
            prev_games = group.iloc[:idx]
            filtered = prev_games[prev_games['game_result'] == result_type]
            
            if len(filtered) >= 2:
                result.append(filtered['corners'].tail(5).mean())
            else:
                result.append(np.nan)
        
        return pd.Series(result, index=group.index)
    
    # Corners quando perde (tendência a atacar mais ou menos?)
    team_stats['avg_corners_when_losing'] = grouped.apply(
        lambda g: avg_corners_when_result(g.sort_values('start_timestamp'), 'loss')
    ).reset_index(level=0, drop=True)
    
    # Corners quando ganha
    team_stats['avg_corners_when_winning'] = grouped.apply(
        lambda g: avg_corners_when_result(g.sort_values('start_timestamp'), 'win')
    ).reset_index(level=0, drop=True)
    
    # Fillna com média geral
    team_stats['avg_corners_when_losing'] = team_stats['avg_corners_when_losing'].fillna(team_stats['avg_corners_general'])
    team_stats['avg_corners_when_winning'] = team_stats['avg_corners_when_winning'].fillna(team_stats['avg_corners_general'])
    
    # 3. Desperation Index = corners quando perde - corners quando ganha
    # Positivo = time ataca mais quando perde (desesperado)
    # Negativo = time recua quando perde (defensivo)
    team_stats['desperation_index'] = team_stats['avg_corners_when_losing'] - team_stats['avg_corners_when_winning']

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
    
    # --- 6. Novas Features V4 ---
    # Fase da Temporada (0=início, 0.5=meio, 1=fim)
    # Assume 38 rodadas padrão, ajusta baseado no timestamp relativo ao torneio
    if 'round' in df.columns:
        df_features = df_features.merge(
            df[['match_id', 'round']], on='match_id', how='left'
        )
        df_features['season_stage'] = df_features['round'].fillna(19) / 38
        df_features['season_stage'] = df_features['season_stage'].clip(0, 1)
    else:
        df_features['season_stage'] = 0.5  # Fallback neutro
    
    # Último Resultado
    df_features['home_last_result'] = df_features['home_last_result'] if 'home_last_result' in df_features.columns else 0.5
    df_features['away_last_result'] = df_features['away_last_result'] if 'away_last_result' in df_features.columns else 0.5
    
    # Posição na Tabela (Proxy: baseado em pontos acumulados)
    # Como não temos posição direta, usamos Win Rate recente como proxy
    df_features['home_form_score'] = df_features['home_avg_goals_general'] * 3  # Simula pontos
    df_features['away_form_score'] = df_features['away_avg_goals_general'] * 3
    df_features['position_diff'] = df_features['home_form_score'] - df_features['away_form_score']
    
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
        'tournament_id',
        
        # V4 Features
        'season_stage',
        'position_diff',
        
        # V5 Features (Auditoria ML - Melhorias)
        'home_decay_weighted_corners', 'away_decay_weighted_corners',  # Decaimento exponencial
        'home_entropy_corners', 'away_entropy_corners',  # Imprevisibilidade
        
        # V6 Features (Strength of Schedule - Qualidade do Adversário)
        'home_sos_rolling', 'away_sos_rolling',  # Força média dos oponentes enfrentados
        'home_opponent_defense_strength', 'away_opponent_defense_strength',  # Fraqueza defensiva do adversário atual
        
        # V7 Features (Game State - Comportamento Histórico)
        'home_desperation_index', 'away_desperation_index',  # Positivo = ataca mais quando perde
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