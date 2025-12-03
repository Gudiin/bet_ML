"""
Feature Store Unificado - Versão 2.2 (Bulletproof)

Correção Crítica:
- Previne duplicação de colunas (ex: criar 'goals_ft_home' duas vezes).
- Remove colunas conflitantes antes do processamento.
"""

import pandas as pd
import numpy as np


def create_advanced_features(
    df: pd.DataFrame, 
    window_short: int = 3, 
    window_long: int = 5
) -> tuple:
    """
    Pipeline unificado de Feature Engineering (Vetorizado e Anti-Leakage).
    """
    # 1. Ordenação Temporal
    df = df.sort_values('start_timestamp').copy()
    
    # 1.5. Normalização de Colunas (Blindada)
    # Define o mapeamento desejado: De -> Para
    desired_mapping = {
        'corners_home_ft': 'corners_ft_home',
        'corners_away_ft': 'corners_ft_away',
        'corners_home_ht': 'corners_ht_home',
        'corners_away_ht': 'corners_ht_away',
        'shots_ot_home_ft': 'shots_ot_ft_home',
        'shots_ot_away_ft': 'shots_ot_ft_away',
        'home_score': 'goals_ft_home',
        'away_score': 'goals_ft_away'
    }

    # Aplica o mapeamento com segurança
    for old_name, new_name in desired_mapping.items():
        # Se a coluna antiga existe...
        if old_name in df.columns:
            # E a nova JÁ existe (ex: criada pelo main.py), removemos a antiga para evitar conflito
            if new_name in df.columns:
                df = df.drop(columns=[old_name])
            # Se a nova NÃO existe, renomeamos a antiga
            else:
                df = df.rename(columns={old_name: new_name})
    
    # Remove duplicatas de colunas caso algo tenha passado (Safety Net)
    df = df.loc[:, ~df.columns.duplicated()]

    # 2. Estratégia "Team-Centric"
    # Mapeamento base para as features
    map_base = {
        'corners_ft': 'corners',       
        'shots_ot_ft': 'shots',        
        'goals_ft': 'goals',           
        'corners_ht': 'corners_ht'     
    }
    
    # Define colunas esperadas
    cols_home_src = [f'{k}_home' for k in map_base.keys()]
    cols_away_src = [f'{k}_away' for k in map_base.keys()]
    
    # Verifica quais colunas realmente temos disponíveis
    valid_cols_home = [c for c in cols_home_src if c in df.columns]
    valid_cols_away = [c for c in cols_away_src if c in df.columns]
    
    # Cria DataFrames parciais
    df_home = df[['match_id', 'start_timestamp', 'home_team_id'] + valid_cols_home].copy()
    df_away = df[['match_id', 'start_timestamp', 'away_team_id'] + valid_cols_away].copy()
    
    # Renomeia para nome genérico (ex: corners_ft_home -> corners)
    rename_home = {f'{k}_home': v for k, v in map_base.items() if f'{k}_home' in valid_cols_home}
    rename_away = {f'{k}_away': v for k, v in map_base.items() if f'{k}_away' in valid_cols_away}
    
    df_home = df_home.rename(columns=rename_home).assign(is_home=1)
    df_away = df_away.rename(columns=rename_away).assign(is_home=0)
    
    df_home = df_home.rename(columns={'home_team_id': 'team_id'})
    df_away = df_away.rename(columns={'away_team_id': 'team_id'})
    
    # Stack
    team_stats = pd.concat([df_home, df_away]).sort_values(['team_id', 'start_timestamp'])
    
    # 3. Engenharia Vetorizada
    grouped = team_stats.groupby('team_id')
    
    # Calcula apenas para as colunas que conseguimos mapear
    present_features = set(team_stats.columns) & {'corners', 'shots', 'goals'}
    
    for col in present_features:
        # Média Longa
        team_stats[f'avg_{col}_l{window_long}'] = grouped[col].transform(
            lambda x: x.shift(1).rolling(window=window_long, min_periods=1).mean()
        )
        
        # Média Curta
        team_stats[f'avg_{col}_l{window_short}'] = grouped[col].transform(
            lambda x: x.shift(1).rolling(window=window_short, min_periods=1).mean()
        )
        
        # Tendência
        team_stats[f'trend_{col}'] = (
            team_stats[f'avg_{col}_l{window_short}'] - 
            team_stats[f'avg_{col}_l{window_long}']
        )
    
    # 4. Reconstrução (Merge)
    cols_to_keep = [c for c in team_stats.columns if 'avg_' in c or 'trend_' in c]
    
    if not cols_to_keep:
        # Se não calculou nada, tenta retornar vazio ou erro explicativo
        raise ValueError(f"Não foi possível calcular features. Colunas encontradas: {list(df.columns)}")

    stats_home = team_stats[team_stats['is_home'] == 1][['match_id'] + cols_to_keep].copy()
    stats_away = team_stats[team_stats['is_home'] == 0][['match_id'] + cols_to_keep].copy()
    
    stats_home = stats_home.rename(columns={c: f'home_{c}' for c in cols_to_keep})
    stats_away = stats_away.rename(columns={c: f'away_{c}' for c in cols_to_keep})
    
    # Merge
    # Tenta identificar target
    target_cols = ['corners_ft_home', 'corners_ft_away']
    if 'corners_ft_home' not in df.columns and 'corners_home_ft' in df.columns:
        target_cols = ['corners_home_ft', 'corners_away_ft']
    
    df_features = df[['match_id', 'start_timestamp'] + target_cols].copy()
    df_features = df_features.rename(columns={target_cols[0]: 'corners_home', target_cols[1]: 'corners_away'})
    
    df_features = df_features.merge(stats_home, on='match_id', how='inner')
    df_features = df_features.merge(stats_away, on='match_id', how='inner')
    
    # 5. Features de Confronto
    if f'home_avg_corners_l{window_long}' in df_features.columns:
        df_features['expected_total_corners'] = (
            df_features[f'home_avg_corners_l{window_long}'] + 
            df_features[f'away_avg_corners_l{window_long}']
        )
        df_features['corners_diff_strength'] = (
            df_features[f'home_avg_corners_l{window_long}'] - 
            df_features[f'away_avg_corners_l{window_long}']
        )
    
    if 'home_trend_corners' in df_features.columns:
        df_features['combined_trend'] = (
            df_features['home_trend_corners'] + 
            df_features['away_trend_corners']
        )
    
    # 6. Finalização
    final_cols = [c for c in df_features.columns if 
                  'avg_' in c or 'trend_' in c or 'expected' in c or 'diff' in c or 'combined' in c]
    
    target = df_features['corners_home'] + df_features['corners_away']
    
    mask_valid = df_features[final_cols].notna().all(axis=1)
    
    return (
        df_features.loc[mask_valid, final_cols], 
        target.loc[mask_valid], 
        df_features.loc[mask_valid, 'start_timestamp']
    )


def prepare_features_for_prediction(
    df_history: pd.DataFrame,
    home_team_id: int,
    away_team_id: int,
    window_short: int = 3,
    window_long: int = 5
) -> pd.DataFrame:
    """
    Prepara features para inferência (Robust Version).
    """
    df_history = df_history.sort_values('start_timestamp').copy()
    
    # Mesma lógica de normalização para consistência
    desired_mapping = {
        'corners_home_ft': 'corners_ft_home',
        'corners_away_ft': 'corners_ft_away',
        'shots_ot_home_ft': 'shots_ot_ft_home',
        'shots_ot_away_ft': 'shots_ot_ft_away',
        'home_score': 'goals_ft_home',
        'away_score': 'goals_ft_away'
    }

    for old_name, new_name in desired_mapping.items():
        if old_name in df_history.columns:
            if new_name in df_history.columns:
                df_history = df_history.drop(columns=[old_name])
            else:
                df_history = df_history.rename(columns={old_name: new_name})
    
    df_history = df_history.loc[:, ~df_history.columns.duplicated()]

    def get_team_stats(team_id: int) -> dict:
        team_games = df_history[
            (df_history['home_team_id'] == team_id) | 
            (df_history['away_team_id'] == team_id)
        ].copy()
        
        # Extração segura
        metrics = {
            'corners': ('corners_ft_home', 'corners_ft_away'),
            'shots': ('shots_ot_ft_home', 'shots_ot_ft_away'),
            'goals': ('goals_ft_home', 'goals_ft_away')
        }
        
        extracted = {k: [] for k in metrics}
        
        for _, row in team_games.iterrows():
            is_home = (row['home_team_id'] == team_id)
            for metric, (col_home, col_away) in metrics.items():
                val = row.get(col_home if is_home else col_away, 0)
                extracted[metric].append(val)
        
        def safe_mean(lst, window):
            if not lst: return 0.0
            recent = lst[-window:]
            return np.mean(recent) if recent else 0.0
        
        stats = {}
        for name, data in extracted.items():
            if not data: continue
            avg_l = safe_mean(data, window_long)
            avg_s = safe_mean(data, window_short)
            stats[f'avg_{name}_l{window_long}'] = avg_l
            stats[f'avg_{name}_l{window_short}'] = avg_s
            stats[f'trend_{name}'] = avg_s - avg_l
            
        return stats
    
    home_stats = get_team_stats(home_team_id)
    away_stats = get_team_stats(away_team_id)
    
    features = {}
    for key, value in home_stats.items():
        features[f'home_{key}'] = value
    for key, value in away_stats.items():
        features[f'away_{key}'] = value
        
    # Confronto
    if f'avg_corners_l{window_long}' in home_stats:
        features['expected_total_corners'] = (
            home_stats.get(f'avg_corners_l{window_long}', 0) + 
            away_stats.get(f'avg_corners_l{window_long}', 0)
        )
        features['corners_diff_strength'] = (
            home_stats.get(f'avg_corners_l{window_long}', 0) - 
            away_stats.get(f'avg_corners_l{window_long}', 0)
        )
        
    if 'trend_corners' in home_stats:
        features['combined_trend'] = (
            home_stats.get('trend_corners', 0) + 
            away_stats.get('trend_corners', 0)
        )
    
    return pd.DataFrame([features])