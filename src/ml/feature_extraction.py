"""
Módulo de Extração de Features para Modelos de ML.

Este módulo centraliza a lógica de cálculo de features para garantir
consistência entre treinamento e inferência (análise/scanner).

Regra de Negócio:
    - CRÍTICO: As features devem ser IDÊNTICAS no treino e na inferência
    - Qualquer mudança aqui afeta TODO o sistema ML
    - Features baseadas em médias dos últimos 5 jogos de cada time

Contexto:
    Criado para resolver inconsistências entre features calculadas
    manualmente em server.py e as calculadas em model_improved.py.
    Antes havia código duplicado em 3 lugares diferentes.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional


def calculate_features_for_match(
    df_history: pd.DataFrame,
    home_team: str,
    away_team: str,
    min_games: int = 3
) -> Optional[List[float]]:
    """
    Calcula features para uma partida específica baseado no histórico.
    
    Regra de Negócio:
        - Usa últimos 5 jogos de cada time (ou menos se não houver)
        - Mínimo de 3 jogos por time para garantir qualidade
        - Features: médias de escanteios, chutes, gols, tendências
    
    Args:
        df_history: DataFrame com histórico de partidas
        home_team: Nome do time da casa
        away_team: Nome do time visitante
        min_games: Mínimo de jogos necessários (padrão: 3)
        
    Returns:
        Lista com 12 features ou None se dados insuficientes
        
    Features (12 total):
        [0-2]: Home - avg corners, shots, goals (últimos 5)
        [3-5]: Away - avg corners, shots, goals (últimos 5)
        [6-7]: Home/Away - avg corners HT (meio-tempo)
        [8]: Total esperado (soma médias de escanteios)
        [9]: Diferença (home - away corners)
        [10]: Tendência home (últimos 3 vs média geral)
        [11]: Tendência away (últimos 3 vs média geral)
    """
    # Filtra jogos do time da casa
    h_games = df_history[
        (df_history['home_team_name'] == home_team) | 
        (df_history['away_team_name'] == home_team)
    ].sort_values('start_timestamp').tail(5)
    
    # Filtra jogos do time visitante
    a_games = df_history[
        (df_history['home_team_name'] == away_team) | 
        (df_history['away_team_name'] == away_team)
    ].sort_values('start_timestamp').tail(5)
    
    # Validação: mínimo de jogos
    if len(h_games) < min_games or len(a_games) < min_games:
        return None
    
    # Extrai estatísticas
    h_c, h_s, h_g, h_cht = _get_team_stats(h_games, home_team)
    a_c, a_s, a_g, a_cht = _get_team_stats(a_games, away_team)
    
    # Calcula médias
    def avg(lst): 
        return sum(lst) / len(lst) if lst else 0
    
    # Monta vetor de features (12 features)
    features = [
        avg(h_c),           # 0: Home avg corners
        avg(h_s),           # 1: Home avg shots
        avg(h_g),           # 2: Home avg goals
        avg(a_c),           # 3: Away avg corners
        avg(a_s),           # 4: Away avg shots
        avg(a_g),           # 5: Away avg goals
        avg(h_cht),         # 6: Home avg corners HT
        avg(a_cht),         # 7: Away avg corners HT
        avg(h_c) + avg(a_c),                    # 8: Total esperado
        avg(h_c) - avg(a_c),                    # 9: Diferença
        avg(h_c[-3:]) - avg(h_c) if len(h_c) >= 3 else 0,  # 10: Tendência home
        avg(a_c[-3:]) - avg(a_c) if len(a_c) >= 3 else 0   # 11: Tendência away
    ]
    
    return features


def _get_team_stats(games: pd.DataFrame, team_name: str) -> Tuple[List, List, List, List]:
    """
    Extrai estatísticas de um time a partir de seus jogos.
    
    Regra de Negócio:
        - Se o time jogou em casa, usa stats de casa
        - Se jogou fora, usa stats de visitante
        - Retorna listas paralelas (mesmo índice = mesmo jogo)
    
    Args:
        games: DataFrame com jogos do time
        team_name: Nome do time
        
    Returns:
        Tupla (corners, shots, goals, corners_ht)
    """
    corners = []
    shots = []
    goals = []
    corners_ht = []
    
    for _, row in games.iterrows():
        if row['home_team_name'] == team_name:
            # Time jogou em casa
            corners.append(row['corners_home_ft'])
            shots.append(row['shots_ot_home_ft'])
            goals.append(row['home_score'])
            corners_ht.append(row['corners_home_ht'])
        else:
            # Time jogou fora
            corners.append(row['corners_away_ft'])
            shots.append(row['shots_ot_away_ft'])
            goals.append(row['away_score'])
            corners_ht.append(row['corners_away_ht'])
    
    return corners, shots, goals, corners_ht


def get_feature_names() -> List[str]:
    """
    Retorna nomes descritivos das features.
    
    Útil para debugging e análise de importância.
    
    Returns:
        Lista com 12 nomes de features
    """
    return [
        'home_avg_corners',
        'home_avg_shots',
        'home_avg_goals',
        'away_avg_corners',
        'away_avg_shots',
        'away_avg_goals',
        'home_avg_corners_ht',
        'away_avg_corners_ht',
        'total_expected_corners',
        'corners_difference',
        'home_trend',
        'away_trend'
    ]


def calculate_rolling_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Prepara features para treinamento usando dados históricos completos.
    
    Regra de Negócio:
        - Usa TODOS os jogos do DataFrame para treino
        - Calcula features rolling (janela móvel) para cada jogo
        - Garante que features sejam consistentes com calculate_features_for_match
    
    Args:
        df: DataFrame com histórico completo de jogos
        
    Returns:
        Tupla (X, y, feature_names)
        - X: DataFrame com features
        - y: Series com target (total de escanteios)
        - feature_names: Lista com nomes das features
    """
    X_list = []
    y_list = []
    
    # Para cada jogo, calcula features baseadas no histórico ANTERIOR
    for idx, row in df.iterrows():
        # Histórico até este jogo (excluindo o jogo atual)
        df_before = df[df['start_timestamp'] < row['start_timestamp']]
        
        if df_before.empty:
            continue
        
        # Calcula features
        features = calculate_features_for_match(
            df_before,
            row['home_team_name'],
            row['away_team_name'],
            min_games=3
        )
        
        if features is None:
            continue
        
        # Target: total de escanteios do jogo
        target = row['corners_home_ft'] + row['corners_away_ft']
        
        X_list.append(features)
        y_list.append(target)
    
    X = pd.DataFrame(X_list, columns=get_feature_names())
    y = pd.Series(y_list)
    
    return X, y, get_feature_names()
