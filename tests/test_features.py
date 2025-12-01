"""
Script de Teste de Melhorias em Features.

Testa incrementalmente:
- A4: Fator casa/fora (separar médias)
- A1: Mais features (chutes, gols)
- A2: Features de diferença (corner_diff, etc.)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from src.database.db_manager import DBManager


def calculate_metrics(y_true, y_pred):
    """Calcula métricas."""
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred)
    }


def print_comparison(results):
    """Exibe comparativo."""
    print(f"\n{'='*70}")
    print(f"   📊 COMPARATIVO DE FEATURES")
    print(f"{'='*70}")
    print(f"\n{'Versão':<40} {'MAE':>8} {'RMSE':>8} {'R²':>10}")
    print("-"*70)
    
    baseline_mae = results[0][1]['MAE']
    for name, metrics in results:
        improvement = ((baseline_mae - metrics['MAE']) / baseline_mae) * 100
        r2_pct = f"{metrics['R2']*100:.1f}%"
        print(f"{name:<40} {metrics['MAE']:>8.3f} {metrics['RMSE']:>8.3f} {r2_pct:>10}")
    
    print("-"*70)
    best = min(results, key=lambda x: x[1]['MAE'])
    print(f"\n🏆 MELHOR: {best[0]} (MAE: {best[1]['MAE']:.3f})")


def load_raw_data():
    """Carrega dados brutos do banco."""
    db = DBManager()
    df = db.get_historical_data()
    db.close()
    return df


def feature_engineering_v1_baseline(df):
    """
    V1 - BASELINE: Features atuais (6 features).
    """
    df = df.sort_values('start_timestamp').copy()
    
    # Reestrutura para perspectiva de time
    matches_home = df[['match_id', 'start_timestamp', 'home_team_id', 
                       'corners_home_ft', 'shots_ot_home_ft', 'home_score']].copy()
    matches_home.columns = ['match_id', 'timestamp', 'team_id', 'corners', 'shots', 'goals']
    matches_home['is_home'] = 1
    
    matches_away = df[['match_id', 'start_timestamp', 'away_team_id', 
                       'corners_away_ft', 'shots_ot_away_ft', 'away_score']].copy()
    matches_away.columns = ['match_id', 'timestamp', 'team_id', 'corners', 'shots', 'goals']
    matches_away['is_home'] = 0
    
    team_stats = pd.concat([matches_home, matches_away]).sort_values(['team_id', 'timestamp'])
    
    # Rolling stats GERAIS (sem separar casa/fora)
    for col in ['corners', 'shots', 'goals']:
        team_stats[f'avg_{col}_5'] = team_stats.groupby('team_id')[col].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
        )
    
    # Merge back
    df_features = df.copy()
    
    home_stats = team_stats[team_stats['is_home'] == 1][
        ['match_id', 'avg_corners_5', 'avg_shots_5', 'avg_goals_5']
    ]
    home_stats.columns = ['match_id', 'home_avg_corners', 'home_avg_shots', 'home_avg_goals']
    df_features = df_features.merge(home_stats, on='match_id', how='left')
    
    away_stats = team_stats[team_stats['is_home'] == 0][
        ['match_id', 'avg_corners_5', 'avg_shots_5', 'avg_goals_5']
    ]
    away_stats.columns = ['match_id', 'away_avg_corners', 'away_avg_shots', 'away_avg_goals']
    df_features = df_features.merge(away_stats, on='match_id', how='left')
    
    df_features = df_features.dropna()
    
    features = ['home_avg_corners', 'home_avg_shots', 'home_avg_goals',
                'away_avg_corners', 'away_avg_shots', 'away_avg_goals']
    
    X = df_features[features]
    y = df_features['corners_home_ft'] + df_features['corners_away_ft']
    
    return X, y


def feature_engineering_v2_home_away(df):
    """
    V2 - A4: Separar médias CASA vs FORA.
    
    Ideia: Time joga diferente em casa vs fora.
    Calcula média de escanteios quando joga EM CASA e quando joga FORA separadamente.
    """
    df = df.sort_values('start_timestamp').copy()
    
    # Para cada time, calcular médias separadas
    matches_home = df[['match_id', 'start_timestamp', 'home_team_id', 
                       'corners_home_ft', 'shots_ot_home_ft', 'home_score']].copy()
    matches_home.columns = ['match_id', 'timestamp', 'team_id', 'corners', 'shots', 'goals']
    matches_home['is_home'] = 1
    
    matches_away = df[['match_id', 'start_timestamp', 'away_team_id', 
                       'corners_away_ft', 'shots_ot_away_ft', 'away_score']].copy()
    matches_away.columns = ['match_id', 'timestamp', 'team_id', 'corners', 'shots', 'goals']
    matches_away['is_home'] = 0
    
    team_stats = pd.concat([matches_home, matches_away]).sort_values(['team_id', 'timestamp'])
    
    # Rolling stats SEPARADOS por casa/fora
    # Média quando joga em casa
    team_stats['avg_corners_home'] = team_stats.groupby(['team_id', 'is_home'])['corners'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    )
    team_stats['avg_shots_home'] = team_stats.groupby(['team_id', 'is_home'])['shots'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    )
    
    # Também mantém média geral
    team_stats['avg_corners_all'] = team_stats.groupby('team_id')['corners'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    )
    team_stats['avg_shots_all'] = team_stats.groupby('team_id')['shots'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    )
    team_stats['avg_goals_all'] = team_stats.groupby('team_id')['goals'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    )
    
    # Merge back
    df_features = df.copy()
    
    # Stats do mandante (que joga em casa)
    home_stats = team_stats[team_stats['is_home'] == 1][
        ['match_id', 'avg_corners_home', 'avg_shots_home', 'avg_corners_all', 'avg_shots_all', 'avg_goals_all']
    ]
    home_stats.columns = ['match_id', 'home_avg_corners_at_home', 'home_avg_shots_at_home',
                          'home_avg_corners', 'home_avg_shots', 'home_avg_goals']
    df_features = df_features.merge(home_stats, on='match_id', how='left')
    
    # Stats do visitante (que joga fora)
    away_stats = team_stats[team_stats['is_home'] == 0][
        ['match_id', 'avg_corners_home', 'avg_shots_home', 'avg_corners_all', 'avg_shots_all', 'avg_goals_all']
    ]
    away_stats.columns = ['match_id', 'away_avg_corners_away', 'away_avg_shots_away',
                          'away_avg_corners', 'away_avg_shots', 'away_avg_goals']
    df_features = df_features.merge(away_stats, on='match_id', how='left')
    
    df_features = df_features.dropna()
    
    # 10 features agora
    features = [
        'home_avg_corners', 'home_avg_shots', 'home_avg_goals',
        'away_avg_corners', 'away_avg_shots', 'away_avg_goals',
        'home_avg_corners_at_home', 'home_avg_shots_at_home',
        'away_avg_corners_away', 'away_avg_shots_away'
    ]
    
    X = df_features[features]
    y = df_features['corners_home_ft'] + df_features['corners_away_ft']
    
    return X, y


def feature_engineering_v3_differences(df):
    """
    V3 - A2: Adicionar features de DIFERENÇA.
    
    Ideia: Capturar vantagem relativa entre os times.
    """
    df = df.sort_values('start_timestamp').copy()
    
    # Mesmo processo de V2
    matches_home = df[['match_id', 'start_timestamp', 'home_team_id', 
                       'corners_home_ft', 'shots_ot_home_ft', 'home_score']].copy()
    matches_home.columns = ['match_id', 'timestamp', 'team_id', 'corners', 'shots', 'goals']
    matches_home['is_home'] = 1
    
    matches_away = df[['match_id', 'start_timestamp', 'away_team_id', 
                       'corners_away_ft', 'shots_ot_away_ft', 'away_score']].copy()
    matches_away.columns = ['match_id', 'timestamp', 'team_id', 'corners', 'shots', 'goals']
    matches_away['is_home'] = 0
    
    team_stats = pd.concat([matches_home, matches_away]).sort_values(['team_id', 'timestamp'])
    
    # Rolling stats
    for col in ['corners', 'shots', 'goals']:
        team_stats[f'avg_{col}'] = team_stats.groupby('team_id')[col].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
        )
    
    # Merge
    df_features = df.copy()
    
    home_stats = team_stats[team_stats['is_home'] == 1][
        ['match_id', 'avg_corners', 'avg_shots', 'avg_goals']
    ]
    home_stats.columns = ['match_id', 'home_avg_corners', 'home_avg_shots', 'home_avg_goals']
    df_features = df_features.merge(home_stats, on='match_id', how='left')
    
    away_stats = team_stats[team_stats['is_home'] == 0][
        ['match_id', 'avg_corners', 'avg_shots', 'avg_goals']
    ]
    away_stats.columns = ['match_id', 'away_avg_corners', 'away_avg_shots', 'away_avg_goals']
    df_features = df_features.merge(away_stats, on='match_id', how='left')
    
    # NOVAS FEATURES: Diferenças
    df_features['corner_diff'] = df_features['home_avg_corners'] - df_features['away_avg_corners']
    df_features['shots_diff'] = df_features['home_avg_shots'] - df_features['away_avg_shots']
    df_features['goals_diff'] = df_features['home_avg_goals'] - df_features['away_avg_goals']
    
    # Total esperado (soma das médias)
    df_features['total_corners_expected'] = df_features['home_avg_corners'] + df_features['away_avg_corners']
    df_features['total_shots_expected'] = df_features['home_avg_shots'] + df_features['away_avg_shots']
    
    df_features = df_features.dropna()
    
    features = [
        'home_avg_corners', 'home_avg_shots', 'home_avg_goals',
        'away_avg_corners', 'away_avg_shots', 'away_avg_goals',
        'corner_diff', 'shots_diff', 'goals_diff',
        'total_corners_expected', 'total_shots_expected'
    ]
    
    X = df_features[features]
    y = df_features['corners_home_ft'] + df_features['corners_away_ft']
    
    return X, y


def feature_engineering_v4_all_combined(df):
    """
    V4 - COMBINADO: Casa/Fora + Diferenças + Mais features.
    """
    df = df.sort_values('start_timestamp').copy()
    
    matches_home = df[['match_id', 'start_timestamp', 'home_team_id', 
                       'corners_home_ft', 'shots_ot_home_ft', 'home_score',
                       'corners_home_ht', 'shots_ot_home_ht']].copy()
    matches_home.columns = ['match_id', 'timestamp', 'team_id', 'corners', 'shots', 'goals',
                            'corners_ht', 'shots_ht']
    matches_home['is_home'] = 1
    
    matches_away = df[['match_id', 'start_timestamp', 'away_team_id', 
                       'corners_away_ft', 'shots_ot_away_ft', 'away_score',
                       'corners_away_ht', 'shots_ot_away_ht']].copy()
    matches_away.columns = ['match_id', 'timestamp', 'team_id', 'corners', 'shots', 'goals',
                            'corners_ht', 'shots_ht']
    matches_away['is_home'] = 0
    
    team_stats = pd.concat([matches_home, matches_away]).sort_values(['team_id', 'timestamp'])
    
    # Rolling stats gerais
    for col in ['corners', 'shots', 'goals', 'corners_ht', 'shots_ht']:
        team_stats[f'avg_{col}'] = team_stats.groupby('team_id')[col].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
        )
    
    # Rolling stats por casa/fora
    team_stats['avg_corners_venue'] = team_stats.groupby(['team_id', 'is_home'])['corners'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    )
    
    # Variância (volatilidade)
    team_stats['var_corners'] = team_stats.groupby('team_id')['corners'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).std()
    )
    
    # Merge
    df_features = df.copy()
    
    home_stats = team_stats[team_stats['is_home'] == 1][
        ['match_id', 'avg_corners', 'avg_shots', 'avg_goals', 
         'avg_corners_ht', 'avg_shots_ht', 'avg_corners_venue', 'var_corners']
    ]
    home_stats.columns = ['match_id', 'home_avg_corners', 'home_avg_shots', 'home_avg_goals',
                          'home_avg_corners_ht', 'home_avg_shots_ht', 
                          'home_avg_corners_at_home', 'home_var_corners']
    df_features = df_features.merge(home_stats, on='match_id', how='left')
    
    away_stats = team_stats[team_stats['is_home'] == 0][
        ['match_id', 'avg_corners', 'avg_shots', 'avg_goals',
         'avg_corners_ht', 'avg_shots_ht', 'avg_corners_venue', 'var_corners']
    ]
    away_stats.columns = ['match_id', 'away_avg_corners', 'away_avg_shots', 'away_avg_goals',
                          'away_avg_corners_ht', 'away_avg_shots_ht',
                          'away_avg_corners_away', 'away_var_corners']
    df_features = df_features.merge(away_stats, on='match_id', how='left')
    
    # Features derivadas
    df_features['corner_diff'] = df_features['home_avg_corners'] - df_features['away_avg_corners']
    df_features['shots_diff'] = df_features['home_avg_shots'] - df_features['away_avg_shots']
    df_features['total_corners_expected'] = df_features['home_avg_corners'] + df_features['away_avg_corners']
    df_features['total_shots_expected'] = df_features['home_avg_shots'] + df_features['away_avg_shots']
    
    # Ratio de primeiro tempo
    df_features['home_ht_ratio'] = df_features['home_avg_corners_ht'] / (df_features['home_avg_corners'] + 0.1)
    df_features['away_ht_ratio'] = df_features['away_avg_corners_ht'] / (df_features['away_avg_corners'] + 0.1)
    
    df_features = df_features.dropna()
    
    features = [
        'home_avg_corners', 'home_avg_shots', 'home_avg_goals',
        'away_avg_corners', 'away_avg_shots', 'away_avg_goals',
        'home_avg_corners_ht', 'home_avg_shots_ht',
        'away_avg_corners_ht', 'away_avg_shots_ht',
        'home_avg_corners_at_home', 'away_avg_corners_away',
        'home_var_corners', 'away_var_corners',
        'corner_diff', 'shots_diff',
        'total_corners_expected', 'total_shots_expected',
        'home_ht_ratio', 'away_ht_ratio'
    ]
    
    X = df_features[features]
    y = df_features['corners_home_ft'] + df_features['corners_away_ft']
    
    return X, y


def test_model(X, y, version_name):
    """Treina e avalia modelo."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    metrics = calculate_metrics(y_test, y_pred)
    
    print(f"\n{version_name}")
    print(f"   Features: {len(X.columns)}")
    print(f"   MAE: {metrics['MAE']:.3f} | RMSE: {metrics['RMSE']:.3f} | R²: {metrics['R2']:.3f}")
    
    return metrics


def main():
    print("="*70)
    print("   TESTE DE MELHORIAS EM FEATURES")
    print("="*70)
    
    df = load_raw_data()
    if df.empty:
        print("❌ Sem dados!")
        return
    
    print(f"\n📊 Dados carregados: {len(df)} registros")
    
    results = []
    
    # V1 - Baseline
    X, y = feature_engineering_v1_baseline(df)
    metrics = test_model(X, y, "V1 - BASELINE (6 features)")
    results.append(("V1 - Baseline", metrics))
    
    # V2 - Casa/Fora
    X, y = feature_engineering_v2_home_away(df)
    metrics = test_model(X, y, "V2 - A4: Casa/Fora (10 features)")
    results.append(("V2 - Casa/Fora", metrics))
    
    # V3 - Diferenças
    X, y = feature_engineering_v3_differences(df)
    metrics = test_model(X, y, "V3 - A2: Diferenças (11 features)")
    results.append(("V3 - Diferenças", metrics))
    
    # V4 - Combinado
    X, y = feature_engineering_v4_all_combined(df)
    metrics = test_model(X, y, "V4 - COMBINADO (20 features)")
    results.append(("V4 - Combinado", metrics))
    
    print_comparison(results)


if __name__ == "__main__":
    main()
