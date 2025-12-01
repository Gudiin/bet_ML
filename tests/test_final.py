"""
Script Final - Combinação das Melhores Melhorias.

Combina:
- Melhores features (V4 combinado)
- Melhor modelo (LightGBM Tuned)
- Ensemble otimizado
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
import lightgbm as lgb

from src.database.db_manager import DBManager


def calculate_metrics(y_true, y_pred):
    """Calcula métricas."""
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred)
    }


def feature_engineering_baseline(df):
    """V1 - Features atuais (baseline)."""
    df = df.sort_values('start_timestamp').copy()
    
    matches_home = df[['match_id', 'start_timestamp', 'home_team_id', 
                       'corners_home_ft', 'shots_ot_home_ft', 'home_score']].copy()
    matches_home.columns = ['match_id', 'timestamp', 'team_id', 'corners', 'shots', 'goals']
    matches_home['is_home'] = 1
    
    matches_away = df[['match_id', 'start_timestamp', 'away_team_id', 
                       'corners_away_ft', 'shots_ot_away_ft', 'away_score']].copy()
    matches_away.columns = ['match_id', 'timestamp', 'team_id', 'corners', 'shots', 'goals']
    matches_away['is_home'] = 0
    
    team_stats = pd.concat([matches_home, matches_away]).sort_values(['team_id', 'timestamp'])
    
    for col in ['corners', 'shots', 'goals']:
        team_stats[f'avg_{col}_5'] = team_stats.groupby('team_id')[col].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
        )
    
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


def feature_engineering_optimized(df):
    """
    Features otimizadas - seleção baseada em importância.
    
    Mantém features mais simples e diretas que funcionam melhor.
    """
    df = df.sort_values('start_timestamp').copy()
    
    matches_home = df[['match_id', 'start_timestamp', 'home_team_id', 
                       'corners_home_ft', 'shots_ot_home_ft', 'home_score',
                       'corners_home_ht']].copy()
    matches_home.columns = ['match_id', 'timestamp', 'team_id', 'corners', 'shots', 'goals', 'corners_ht']
    matches_home['is_home'] = 1
    
    matches_away = df[['match_id', 'start_timestamp', 'away_team_id', 
                       'corners_away_ft', 'shots_ot_away_ft', 'away_score',
                       'corners_away_ht']].copy()
    matches_away.columns = ['match_id', 'timestamp', 'team_id', 'corners', 'shots', 'goals', 'corners_ht']
    matches_away['is_home'] = 0
    
    team_stats = pd.concat([matches_home, matches_away]).sort_values(['team_id', 'timestamp'])
    
    # Rolling stats com janelas diferentes
    for col in ['corners', 'shots', 'goals']:
        # Últimos 5 jogos
        team_stats[f'avg_{col}_5'] = team_stats.groupby('team_id')[col].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
        )
        # Últimos 3 jogos (tendência recente)
        team_stats[f'avg_{col}_3'] = team_stats.groupby('team_id')[col].transform(
            lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
        )
    
    # Média HT
    team_stats['avg_corners_ht_5'] = team_stats.groupby('team_id')['corners_ht'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    )
    
    df_features = df.copy()
    
    home_stats = team_stats[team_stats['is_home'] == 1][
        ['match_id', 'avg_corners_5', 'avg_shots_5', 'avg_goals_5', 
         'avg_corners_3', 'avg_corners_ht_5']
    ]
    home_stats.columns = ['match_id', 'home_avg_corners', 'home_avg_shots', 'home_avg_goals',
                          'home_avg_corners_3', 'home_avg_corners_ht']
    df_features = df_features.merge(home_stats, on='match_id', how='left')
    
    away_stats = team_stats[team_stats['is_home'] == 0][
        ['match_id', 'avg_corners_5', 'avg_shots_5', 'avg_goals_5',
         'avg_corners_3', 'avg_corners_ht_5']
    ]
    away_stats.columns = ['match_id', 'away_avg_corners', 'away_avg_shots', 'away_avg_goals',
                          'away_avg_corners_3', 'away_avg_corners_ht']
    df_features = df_features.merge(away_stats, on='match_id', how='left')
    
    # Features derivadas simples
    df_features['total_corners_expected'] = df_features['home_avg_corners'] + df_features['away_avg_corners']
    df_features['corner_diff'] = df_features['home_avg_corners'] - df_features['away_avg_corners']
    
    # Tendência (diferença entre média 3 e média 5)
    df_features['home_trend'] = df_features['home_avg_corners_3'] - df_features['home_avg_corners']
    df_features['away_trend'] = df_features['away_avg_corners_3'] - df_features['away_avg_corners']
    
    df_features = df_features.dropna()
    
    features = [
        'home_avg_corners', 'home_avg_shots', 'home_avg_goals',
        'away_avg_corners', 'away_avg_shots', 'away_avg_goals',
        'home_avg_corners_ht', 'away_avg_corners_ht',
        'total_corners_expected', 'corner_diff',
        'home_trend', 'away_trend'
    ]
    
    X = df_features[features]
    y = df_features['corners_home_ft'] + df_features['corners_away_ft']
    
    return X, y


def test_baseline(X, y):
    """Modelo baseline (RF com features atuais)."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return calculate_metrics(y_test, y_pred)


def test_optimized(X, y):
    """Modelo otimizado (LightGBM com features melhoradas)."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # LightGBM com parâmetros otimizados
    model = lgb.LGBMRegressor(
        n_estimators=50,
        learning_rate=0.01,
        max_depth=3,
        num_leaves=15,
        subsample=0.8,
        random_state=42,
        verbosity=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return calculate_metrics(y_test, y_pred), model, X_test, y_test


def test_ensemble_optimized(X, y):
    """Ensemble otimizado com pesos."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=50, learning_rate=0.01, max_depth=3,
        num_leaves=15, subsample=0.8, random_state=42, verbosity=-1
    )
    lgb_model.fit(X_train, y_train)
    lgb_pred = lgb_model.predict(X_test)
    
    # XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=50, learning_rate=0.01, max_depth=3,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    
    # Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=100, max_depth=5, min_samples_leaf=2,
        min_samples_split=10, random_state=42
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    
    # Ensemble com pesos (dando mais peso para LightGBM)
    y_pred = 0.4 * lgb_pred + 0.35 * xgb_pred + 0.25 * rf_pred
    
    return calculate_metrics(y_test, y_pred)


def analyze_predictions(y_test, y_pred):
    """Analisa distribuição dos erros."""
    errors = np.abs(y_test - y_pred)
    
    print(f"\n📊 Análise de Erros:")
    print(f"   Erro médio: {errors.mean():.2f} escanteios")
    print(f"   Erro mediano: {np.median(errors):.2f} escanteios")
    print(f"   Erro máximo: {errors.max():.2f} escanteios")
    print(f"   Erro mínimo: {errors.min():.2f} escanteios")
    
    # Distribuição
    print(f"\n   Distribuição de acertos:")
    print(f"   ±1 escanteio: {(errors <= 1).mean()*100:.1f}%")
    print(f"   ±2 escanteios: {(errors <= 2).mean()*100:.1f}%")
    print(f"   ±3 escanteios: {(errors <= 3).mean()*100:.1f}%")


def main():
    print("="*70)
    print("   TESTE FINAL - COMBINAÇÃO DAS MELHORES MELHORIAS")
    print("="*70)
    
    # Carrega dados
    db = DBManager()
    df = db.get_historical_data()
    db.close()
    
    print(f"\n📊 Dados carregados: {len(df)} registros")
    
    results = {}
    
    # 1. Baseline (RF + features atuais)
    print("\n" + "-"*70)
    print("1️⃣ BASELINE: Random Forest + 6 Features Atuais")
    print("-"*70)
    X_base, y_base = feature_engineering_baseline(df)
    print(f"   Features: {list(X_base.columns)}")
    metrics_base = test_baseline(X_base, y_base)
    results['Baseline (RF)'] = metrics_base
    print(f"   MAE: {metrics_base['MAE']:.3f} | R²: {metrics_base['R2']:.3f}")
    
    # 2. LightGBM + features atuais
    print("\n" + "-"*70)
    print("2️⃣ LightGBM Tuned + 6 Features Atuais")
    print("-"*70)
    metrics_lgb_base, _, _, _ = test_optimized(X_base, y_base)
    results['LightGBM (6 feat)'] = metrics_lgb_base
    print(f"   MAE: {metrics_lgb_base['MAE']:.3f} | R²: {metrics_lgb_base['R2']:.3f}")
    
    # 3. LightGBM + features otimizadas
    print("\n" + "-"*70)
    print("3️⃣ LightGBM Tuned + 12 Features Otimizadas")
    print("-"*70)
    X_opt, y_opt = feature_engineering_optimized(df)
    print(f"   Features: {list(X_opt.columns)}")
    metrics_lgb_opt, model, X_test, y_test = test_optimized(X_opt, y_opt)
    results['LightGBM (12 feat)'] = metrics_lgb_opt
    print(f"   MAE: {metrics_lgb_opt['MAE']:.3f} | R²: {metrics_lgb_opt['R2']:.3f}")
    
    # 4. Ensemble otimizado
    print("\n" + "-"*70)
    print("4️⃣ Ensemble (LGB 40% + XGB 35% + RF 25%) + 12 Features")
    print("-"*70)
    metrics_ensemble = test_ensemble_optimized(X_opt, y_opt)
    results['Ensemble Otimizado'] = metrics_ensemble
    print(f"   MAE: {metrics_ensemble['MAE']:.3f} | R²: {metrics_ensemble['R2']:.3f}")
    
    # Comparativo final
    print("\n" + "="*70)
    print("   📊 RESULTADO FINAL")
    print("="*70)
    
    print(f"\n{'Configuração':<30} {'MAE':>10} {'RMSE':>10} {'R²':>10} {'Δ MAE':>10}")
    print("-"*70)
    
    baseline_mae = results['Baseline (RF)']['MAE']
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['MAE'])
    
    for name, metrics in sorted_results:
        improvement = ((baseline_mae - metrics['MAE']) / baseline_mae) * 100
        r2_str = f"{metrics['R2']:.3f}"
        delta = f"{improvement:+.1f}%" if name != 'Baseline (RF)' else "-"
        print(f"{name:<30} {metrics['MAE']:>10.3f} {metrics['RMSE']:>10.3f} {r2_str:>10} {delta:>10}")
    
    # Análise detalhada do melhor
    print("\n" + "="*70)
    best = sorted_results[0]
    print(f"🏆 MELHOR: {best[0]}")
    improvement = ((baseline_mae - best[1]['MAE']) / baseline_mae) * 100
    print(f"   Melhoria total: {improvement:.1f}% sobre baseline")
    
    # Análise de erros
    y_pred = model.predict(X_test)
    analyze_predictions(y_test.values, y_pred)
    
    # Feature importance
    print("\n📊 Importância das Features (LightGBM):")
    importance = pd.DataFrame({
        'feature': X_opt.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for _, row in importance.head(6).iterrows():
        print(f"   {row['feature']:<25} {row['importance']:.3f}")


if __name__ == "__main__":
    main()
