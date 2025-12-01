"""
Script de Benchmark para Avaliar Melhorias no Modelo de ML.

Este script mede a performance do modelo atual (baseline) e permite
comparar com versões melhoradas incrementalmente.

Métricas avaliadas:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score
- MAPE (Mean Absolute Percentage Error)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from src.database.db_manager import DBManager
from src.ml.feature_engineering import prepare_training_data


def calculate_metrics(y_true, y_pred):
    """Calcula todas as métricas de avaliação."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    # MAPE (evita divisão por zero)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape
    }


def print_metrics(metrics, title="Métricas"):
    """Exibe métricas formatadas."""
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    print(f"  MAE  (Erro Médio Absoluto): {metrics['MAE']:.3f} escanteios")
    print(f"  RMSE (Raiz do Erro Quadrático): {metrics['RMSE']:.3f}")
    print(f"  R²   (Variância Explicada): {metrics['R2']:.3f} ({metrics['R2']*100:.1f}%)")
    print(f"  MAPE (Erro Percentual Médio): {metrics['MAPE']:.1f}%")
    print(f"{'='*50}")


def baseline_model(X, y):
    """
    Modelo baseline: Random Forest com configuração atual.
    """
    print("\n🔍 Testando MODELO BASELINE (Random Forest atual)...")
    
    # Split simples (como está hoje)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    metrics = calculate_metrics(y_test, y_pred)
    print_metrics(metrics, "BASELINE - Random Forest (Split Aleatório)")
    
    return metrics


def test_time_series_cv(X, y):
    """
    B3 - Cross-validation temporal com TimeSeriesSplit.
    """
    print("\n🔍 Testando B3 - TimeSeriesSplit (Cross-validation temporal)...")
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # TimeSeriesSplit mantém ordem temporal
    tscv = TimeSeriesSplit(n_splits=5)
    
    all_y_true = []
    all_y_pred = []
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
    
    metrics = calculate_metrics(np.array(all_y_true), np.array(all_y_pred))
    print_metrics(metrics, "B3 - TimeSeriesSplit (5 folds)")
    
    return metrics


def load_data():
    """Carrega dados do banco."""
    print("📊 Carregando dados do banco...")
    db = DBManager()
    df = db.get_historical_data()
    db.close()
    
    if df.empty:
        print("❌ Banco vazio! Execute primeiro a atualização (opção 1 no menu).")
        return None, None
    
    print(f"   Registros carregados: {len(df)}")
    
    X, y, df_processed = prepare_training_data(df)
    print(f"   Amostras após feature engineering: {len(X)}")
    print(f"   Features: {list(X.columns)}")
    print(f"   Target (média): {y.mean():.2f} escanteios")
    
    return X, y


def main():
    """Executa benchmark completo."""
    print("="*60)
    print("   BENCHMARK DE MODELOS DE ML - PREVISÃO DE ESCANTEIOS")
    print("="*60)
    
    X, y = load_data()
    if X is None:
        return
    
    results = {}
    
    # 1. Baseline
    results['baseline'] = baseline_model(X, y)
    
    # 2. TimeSeriesSplit
    results['timeseries_cv'] = test_time_series_cv(X, y)
    
    # Comparativo
    print("\n" + "="*60)
    print("   📊 COMPARATIVO DE RESULTADOS")
    print("="*60)
    
    print(f"\n{'Modelo':<30} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print("-"*60)
    for name, metrics in results.items():
        print(f"{name:<30} {metrics['MAE']:>8.3f} {metrics['RMSE']:>8.3f} {metrics['R2']:>8.3f}")
    
    # Calcula melhoria
    baseline_mae = results['baseline']['MAE']
    for name, metrics in results.items():
        if name != 'baseline':
            improvement = ((baseline_mae - metrics['MAE']) / baseline_mae) * 100
            status = "✅ MELHOROU" if improvement > 0 else "❌ PIOROU"
            print(f"\n{name}: {status} {abs(improvement):.1f}% no MAE")


if __name__ == "__main__":
    main()
