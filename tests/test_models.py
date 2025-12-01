"""
Script de Teste de Diferentes Modelos de ML.

Testa:
- Random Forest (baseline)
- XGBoost
- LightGBM
- Ensemble (média dos 3)
- Com e sem hyperparameter tuning
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Tentar importar XGBoost e LightGBM
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠️ XGBoost não instalado")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("⚠️ LightGBM não instalado")

from src.database.db_manager import DBManager
from src.ml.feature_engineering import prepare_training_data


def calculate_metrics(y_true, y_pred):
    """Calcula métricas."""
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred)
    }


def load_data():
    """Carrega dados."""
    db = DBManager()
    df = db.get_historical_data()
    db.close()
    
    X, y, _ = prepare_training_data(df)
    return X, y


def test_random_forest(X_train, X_test, y_train, y_test):
    """Random Forest baseline."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return calculate_metrics(y_test, y_pred), model


def test_random_forest_tuned(X_train, X_test, y_train, y_test):
    """Random Forest com tuning."""
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    model = RandomForestRegressor(random_state=42)
    grid = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    print(f"   Melhores parâmetros RF: {grid.best_params_}")
    
    y_pred = grid.predict(X_test)
    return calculate_metrics(y_test, y_pred), grid.best_estimator_


def test_gradient_boosting(X_train, X_test, y_train, y_test):
    """Gradient Boosting (sklearn)."""
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return calculate_metrics(y_test, y_pred), model


def test_xgboost(X_train, X_test, y_train, y_test):
    """XGBoost."""
    if not HAS_XGB:
        return None, None
    
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return calculate_metrics(y_test, y_pred), model


def test_xgboost_tuned(X_train, X_test, y_train, y_test):
    """XGBoost com tuning."""
    if not HAS_XGB:
        return None, None
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    model = xgb.XGBRegressor(random_state=42, verbosity=0)
    grid = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    print(f"   Melhores parâmetros XGB: {grid.best_params_}")
    
    y_pred = grid.predict(X_test)
    return calculate_metrics(y_test, y_pred), grid.best_estimator_


def test_lightgbm(X_train, X_test, y_train, y_test):
    """LightGBM."""
    if not HAS_LGB:
        return None, None
    
    model = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        verbosity=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return calculate_metrics(y_test, y_pred), model


def test_lightgbm_tuned(X_train, X_test, y_train, y_test):
    """LightGBM com tuning."""
    if not HAS_LGB:
        return None, None
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7, -1],
        'num_leaves': [15, 31, 63],
        'subsample': [0.8, 1.0]
    }
    
    model = lgb.LGBMRegressor(random_state=42, verbosity=-1)
    grid = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    print(f"   Melhores parâmetros LGB: {grid.best_params_}")
    
    y_pred = grid.predict(X_test)
    return calculate_metrics(y_test, y_pred), grid.best_estimator_


def test_ensemble(X_train, X_test, y_train, y_test, models):
    """Ensemble - média das previsões."""
    predictions = []
    
    for name, model in models.items():
        if model is not None:
            pred = model.predict(X_test)
            predictions.append(pred)
    
    if not predictions:
        return None, None
    
    # Média das previsões
    y_pred = np.mean(predictions, axis=0)
    return calculate_metrics(y_test, y_pred), None


def test_ridge(X_train, X_test, y_train, y_test):
    """Ridge Regression (linear)."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    return calculate_metrics(y_test, y_pred), model


def main():
    print("="*70)
    print("   TESTE DE MODELOS DE MACHINE LEARNING")
    print("="*70)
    
    X, y = load_data()
    print(f"\n📊 Dados: {len(X)} amostras, {len(X.columns)} features")
    print(f"   Target médio: {y.mean():.2f} escanteios")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    results = {}
    models = {}
    
    # 1. Random Forest (baseline)
    print("\n🔍 Testando Random Forest (baseline)...")
    metrics, model = test_random_forest(X_train, X_test, y_train, y_test)
    results['RF Baseline'] = metrics
    models['rf'] = model
    print(f"   MAE: {metrics['MAE']:.3f} | R²: {metrics['R2']:.3f}")
    
    # 2. Random Forest Tuned
    print("\n🔍 Testando Random Forest (tuned)...")
    metrics, model = test_random_forest_tuned(X_train, X_test, y_train, y_test)
    results['RF Tuned'] = metrics
    models['rf_tuned'] = model
    print(f"   MAE: {metrics['MAE']:.3f} | R²: {metrics['R2']:.3f}")
    
    # 3. Gradient Boosting
    print("\n🔍 Testando Gradient Boosting...")
    metrics, model = test_gradient_boosting(X_train, X_test, y_train, y_test)
    results['Gradient Boosting'] = metrics
    models['gb'] = model
    print(f"   MAE: {metrics['MAE']:.3f} | R²: {metrics['R2']:.3f}")
    
    # 4. XGBoost
    if HAS_XGB:
        print("\n🔍 Testando XGBoost...")
        metrics, model = test_xgboost(X_train, X_test, y_train, y_test)
        if metrics:
            results['XGBoost'] = metrics
            models['xgb'] = model
            print(f"   MAE: {metrics['MAE']:.3f} | R²: {metrics['R2']:.3f}")
        
        print("\n🔍 Testando XGBoost (tuned)...")
        metrics, model = test_xgboost_tuned(X_train, X_test, y_train, y_test)
        if metrics:
            results['XGBoost Tuned'] = metrics
            models['xgb_tuned'] = model
            print(f"   MAE: {metrics['MAE']:.3f} | R²: {metrics['R2']:.3f}")
    
    # 5. LightGBM
    if HAS_LGB:
        print("\n🔍 Testando LightGBM...")
        metrics, model = test_lightgbm(X_train, X_test, y_train, y_test)
        if metrics:
            results['LightGBM'] = metrics
            models['lgb'] = model
            print(f"   MAE: {metrics['MAE']:.3f} | R²: {metrics['R2']:.3f}")
        
        print("\n🔍 Testando LightGBM (tuned)...")
        metrics, model = test_lightgbm_tuned(X_train, X_test, y_train, y_test)
        if metrics:
            results['LightGBM Tuned'] = metrics
            models['lgb_tuned'] = model
            print(f"   MAE: {metrics['MAE']:.3f} | R²: {metrics['R2']:.3f}")
    
    # 6. Ridge (linear)
    print("\n🔍 Testando Ridge Regression...")
    metrics, model = test_ridge(X_train, X_test, y_train, y_test)
    results['Ridge'] = metrics
    print(f"   MAE: {metrics['MAE']:.3f} | R²: {metrics['R2']:.3f}")
    
    # 7. Ensemble
    print("\n🔍 Testando Ensemble (média)...")
    ensemble_models = {k: v for k, v in models.items() if v is not None}
    metrics, _ = test_ensemble(X_train, X_test, y_train, y_test, ensemble_models)
    if metrics:
        results['Ensemble'] = metrics
        print(f"   MAE: {metrics['MAE']:.3f} | R²: {metrics['R2']:.3f}")
    
    # Comparativo final
    print("\n" + "="*70)
    print("   📊 COMPARATIVO FINAL")
    print("="*70)
    
    print(f"\n{'Modelo':<25} {'MAE':>10} {'RMSE':>10} {'R²':>10}")
    print("-"*60)
    
    baseline_mae = results['RF Baseline']['MAE']
    
    # Ordena por MAE
    sorted_results = sorted(results.items(), key=lambda x: x[1]['MAE'])
    
    for name, metrics in sorted_results:
        improvement = ((baseline_mae - metrics['MAE']) / baseline_mae) * 100
        r2_str = f"{metrics['R2']:.3f}"
        if name == 'RF Baseline':
            print(f"{name:<25} {metrics['MAE']:>10.3f} {metrics['RMSE']:>10.3f} {r2_str:>10} (baseline)")
        else:
            status = "✅" if improvement > 0 else "❌"
            print(f"{name:<25} {metrics['MAE']:>10.3f} {metrics['RMSE']:>10.3f} {r2_str:>10} {status} {improvement:+.1f}%")
    
    print("-"*60)
    
    best = sorted_results[0]
    print(f"\n🏆 MELHOR MODELO: {best[0]}")
    print(f"   MAE: {best[1]['MAE']:.3f} | RMSE: {best[1]['RMSE']:.3f} | R²: {best[1]['R2']:.3f}")
    
    if best[0] != 'RF Baseline':
        improvement = ((baseline_mae - best[1]['MAE']) / baseline_mae) * 100
        print(f"   Melhoria sobre baseline: {improvement:.1f}%")


if __name__ == "__main__":
    main()
