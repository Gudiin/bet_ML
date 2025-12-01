"""
Script de Teste - Integração ML + Monte Carlo.

Testa D2: Usar previsão ML como input do λ no Monte Carlo
para melhorar a precisão das probabilidades.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from scipy.stats import poisson, nbinom
import warnings
warnings.filterwarnings('ignore')

from src.database.db_manager import DBManager
from src.ml.model_improved import ImprovedCornerPredictor, prepare_improved_features


def monte_carlo_original(lambda_val, var_val, n_sims=10000):
    """Monte Carlo original (sem ML)."""
    if var_val > lambda_val:
        p = lambda_val / var_val
        n = (lambda_val ** 2) / (var_val - lambda_val)
        return nbinom.rvs(n, p, size=n_sims)
    else:
        return poisson.rvs(lambda_val, size=n_sims)


def monte_carlo_ml_enhanced(ml_prediction, historical_var, n_sims=10000):
    """
    Monte Carlo melhorado usando previsão ML como λ.
    
    A previsão do ML é mais precisa que a média histórica simples,
    então usamos ela como centro da distribuição.
    """
    lambda_val = ml_prediction
    
    # Usa variância histórica mas com λ do ML
    if historical_var > lambda_val:
        p = lambda_val / historical_var
        n = (lambda_val ** 2) / (historical_var - lambda_val)
        return nbinom.rvs(n, p, size=n_sims)
    else:
        return poisson.rvs(lambda_val, size=n_sims)


def monte_carlo_weighted(ml_prediction, hist_mean, hist_var, weight_ml=0.6, n_sims=10000):
    """
    Monte Carlo com λ ponderado entre ML e histórico.
    
    λ = weight_ml * ML_prediction + (1-weight_ml) * historical_mean
    
    Isso combina a precisão do ML com a estabilidade do histórico.
    """
    lambda_val = weight_ml * ml_prediction + (1 - weight_ml) * hist_mean
    
    if hist_var > lambda_val:
        p = lambda_val / hist_var
        n = (lambda_val ** 2) / (hist_var - lambda_val)
        return nbinom.rvs(n, p, size=n_sims)
    else:
        return poisson.rvs(lambda_val, size=n_sims)


def evaluate_probabilities(simulations, actual_total, lines=[8.5, 9.5, 10.5, 11.5]):
    """
    Avalia qualidade das probabilidades calculadas.
    
    Para cada linha, verifica se o resultado real caiu dentro
    do esperado baseado na probabilidade calculada.
    """
    results = []
    for line in lines:
        prob_over = (simulations > line).mean()
        actual_over = actual_total > line
        
        results.append({
            'line': line,
            'prob_over': prob_over,
            'actual_over': actual_over,
            'correct': (prob_over > 0.5) == actual_over
        })
    
    return results


def test_on_historical_data():
    """Testa as diferentes abordagens em dados históricos."""
    
    # Carrega dados
    db = DBManager()
    df = db.get_historical_data()
    db.close()
    
    if df.empty:
        print("❌ Banco vazio!")
        return
    
    print(f"📊 Dados carregados: {len(df)} registros")
    
    # Prepara features
    X, y, features = prepare_improved_features(df)
    
    # Treina modelo
    print("\n🤖 Treinando modelo melhorado...")
    predictor = ImprovedCornerPredictor(use_ensemble=False)
    predictor.train(X, y)
    
    # Testa em amostra de jogos
    print("\n" + "="*70)
    print("   TESTE DE INTEGRAÇÃO ML + MONTE CARLO")
    print("="*70)
    
    # Pega últimos 20% dos dados como teste
    n_test = int(len(df) * 0.2)
    df_test = df.tail(n_test).copy()
    
    results_original = {'correct': 0, 'total': 0}
    results_ml = {'correct': 0, 'total': 0}
    results_weighted = {'correct': 0, 'total': 0}
    
    errors_original = []
    errors_ml = []
    errors_weighted = []
    
    print(f"\nTestando em {len(df_test)} jogos...")
    
    for idx, row in df_test.iterrows():
        actual_total = row['corners_home_ft'] + row['corners_away_ft']
        
        # Histórico (média dos jogos anteriores)
        hist_mean = y.mean()
        hist_var = y.var()
        
        # Previsão ML
        # Simula features do jogo (usando dados já conhecidos para teste)
        X_game = X.loc[[idx]] if idx in X.index else None
        if X_game is None:
            continue
        
        ml_pred = predictor.predict(X_game)[0]
        
        # Monte Carlo Original (só histórico)
        sims_original = monte_carlo_original(hist_mean, hist_var)
        
        # Monte Carlo ML (λ = ML prediction)
        sims_ml = monte_carlo_ml_enhanced(ml_pred, hist_var)
        
        # Monte Carlo Weighted (60% ML + 40% histórico)
        sims_weighted = monte_carlo_weighted(ml_pred, hist_mean, hist_var, weight_ml=0.6)
        
        # Avalia
        lines = [8.5, 9.5, 10.5, 11.5]
        
        for line in lines:
            # Original
            prob_orig = (sims_original > line).mean()
            pred_orig = prob_orig > 0.5
            correct_orig = pred_orig == (actual_total > line)
            results_original['correct'] += int(correct_orig)
            results_original['total'] += 1
            
            # ML
            prob_ml = (sims_ml > line).mean()
            pred_ml = prob_ml > 0.5
            correct_ml = pred_ml == (actual_total > line)
            results_ml['correct'] += int(correct_ml)
            results_ml['total'] += 1
            
            # Weighted
            prob_weighted = (sims_weighted > line).mean()
            pred_weighted = prob_weighted > 0.5
            correct_weighted = pred_weighted == (actual_total > line)
            results_weighted['correct'] += int(correct_weighted)
            results_weighted['total'] += 1
        
        # Erro da previsão central
        errors_original.append(abs(actual_total - hist_mean))
        errors_ml.append(abs(actual_total - ml_pred))
        errors_weighted.append(abs(actual_total - (0.6*ml_pred + 0.4*hist_mean)))
    
    # Resultados
    print("\n" + "-"*70)
    print("📊 TAXA DE ACERTO NAS PREVISÕES (Over/Under)")
    print("-"*70)
    
    acc_orig = results_original['correct'] / results_original['total'] * 100
    acc_ml = results_ml['correct'] / results_ml['total'] * 100
    acc_weighted = results_weighted['correct'] / results_weighted['total'] * 100
    
    print(f"\n{'Método':<30} {'Acertos':>10} {'Total':>10} {'Taxa':>10}")
    print("-"*60)
    print(f"{'Original (só histórico)':<30} {results_original['correct']:>10} {results_original['total']:>10} {acc_orig:>9.1f}%")
    print(f"{'ML Enhanced (λ=ML)':<30} {results_ml['correct']:>10} {results_ml['total']:>10} {acc_ml:>9.1f}%")
    print(f"{'Weighted (60% ML + 40% hist)':<30} {results_weighted['correct']:>10} {results_weighted['total']:>10} {acc_weighted:>9.1f}%")
    
    print("\n" + "-"*70)
    print("📊 ERRO MÉDIO NA PREVISÃO DO TOTAL")
    print("-"*70)
    
    mae_orig = np.mean(errors_original)
    mae_ml = np.mean(errors_ml)
    mae_weighted = np.mean(errors_weighted)
    
    print(f"\n{'Método':<30} {'MAE (escanteios)':>20}")
    print("-"*50)
    print(f"{'Original (média histórica)':<30} {mae_orig:>20.2f}")
    print(f"{'ML Prediction':<30} {mae_ml:>20.2f}")
    print(f"{'Weighted (60% ML)':<30} {mae_weighted:>20.2f}")
    
    # Melhor método
    print("\n" + "="*70)
    best_acc = max([('Original', acc_orig), ('ML Enhanced', acc_ml), ('Weighted', acc_weighted)], key=lambda x: x[1])
    best_mae = min([('Original', mae_orig), ('ML', mae_ml), ('Weighted', mae_weighted)], key=lambda x: x[1])
    
    print(f"🏆 MELHOR TAXA DE ACERTO: {best_acc[0]} ({best_acc[1]:.1f}%)")
    print(f"🏆 MENOR ERRO MÉDIO: {best_mae[0]} (MAE: {best_mae[1]:.2f})")
    
    # Melhoria
    improvement_acc = ((acc_weighted - acc_orig) / acc_orig) * 100
    improvement_mae = ((mae_orig - mae_weighted) / mae_orig) * 100
    
    print(f"\n📈 Melhoria Weighted vs Original:")
    print(f"   Taxa de acerto: {improvement_acc:+.1f}%")
    print(f"   MAE: {improvement_mae:+.1f}%")


if __name__ == "__main__":
    test_on_historical_data()
