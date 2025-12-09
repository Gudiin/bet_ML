import sys
import os
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.database.db_manager import DBManager
from src.ml.features_v2 import create_advanced_features
from src.ml.model_v2 import ProfessionalPredictor

def run_validation():
    print("="*60)
    print("🚀 AUTOMATED VALIDATION: OPTION 4 (Optuna + Transfer Learning)")
    print("="*60)
    
    # 1. Load Data
    print("\n📦 Loading Data...")
    db = DBManager()
    df = db.get_historical_data()
    db.close()
    
    if df.empty:
        print("❌ Database is empty!")
        return

    # Fix columns
    if 'home_score' in df.columns and 'goals_ft_home' not in df.columns:
        df['goals_ft_home'] = df['home_score']
    if 'away_score' in df.columns and 'goals_ft_away' not in df.columns:
        df['goals_ft_away'] = df['away_score']
        
    print(f"✅ Loaded {len(df)} matches.")

    # 2. Generate Features (Testing Dynamic Windows + Historical Standings)
    print("\n🔧 Generating Advanced Features (Dynamic Windows + Historical Standings)...")
    try:
        # Note: window arguments are mostly legacy aliases now, but we keep them
        X, y, df_meta = create_advanced_features(df, window_short=3, window_long=5)
        timestamps = df_meta['start_timestamp']
        
        # Extract Odds for Evaluation (if available)
        odds = None
        if 'odds_home' in df_meta.columns:
            odds = df_meta[['odds_home', 'odds_draw', 'odds_away']]
            print("   ✅ Odds Data Extracted (for ROI Calculation)")
            
        print(f"✅ Features Generated: {X.shape[1]} columns")
        
        # Check if new features exist
        new_cols = ['position_diff', 'home_league_pos', 'home_avg_corners_20g']
        for col in new_cols:
            if col in X.columns:
                print(f"   ✓ Feature verified: {col}")
            else:
                print(f"   ⚠️ Feature MISSING: {col}")
                
    except Exception as e:
        print(f"❌ Feature Generation Failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Validation Run (Optuna + Transfer Learning)
    predictor = ProfessionalPredictor()
    
    # We use a small number of trials/epochs for validation speed
    N_TRIALS = 5 
    print(f"\n🔥 FASE 1: Optuna Optimization ({N_TRIALS} trials for validation)...")
    
    try:
        # Note: Optuna minimizes MAE, doesn't strictly need odds, but we can pass if updated
        best_params = predictor.optimize_hyperparameters(X, y, timestamps, n_trials=N_TRIALS)
        print(f"✅ Optuna Success! Best params: {best_params}")
    except Exception as e:
        print(f"❌ Optuna Failed: {e}")
        return

    # 4. Transfer Learning
    print("\n🌍 FASE 2: Transfer Learning (Global + League Fine-tuning)...")
    try:
        tournament_ids = X['tournament_id'] if 'tournament_id' in X.columns else None
        
        metrics = predictor.train_global_and_finetune(X, y, timestamps, tournament_ids, odds=odds)
        
        print("\n✅ Transfer Learning Success!")
        print(f"   Global MAE: {metrics['global']['mae_test']:.4f}")
        print(f"   Leagues Tuned: {len(metrics['leagues'])}")
        
    except Exception as e:
        print(f"❌ Transfer Learning Failed: {e}")
        return

    print("\n" + "="*60)
    print("✅✅ PROJECT VALIDATION: SUCCESS ✅✅")
    print("All advanced ML components are working correctly.")
    print("="*60)

if __name__ == "__main__":
    run_validation()
