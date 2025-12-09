import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.database.db_manager import DBManager
from src.ml.features_v2 import create_advanced_features

def debug_counts():
    db = DBManager()
    df = db.get_historical_data()
    print(f"1. Raw DB Count: {len(df)}")
    
    # Check per league
    print("\nCounts per tournament_id (Raw):")
    print(df['tournament_id'].value_counts().head(10))
    
    print("\n2. Generating Features...")
    try:
        X, y, df_meta = create_advanced_features(df)
        print(f"3. Final X Count: {len(X)}")
        
        print("\nCounts per tournament_id (Final X):")
        if 'tournament_id' in X.columns:
            print(X['tournament_id'].value_counts().head(10))
        else:
             print("tournament_id dropped from X!")
             
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_counts()
