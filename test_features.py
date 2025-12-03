"""Script de teste rápido para debug"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.database.db_manager import DBManager

# Carrega dados
print("Carregando dados...")
db = DBManager()
df = db.get_historical_data()
df['start_timestamp'] = pd.to_datetime(df['start_timestamp'], unit='s')
df['goals_ft_home'] = df['home_score']
df['goals_ft_away'] = df['away_score']
db.close()

print(f"Total de jogos: {len(df)}")

# Testa feature engineering
print("\nCriando features...")
try:
    from src.ml.features_v2 import create_advanced_features
    X, y, timestamps = create_advanced_features(df)
    print(f"Sucesso! Features shape: {X.shape}")
except Exception as e:
    print(f"Erro: {str(e)[:200]}")

