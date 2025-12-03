"""Debug detalhado"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.database.db_manager import DBManager

# Carrega dados
db = DBManager()
df = db.get_historical_data()
df['start_timestamp'] = pd.to_datetime(df['start_timestamp'], unit='s')
df['goals_ft_home'] = df['home_score']
df['goals_ft_away'] = df['away_score']
db.close()

# Normaliza nomes
column_mapping = {
    'corners_home_ft': 'corners_ft_home',
    'corners_away_ft': 'corners_ft_away',
    'corners_home_ht': 'corners_ht_home',
    'corners_away_ht': 'corners_ht_away',
    'shots_ot_home_ft': 'shots_ot_ft_home',
    'shots_ot_away_ft': 'shots_ot_ft_away',
}

existing_renames = {k: v for k, v in column_mapping.items() if k in df.columns}
df = df.rename(columns=existing_renames)

df = df.sort_values('start_timestamp').copy()

# Cria visão team-centric
cols_metrics = ['corners_ft', 'shots_ot_ft', 'goals_ft', 'corners_ht']

df_home = df[['match_id', 'start_timestamp', 'home_team_id'] + 
             [f'{c}_home' for c in cols_metrics]].copy()

print("Colunas df_home:", list(df_home.columns))

# Renomeia
rename_map_home = {f'{c}_home': c.split('_')[0] for c in cols_metrics}
print("Rename map:", rename_map_home)

df_home = df_home.rename(columns=rename_map_home)
print("Colunas após rename:", list(df_home.columns))
print("Duplicadas?", df_home.columns.duplicated().any())
