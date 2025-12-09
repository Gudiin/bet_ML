import sys
import os
import pandas as pd
import sqlite3

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.database.db_manager import DBManager
from src.data.external.manager import ExternalDataManager

def debug_brazil():
    print("🕵️ DEBUGGING BRAZIL IMPORT")
    
    # 1. Load CSV
    manager = ExternalDataManager()
    df_ext = manager.load_combined_data('BRA')
    if df_ext.empty:
        print("❌ BRA.csv not found or empty.")
        return
        
    df_ext = df_ext.rename(columns={'Home': 'HomeTeam', 'Away': 'AwayTeam'})
    print(f"\n📄 CSV Sample (first 2):")
    print(df_ext[['Date', 'HomeTeam', 'AwayTeam']].head(2))
    
    csv_teams = df_ext['HomeTeam'].unique()[:5]
    print(f"👉 CSV Teams (Sample): {csv_teams}")
    
    # 2. Load DB
    db = DBManager()
    conn = db.connect()
    
    # Get Tournament ID
    df_tourn = pd.read_sql_query("SELECT tournament_id, tournament_name FROM matches WHERE tournament_name LIKE '%Brasileir%'", conn)
    print(f"\n🏆 DB Tournaments found: {df_tourn['tournament_name'].unique()}")
    
    if df_tourn.empty:
        print("❌ No Brasileirão in DB.")
        return
        
    t_id = df_tourn.iloc[0]['tournament_id']
    
    # Get Matches
    df_db = pd.read_sql_query(f"SELECT start_timestamp, home_team_name, away_team_name FROM matches WHERE tournament_id = {t_id} LIMIT 5", conn)
    
    df_db['date_str'] = pd.to_datetime(df_db['start_timestamp'], unit='s').dt.strftime('%Y-%m-%d')
    print(f"\n🗄️ DB Sample (first 2):")
    print(df_db[['date_str', 'home_team_name', 'away_team_name']].head(2))
    
    db_teams = pd.read_sql_query(f"SELECT DISTINCT home_team_name FROM matches WHERE tournament_id = {t_id}", conn)['home_team_name'].tolist()[:5]
    print(f"👉 DB Teams (Sample): {db_teams}")
    
    # 3. Test Date Conversion
    print("\n📅 Date Diagnosis:")
    first_csv_date = df_ext['Date'].iloc[0]
    print(f"   Raw CSV Date 0: {first_csv_date} (Type: {type(first_csv_date)})")
    
    # Try parsing
    try:
        parsed = pd.to_datetime(df_ext['Date'], dayfirst=True, errors='coerce')
        print(f"   Parsed CSV Date 0: {parsed.iloc[0]}")
    except Exception as e:
        print(f"   ❌ Parse Error: {e}")

if __name__ == "__main__":
    debug_brazil()
