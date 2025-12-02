import sys
import os
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from database.db_manager import DBManager
from scrapers.sofascore import SofaScoreScraper

def debug_mismatch():
    print("INICIANDO DIAGNOSTICO DE DADOS...")
    
    # 1. Check DB Content
    db = DBManager()
    df = db.get_historical_data()
    print(f"Total de jogos no historico (DataFrame): {len(df)}")
    
    if df.empty:
        print("ERRO CRITICO: DataFrame do historico esta vazio!")
        return

    # Amostra de times no banco
    db_teams = set(df['home_team_name'].unique())
    print(f"Times unicos no banco: {len(db_teams)}")
    # print(f"   Exemplos: {list(db_teams)[:5]}") # Pode ter acento
    
    # 2. Check Scraper Output (Live)
    print("\nBuscando jogos de HOJE no Scraper...")
    scraper = SofaScoreScraper(headless=True)
    
    # Simula a chamada do Scanner
    date_str = datetime.now().strftime('%Y-%m-%d')
    # Top 7 leagues IDs
    leagues = [325, 390, 17, 8, 31, 35, 34, 23] 
    
    try:
        scraper.start()
        matches = scraper.get_scheduled_matches(date_str, leagues)
        print(f"Jogos encontrados hoje: {len(matches)}")
        
        if not matches:
            print("Nenhum jogo hoje para comparar. Tente mudar a data no script se necessario.")
        
        mismatches = 0
        for m in matches:
            home = m['home_team']
            away = m['away_team']
            
            has_home = home in db_teams
            has_away = away in db_teams
            
            status_icon = "[OK]" if (has_home and has_away) else "[MISSING]"
            print(f"{status_icon} {home} ({'OK' if has_home else 'MISSING'}) vs {away} ({'OK' if has_away else 'MISSING'})")
            
            if not has_home or not has_away:
                mismatches += 1
                
        if mismatches > 0:
            print(f"\nALERTA: {mismatches} jogos tem times nao encontrados no banco (Mismatch de Nome).")
            print("   Isso explica por que o Scanner pula esses jogos.")
        else:
            print("\nTodos os times foram encontrados no banco. O problema pode ser outro (ex: filtro de data no historico).")
            
    finally:
        scraper.stop()
        db.close()

if __name__ == "__main__":
    debug_mismatch()
