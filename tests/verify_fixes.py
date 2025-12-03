
import sys
import os
sys.path.insert(0, os.getcwd())

from src.scrapers.sofascore import SofaScoreScraper
from datetime import datetime

def test_scraper():
    print("Testing Scraper Fixes...")
    scraper = SofaScoreScraper(headless=True)
    
    try:
        scraper.start()
        
        # Test 1: Season ID for Brasileirão 2025 (or 2024 if 2025 not avail)
        # Brasileirão ID is 325
        print("\nTest 1: Get Season ID for Brasileirão (325)...")
        s_id = scraper.get_season_id(325, "2024")
        print(f"Season ID for 2024: {s_id}")
        
        s_id_25 = scraper.get_season_id(325, "2025")
        print(f"Season ID for 2025: {s_id_25}")
        
        # Test 2: Get Scheduled Matches for Today
        today = datetime.now().strftime('%Y-%m-%d')
        print(f"\nTest 2: Get Scheduled Matches for {today}...")
        matches = scraper.get_scheduled_matches(today, [325, 17]) # Brasileirão, Premier League
        
        print(f"Found {len(matches)} matches.")
        for m in matches[:3]:
            print(f" - {m['home_team']} vs {m['away_team']} ({m['status']})")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        scraper.stop()

if __name__ == "__main__":
    test_scraper()
