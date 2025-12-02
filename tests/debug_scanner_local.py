import sys
import os
import time
import random
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.database.db_manager import DBManager
from src.scrapers.sofascore import SofaScoreScraper

def debug_scanner():
    print("🚀 Starting Local Debug Scanner...")
    
    db = DBManager()
    scraper = SofaScoreScraper(headless=True)
    
    try:
        scraper.start()
        
        # Leagues
        leagues_filter = [325, 390, 17, 8, 31, 35, 34, 23]
        date_str = datetime.now().strftime('%Y-%m-%d') # Today/Tomorrow logic simplified
        
        print("🔍 Fetching matches...")
        matches = scraper.get_scheduled_matches(date_str, leagues_filter)
        
        if not matches:
            print("❌ No matches found.")
            return

        print(f"✅ Found {len(matches)} matches. Processing first 3...")
        
        # Process only first 3 for debug
        for i, match in enumerate(matches[:3]):
            print(f"\n--- Processing Match {i+1} ---")
            print(f"Match: {match['home_team']} vs {match['away_team']}")
            
            match_id_val = match.get('match_id')
            if not match_id_val:
                match_id_val = int(hash(match['home_team'] + date_str) % 100000000)
            else:
                match_id_val = int(match_id_val)
                
            print(f"Match ID: {match_id_val}")
            
            # Save Match
            try:
                try:
                    ts = int(datetime.strptime(match['start_time'], '%Y-%m-%d %H:%M').timestamp())
                except:
                    ts = int(time.time())

                match_data = {
                    'id': match_id_val,
                    'tournament': match.get('tournament', 'Unknown'),
                    'season_id': 0,
                    'round': 0,
                    'status': 'scheduled',
                    'timestamp': ts,
                    'home_id': 0,
                    'home_name': match['home_team'],
                    'away_id': 0,
                    'away_name': match['away_team'],
                    'home_score': 0,
                    'away_score': 0
                }
                print("Saving match...")
                db.save_match(match_data)
                print("✅ Match saved.")
            except Exception as e:
                print(f"❌ Error saving match: {e}")
                
            # Save Prediction
            try:
                ml_prediction = 10.5 # Dummy
                confidence = 0.85 # Dummy
                best_bet = 'Over 9.5'
                
                print("Saving prediction...")
                db.save_prediction(
                    match_id=match_id_val,
                    prediction_type='ML_Scanner_Debug', # Unique type for debug
                    predicted_value=ml_prediction,
                    expected_value=best_bet,
                    probability=confidence,
                    odds=1.85,
                    category='Scanner',
                    market_group='Corners',
                    status='PENDING',
                    verbose=True
                )
                print("✅ Prediction saved.")
            except Exception as e:
                print(f"❌ Error saving prediction: {e}")
                import traceback
                traceback.print_exc()
                
    except Exception as e:
        print(f"❌ Global Error: {e}")
    finally:
        scraper.stop()
        db.close()

if __name__ == "__main__":
    debug_scanner()
