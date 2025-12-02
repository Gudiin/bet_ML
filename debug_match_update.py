from src.scrapers.sofascore import SofaScoreScraper
from src.database.db_manager import DBManager
import json

def debug_specific_match():
    match_id = "14083139"
    
    print(f"🔍 Debugging Match ID: {match_id}")
    print("=" * 60)
    
    scraper = SofaScoreScraper(headless=True)
    db = DBManager()
    
    try:
        scraper.start()
        
        # 1. Get Match Details
        print("\n1️⃣ Fetching match details...")
        api_url = f"https://www.sofascore.com/api/v1/event/{match_id}"
        ev_data = scraper._fetch_api(api_url)
        
        if not ev_data or 'event' not in ev_data:
            print("❌ Failed to fetch match data")
            return
            
        ev = ev_data['event']
        match_name = f"{ev['homeTeam']['name']} vs {ev['awayTeam']['name']}"
        status = ev.get('status', {}).get('type', 'unknown')
        
        print(f"✅ Match: {match_name}")
        print(f"   Status: {status}")
        print(f"   Score: {ev.get('homeScore', {}).get('display', 0)} - {ev.get('awayScore', {}).get('display', 0)}")
        
        # 2. Save Match Info
        print("\n2️⃣ Saving match info to database...")
        match_data = {
            'id': match_id,
            'tournament': ev.get('tournament', {}).get('name', 'Unknown'),
            'season_id': ev.get('season', {}).get('id', 0),
            'round': ev.get('roundInfo', {}).get('round', 0),
            'status': status,
            'timestamp': ev.get('startTimestamp', 0),
            'home_id': ev['homeTeam']['id'],
            'home_name': ev['homeTeam']['name'],
            'away_id': ev['awayTeam']['id'],
            'away_name': ev['awayTeam']['name'],
            'home_score': ev.get('homeScore', {}).get('display', 0),
            'away_score': ev.get('awayScore', {}).get('display', 0)
        }
        db.save_match(match_data)
        print("✅ Match info saved")
        
        # 3. Get Stats
        print("\n3️⃣ Fetching match statistics...")
        stats = scraper.get_match_stats(match_id)
        
        print("\n📊 Extracted Stats:")
        print(json.dumps(stats, indent=2))
        
        # 4. Save Stats
        print("\n4️⃣ Saving statistics to database...")
        db.save_stats(match_id, stats)
        print("✅ Stats saved successfully!")
        
        # 5. Verify in Database
        print("\n5️⃣ Verifying data in database...")
        conn = db.connect()
        import pandas as pd
        
        query = "SELECT * FROM match_stats WHERE match_id = ?"
        result = pd.read_sql_query(query, conn, params=(match_id,))
        
        if not result.empty:
            print("✅ Data found in database:")
            print(f"   Corners: {result.iloc[0]['corners_home_ft']} vs {result.iloc[0]['corners_away_ft']}")
            print(f"   Possession: {result.iloc[0]['possession_home']}% vs {result.iloc[0]['possession_away']}%")
            print(f"   Total Shots: {result.iloc[0]['total_shots_home']} vs {result.iloc[0]['total_shots_away']}")
            print(f"   Fouls: {result.iloc[0]['fouls_home']} vs {result.iloc[0]['fouls_away']}")
            print(f"   Yellow Cards: {result.iloc[0]['yellow_cards_home']} vs {result.iloc[0]['yellow_cards_away']}")
            print(f"   Red Cards: {result.iloc[0]['red_cards_home']} vs {result.iloc[0]['red_cards_away']}")
            print(f"   Big Chances: {result.iloc[0]['big_chances_home']} vs {result.iloc[0]['big_chances_away']}")
            print(f"   xG: {result.iloc[0]['expected_goals_home']} vs {result.iloc[0]['expected_goals_away']}")
        else:
            print("❌ No data found in database!")
        
        db.close()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        scraper.stop()

if __name__ == "__main__":
    debug_specific_match()
