from src.scrapers.sofascore import SofaScoreScraper
import json
import time

def debug_scraper():
    scraper = SofaScoreScraper(headless=True)
    scraper.start()
    
    try:
        # Try to find a finished match to analyze
        # We'll search for a popular league or just use a known ID if we had one.
        # Let's try to get matches from a known tournament ID if possible, or just use a hardcoded recent match ID if we can find one.
        # Since I don't have a match ID, I'll try to fetch matches from "Brasileirão Série A" (Tournament 325) or "Premier League" (17)
        # Premier League ID: 17, Season 25/26 ID: 61627 (approx, might need to fetch)
        
        # Let's just try to fetch a specific match URL that is likely to exist or be finished.
        # Or better, let's use the scraper's search functionality if it exists? No.
        # Let's try to get matches for today/yesterday.
        
        print("Fetching matches...")
        # Premier League (17), Season 24/25 (52186) - Let's guess IDs or use get_matches logic
        # Actually, let's just try to hit the API for a known recent match.
        # Liverpool vs Man City (Example) - ID: 12437339 (just a guess, probably wrong)
        
        # Better approach: Use the scraper to get a list of matches for a league and pick the first finished one.
        # Brasileirão Série A: 325. Season 2024: 57478
        matches = scraper.get_matches(325, 57478) 
        
        finished_matches = [m for m in matches if m['status']['type'] == 'finished']
        
        if not finished_matches:
            print("No finished matches found in Brasileirão. Trying Premier League (17, 61627)...")
            matches = scraper.get_matches(17, 61627)
            finished_matches = [m for m in matches if m['status']['type'] == 'finished']
            
        if not finished_matches:
            print("No finished matches found. Exiting.")
            return

        target_match = finished_matches[0]
        match_id = target_match['id']
        print(f"Analyzing match: {target_match['homeTeam']['name']} vs {target_match['awayTeam']['name']} (ID: {match_id})")
        
        # Get Stats
        print("Fetching stats...")
        stats = scraper.get_match_stats(match_id)
        print("\nExtracted Stats:")
        print(json.dumps(stats, indent=2))
        
        # Fetch Raw Data to debug keys
        print("\nFetching Raw Statistics Data for debugging...")
        api_url = f"https://www.sofascore.com/api/v1/event/{match_id}/statistics"
        raw_data = scraper._fetch_api(api_url)
        
        if raw_data and 'statistics' in raw_data:
            print("\nRaw Data Keys (Groups):")
            for item in raw_data['statistics']:
                print(f"Period: {item['period']}")
                for group in item.get('groups', []):
                    print(f"  Group: {group.get('groupName')}")
                    for stat_item in group.get('statisticsItems', []):
                        print(f"    - {stat_item.get('name')}: Home={stat_item.get('home')}, Away={stat_item.get('away')}")
        else:
            print("Failed to fetch raw statistics data.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        scraper.stop()

if __name__ == "__main__":
    debug_scraper()
