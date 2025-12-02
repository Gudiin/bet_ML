import requests
import json
from datetime import datetime

def test_schedule_endpoint():
    date_str = datetime.now().strftime('%Y-%m-%d')
    url = f"https://www.sofascore.com/api/v1/sport/football/scheduled-events/{date_str}"
    
    print(f"Testing URL: {url}")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Referer': 'https://www.sofascore.com/',
        'Origin': 'https://www.sofascore.com'
    }
    
    try:
        response = requests.get(url, headers=headers)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            events = data.get('events', [])
            print(f"Total Events Found: {len(events)}")
            
            # Filter for major leagues to see if we find the missing ones
            # Brasileirão: 325, Premier: 17, LaLiga: 8
            target_ids = [325, 17, 8]
            found_count = 0
            
            for evt in events:
                t_id = evt.get('tournament', {}).get('uniqueTournament', {}).get('id')
                if t_id in target_ids:
                    print(f"Found: {evt['homeTeam']['name']} vs {evt['awayTeam']['name']} (League {t_id})")
                    found_count += 1
            
            print(f"Target League Matches Found: {found_count}")
        else:
            print("Failed to fetch data.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_schedule_endpoint()
