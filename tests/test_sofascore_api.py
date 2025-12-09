import requests
import json

# Test SofaScore API directly
match_id = "14025165"  # Fulham vs Crystal Palace
url = f"https://www.sofascore.com/api/v1/event/{match_id}"

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Referer': 'https://www.sofascore.com/'
}

try:
    r = requests.get(url, headers=headers, timeout=10)
    d = r.json()
    ev = d.get('event', {})
    
    print(f"Match: {ev.get('homeTeam', {}).get('name')} vs {ev.get('awayTeam', {}).get('name')}")
    
    status = ev.get('status', {})
    print(f"\nStatus Object:")
    print(json.dumps(status, indent=2))
    
    print(f"\nStatus Description: '{status.get('description')}'")
    print(f"Status Type: '{status.get('type')}'")
    print(f"Status Code: '{status.get('code')}'")
    
    print(f"\nStart Timestamp: {ev.get('startTimestamp')}")
    
    import time
    start_ts = ev.get('startTimestamp', 0)
    if start_ts:
        elapsed = int(time.time()) - start_ts
        print(f"Elapsed time: {elapsed} seconds ({elapsed // 60} minutes)")
        
except Exception as e:
    print(f"Error: {e}")
