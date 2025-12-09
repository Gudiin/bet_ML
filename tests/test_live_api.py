import requests

# Get list of analyzed matches
url = "http://localhost:5000/api/analyses"
r = requests.get(url, timeout=10)
matches = r.json()

# Find live matches
live_matches = [m for m in matches if m.get("status") == "inprogress"]
print(f"Found {len(live_matches)} live matches")

for lm in live_matches[:3]:
    mid = lm.get("match_id")
    print(f"\nChecking match ID: {mid}")
    
    # Get match result
    url2 = f"http://localhost:5000/api/match/result/{mid}"
    r2 = requests.get(url2, timeout=10)
    d = r2.json()
    m = d.get("match", {})
    
    print(f"  Match: {m.get('home_team_name')} vs {m.get('away_team_name')}")
    print(f"  Status: {m.get('status')}")
    print(f"  Score: {m.get('home_score')} - {m.get('away_score')}")
    print(f"  Match Minute: {m.get('match_minute', 'NOT FOUND')}")
