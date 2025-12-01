import sys
import os
import json
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.scrapers.sofascore import SofaScoreScraper

scraper = SofaScoreScraper(headless=False)
scraper.start()

t_id = 325 # Brasileirão Betano
url = f"https://www.sofascore.com/api/v1/unique-tournament/{t_id}/seasons"
print(f"Fetching seasons for ID {t_id}...")
data = scraper._fetch_api(url)

with open('debug_seasons.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
print("Output saved to debug_seasons.json")

scraper.stop()
