import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.scrapers.sofascore import SofaScoreScraper

scraper = SofaScoreScraper(headless=False) # Run with head to see what happens
scraper.start()

print("Searching for tournament...")
# Try a broader search first
url = "https://www.sofascore.com/api/v1/search/Brasileirão"
data = scraper._fetch_api(url)
print("Search Results for 'Brasileirão':")
import json
with open('debug_output.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
print("Output saved to debug_output.json")

scraper.stop()
