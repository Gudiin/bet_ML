import sys
import os
import inspect

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.scrapers.sofascore import SofaScoreScraper
    from src.main import update_database, update_specific_league, update_full_history, train_model
    from src.ml.model_improved import ImprovedCornerPredictor
    
    print("✅ Imports successful")
    
    # Check SofaScoreScraper methods
    scraper = SofaScoreScraper(headless=True)
    if hasattr(scraper, 'get_matches') and hasattr(scraper, 'get_season_id'):
        sig_matches = inspect.signature(scraper.get_matches)
        sig_season = inspect.signature(scraper.get_season_id)
        print(f"✅ SofaScoreScraper methods present")
        print(f"   get_matches signature: {sig_matches}")
        print(f"   get_season_id signature: {sig_season}")
    else:
        print("❌ SofaScoreScraper methods missing")

    # Check main.py functions
    if inspect.isfunction(update_specific_league) and inspect.isfunction(update_full_history):
        print("✅ main.py new menu functions present")
    else:
        print("❌ main.py new menu functions missing")
        
    # Check ImprovedCornerPredictor
    predictor = ImprovedCornerPredictor()
    if hasattr(predictor, 'train_with_optimization'):
        print("✅ ImprovedCornerPredictor.train_with_optimization present")
    else:
        print("❌ ImprovedCornerPredictor.train_with_optimization missing")

except ImportError as e:
    print(f"❌ ImportError: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
