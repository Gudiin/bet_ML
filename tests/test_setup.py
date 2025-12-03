import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.database.db_manager import DBManager
    from src.scrapers.sofascore import SofaScoreScraper
    from src.ml.model import CornerPredictor
    print("Imports successful!")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

try:
    db = DBManager()
    print("Database created/connected successfully.")
    db.close()
except Exception as e:
    print(f"Database error: {e}")
    sys.exit(1)

print("Setup verification passed.")
