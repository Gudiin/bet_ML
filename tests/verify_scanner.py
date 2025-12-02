import requests
import time
import sqlite3
import sys

def verify():
    print("🚀 Triggering Scanner via API...")
    try:
        response = requests.post('http://localhost:5000/api/scanner/start', 
                               json={'date': 'tomorrow', 'leagues': 'top7'})
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"❌ Failed to trigger scanner: {e}")
        return

    print("⏳ Waiting 60 seconds for scanner to run...")
    time.sleep(60)

    print("🔍 Checking Database...")
    try:
        conn = sqlite3.connect('data/football_data.db')
        cursor = conn.cursor()
        
        # Check matches
        cursor.execute("SELECT COUNT(*) FROM matches WHERE status='scheduled'")
        matches_count = cursor.fetchone()[0]
        print(f"✅ Scheduled Matches in DB: {matches_count}")
        
        # Check predictions
        cursor.execute("SELECT prediction_type, COUNT(*) FROM predictions GROUP BY prediction_type")
        counts = cursor.fetchall()
        print(f"✅ Predictions by Type: {counts}")
        
        cursor.execute("SELECT COUNT(*) FROM predictions")
        total_preds = cursor.fetchone()[0]
        print(f"✅ Total Predictions in DB: {total_preds}")
        
        if total_preds > 0:
            print("🎉 Found predictions!")
            # Show sample
            cursor.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 3")
            print(f"Sample: {cursor.fetchall()}")
        else:
            print("❌ FAILURE: No predictions found at all.")
            
        conn.close()
    except Exception as e:
        print(f"❌ Database check failed: {e}")

if __name__ == "__main__":
    verify()
