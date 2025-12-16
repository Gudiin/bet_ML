import sqlite3
import os
import sys

DB_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'football_data.db')

def connect_db():
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found at: {DB_PATH}")
        return None
    return sqlite3.connect(DB_PATH)

def check_predictions(conn):
    """
    Manually implements the logic from DBManager.check_predictions
    without depending on pandas or the rest of the app.
    """
    print("    - Running verification logic...")
    cursor = conn.cursor()
    
    # Logic copied from db_manager.py
    query = '''
        SELECT p.id, p.match_id, p.prediction_value, p.prediction_label, p.market_group,
               s.corners_home_ft, s.corners_away_ft, s.corners_home_ht, s.corners_away_ht
        FROM predictions p
        JOIN matches m ON p.match_id = m.match_id
        LEFT JOIN match_stats s ON m.match_id = s.match_id
        WHERE (p.is_correct IS NULL OR p.status = 'PENDING')
          AND m.status = 'finished'
          AND s.corners_home_ft IS NOT NULL
    '''
    
    cursor.execute(query)
    rows = cursor.fetchall()
    
    if not rows:
        print("    - No verifiable predictions found.")
        return

    print(f"    - verifying {len(rows)} predictions...")
    
    updates = 0
    for row in rows:
        pred_id, match_id, pred_val, pred_label, market_group, h_corners_ft, a_corners_ft, h_corners_ht, a_corners_ht = row
        
        # Garante valores numéricos (fallback para 0 se None)
        h_corners_ft = h_corners_ft or 0
        a_corners_ft = a_corners_ft or 0
        h_corners_ht = h_corners_ht or 0
        a_corners_ht = a_corners_ht or 0
        
        # Determina qual valor usar baseado no market_group
        market_group_lower = (market_group or '').lower()
        
        if 'mandante' in market_group_lower or 'home' in market_group_lower:
            corners_value = h_corners_ft
        elif 'visitante' in market_group_lower or 'away' in market_group_lower:
            corners_value = a_corners_ft
        elif '1' in market_group_lower or 'ht' in market_group_lower or 'primeiro' in market_group_lower:
            corners_value = h_corners_ht + a_corners_ht
        elif '2' in market_group_lower or 'segundo' in market_group_lower:
            corners_value = (h_corners_ft - h_corners_ht) + (a_corners_ft - a_corners_ht)
        else:
            corners_value = h_corners_ft + a_corners_ft
        
        is_over = 'over' in pred_label.lower() if pred_label else False
        is_under = 'under' in pred_label.lower() if pred_label else False
        
        is_correct = False
        if pred_val is not None and pred_val > 0:
            line = pred_val
            if is_over:
                is_correct = corners_value > line
            elif is_under:
                is_correct = corners_value < line
        
        status = 'GREEN' if is_correct else 'RED'
        cursor.execute("UPDATE predictions SET is_correct = ?, status = ? WHERE id = ?", (is_correct, status, pred_id))
        updates += 1
        
    conn.commit()
    print(f"    - [OK] Verified and updated {updates} predictions.")

def main():
    print(f"[INFO] Starting Pending Predictions Fix on {DB_PATH}")
    
    conn = connect_db()
    if not conn:
        return
    cursor = conn.cursor()
    
    try:
        # 1. Diagnose
        print("\n[INFO] Diagnosing Pending Predictions...")
        query_diag = """
        SELECT 
            COUNT(p.id) as pending_count,
            SUM(CASE WHEN s.match_id IS NOT NULL THEN 1 ELSE 0 END) as has_stats,
            SUM(CASE WHEN s.match_id IS NULL THEN 1 ELSE 0 END) as missing_stats
        FROM predictions p
        JOIN matches m ON p.match_id = m.match_id
        LEFT JOIN match_stats s ON m.match_id = s.match_id
        WHERE (p.status = 'PENDING' OR p.is_correct IS NULL)
          AND m.status = 'finished'
        """
        cursor.execute(query_diag)
        row = cursor.fetchone()
        
        pending_count = row[0] or 0
        has_stats = row[1] or 0
        missing_stats = row[2] or 0
        
        print(f"    - Total Pending (Finished Matches): {pending_count}")
        print(f"    - Have Stats (Ready to Verify):     {has_stats}")
        print(f"    - Missing Stats (Need Update):      {missing_stats}")
        
        if pending_count == 0:
            print("\n[OK] No pending predictions found for finished matches.")
            return

        # 2. Fix
        if has_stats > 0:
            print(f"\n[ACTION] Attempting to verify {has_stats} predictions with existing stats...")
            check_predictions(conn)
        
        # 3. Report Remaining
        cursor.execute(query_diag)
        row_after = cursor.fetchone()
        remaining = row_after[0] or 0
        missing_stats_after = row_after[2] or 0
        
        print("\n[INFO] Status After Fix:")
        print(f"    - Remaining Pending: {remaining}")
        
        if missing_stats_after > 0:
            print(f"\n[WARN] {missing_stats_after} matches are missing statistics. Examples:")
            query_missing = """
            SELECT m.match_id, m.home_team_name, m.away_team_name, m.start_timestamp
            FROM predictions p
            JOIN matches m ON p.match_id = m.match_id
            LEFT JOIN match_stats s ON m.match_id = s.match_id
            WHERE (p.status = 'PENDING' OR p.is_correct IS NULL)
              AND m.status = 'finished'
              AND s.match_id IS NULL
            GROUP BY m.match_id
            ORDER BY m.start_timestamp DESC
            LIMIT 10
            """
            cursor.execute(query_missing)
            rows = cursor.fetchall()
            for r in rows:
                import datetime
                dt = datetime.datetime.fromtimestamp(r[3]).strftime('%Y-%m-%d %H:%M')
                # Use ascii safe encoding for print if needed, but python 3 usually handles it well on modern terminals
                # replacing potential unicode chars just in case
                home = str(r[1]).encode('ascii', 'replace').decode()
                away = str(r[2]).encode('ascii', 'replace').decode()
                print(f"    - ID {r[0]}: {home} vs {away} ({dt})")

    finally:
        conn.close()

if __name__ == "__main__":
    main()
