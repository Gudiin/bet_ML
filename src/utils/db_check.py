import sqlite3
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

DB_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'football_data.db')

def connect_db():
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found at: {DB_PATH}")
        return None
    return sqlite3.connect(DB_PATH)

def check_duplicate_games_logical(conn):
    print("\n[INFO] Checking for Potentially Duplicate Games (Logical)...")
    cursor = conn.cursor()
    query = """
    SELECT 
        home_team_name, 
        away_team_name, 
        start_timestamp, 
        COUNT(*) as count,
        GROUP_CONCAT(match_id) as match_ids,
        GROUP_CONCAT(tournament_name) as tournaments
    FROM matches
    GROUP BY home_team_name, away_team_name, start_timestamp
    HAVING count > 1
    ORDER BY count DESC
    """
    cursor.execute(query)
    rows = cursor.fetchall()
    
    if not rows:
        print("[OK] No logical duplicate games found (Same Home/Away/Time).")
    else:
        print(f"[WARN] Found {len(rows)} sets of duplicate games:")
        print(f"{'Home':<30} | {'Away':<30} | {'Timestamp':<12} | {'Count':<5} | {'IDs'}")
        print("-" * 120)
        for row in rows:
            home, away, ts, count, ids, tournaments = row
            print(f"{str(home)[:30]:<30} | {str(away)[:30]:<30} | {str(ts):<12} | {str(count):<5} | {str(ids)}")

def check_league_consistency(conn):
    print("\n[INFO] Checking League ID Consistency...")
    cursor = conn.cursor()
    
    # Check 1: One ID, Multiple Names
    query_id_names = """
    SELECT 
        tournament_id, 
        COUNT(DISTINCT tournament_name) as name_count,
        GROUP_CONCAT(DISTINCT tournament_name) as names
    FROM matches
    WHERE tournament_id IS NOT NULL
    GROUP BY tournament_id
    HAVING name_count > 1
    """
    cursor.execute(query_id_names)
    rows_id_names = cursor.fetchall()
    
    if not rows_id_names:
        print("[OK] League IDs are consistent (1 ID -> 1 Name).")
    else:
        print(f"[WARN] Found {len(rows_id_names)} League IDs with multiple names:")
        for row in rows_id_names:
            print(f"ID {row[0]}: {row[2]}")

    # Check 2: One Name, Multiple IDs
    query_name_ids = """
    SELECT 
        tournament_name, 
        COUNT(DISTINCT tournament_id) as id_count,
        GROUP_CONCAT(DISTINCT tournament_id) as ids
    FROM matches
    WHERE tournament_id IS NOT NULL
    GROUP BY tournament_name
    HAVING id_count > 1
    """
    cursor.execute(query_name_ids)
    rows_name_ids = cursor.fetchall()
    
    if not rows_name_ids:
        print("[OK] League Names are consistent (1 Name -> 1 ID).")
    else:
        print(f"[WARN] Found {len(rows_name_ids)} League Names with multiple IDs:")
        for row in rows_name_ids:
            print(f"Name '{row[0]}': IDs {row[2]}")

def check_orphaned_stats(conn):
    print("\n[INFO] Checking for Orphaned Match Stats...")
    cursor = conn.cursor()
    query = """
    SELECT s.match_id
    FROM match_stats s
    LEFT JOIN matches m ON s.match_id = m.match_id
    WHERE m.match_id IS NULL
    """
    cursor.execute(query)
    rows = cursor.fetchall()
    
    if not rows:
        print("[OK] No orphaned match stats found.")
    else:
        print(f"[WARN] Found {len(rows)} match_stats entries without a corresponding match in 'matches' table:")
        print([r[0] for r in rows[:10]])
        if len(rows) > 10:
            print(f"... and {len(rows) - 10} more.")

def check_orphaned_predictions(conn):
    print("\n[INFO] Checking for Orphaned Predictions...")
    cursor = conn.cursor()
    query = """
    SELECT p.id, p.match_id
    FROM predictions p
    LEFT JOIN matches m ON p.match_id = m.match_id
    WHERE m.match_id IS NULL
    """
    cursor.execute(query)
    rows = cursor.fetchall()
    
    if not rows:
        print("[OK] No orphaned predictions found.")
    else:
        print(f"[WARN] Found {len(rows)} predictions without a corresponding match:")
        print([f"ID:{r[0]}/Match:{r[1]}" for r in rows[:10]])

def check_data_quality(conn):
    print("\n[INFO] Checking for Data Quality Issues (Negative Scores, Invalid Dates)...")
    cursor = conn.cursor()
    
    # Check 1: Negative Scores
    cursor.execute("""
        SELECT match_id, home_score, away_score 
        FROM matches 
        WHERE home_score < 0 OR away_score < 0
    """)
    rows_neg = cursor.fetchall()
    
    if not rows_neg:
        print("[OK] No negative scores found.")
    else:
        print(f"[WARN] Found {len(rows_neg)} matches with negative scores:")
        print(rows_neg)

    # Check 2: Invalid Timestamps (Before 2000 or after 2030)
    # Timestamp for 2000-01-01: 946684800
    # Timestamp for 2030-01-01: 1893456000
    cursor.execute("""
        SELECT match_id, start_timestamp 
        FROM matches 
        WHERE start_timestamp < 946684800 OR start_timestamp > 1893456000
    """)
    rows_dates = cursor.fetchall()
    
    if not rows_dates:
        print("[OK] All match timestamps seem reasonable (2000-2030).")
    else:
        print(f"[WARN] Found {len(rows_dates)} matches with suspicious timestamps:")
        print(rows_dates)

def main():
    print(f"[INFO] Starting Database Consistency Check on: {DB_PATH}")
    conn = connect_db()
    if not conn:
        return

    try:
        check_duplicate_games_logical(conn)
        check_league_consistency(conn)
        check_orphaned_stats(conn)
        check_orphaned_predictions(conn)
        check_data_quality(conn)
    finally:
        conn.close()
        print("\n[INFO] Check Complete.")

if __name__ == "__main__":
    main()
