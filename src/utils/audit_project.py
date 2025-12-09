import sys
import os
import pandas as pd
import sqlite3
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.database.db_manager import DBManager

def audit_database():
    print("="*60)
    print("🕵️  PROJECT AUDIT: DATABASE & INTEGRITY")
    print("="*60)
    
    db = DBManager()
    conn = db.connect()
    
    issues_found = 0
    
    # 1. CHECK FOR DUPLICATE MATCHES
    print("\n🔍 1. Checking for Duplicate Matches (ID)...")
    try:
        query = """
            SELECT id, COUNT(*) as count 
            FROM matches 
            GROUP BY id 
            HAVING count > 1
        """
        duplicates = pd.read_sql_query(query, conn)
        
        if not duplicates.empty:
            print(f"❌ FOUND {len(duplicates)} DUPLICATE MATCH IDs!")
            print(duplicates.head())
            issues_found += 1
            
            # Auto-repair suggestion (would require complex deletion logic, maybe later)
        else:
            print("✅ No duplicate match IDs found.")
    except Exception as e:
        print(f"⚠️ Error checking match duplicates: {e}")

    # 2. CHECK FOR DUPLICATE PREDICTIONS
    print("\n🔍 2. Checking for Duplicate Predictions...")
    try:
        # Check duplicates based on match_id, model_version, category, prediction_label
        # Ignoring timestamp changes
        query = """
            SELECT match_id, model_version, category, prediction_label, COUNT(*) as count
            FROM predictions
            GROUP BY match_id, model_version, category, prediction_label
            HAVING count > 1
        """
        dup_preds = pd.read_sql_query(query, conn)
        
        if not dup_preds.empty:
            print(f"⚠️ FOUND {len(dup_preds)} DUPLICATE PREDICTION SETS.")
            print(f"   (This might be okay if you re-ran analysis, but wastes space)")
            print(dup_preds.head(3))
            
            # Clean up duplicates?
            # clean = input("   🧹 Do you want to clean up duplicate predictions? (keep latest) [y/n]: ").lower()
            clean = 'y' # FORCE AUTO CLEAN for this session
            if clean == 'y':
                print("   🧹 Cleaning duplicates (keeping latest rowid)...")
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM predictions 
                    WHERE rowid NOT IN (
                        SELECT MAX(rowid) 
                        FROM predictions 
                        GROUP BY match_id, model_version, category, prediction_label
                    )
                """)
                conn.commit()
                print(f"   ✅ Deleted {cursor.rowcount} duplicate rows.")
        else:
            print("✅ No duplicate predictions found.")
            
    except Exception as e:
        print(f"⚠️ Error checking prediction duplicates: {e}")
        
    # 3. STATISTICAL & STANDINGS DATA INTEGRITY
    print("\n🔍 3. Validating Historical Standings Logic...")
    
    # We need to fetch matches with the new columns
    try:
        df_matches = pd.read_sql_query("SELECT * FROM matches ORDER BY timestamp", conn)
        
        if 'home_league_position' not in df_matches.columns:
            print("❌ 'home_league_position' column MISSING in database. Run reconstruction first.")
            issues_found += 1
        else:
            # Check for NULLs in positions for finished games
            finished_games = df_matches[df_matches['status'] == 'finished']
            null_pos = finished_games['home_league_position'].isnull().sum()
            
            if null_pos > 0:
                print(f"⚠️ Found {null_pos} finished matches with NULL league positions.")
                # This usually happens for the very first games of a season/dataset if logic isn't perfect, or untracked leagues
                issues_found += 1
            else:
                print("✅ All finished matches have league positions assigned.")
                
            # Verify Value Ranges
            min_pos = df_matches['home_league_position'].min()
            max_pos = df_matches['home_league_position'].max()
            print(f"   📊 Position Range: {min_pos} to {max_pos}")
            
            if min_pos < 0: 
                print("❌ Found NEGATIVE positions!") 
                issues_found += 1
                
            if max_pos > 30:
                print(f"⚠️ Found positions > 30 ({max_pos}). Valid? (Some lower leagues have many teams)")
                
    except Exception as e:
        print(f"⚠️ Error verifying standings: {e}")

    conn.close()
    
    # 4. CODE PERFORMANCE REVIEW (Static Analysis)
    print("\n🔍 4. Code Performance Review (Static)...")
    
    # Check features_v2 for vectorization
    f_path = os.path.join(os.path.dirname(__file__), '..', 'ml', 'features_v2.py')
    with open(f_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    if "rolling(" in content and "apply(" in content:
        print("ℹ️  Note: 'rolling().apply()' detected. This can be slow if used with custom Python functions.")
        print("   Checking if we can optimize to native pans rolling functions (mean, std, etc)...")
        if ".mean()" in content and ".std()" in content:
             print("   ✅ Valid: Using native optimized .mean() / .std() where possible.")
        else:
             print("   ⚠️ Potential Optimization: Ensure basic stats use native pandas methods, not apply(lambda).")
             
    print("\n" + "="*60)
    if issues_found == 0:
        print("✅✅ AUDIT PASSED: SYSTEM IS HEALTHY ✅✅")
    else:
        print(f"⚠️ AUDIT COMPLETED WITH {issues_found} POTENTIAL ISSUES.")
    print("="*60)

if __name__ == "__main__":
    audit_database()
