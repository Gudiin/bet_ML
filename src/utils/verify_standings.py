import sys
import os
import pandas as pd
import sqlite3

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.database.db_manager import DBManager

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

def verify_standings(target_tournament="Brasileirão Série A", target_season_year="2024"):
    print("="*60)
    print(f"🕵️  VERIFYING STANDINGS: {target_tournament} ({target_season_year})")
    print("="*60)
    
    db = DBManager()
    conn = db.connect()
    
    # 1. Get Tournament ID
    # Try with name matching
    query = f"SELECT DISTINCT tournament_id, tournament_name FROM matches WHERE tournament_name LIKE '%{target_tournament}%' LIMIT 1"
    res = pd.read_sql_query(query, conn)
    
    if res.empty:
        print(f"❌ Tournament '{target_tournament}' not found in database.")
        print("   Available tournaments (sample):")
        print(pd.read_sql_query("SELECT DISTINCT tournament_name FROM matches LIMIT 10", conn))
        return

    t_id = res.iloc[0]['tournament_id']
    t_name = res.iloc[0]['tournament_name']
    print(f"✅ Found Tournament: {t_name} (ID: {t_id})")

    # 2. Get Matches for this tournament
    # Filter by date/season implicitly or just take the whole history for that ID and deduce season
    matches_query = f"""
        SELECT 
            match_id, season_id, start_timestamp, 
            home_team_id, home_team_name, away_team_id, away_team_name,
            home_score, away_score, status
        FROM matches 
        WHERE tournament_id = {t_id} AND status = 'finished'
        ORDER BY start_timestamp ASC
    """
    df = pd.read_sql_query(matches_query, conn)
    
    if df.empty:
        print("❌ No matches found for this tournament.")
        return

    # Identify Season (assuming season_id groups them, or just use year from timestamp)
    df['year'] = pd.to_datetime(df['start_timestamp'], unit='s').dt.year
    
    # Filter for target year if specified, or take the most recent one
    if target_season_year:
        df_season = df[df['year'] == int(target_season_year)]
        if df_season.empty:
             print(f"⚠️ No matches for year {target_season_year}. Using most recent year: {df['year'].max()}")
             df_season = df[df['year'] == df['year'].max()]
    else:
         df_season = df
         
    print(f"📊 Processing {len(df_season)} matches for Season {df_season['year'].iloc[0]}...")

    # 3. Reconstruct State
    teams = {} # {id: {name, p, j, v, e, d, gp, gc, sg}}
    
    for _, match in df_season.iterrows():
        hid, hname = match['home_team_id'], match['home_team_name']
        aid, aname = match['away_team_id'], match['away_team_name']
        hs, as_ = match['home_score'], match['away_score']
        
        # Init
        if hid not in teams: teams[hid] = {'name': hname, 'p': 0, 'j': 0, 'v': 0, 'e': 0, 'd': 0, 'gp': 0, 'gc': 0, 'sg': 0}
        if aid not in teams: teams[aid] = {'name': aname, 'p': 0, 'j': 0, 'v': 0, 'e': 0, 'd': 0, 'gp': 0, 'gc': 0, 'sg': 0}
        
        # Update
        teams[hid]['j'] += 1
        teams[aid]['j'] += 1
        teams[hid]['gp'] += hs
        teams[hid]['gc'] += as_
        teams[aid]['gp'] += as_
        teams[aid]['gc'] += hs
        teams[hid]['sg'] = teams[hid]['gp'] - teams[hid]['gc']
        teams[aid]['sg'] = teams[aid]['gp'] - teams[aid]['gc']
        
        if hs > as_:
            teams[hid]['p'] += 3
            teams[hid]['v'] += 1
            teams[aid]['d'] += 1
        elif as_ > hs:
            teams[aid]['p'] += 3
            teams[aid]['v'] += 1
            teams[hid]['d'] += 1
        else:
            teams[hid]['p'] += 1
            teams[hid]['e'] += 1
            teams[aid]['p'] += 1
            teams[aid]['e'] += 1

    # 4. Sort and Print
    standings = list(teams.values())
    standings.sort(key=lambda x: (x['p'], x['v'], x['sg'], x['gp']), reverse=True)
    
    output_path = os.path.join(os.path.dirname(__file__), 'standings_output.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n🏆 FINAL RECONSTRUCTED TABLE:\n")
        f.write(f"{'Pos':<4} {'Team':<25} {'P':<4} {'J':<4} {'V':<4} {'E':<4} {'D':<4} {'SG':<4}\n")
        f.write("-" * 60 + "\n")
        
        for i, team in enumerate(standings, 1):
            line = f"{i:<4} {team['name']:<25} {team['p']:<4} {team['j']:<4} {team['v']:<4} {team['e']:<4} {team['d']:<4} {team['sg']:<4}\n"
            f.write(line)
            print(line.strip()) # Also print to console
            
        f.write("-" * 60 + "\n")
        f.write("ℹ️  Please compare the Top 5 and Bottom 5 with the real table on Google/SofaScore.\n")
    
    print(f"✅ Output saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", default="Brasileirão Série A", help="League name")
    parser.add_argument("--year", default="2024", help="Year")
    args = parser.parse_args()
    
    verify_standings(args.league, args.year)
