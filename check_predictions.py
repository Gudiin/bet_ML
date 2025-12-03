import sqlite3
import pandas as pd

conn = sqlite3.connect('data/football_data.db')

# Check recent predictions
print("=" * 80)
print("RECENT PREDICTIONS (Last 30)")
print("=" * 80)

query = """
SELECT match_id, market, probability, category, market_group
FROM predictions 
WHERE category IN ('Top7', 'Suggestion_Easy', 'Suggestion_Medium', 'Suggestion_Hard')
ORDER BY id DESC 
LIMIT 30
"""

df = pd.read_sql_query(query, conn)
print(df.to_string())

# Count Over vs Under
print("\n" + "=" * 80)
print("OVER vs UNDER DISTRIBUTION")
print("=" * 80)

query2 = """
SELECT 
    CASE 
        WHEN market LIKE '%Over%' THEN 'OVER'
        WHEN market LIKE '%Under%' THEN 'UNDER'
        ELSE 'OTHER'
    END as bet_type,
    COUNT(*) as count,
    AVG(probability) as avg_prob
FROM predictions
WHERE category = 'Top7'
GROUP BY bet_type
"""

df2 = pd.read_sql_query(query2, conn)
print(df2.to_string())

conn.close()
