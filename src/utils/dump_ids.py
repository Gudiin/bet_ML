import sqlite3
import pandas as pd

conn = sqlite3.connect('data/football_data.db')
df = pd.read_sql_query('SELECT tournament_id, tournament_name, COUNT(*) as qtd FROM matches GROUP BY tournament_id ORDER BY qtd DESC', conn)
with open('ids.txt', 'w', encoding='utf-8') as f:
    f.write(df.to_string())
conn.close()
