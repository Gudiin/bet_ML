import sqlite3
import pandas as pd
from datetime import datetime
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.database.db_manager import DBManager

def reconstruct_standings():
    """
    Reconstrói a tabela de classificação histórica passo a passo.
    Regra: Para cada jogo, a posição é calculada baseada APENAS nos jogos anteriores.
    """
    print("="*60)
    print("⏳ INICIANDO RECONSTRUÇÃO HISTÓRICA DE CLASSIFICAÇÃO")
    print("="*60)
    
    db = DBManager()
    conn = db.connect()
    cursor = conn.cursor()
    
    # Busca todos os jogos terminados, ordenados por torneio e data
    query = """
    SELECT 
        match_id, tournament_id, season_id, start_timestamp, 
        home_team_id, away_team_id, 
        home_score, away_score, status
    FROM matches 
    WHERE status = 'finished'
    ORDER BY tournament_id, season_id, start_timestamp ASC
    """
    
    df_matches = pd.read_sql_query(query, conn)
    
    if df_matches.empty:
        print("❌ Nenhum jogo encontrado para reconstrução.")
        return

    print(f"📊 Processando {len(df_matches)} partidas...")
    
    # Estrutura para manter o estado atual de cada torneio
    # {tournament_id_season_id: {team_id: {points, games, wins, draws, losses, gf, ga}}}
    tournament_states = {}
    
    updates = []
    
    for idx, match in df_matches.iterrows():
        tourn_key = f"{match['tournament_id']}_{match['season_id']}"
        
        if tourn_key not in tournament_states:
            tournament_states[tourn_key] = {}
            
        home_id = match['home_team_id']
        away_id = match['away_team_id']
        
        # Inicializa times se não existirem
        if home_id not in tournament_states[tourn_key]:
            tournament_states[tourn_key][home_id] = {'p': 0, 'j': 0, 'v': 0, 'e': 0, 'd': 0, 'gp': 0, 'gc': 0, 'sg': 0}
        if away_id not in tournament_states[tourn_key]:
            tournament_states[tourn_key][away_id] = {'p': 0, 'j': 0, 'v': 0, 'e': 0, 'd': 0, 'gp': 0, 'gc': 0, 'sg': 0}
            
        # --- 1. CAPTURA O ESTADO ATUAL (ANTES DO JOGO) ---
        # Calcula classificação no momento
        standings = []
        for tid, stats in tournament_states[tourn_key].items():
            standings.append({
                'team_id': tid,
                'points': stats['p'],
                'wins': stats['v'],
                'goal_diff': stats['sg'],
                'goals_for': stats['gp']
            })
            
        # Ordenação oficial típica: Pontos > Vitórias > Saldo > Gols Pró
        standings.sort(key=lambda x: (
            x['points'], 
            x['wins'], 
            x['goal_diff'], 
            x['goals_for']
        ), reverse=True)
        
        # Encontra posição dos envolvidos
        home_pos = next((i+1 for i, t in enumerate(standings) if t['team_id'] == home_id), len(standings) + 1)
        away_pos = next((i+1 for i, t in enumerate(standings) if t['team_id'] == away_id), len(standings) + 1)
        
        # Se for o primeiro jogo do time, posição é teórica (meio da tabela ou último)
        # Vamos assumir que se games=0, a posição não é muito informativa, mas salvamos mesmo assim
        
        updates.append((int(home_pos), int(away_pos), int(match['match_id'])))
        
        # --- 2. ATUALIZA O ESTADO (DEPOIS DO JOGO) ---
        h_score = match['home_score']
        a_score = match['away_score']
        
        # Home Update
        tournament_states[tourn_key][home_id]['j'] += 1
        tournament_states[tourn_key][home_id]['gp'] += h_score
        tournament_states[tourn_key][home_id]['gc'] += a_score
        tournament_states[tourn_key][home_id]['sg'] = tournament_states[tourn_key][home_id]['gp'] - tournament_states[tourn_key][home_id]['gc']
        
        # Away Update
        tournament_states[tourn_key][away_id]['j'] += 1
        tournament_states[tourn_key][away_id]['gp'] += a_score
        tournament_states[tourn_key][away_id]['gc'] += h_score
        tournament_states[tourn_key][away_id]['sg'] = tournament_states[tourn_key][away_id]['gp'] - tournament_states[tourn_key][away_id]['gc']
        
        if h_score > a_score:
            tournament_states[tourn_key][home_id]['p'] += 3
            tournament_states[tourn_key][home_id]['v'] += 1
            tournament_states[tourn_key][away_id]['d'] += 1
        elif a_score > h_score:
            tournament_states[tourn_key][away_id]['p'] += 3
            tournament_states[tourn_key][away_id]['v'] += 1
            tournament_states[tourn_key][home_id]['d'] += 1
        else:
            tournament_states[tourn_key][home_id]['p'] += 1
            tournament_states[tourn_key][home_id]['e'] += 1
            tournament_states[tourn_key][away_id]['p'] += 1
            tournament_states[tourn_key][away_id]['e'] += 1
            
        if idx % 500 == 0:
            print(f"   Processados {idx}/{len(df_matches)} jogos...")

    print("💾 Salvando dados no banco...")
    
    # Batch Update
    cursor.executemany("""
        UPDATE matches 
        SET home_league_position = ?, 
            away_league_position = ? 
        WHERE match_id = ?
    """, updates)
    
    conn.commit()
    conn.close()
    
    print("✅ Reconstrução concluída com sucesso!")
    print(f"   Total atualizado: {len(updates)} partidas.")

if __name__ == "__main__":
    reconstruct_standings()
