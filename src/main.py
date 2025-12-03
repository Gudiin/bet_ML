"""
Módulo Principal - Sistema de Previsão de Escanteios com Machine Learning.
"""

import sys
import os
import pandas as pd
import re
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.database.db_manager import DBManager
from src.scrapers.sofascore import SofaScoreScraper
from src.ml.feature_engineering import prepare_training_data
from src.ml.model import CornerPredictor
from src.ml.model_improved import ImprovedCornerPredictor, prepare_improved_features
from src.analysis.statistical import StatisticalAnalyzer, Colors


def load_leagues_config() -> list:
    try:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'clubes_sofascore.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('competicoes', [])
    except Exception as e:
        print(f"Erro ao carregar config de ligas: {e}")
        return []

def update_database(league_name: str = "Brasileirão Série A", season_year: str = "2025") -> None:
    """
    Atualiza o banco de dados com inteligência incremental.
    """
    db = DBManager()
    
    # Check for feedback loop updates first
    print("Verificando resultados de previsões anteriores...")
    db.check_predictions()
    
    scraper = SofaScoreScraper(headless=True)
    
    try:
        scraper.start()
        
        # 1. Get Tournament/Season IDs
        t_id = scraper.get_tournament_id(league_name)
        if not t_id:
            print("Torneio não encontrado.")
            return
            
        s_id = scraper.get_season_id(t_id, season_year)
        if not s_id:
            print("Temporada não encontrada.")
            return
            
        print(f"ID Torneio: {t_id}, ID Temporada: {s_id}")
        
        # --- VERIFICAÇÃO DE INTEGRIDADE (NOVO) ---
        stats = db.get_season_stats(s_id)
        total_matches_db = stats['total_matches']
        last_round_db = stats['last_round']
        
        print(f"Status Atual no DB: {total_matches_db} jogos, Última Rodada: {last_round_db}")
        
        # Lógica: Se já tem +370 jogos e não é 2025/25/26, considera completo
        is_current_season = "2025" in season_year or "25/26" in season_year
        if total_matches_db > 370 and not is_current_season:
            print(f"✅ Temporada {season_year} já está completa no banco ({total_matches_db} jogos). Pulando...")
            return

        # Define rodada inicial (Incremental)
        start_round = 1
        if last_round_db > 0:
            # Começa da última rodada para garantir atualizações de jogos adiados/pendentes
            start_round = last_round_db
            print(f"⏩ Retomando atualização a partir da rodada {start_round}...")
        # ----------------------------------------
        
        # 2. Get Matches (passando start_round)
        matches = scraper.get_matches(t_id, s_id, start_round=start_round)
        print(f"Encontrados {len(matches)} jogos novos/atualizados.")
        
        # 3. Process Matches & Stats
        for i, m in enumerate(matches):
            if m['status']['type'] == 'finished':
                print(f"[{i+1}/{len(matches)}] Processando {m['homeTeam']['name']} vs {m['awayTeam']['name']}...")
                
                # Save Match Info
                match_data = {
                    'id': m['id'],
                    'tournament': m['tournament']['name'],
                    'season_id': s_id,
                    'round': m['roundInfo']['round'],
                    'status': 'finished',
                    'timestamp': m['startTimestamp'],
                    'home_id': m['homeTeam']['id'],
                    'home_name': m['homeTeam']['name'],
                    'away_id': m['awayTeam']['id'],
                    'away_name': m['awayTeam']['name'],
                    'home_score': m['homeScore']['display'],
                    'away_score': m['awayScore']['display']
                }
                db.save_match(match_data)
                
                # Get & Save Stats
                stats = scraper.get_match_stats(m['id'])
                db.save_stats(m['id'], stats)
                
    except Exception as e:
        print(f"Erro: {e}")
    finally:
        scraper.stop()
        db.close()

def get_current_season(league_name: str) -> str:
    european_leagues = [
        'Premier League', 'La Liga', 'Bundesliga', 
        'Serie A', 'Ligue 1', 'Serie A 25/26'
    ]
    if league_name in european_leagues:
        return "25/26"
    return "2025"

def get_seasons_list(league_name: str) -> list:
    """Retorna lista das 3 últimas temporadas para a liga."""
    european_leagues = [
        'Premier League', 'La Liga', 'Bundesliga', 
        'Serie A', 'Ligue 1', 'Serie A 25/26'
    ]
    if league_name in european_leagues:
        return ["25/26", "24/25", "23/24"]
    return ["2025", "2024", "2023"]

def update_specific_league() -> None:
    print("\n=== SELECIONE A LIGA ===")
    leagues = [
        "Brasileirão Série A",
        "Brasileirão Série B",
        "Premier League",
        "La Liga",
        "Bundesliga",
        "Serie A 25/26",
        "Ligue 1",
        "Liga Profesional (Argentina)"
    ]
    
    for i, league in enumerate(leagues, 1):
        print(f"{i}. {league}")
        
    try:
        choice = int(input("\nEscolha o número da liga: "))
        if 1 <= choice <= len(leagues):
            selected_league = leagues[choice-1]
            seasons = get_seasons_list(selected_league)
            
            print(f"\n🔄 Iniciando atualização completa (3 anos) para: {selected_league}")
            for season in seasons:
                print(f"\n>>> Temporada {season} <<<")
                update_database(league_name=selected_league, season_year=season)
                
            print(f"\n✅ Atualização concluída para {selected_league}!")
        else:
            print("Opção inválida.")
    except ValueError:
        print("Digite um número válido.")

def update_all_leagues() -> None:
    """Percorre e atualiza TODAS as ligas cadastradas."""
    leagues = [
        "Brasileirão Série A",
        "Brasileirão Série B",
        "Premier League",
        "La Liga",
        "Bundesliga",
        "Serie A 25/26",
        "Ligue 1",
        "Liga Profesional (Argentina)"
    ]
    
    print(f"\n🚀 INICIANDO ATUALIZAÇÃO EM MASSA ({len(leagues)} LIGAS)...")
    print("Isso pode demorar. O sistema pulará automaticamente temporadas já baixadas.")
    
    for league in leagues:
        seasons = get_seasons_list(league)
        print(f"\n{'#'*50}")
        print(f"⚽ LIGA: {league}")
        print(f"{'#'*50}")
        
        for season in seasons:
            print(f"\n>> Verificando Temporada {season}...")
            update_database(league_name=league, season_year=season)

    print("\n🏁 ATUALIZAÇÃO GERAL CONCLUÍDA!")

def train_model() -> None:
    db = DBManager()
    df = db.get_historical_data()
    db.close()
    
    if df.empty:
        print("Banco de dados vazio. Execute a atualização primeiro.")
        return
        
    print(f"Carregados {len(df)} registros para treino.")
    print("\n=== OPÇÕES DE TREINAMENTO ===")
    print("1. Treinamento Rápido (Padrão)")
    print("2. Treinamento Otimizado (Cross-Validation + Tuning)")
    
    choice = input("Escolha uma opção (1 ou 2): ")
    
    if choice == '2':
        print("\nPreparando features avançadas...")
        X, y, _ = prepare_improved_features(df)
        predictor = ImprovedCornerPredictor(use_ensemble=False)
        predictor.train_with_optimization(X, y)
    else:
        print("\nTreinando modelo padrão...")
        X, y, _ = prepare_training_data(df)
        predictor = CornerPredictor()
        predictor.train(X, y)

def analyze_match_url() -> None:
    url = input("Cole a URL do jogo do SofaScore: ")
    match_id_search = re.search(r'id:(\d+)', url)
    
    if not match_id_search:
        print("ID do jogo não encontrado na URL.")
        return

    match_id = match_id_search.group(1)
    print(f"Analisando jogo ID: {match_id}...")
    
    scraper = SofaScoreScraper(headless=True)
    try:
        scraper.start()
        
        api_url = f"https://www.sofascore.com/api/v1/event/{match_id}"
        ev_data = scraper._fetch_api(api_url)
        
        if not ev_data or 'event' not in ev_data:
            print("Erro ao buscar dados do jogo.")
            return
            
        ev = ev_data['event']
        home_id = ev['homeTeam']['id']
        away_id = ev['awayTeam']['id']
        match_name = f"{ev['homeTeam']['name']} vs {ev['awayTeam']['name']}"
        print(f"Jogo: {match_name}")
        
        db = DBManager()
        match_data = {
            'id': match_id,
            'tournament': ev.get('tournament', {}).get('name', 'Unknown'),
            'season_id': ev.get('season', {}).get('id', 0),
            'round': ev.get('roundInfo', {}).get('round', 0),
            'status': 'finished',
            'timestamp': ev.get('startTimestamp', 0),
            'home_id': home_id,
            'home_name': ev['homeTeam']['name'],
            'away_id': away_id,
            'away_name': ev['awayTeam']['name'],
            'home_score': ev.get('homeScore', {}).get('display', 0),
            'away_score': ev.get('awayScore', {}).get('display', 0)
        }
        db.save_match(match_data)
        db.close()
        
        print("Coletando histórico recente...")
        db = DBManager()
        df = db.get_historical_data()
        db.close()
        
        if df.empty:
            print("Banco de dados vazio.")
            return

        home_games = df[(df['home_team_id'] == home_id) | (df['away_team_id'] == home_id)].tail(5)
        away_games = df[(df['home_team_id'] == away_id) | (df['away_team_id'] == away_id)].tail(5)
        
        if len(home_games) < 3 or len(away_games) < 3:
            print("Dados insuficientes no histórico.")
        
        db = DBManager()
        db.delete_predictions(match_id)
        db.close()
        
        ml_prediction = 0
        try:
            predictor_v2 = ImprovedCornerPredictor()
            if predictor_v2.load_model():
                def get_team_stats(games, team_id):
                    stats = {'corners': [], 'shots': [], 'goals': [], 'corners_ht': []}
                    for _, row in games.iterrows():
                        if row['home_team_id'] == team_id:
                            stats['corners'].append(row['corners_home_ft'])
                            stats['shots'].append(row['shots_ot_home_ft'])
                            stats['goals'].append(row['home_score'])
                            stats['corners_ht'].append(row['corners_home_ht'])
                        else:
                            stats['corners'].append(row['corners_away_ft'])
                            stats['shots'].append(row['shots_ot_away_ft'])
                            stats['goals'].append(row['away_score'])
                            stats['corners_ht'].append(row['corners_away_ht'])
                    def avg(lst): return sum(lst) / len(lst) if lst else 0
                    return {k: avg(v) for k, v in stats.items()}, {k: avg(v[-3:]) for k, v in stats.items()}

                h_stats_5, h_stats_3 = get_team_stats(home_games, home_id)
                a_stats_5, a_stats_3 = get_team_stats(away_games, away_id)
                
                features = [
                    h_stats_5['corners'], h_stats_5['shots'], h_stats_5['goals'],
                    a_stats_5['corners'], a_stats_5['shots'], a_stats_5['goals'],
                    h_stats_5['corners_ht'], a_stats_5['corners_ht'],
                    h_stats_5['corners'] + a_stats_5['corners'],
                    h_stats_5['corners'] - a_stats_5['corners'],
                    h_stats_3['corners'] - h_stats_5['corners'],
                    a_stats_3['corners'] - a_stats_5['corners']
                ]
                
                X_new = [features]
                pred = predictor_v2.predict(X_new)
                ml_prediction = pred[0]
                print(f"\n🤖 Previsão da IA (LightGBM): {ml_prediction:.2f} Escanteios")
                
                db = DBManager()
                db.save_prediction(match_id, 'ML', ml_prediction, f"Over {int(ml_prediction)}", 0.0, verbose=True)
                db.close()
        except Exception as e:
            print(f"Erro ao usar modelo V2: {e}")
            pass

        if ml_prediction == 0:
            predictor = CornerPredictor()
            if predictor.load_model():
                # Falta implementar fallback corretamente ou manter como estava
                pass
            
        analyzer = StatisticalAnalyzer()
        
        def prepare_team_df(games, team_id):
            data = []
            for _, row in games.iterrows():
                is_home = row['home_team_id'] == team_id
                data.append({
                    'corners_ft': row['corners_home_ft'] if is_home else row['corners_away_ft'],
                    'corners_ht': row['corners_home_ht'] if is_home else row['corners_away_ht'],
                    'corners_2t': (row['corners_home_ft'] - row['corners_home_ht']) if is_home else (row['corners_away_ft'] - row['corners_away_ht']),
                    'shots_ht': row['shots_ot_home_ht'] if is_home else row['shots_ot_away_ht']
                })
            return pd.DataFrame(data)

        df_h_stats = prepare_team_df(home_games, home_id)
        df_a_stats = prepare_team_df(away_games, away_id)

        top_picks, suggestions = analyzer.analyze_match(df_h_stats, df_a_stats, ml_prediction=ml_prediction, match_name=match_name)
        
        db = DBManager()
        for pick in top_picks:
            db.save_prediction(match_id, 'Statistical', 0, pick['Seleção'], pick['Prob'], odds=pick['Odd'], category='Top7', market_group=pick['Mercado'])
            
        for level, pick in suggestions.items():
            if pick:
                db.save_prediction(match_id, 'Statistical', 0, pick['Seleção'], pick['Prob'], odds=pick['Odd'], category=f"Suggestion_{level}", market_group=pick['Mercado'])
        
        print("✅ Previsões salvas no banco de dados.")
        db.close()

    except Exception as e:
        print(f"Erro na análise: {e}")
    finally:
        scraper.stop()

def retrieve_analysis() -> None:
    match_id = input("Digite o ID do jogo: ")
    db = DBManager()
    conn = db.connect()
    
    match_query = "SELECT home_team_name, away_team_name FROM matches WHERE match_id = ?"
    match_info = pd.read_sql_query(match_query, conn, params=(match_id,))
    
    match_name = None
    if not match_info.empty:
        match_name = f"{match_info.iloc[0]['home_team_name']} vs {match_info.iloc[0]['away_team_name']}"
    
    query_ml = "SELECT predicted_value FROM predictions WHERE match_id = ? AND prediction_type = 'ML'"
    ml_pred = pd.read_sql_query(query_ml, conn, params=(match_id,))
    
    if not ml_pred.empty:
        print(f"\n🤖 Previsão da IA (Random Forest): {ml_pred.iloc[0]['predicted_value']:.2f} Escanteios")

    query_top7 = "SELECT market_group, market, probability, odds, status FROM predictions WHERE match_id = ? AND category = 'Top7' ORDER BY probability DESC"
    top7 = pd.read_sql_query(query_top7, conn, params=(match_id,))
    
    if not top7.empty:
        if match_name:
             print(f"\n⚽ {Colors.BOLD}{match_name}{Colors.RESET}")
        print(f"🏆 {Colors.BOLD}TOP 7 OPORTUNIDADES (RECUPERADO){Colors.RESET}")
        tabela_display = []
        for _, row in top7.iterrows():
            prob = row['probability']
            tipo = "OVER" if "Over" in row['market'] else "UNDER"
            cor = Colors.GREEN if tipo == "OVER" else Colors.CYAN
            seta = "▲" if tipo == "OVER" else "▼"
            
            m_group = row['market_group'] if row['market_group'] else "RECUPERADO"
            
            status = row['status']
            if status == 'GREEN':
                status_fmt = f"{Colors.GREEN}✓ GREEN{Colors.RESET}"
            elif status == 'RED':
                status_fmt = f"{Colors.RED}✗ RED{Colors.RESET}"
            else:
                status_fmt = f"{Colors.YELLOW}PENDING{Colors.RESET}"
            
            linha_fmt = f"{cor}{row['market']}{Colors.RESET}"
            prob_fmt = f"{prob * 100:.1f}%"
            odd_fmt = f"{Colors.BOLD}@{row['odds']:.2f}{Colors.RESET}"
            direcao_fmt = f"{cor}{seta} {tipo}{Colors.RESET}"
            
            tabela_display.append([m_group, linha_fmt, prob_fmt, odd_fmt, direcao_fmt, status_fmt])
            
        headers = ["MERCADO", "LINHA", "PROB.", "ODD JUSTA", "TIPO", "STATUS"]
        from tabulate import tabulate
        print(tabulate(tabela_display, headers=headers, tablefmt="fancy_grid", stralign="center"))
    else:
        print("Nenhuma análise Top 7 encontrada para este ID.")

    query_sugg = "SELECT category, market_group, market, probability, odds, status FROM predictions WHERE match_id = ? AND category LIKE 'Suggestion_%'"
    suggs = pd.read_sql_query(query_sugg, conn, params=(match_id,))
    
    if not suggs.empty:
        print(f"\n🎯 {Colors.BOLD}SUGESTÕES DA IA (RECUPERADO):{Colors.RESET}")
        for _, row in suggs.iterrows():
            level = row['category'].split('_')[1]
            cor_nivel = Colors.GREEN if level == "Easy" else (Colors.YELLOW if level == "Medium" else Colors.RED)
            m_group = row['market_group'] if row['market_group'] else ""
            
            status = row['status']
            if status == 'GREEN':
                status_fmt = f"[{Colors.GREEN}✓ GREEN{Colors.RESET}]"
            elif status == 'RED':
                status_fmt = f"[{Colors.RED}✗ RED{Colors.RESET}]"
            else:
                status_fmt = f"[{Colors.YELLOW}PENDING{Colors.RESET}]"
                
            print(f"{cor_nivel}[{level.upper()}]{Colors.RESET} {m_group} - {row['market']} (@{row['odds']:.2f}) | Prob: {row['probability']*100:.1f}% {status_fmt}")
    else:
        print("Nenhuma sugestão da IA encontrada para este ID.")
        
    db.close()

def update_match_by_url() -> None:
    url = input("Cole a URL do jogo do SofaScore: ")
    match_id_search = re.search(r'id:(\d+)', url)
    
    if not match_id_search:
        print("ID do jogo não encontrado na URL.")
        return

    match_id = match_id_search.group(1)
    print(f"Atualizando jogo ID: {match_id}...")
    
    scraper = SofaScoreScraper(headless=True)
    db = DBManager()
    
    try:
        scraper.start()
        
        api_url = f"https://www.sofascore.com/api/v1/event/{match_id}"
        ev_data = scraper._fetch_api(api_url)
        
        if not ev_data or 'event' not in ev_data:
            print("Erro ao buscar dados do jogo.")
            return
            
        ev = ev_data['event']
        match_name = f"{ev['homeTeam']['name']} vs {ev['awayTeam']['name']}"
        print(f"Jogo: {match_name} (Status: {ev['status']['type']})")
        
        match_data = {
            'id': match_id,
            'tournament': ev.get('tournament', {}).get('name', 'Unknown'),
            'season_id': ev.get('season', {}).get('id', 0),
            'round': ev.get('roundInfo', {}).get('round', 0),
            'status': ev.get('status', {}).get('type', 'unknown'),
            'timestamp': ev.get('startTimestamp', 0),
            'home_id': ev['homeTeam']['id'],
            'home_name': ev['homeTeam']['name'],
            'away_id': ev['awayTeam']['id'],
            'away_name': ev['awayTeam']['name'],
            'home_score': ev.get('homeScore', {}).get('display', 0),
            'away_score': ev.get('awayScore', {}).get('display', 0)
        }
        db.save_match(match_data)
        print("✅ Dados da partida atualizados.")
        
        if ev['status']['type'] == 'finished':
            print("Coletando estatísticas finais...")
            stats = scraper.get_match_stats(match_id)
            db.save_stats(match_id, stats)
            print("✅ Estatísticas salvas.")
            
            print("\nVerificando apostas pendentes...")
            db.check_predictions()
        else:
            print("⚠️ Jogo não finalizado. Estatísticas completas podem não estar disponíveis.")
            
    except Exception as e:
        print(f"Erro ao atualizar jogo: {e}")
    finally:
        scraper.stop()
        db.close()

def main():
    while True:
        print("\n" + "═" * 50)
        print(f"{Colors.BOLD}🤖 SISTEMA DE PREVISÃO DE ESCANTEIOS (ML){Colors.RESET}")
        print("═" * 50)
        print("1. Atualizar Campeonato Brasileiro Serie A")
        print("2. Treinar Modelo de IA")
        print("3. Analisar Jogo (URL)")
        print("4. Consultar Análise (ID)")
        print("5. Atualizar Liga Específica (3 Anos)")
        print("6. Atualizar Jogo Específico (URL)")
        print("9. 🚀 Atualizar TODAS as Ligas (3 Anos - Batch)")
        print("0. Sair")
        
        choice = input("\nEscolha uma opção: ")
        
        if choice == '1':
            update_database()
        elif choice == '2':
            train_model()
        elif choice == '3':
            analyze_match_url()
        elif choice == '4':
            retrieve_analysis()
        elif choice == '5':
            update_specific_league()
        elif choice == '6':
            update_match_by_url()
        elif choice == '9':
            update_all_leagues()
        elif choice == '0':
            print("Saindo...")
            break
        else:
            print("Opção inválida.")

if __name__ == "__main__":
    main()