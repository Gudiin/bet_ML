"""
Módulo Principal - Sistema de Previsão de Escanteios com Machine Learning.
"""

import sys
import os
import pandas as pd
import re
import json
import traceback

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.database.db_manager import DBManager
from src.scrapers.sofascore import SofaScoreScraper
from src.analysis.statistical import StatisticalAnalyzer, Colors

# Imports de ML (Profissional V2)
from src.ml.features_v2 import create_advanced_features, prepare_features_for_prediction
from src.ml.model_v2 import ProfessionalPredictor


def _fix_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Helper para corrigir nomes de colunas do banco para o formato esperado pelo ML.
    O banco usa 'home_score', mas o feature engineering espera 'goals_ft_home'.
    """
    if df.empty:
        return df
        
    df = df.copy()
    
    # Mapeamento de Gols (O erro estava aqui!)
    if 'home_score' in df.columns and 'goals_ft_home' not in df.columns:
        df['goals_ft_home'] = df['home_score']
    if 'away_score' in df.columns and 'goals_ft_away' not in df.columns:
        df['goals_ft_away'] = df['away_score']
        
    return df


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
        
        # --- VERIFICAÇÃO DE INTEGRIDADE ---
        stats = db.get_season_stats(s_id)
        total_matches_db = stats['total_matches']
        last_round_db = stats['last_round']
        
        print(f"Status Atual no DB: {total_matches_db} jogos, Última Rodada: {last_round_db}")
        
        # Lógica: Se já tem +370 jogos e não é temporada atual, considera completo
        is_current_season = "2025" in season_year or "25/26" in season_year
        if total_matches_db > 370 and not is_current_season:
            print(f"✅ Temporada {season_year} já está completa no banco ({total_matches_db} jogos). Pulando...")
            return

        # Define rodada inicial (Incremental)
        start_round = 1
        if last_round_db > 0:
            start_round = last_round_db
            print(f"⏩ Retomando atualização a partir da rodada {start_round}...")
        
        # 2. Get Matches
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
                    'tournament_id': m['tournament']['id'], # Novo campo
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

def train_model() -> None:
    """
    Treina o modelo de Machine Learning utilizando o pipeline Professional V2.
    
    Regra de Negócio:
        O treinamento usa LightGBM com objetivo Poisson (adequado para contagem)
        e validação temporal estrita (TimeSeriesSplit) para evitar data leakage.
        
    Pipeline:
        1. Carrega dados históricos do banco
        2. Gera features avançadas (Home/Away, H2H, Momentum)
        3. Treina com validação cruzada temporal (5 folds)
        4. Salva modelo em data/corner_model_v2_professional.pkl
    """
    db = DBManager()
    df = db.get_historical_data()
    db.close()
    
    if df.empty:
        print("Banco de dados vazio. Execute a atualização primeiro.")
        return
        
    print(f"Carregados {len(df)} registros para treino.")
    
    # Correção de nomes de colunas para compatibilidade
    df = _fix_column_names(df)
    
    print("\n🚀 Iniciando Treinamento Profissional V2...")
    print("🔧 Gerando features avançadas (Home/Away, H2H, Momentum)...")
    
    try:
        X, y, timestamps = create_advanced_features(df, window_short=3, window_long=5)
        
        print(f"📊 Features geradas: {X.shape[1]} colunas, {X.shape[0]} amostras")
        
        # Treina com validação temporal
        predictor = ProfessionalPredictor()
        predictor.train_time_series_split(X, y, timestamps)
        
    except Exception as e:
        print(f"❌ Erro fatal no treinamento: {e}")
        traceback.print_exc()

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
            'tournament_id': ev.get('tournament', {}).get('id', 0), # Novo campo
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

        # --- CORREÇÃO DE NOMES DE COLUNAS TAMBÉM NA ANÁLISE ---
        df = _fix_column_names(df)
        # ------------------------------------------------------

        home_games = df[(df['home_team_id'] == home_id) | (df['away_team_id'] == home_id)].tail(5)
        away_games = df[(df['home_team_id'] == away_id) | (df['away_team_id'] == away_id)].tail(5)
        
        if len(home_games) < 3 or len(away_games) < 3:
            print("Dados insuficientes no histórico.")
        
        db = DBManager()
        db.delete_predictions(match_id)
        db.close()
        
        ml_prediction = 0
        
        # 1. Preparar Features usando a V2
        try:
            print(f"Gerando features avançadas para {match_name}...")
            
            # Cria instância do DB para passar ao features_v2
            db_for_features = DBManager()
            features_df = prepare_features_for_prediction(
                home_id=home_id,
                away_id=away_id,
                db_manager=db_for_features,
                window_long=5
            )
            db_for_features.close()
            
            # 2. Carregar e Usar o Modelo Profissional
            predictor = ProfessionalPredictor()
            
            if predictor.load_model():
                # Faz a previsão
                pred_array = predictor.predict(features_df)
                ml_prediction = float(pred_array[0])
                
                print(f"\n🤖 Previsão da IA (Professional V2): {ml_prediction:.2f} Escanteios")
                
                # Salva no banco
                db = DBManager()
                db.save_prediction(
                    match_id, 
                    'ML_V2', 
                    ml_prediction, 
                    f"Over {int(ml_prediction)}", 
                    0.0, 
                    category="Professional",
                    verbose=True
                )
                db.close()
            else:
                print(f"{Colors.RED}⚠️ Modelo Profissional não encontrado. Treine-o primeiro (Opção 2).{Colors.RESET}")
                
        except Exception as e:
            print(f"{Colors.RED}❌ Erro na Predição ML: {e}{Colors.RESET}")
            traceback.print_exc()

        # Statistical analysis continues
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

        # Extrai métricas avançadas do DataFrame de features (se disponível)
        # Essas métricas serão usadas pelo Lambda Híbrido no Monte Carlo
        advanced_metrics = None
        if 'features_df' in dir() and features_df is not None and not features_df.empty:
            try:
                advanced_metrics = {
                    'home_avg_corners_home': float(features_df['home_avg_corners_home'].iloc[0]),
                    'away_avg_corners_away': float(features_df['away_avg_corners_away'].iloc[0]),
                    'home_avg_corners_conceded_home': float(features_df['home_avg_corners_conceded_home'].iloc[0]),
                    'away_avg_corners_conceded_away': float(features_df['away_avg_corners_conceded_away'].iloc[0]),
                    'home_avg_corners_h2h': float(features_df['home_avg_corners_h2h'].iloc[0]),
                    'away_avg_corners_h2h': float(features_df['away_avg_corners_h2h'].iloc[0]),
                    'home_avg_corners_general': float(features_df['home_avg_corners_general'].iloc[0]),
                    'away_avg_corners_general': float(features_df['away_avg_corners_general'].iloc[0]),
                }
                print(f"\n✅ Métricas avançadas extraídas para Monte Carlo Híbrido")
            except Exception as e:
                print(f"\n⚠️ Métricas avançadas não disponíveis: {e}")
                advanced_metrics = None

        top_picks, suggestions = analyzer.analyze_match(
            df_h_stats, df_a_stats, 
            ml_prediction=ml_prediction, 
            match_name=match_name,
            advanced_metrics=advanced_metrics
        )
        
        # Helper to extract line value from label (e.g., "Over 3.5" -> 3.5)
        def extract_line(label: str) -> float:
            import re
            match = re.search(r'(\d+\.?\d*)', label)
            return float(match.group(1)) if match else 0.0
        
        db = DBManager()
        for pick in top_picks:
            line_value = extract_line(pick['Seleção'])
            db.save_prediction(match_id, 'Statistical', line_value, pick['Seleção'], pick['Prob'], odds=pick['Odd'], category='Top7', market_group=pick['Mercado'])
            
        for level, pick in suggestions.items():
            if pick:
                line_value = extract_line(pick['Seleção'])
                db.save_prediction(match_id, 'Statistical', line_value, pick['Seleção'], pick['Prob'], odds=pick['Odd'], category=f"Suggestion_{level}", market_group=pick['Mercado'])
        
        print("✅ Previsões salvas no banco de dados.")
        
        # Se o jogo já acabou, verifica se acertou imediatamente
        if ev.get('status', {}).get('type') == 'finished':
            print("🏁 Jogo finalizado. Verificando acertos...")
            db.check_predictions()
            
        db.close()

    except Exception as e:
        print(f"Erro na análise: {e}")
    finally:
        scraper.stop()

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
            'tournament_id': ev.get('tournament', {}).get('id', 0), # Novo campo
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

def retrieve_analysis() -> None:
    user_input = input("Digite o ID do jogo (ou cole a URL): ")
    
    # Tenta extrair ID se for URL
    match_id_search = re.search(r'id:(\d+)', user_input)
    if match_id_search:
        match_id = match_id_search.group(1)
    else:
        # Tenta usar o input direto (limpando espaços)
        match_id = user_input.strip()
        if not match_id.isdigit():
             print("❌ ID inválido. Certifique-se de colar a URL correta ou apenas os números.")
             return
    
    db = DBManager()
    conn = db.connect()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT prediction_label, confidence, odds, category, market_group, model_version, prediction_value
        FROM predictions 
        WHERE match_id = ?
        ORDER BY confidence DESC
    ''', (match_id,))
    
    rows = cursor.fetchall()
    db.close()
    
    if not rows:
        print("Nenhuma análise encontrada para este jogo.")
        return
        
    print(f"\n📊 Análise para o Jogo {match_id}:")
    print("-" * 50)
    
    # Agrupa por categoria
    ml_pred = None
    stats_preds = []
    
    for row in rows:
        label, conf, odds, cat, market, model, val = row
        if cat == 'Professional' or model == 'ML_V2':
            ml_pred = (val, label)
        else:
            stats_preds.append((label, conf, odds, cat, market))
            
    if ml_pred:
        print(f"🤖 IA (Professional V2): {ml_pred[0]:.2f} Escanteios ({ml_pred[1]})")
        print("-" * 50)
        
    print("📈 Oportunidades Estatísticas:")
    for label, conf, odds, cat, market in stats_preds:
        print(f"   • {label:<20} | Prob: {conf:>6.1%} | Odd: {odds:>5.2f} | [{cat}]")
    print("-" * 50)

def update_specific_league() -> None:
    league_name = input("Nome da Liga (ex: 'Brasileirão Série A'): ")
    years = ["2023", "2024", "2025"] # Exemplo de 3 anos
    
    print(f"Atualizando {league_name} para os anos: {years}")
    for year in years:
        print(f"\n📅 Processando Temporada {year}...")
        update_database(league_name, year)

def update_all_leagues() -> None:
    leagues = load_leagues_config()
    years = ["2023", "2024", "2025"]  # Últimos 3 anos
    
    print(f"🚀 Iniciando atualização em lote de {len(leagues)} ligas...")
    
    for league in leagues:
        league_name = league['torneio']  # Chave correta do JSON
        print(f"\n🏆 Liga: {league_name}")
        for year in years:
            print(f"   📅 Temporada {year}...")
            update_database(league_name, year)
            
    print("\n✅ Atualização em lote concluída!")
    print("\n✅ Atualização em lote concluída!")

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