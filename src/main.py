"""
Módulo Principal - Sistema de Previsão de Escanteios com Machine Learning.

Este arquivo é o "Maestro" da orquestra. Ele coordena todos os outros módulos
(Scraper, Banco de Dados, IA, Estatística) para fazer o sistema funcionar.

É aqui que o usuário interage com o sistema via terminal (CLI).

Funcionalidades Principais:
---------------------------
1. **Atualização do Banco de Dados**:
   - Chama o Scraper (Olheiro) para buscar novos jogos.
   - Salva tudo no Banco de Dados (Caderno).
   
2. **Treinamento do Modelo de IA**:
   - Pega os dados históricos.
   - Ensina o computador a reconhecer padrões.
   - Salva o "cérebro" treinado em um arquivo.
   
3. **Análise de Jogo**:
   - O usuário cola o link de um jogo.
   - O sistema busca os dados desse jogo.
   - A IA faz uma previsão.
   - O Monte Carlo faz simulações.
   - O sistema sugere as melhores apostas.

Como Usar:
----------
Basta executar este arquivo e seguir o menu interativo:
    $ python src/main.py

Fluxo de Dados:
---------------
    [Usuário] -> [main.py] -> [SofaScoreScraper] -> [Internet]
                     |
                     v
               [DBManager] <-> [Arquivo .db]
                     |
                     v
               [CornerPredictor] (IA)
                     |
                     v
            [StatisticalAnalyzer] (Estatística)
                     |
                     v
                [Resultado Final]

Author: Projeto Bet Team
Version: 1.0.0
"""

import sys
import os
import pandas as pd
import re
import json

#
# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.database.db_manager import DBManager
from src.scrapers.sofascore import SofaScoreScraper
from src.ml.feature_engineering import prepare_training_data
from src.ml.model import CornerPredictor
from src.ml.model_improved import ImprovedCornerPredictor, prepare_improved_features
from src.analysis.statistical import StatisticalAnalyzer, Colors


def load_leagues_config() -> list:
    """Carrega configurações de ligas do arquivo JSON."""
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
    Atualiza o banco de dados com partidas e estatísticas do Brasileirão.
    
    Esta função executa o pipeline completo de web scraping para coletar
    dados da temporada atual do Brasileirão Série A via API do SofaScore.
    
    Fluxo de Execução:
    ------------------
    1. **Feedback Loop**: Verifica resultados de previsões pendentes e 
       atualiza status (GREEN/RED) baseado nos resultados reais
       
    2. **Identificação do Torneio**: Busca o ID do Brasileirão e temporada 2025
       na API do SofaScore
       
    3. **Coleta de Partidas**: Obtém lista de todas as partidas da temporada
    
    4. **Processamento**: Para cada jogo finalizado:
       - Salva informações básicas (times, placar, data)
       - Coleta estatísticas detalhadas (escanteios, chutes, etc.)
       - Persiste no banco de dados SQLite
    
    Regras de Negócio:
    ------------------
    - Apenas jogos com status 'finished' são processados
    - O scraper usa rate limiting (0.5-1.5s) para evitar bloqueio
    - Dados são salvos com upsert (INSERT OR REPLACE)
    - O feedback loop é executado ANTES do scraping para atualizar previsões
    
    Pipeline de Dados:
    ------------------
    ::
    
        SofaScore API ─▶ Scraper ─▶ Parser ─▶ DBManager ─▶ SQLite
        
        Tabelas atualizadas:
        ├── matches: informações básicas do jogo
        └── match_stats: estatísticas detalhadas (escanteios, chutes, etc.)
    
    Tempo Estimado:
    ---------------
    - ~10-15 minutos para temporada completa (~380 jogos)
    - ~1-2 segundos por jogo (rate limiting)
    
    Raises:
        Exception: Erro genérico capturado e exibido no console.
                   O scraper é encerrado graciosamente em caso de erro.
    
    Example:
        >>> update_database()
        Verificando resultados de previsões anteriores...
        ID Torneio: 325, ID Temporada: 58766
        Encontrados 150 jogos.
        [1/150] Processando Flamengo vs Palmeiras...
        [2/150] Processando Corinthians vs São Paulo...
        ...
    
    Note:
        Esta função deve ser executada periodicamente para manter o banco
        atualizado com novos jogos. Recomenda-se executar antes de análises
        importantes para garantir dados mais recentes.
        
        O navegador Playwright é executado em modo headless por padrão.
        Para debug visual, altere headless=False no construtor do scraper.
    
    See Also:
        - :class:`SofaScoreScraper`: Classe responsável pelo web scraping
        - :class:`DBManager`: Gerenciador de persistência
        - :meth:`DBManager.check_predictions`: Feedback loop de previsões
    """
    db = DBManager()
    
    # Check for feedback loop updates first
    print("Verificando resultados de previsões anteriores...")
    db.check_predictions()
    
    scraper = SofaScoreScraper(headless=True) # Set headless=False to debug
    
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
        
        # 2. Get Matches
        matches = scraper.get_matches(t_id, s_id)
        print(f"Encontrados {len(matches)} jogos.")
        
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
    """
    Determina a temporada atual baseada na liga.
    
    Regra de Negócio:
        - Ligas Europeias (La Liga, Premier, etc): Usam formato "25/26"
        - Ligas Brasileiras/Sul-americanas: Usam formato "2025"
        
    Args:
        league_name: Nome da liga.
        
    Returns:
        str: String da temporada (ex: "25/26" ou "2025").
    """
    european_leagues = [
        'Premier League', 'La Liga', 'Bundesliga', 
        'Serie A', 'Ligue 1', 'Serie A 25/26'
    ]
    
    if league_name in european_leagues:
        return "25/26"
    return "2025"

def update_specific_league() -> None:
    """
    Menu para atualizar uma liga específica.
    """
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
            
            # Determina temporada automaticamente
            season = get_current_season(selected_league)
            print(f"\nIniciando atualização para: {selected_league} (Temporada {season})")
            
            update_database(league_name=selected_league, season_year=season)
        else:
            print("Opção inválida.")
    except ValueError:
        print("Digite um número válido.")

def update_full_history() -> None:
    """Atualiza histórico de 3 anos para todas as ligas (Demorado!)."""
    leagues = load_leagues_config()
    years = ["2025", "2024", "2023"]
    
    print(f"\n⚠️  ATENÇÃO: Isso atualizará {len(leagues)} ligas por {len(years)} temporadas.")
    print("Isso pode levar VÁRIAS HORAS. Tem certeza?")
    confirm = input("Digite 'SIM' para continuar: ")
    
    if confirm != 'SIM':
        print("Operação cancelada.")
        return
        
    for league in leagues:
        for year in years:
            print(f"\n>>> Atualizando {league['torneio']} - Temporada {year} <<<")
            update_database(league_name=league['torneio'], season_year=year)

def train_model() -> None:
    """
    Treina o modelo de Machine Learning.
    
    Opções:
    1. Treinamento Rápido (Random Forest Padrão)
    2. Treinamento Otimizado (LightGBM + CV + Tuning)
    """
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
        predictor = ImprovedCornerPredictor(use_ensemble=False) # Start simple, maybe add ensemble option later
        predictor.train_with_optimization(X, y)
    else:
        print("\nTreinando modelo padrão...")
        X, y, _ = prepare_training_data(df)
        predictor = CornerPredictor()
        predictor.train(X, y)


def analyze_match_url() -> None:
    """
    Analisa uma partida específica a partir de sua URL do SofaScore.
    
    Esta é a função principal de análise que combina Machine Learning e
    Estatística Avançada (Monte Carlo) para gerar previsões completas
    de escanteios para uma partida específica.
    
    Fluxo de Execução:
    ------------------
    1. **Input**: Solicita URL do jogo no formato do SofaScore
       - Exemplo: https://www.sofascore.com/flamengo-palmeiras/id:12345678
       
    2. **Extração**: Extrai o match_id da URL via regex
    
    3. **Coleta de Dados**: Via API do SofaScore:
       - Informações do evento (times, data, placar)
       - Histórico dos últimos 5 jogos de cada time
       
    4. **Previsão ML**: Usa o Random Forest treinado para prever total
    
    5. **Análise Monte Carlo**: Executa 10.000 simulações para cada mercado:
       - Jogo Completo (Over/Under)
       - Primeiro Tempo (HT)
       - Segundo Tempo (2T)
       - Total Mandante
       - Total Visitante
       
    6. **Geração de Sugestões**: Categoriza oportunidades em:
       - 🟢 EASY: Alta probabilidade (>70%), odds baixas
       - 🟡 MEDIUM: Probabilidade média (50-75%)
       - 🔴 HARD: Probabilidade menor, value bet
       
    7. **Persistência**: Salva todas as previsões no banco para feedback loop
    
    Regras de Negócio:
    ------------------
    - URL deve conter "id:" seguido do match_id numérico
    - Requer mínimo de 3 jogos no histórico de cada time
    - Previsões anteriores do mesmo jogo são deletadas antes de nova análise
    - Usa λ = 0.6 * média_geral + 0.4 * média_5_jogos para Monte Carlo
    - Score = Probabilidade * (1 - CV * fator) penaliza variância alta
    
    Cálculo da Previsão ML:
    -----------------------
    ::
    
        Features de entrada:
        X = [home_avg_corners, home_avg_shots, home_avg_goals,
             away_avg_corners, away_avg_shots, away_avg_goals]
             
        Modelo: RandomForestRegressor(n_estimators=100)
        
        Previsão = média das 100 árvores de decisão
    
    Cálculo Monte Carlo:
    --------------------
    ::
    
        1. Determina distribuição:
           - Se variância > média: Binomial Negativa (overdispersion)
           - Se variância ≤ média: Poisson
           
        2. Gera 10.000 amostras aleatórias
        
        3. Calcula probabilidade:
           P(Over X.5) = count(amostras > X.5) / 10.000
           
        4. Calcula odd justa:
           Odd = 1 / Probabilidade
    
    Formato de Saída:
    -----------------
    ::
    
        🤖 Previsão da IA (Random Forest): 10.82 Escanteios
        
        🏆 TOP 7 OPORTUNIDADES (DATA DRIVEN)
        ╒═══════════════╤══════════╤═════════╤═══════════╤══════════╕
        │    MERCADO    │  LINHA   │  PROB.  │ ODD JUSTA │   TIPO   │
        ╞═══════════════╪══════════╪═════════╪═══════════╪══════════╡
        │ JOGO COMPLETO │ Over 9.5 │  78.2%  │   @1.28   │ ▲ OVER   │
        │ ...           │ ...      │  ...    │   ...     │ ...      │
        ╘═══════════════╧══════════╧═════════╧═══════════╧══════════╛
        
        🎯 SUGESTÕES DA IA:
        [EASY] Over 8.5 (@1.35) | Prob: 82%
        [MEDIUM] Over 10.5 (@1.85) | Prob: 62%
        [HARD] Over 12.5 (@2.45) | Prob: 38%
    
    Raises:
        Exception: Erro capturado e exibido no console.
                   Scraper é encerrado graciosamente.
    
    Example:
        >>> analyze_match_url()
        Cole a URL do jogo do SofaScore: https://www.sofascore.com/game/id:12345678
        Analisando jogo ID: 12345678...
        Jogo: Flamengo vs Palmeiras
        Coletando histórico recente...
        Média Escanteios (Últimos 5): Casa 5.2 | Fora 4.8
        
        🤖 Previsão da IA (Random Forest): 10.82 Escanteios
        ...
        ✅ Previsões salvas no banco de dados.
    
    Warning:
        Execute train_model() ANTES para garantir previsões ML disponíveis.
        Sem modelo treinado, apenas análise estatística será exibida.
    
    See Also:
        - :class:`StatisticalAnalyzer`: Análise Monte Carlo
        - :meth:`StatisticalAnalyzer.analyze_match`: Executa simulações
        - :meth:`StatisticalAnalyzer.generate_suggestions`: Gera sugestões
    """
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
        
        # Get Match Details
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
        
        # Save Match Info to DB (for retrieval in Option 4)
        db = DBManager()
        match_data = {
            'id': match_id,
            'tournament': ev.get('tournament', {}).get('name', 'Unknown'),
            'season_id': ev.get('season', {}).get('id', 0),
            'round': ev.get('roundInfo', {}).get('round', 0),
            'status': 'finished', # Assuming finished for analysis context or update later
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
        
        # Get Last Games for Home and Away
        print("Coletando histórico recente...")
        
        db = DBManager()
        df = db.get_historical_data()
        db.close()
        
        if df.empty:
            print("Banco de dados vazio. Treine o modelo primeiro para melhores resultados.")
            return

        # Filter for Home Team
        home_games = df[(df['home_team_id'] == home_id) | (df['away_team_id'] == home_id)].tail(5)
        away_games = df[(df['home_team_id'] == away_id) | (df['away_team_id'] == away_id)].tail(5)
        
        if len(home_games) < 3 or len(away_games) < 3:
            print("Dados insuficientes no histórico para análise precisa.")
        
        # Calculate averages for ML
        def get_avg_corners(games, team_id):
            corners = []
            for _, row in games.iterrows():
                if row['home_team_id'] == team_id:
                    corners.append(row['corners_home_ft'])
                else:
                    corners.append(row['corners_away_ft'])
            return sum(corners) / len(corners) if corners else 0

        h_avg_corners = get_avg_corners(home_games, home_id)
        a_avg_corners = get_avg_corners(away_games, away_id)
        
        print(f"Média Escanteios (Últimos 5): Casa {h_avg_corners:.1f} | Fora {a_avg_corners:.1f}")
        
        # Clear old predictions for this match to avoid duplicates
        db = DBManager()
        db.delete_predictions(match_id)
        db.close()
        
        # ML Prediction
        # ML Prediction
        # Tenta usar modelo melhorado primeiro
        ml_prediction = 0
        
        try:
            predictor_v2 = ImprovedCornerPredictor()
            if predictor_v2.load_model():
                # Calculate detailed stats for ML (Same logic as server.py)
                def get_team_stats(games, team_id):
                    stats = {
                        'corners': [], 'shots': [], 'goals': [], 'corners_ht': []
                    }
                    
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
                    
                    # Helper to calc avg
                    def avg(lst): return sum(lst) / len(lst) if lst else 0
                    
                    # Avg last 5
                    avg_5 = {k: avg(v) for k, v in stats.items()}
                    
                    # Avg last 3 (for trend)
                    avg_3 = {k: avg(v[-3:]) for k, v in stats.items()}
                    
                    return avg_5, avg_3

                h_stats_5, h_stats_3 = get_team_stats(home_games, home_id)
                a_stats_5, a_stats_3 = get_team_stats(away_games, away_id)
                
                features = [
                    h_stats_5['corners'], h_stats_5['shots'], h_stats_5['goals'],
                    a_stats_5['corners'], a_stats_5['shots'], a_stats_5['goals'],
                    h_stats_5['corners_ht'], a_stats_5['corners_ht'],
                    h_stats_5['corners'] + a_stats_5['corners'], # total_expected
                    h_stats_5['corners'] - a_stats_5['corners'], # diff
                    h_stats_3['corners'] - h_stats_5['corners'], # home_trend
                    a_stats_3['corners'] - a_stats_5['corners']  # away_trend
                ]
                
                X_new = [features]
                pred = predictor_v2.predict(X_new)
                ml_prediction = pred[0]
                print(f"\n🤖 Previsão da IA (LightGBM): {ml_prediction:.2f} Escanteios")
                
                # Save ML Prediction
                db = DBManager()
                db.save_prediction(match_id, 'ML', ml_prediction, f"Over {int(ml_prediction)}", 0.0, verbose=True)
                db.close()
        except Exception as e:
            print(f"Erro ao usar modelo V2: {e}")
            pass

        if ml_prediction == 0:
            predictor = CornerPredictor()
            if predictor.load_model():
                # Fallback features
                X_new = [[h_avg_corners, 0, 0, a_avg_corners, 0, 0]] 
                pred = predictor.predict(X_new)
                ml_prediction = pred[0]
                print(f"\n🤖 Previsão da IA (Random Forest): {ml_prediction:.2f} Escanteios")
                
                # Save ML Prediction
                db = DBManager()
                db.save_prediction(match_id, 'ML', ml_prediction, f"Over {int(ml_prediction)}", 0.0, verbose=True)
                db.close()
            
        # Statistical Analysis
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

        # Run Analysis (Pass ML Prediction for alignment)
        top_picks, suggestions = analyzer.analyze_match(df_h_stats, df_a_stats, ml_prediction=ml_prediction, match_name=match_name)
        
        # Save Predictions (Feedback Loop)
        db = DBManager()
        
        # 1. Save Top 7 Opportunities
        for pick in top_picks:
            db.save_prediction(
                match_id, 
                'Statistical', 
                0, 
                pick['Seleção'], 
                pick['Prob'],
                odds=pick['Odd'],
                category='Top7',
                market_group=pick['Mercado']
            )
            
        # 2. Save AI Suggestions
        # suggestions already generated by analyze_match using full list
        for level, pick in suggestions.items():
            if pick:
                db.save_prediction(
                    match_id,
                    'Statistical',
                    0,
                    pick['Seleção'],
                    pick['Prob'],
                    odds=pick['Odd'],
                    category=f"Suggestion_{level}",
                    market_group=pick['Mercado']
                )
        
        print("✅ Previsões salvas no banco de dados.")
        db.close()

    except Exception as e:
        print(f"Erro na análise: {e}")
    finally:
        scraper.stop()


def retrieve_analysis() -> None:
    """
    Recupera e exibe uma análise previamente salva no banco de dados.
    
    Esta função permite consultar análises realizadas anteriormente,
    buscando no banco de dados SQLite todas as previsões associadas
    a um determinado match_id.
    
    Fluxo de Execução:
    ------------------
    1. **Input**: Solicita o ID do jogo (match_id numérico)
    
    2. **Busca Match**: Recupera informações básicas do jogo (times)
    
    3. **Busca ML Prediction**: Recupera previsão do Random Forest
    
    4. **Busca Top 7**: Recupera as 7 melhores oportunidades estatísticas
       ordenadas por probabilidade decrescente
       
    5. **Busca Sugestões**: Recupera sugestões categorizadas (Easy/Medium/Hard)
    
    6. **Exibição**: Formata e exibe resultados em tabela colorida
    
    Estrutura de Dados Recuperados:
    -------------------------------
    ::
    
        Tabela predictions:
        ├── prediction_type = 'ML'
        │   └── predicted_value: valor numérico da previsão
        │
        ├── category = 'Top7'
        │   ├── market_group: grupo do mercado (JOGO COMPLETO, HT, etc.)
        │   ├── market: linha específica (Over 9.5, Under 10.5, etc.)
        │   ├── probability: probabilidade calculada
        │   └── odds: odd justa
        │
        └── category = 'Suggestion_Easy/Medium/Hard'
            └── mesmos campos do Top7
    
    Formato de Saída:
    -----------------
    ::
    
        🤖 Previsão da IA (Random Forest): 10.82 Escanteios
        
        ⚽ Flamengo vs Palmeiras
        🏆 TOP 7 OPORTUNIDADES (RECUPERADO)
        ╒═══════════════╤══════════╤═════════╤═══════════╤══════════╕
        │    MERCADO    │  LINHA   │  PROB.  │ ODD JUSTA │   TIPO   │
        ╞═══════════════╪══════════╪═════════╪═══════════╪══════════╡
        │ JOGO COMPLETO │ Over 9.5 │  78.2%  │   @1.28   │ ▲ OVER   │
        ╘═══════════════╧══════════╧═════════╧═══════════╧══════════╛
        
        🎯 SUGESTÕES DA IA (RECUPERADO):
        [EASY] JOGO COMPLETO - Over 8.5 (@1.35) | Prob: 82.0%
        [MEDIUM] JOGO COMPLETO - Over 10.5 (@1.85) | Prob: 62.0%
        [HARD] JOGO COMPLETO - Over 12.5 (@2.45) | Prob: 38.0%
    
    Regras de Negócio:
    ------------------
    - Match ID deve existir na tabela matches
    - Exibe mensagem apropriada se não houver dados
    - Coloração ANSI aplicada conforme tipo:
      - Verde (▲ OVER): mercados Over
      - Ciano (▼ UNDER): mercados Under
      - Níveis: Verde=Easy, Amarelo=Medium, Vermelho=Hard
    
    Casos de Uso:
    -------------
    - Revisar análise feita anteriormente
    - Comparar previsão com resultado real
    - Acompanhar histórico de análises
    - Verificar status de previsões (PENDING/GREEN/RED)
    
    Dependencies:
        - tabulate: Biblioteca para formatação de tabelas
    
    Example:
        >>> retrieve_analysis()
        Digite o ID do jogo: 12345678
        
        🤖 Previsão da IA (Random Forest): 10.82 Escanteios
        
        ⚽ Flamengo vs Palmeiras
        🏆 TOP 7 OPORTUNIDADES (RECUPERADO)
        ...
    
    Note:
        O match_id pode ser encontrado na URL do SofaScore ou em
        análises anteriores. É um número inteiro único por partida.
        
        Se nenhuma análise for encontrada, verifique se o jogo foi
        analisado anteriormente via opção 3 do menu.
    
    See Also:
        - :meth:`analyze_match_url`: Função que gera as análises
        - :class:`DBManager`: Gerenciador de banco de dados
    """
    match_id = input("Digite o ID do jogo: ")
    db = DBManager()
    conn = db.connect()
    
    # Get Match Details
    match_query = "SELECT home_team_name, away_team_name FROM matches WHERE match_id = ?"
    match_info = pd.read_sql_query(match_query, conn, params=(match_id,))
    
    match_name = None
    if not match_info.empty:
        match_name = f"{match_info.iloc[0]['home_team_name']} vs {match_info.iloc[0]['away_team_name']}"
    
    # Fetch ML Prediction
    query_ml = "SELECT predicted_value FROM predictions WHERE match_id = ? AND prediction_type = 'ML'"
    ml_pred = pd.read_sql_query(query_ml, conn, params=(match_id,))
    
    if not ml_pred.empty:
        print(f"\n🤖 Previsão da IA (Random Forest): {ml_pred.iloc[0]['predicted_value']:.2f} Escanteios")

    # Fetch Top 7
    query_top7 = "SELECT market_group, market, probability, odds, status FROM predictions WHERE match_id = ? AND category = 'Top7' ORDER BY probability DESC"
    top7 = pd.read_sql_query(query_top7, conn, params=(match_id,))
    
    if not top7.empty:
        if match_name:
             print(f"\n⚽ {Colors.BOLD}{match_name}{Colors.RESET}")
        print(f"🏆 {Colors.BOLD}TOP 7 OPORTUNIDADES (RECUPERADO){Colors.RESET}")
        tabela_display = []
        for _, row in top7.iterrows():
            prob = row['probability']
            # Reconstruct Type based on market string (simple heuristic)
            tipo = "OVER" if "Over" in row['market'] else "UNDER"
            cor = Colors.GREEN if tipo == "OVER" else Colors.CYAN
            seta = "▲" if tipo == "OVER" else "▼"
            
            # Use market_group if available, else 'RECUPERADO'
            m_group = row['market_group'] if row['market_group'] else "RECUPERADO"
            
            # Status formatting
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

    # Fetch Suggestions
    query_sugg = "SELECT category, market_group, market, probability, odds, status FROM predictions WHERE match_id = ? AND category LIKE 'Suggestion_%'"
    suggs = pd.read_sql_query(query_sugg, conn, params=(match_id,))
    
    if not suggs.empty:
        print(f"\n🎯 {Colors.BOLD}SUGESTÕES DA IA (RECUPERADO):{Colors.RESET}")
        for _, row in suggs.iterrows():
            level = row['category'].split('_')[1]
            cor_nivel = Colors.GREEN if level == "Easy" else (Colors.YELLOW if level == "Medium" else Colors.RED)
            m_group = row['market_group'] if row['market_group'] else ""
            
            # Status formatting
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


def main() -> None:
    """
    Função principal que exibe o menu interativo e coordena o sistema.
    
    Esta função implementa um loop infinito que apresenta um menu de
    opções ao usuário e direciona para a função apropriada baseado
    na escolha. O loop continua até que o usuário selecione "Sair".
    
    Menu de Opções:
    ---------------
    ::
    
        --- SISTEMA DE PREVISÃO DE ESCANTEIOS (ML) ---
        1. Atualizar Banco de Dados (Brasileirão 2025 - Padrão)
        2. Atualizar Liga Específica
        3. Atualizar Histórico Completo (3 Anos - Todas Ligas)
        4. Treinar Modelo de IA
        5. Analisar Jogo (URL)
        6. Consultar Análise (ID)
        7. Sair
    
    Mapeamento de Opções:
    ---------------------
    - **Opção 1** → :func:`update_database`: 
      Web scraping do Brasileirão via SofaScore
      
    - **Opção 2** → :func:`train_model`: 
      Treinamento do Random Forest com dados históricos
      
    - **Opção 3** → :func:`analyze_match_url`: 
      Análise completa (ML + Monte Carlo) de um jogo específico
      
    - **Opção 4** → :func:`retrieve_analysis`: 
      Consulta de análises salvas no banco
      
    - **Opção 5** → Encerra o programa
    
    Fluxo Recomendado para Primeiro Uso:
    ------------------------------------
    ::
    
        1. Opção 1: Atualizar Banco (~10-15 min)
           ↓
        2. Opção 2: Treinar Modelo (~1 min)
           ↓
        3. Opção 3: Analisar Jogos (repetir conforme necessário)
           ↓
        4. Opção 4: Consultar análises anteriores
    
    Regras de Negócio:
    ------------------
    - Input inválido exibe "Opção inválida." e retorna ao menu
    - Não há validação de dependências (ex: treinar sem dados)
    - Cada função trata seus próprios erros internamente
    - O programa só encerra com escolha explícita (opção 5)
    
    Example:
        >>> main()
        
        --- SISTEMA DE PREVISÃO DE ESCANTEIOS (ML) ---
        1. Atualizar Banco de Dados (Scraping Completo)
        2. Treinar Modelo de IA
        3. Analisar Jogo (URL)
        4. Consultar Análise (ID)
        5. Sair
        
        Escolha uma opção: 1
        Verificando resultados de previsões anteriores...
        ...
    
    Warning:
        Para melhores resultados, siga a ordem recomendada:
        1 → 2 → 3 → 4
        
        Executar opção 3 sem dados ou modelo pode resultar em
        análises incompletas ou imprecisas.
    
    See Also:
        - :func:`update_database`: Coleta de dados
    """
def update_match_by_url() -> None:
    """
    Atualiza uma partida específica no banco de dados via URL.
    
    Esta função permite atualizar o status e estatísticas de um jogo específico
    sem precisar rodar o scraper para o campeonato inteiro. É útil para
    obter feedback rápido (GREEN/RED) logo após o término de uma partida.
    
    Fluxo:
    1. Solicita URL.
    2. Baixa dados do jogo (placar, status).
    3. Baixa estatísticas (escanteios, chutes).
    4. Salva no banco.
    5. Executa check_predictions() para validar apostas.
    """
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
        
        # 1. Get Match Details
        api_url = f"https://www.sofascore.com/api/v1/event/{match_id}"
        ev_data = scraper._fetch_api(api_url)
        
        if not ev_data or 'event' not in ev_data:
            print("Erro ao buscar dados do jogo.")
            return
            
        ev = ev_data['event']
        match_name = f"{ev['homeTeam']['name']} vs {ev['awayTeam']['name']}"
        print(f"Jogo: {match_name} (Status: {ev['status']['type']})")
        
        # 2. Save Match Info
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
        
        # 3. Get & Save Stats
        if ev['status']['type'] == 'finished':
            print("Coletando estatísticas finais...")
            stats = scraper.get_match_stats(match_id)
            db.save_stats(match_id, stats)
            print("✅ Estatísticas salvas.")
            
            # 4. Trigger Feedback Loop
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
    """Função principal do CLI."""
    while True:
        print("\n" + "═" * 50)
        print(f"{Colors.BOLD}🤖 SISTEMA DE PREVISÃO DE ESCANTEIOS (ML){Colors.RESET}")
        print("═" * 50)
        print("1. Atualizar Banco de Dados (Liga Completa)")
        print("2. Treinar Modelo de IA")
        print("3. Analisar Jogo (URL)")
        print("4. Consultar Análise (ID)")
        print("5. Atualizar Liga Específica")
        print("6. Atualizar Jogo Específico (URL) [NOVO]")
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
        elif choice == '0':
            print("Saindo...")
            break
        else:
            print("Opção inválida.")

if __name__ == "__main__":
    main()
