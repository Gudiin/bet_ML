"""
Servidor Web com SSE (Server-Sent Events) para Interface de Controle.

Este módulo é o "Consultor" do sistema. Ele cria um site local onde você pode:
1. Ver os logs coloridos em tempo real (como no terminal).
2. Clicar em botões para iniciar análises e treinamentos.
3. Ver os resultados das previsões de forma bonita e organizada.

Conceitos Principais:
---------------------
1. **Flask**:
   Uma biblioteca Python que cria um servidor web. É o que faz o seu navegador
   conseguir conversar com o nosso código Python.

2. **SSE (Server-Sent Events)**:
   Uma tecnologia que permite o servidor "empurrar" dados para o navegador.
   É isso que faz os logs aparecerem na tela sem você precisar recarregar a página.

3. **API REST**:
   Um conjunto de "endereços" (rotas) que o site usa para pedir coisas ao servidor.
   Ex: Quando você clica em "Analisar", o site manda um pedido para '/api/match/analyze'.

Regras de Negócio:
------------------
- O servidor roda localmente (localhost).
- Ele mantém uma fila de mensagens para garantir que você não perca nenhum log.
"""

import sys
import os
import json
import threading
import queue
import time
import re
from datetime import datetime
from functools import wraps
from typing import Generator, Dict, Any, Optional, Callable

# Flask imports
from flask import Flask, render_template, jsonify, request, Response, stream_with_context

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.database.db_manager import DBManager
from src.scrapers.sofascore import SofaScoreScraper
from src.ml.feature_engineering import prepare_training_data
from src.ml.model import CornerPredictor
from src.analysis.statistical import StatisticalAnalyzer
import pandas as pd

# ============================================================================
# Configuração Global
# ============================================================================

app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')

# Fila de logs para SSE
log_queue: queue.Queue = queue.Queue()

# Estado global do sistema
system_state = {
    'is_running': False,
    'current_task': None,
    'progress': 0,
    'last_result': None,
    'config': {
        'headless': True,
        'num_simulations': 10000,
        'history_games': 5,
        'model_type': 'lightgbm',  # random_forest, lightgbm, xgboost
        'use_improved_model': True,
        'confidence_threshold': 0.65
    }
}

# Lock para operações thread-safe
state_lock = threading.Lock()


# ============================================================================
# Logger Customizado para SSE
# ============================================================================

class SSELogger:
    """
    Logger que envia mensagens para a fila SSE.
    
    Intercepta prints do sistema e redireciona para:
    1. Console (comportamento normal)
    2. Fila SSE (para interface web)
    """
    
    def __init__(self, log_queue: queue.Queue):
        self.log_queue = log_queue
        self.original_stdout = sys.stdout
        
    def write(self, message: str) -> None:
        """Escreve mensagem no console e na fila SSE."""
        if message.strip():
            # Envia para console original
            self.original_stdout.write(message)
            
            # Envia para fila SSE
            log_entry = {
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'message': message.strip(),
                'type': self._detect_type(message)
            }
            self.log_queue.put(log_entry)
    
    def _detect_type(self, message: str) -> str:
        """Detecta tipo de log baseado no conteúdo."""
        message_lower = message.lower()
        if 'erro' in message_lower or 'error' in message_lower:
            return 'error'
        elif '✅' in message or 'sucesso' in message_lower:
            return 'success'
        elif '⚠' in message or 'warning' in message_lower:
            return 'warning'
        elif '🤖' in message or '🏆' in message or '🎯' in message:
            return 'highlight'
        else:
            return 'info'
    
    def flush(self) -> None:
        """Flush do buffer."""
        self.original_stdout.flush()


def emit_log(message: str, log_type: str = 'info') -> None:
    """
    Emite log diretamente para a fila SSE.
    
    Args:
        message: Mensagem a ser logada
        log_type: Tipo do log (info, success, error, warning, highlight)
    """
    log_entry = {
        'timestamp': datetime.now().strftime('%H:%M:%S'),
        'message': message,
        'type': log_type
    }
    log_queue.put(log_entry)


def update_progress(progress: int, task: str = None) -> None:
    """
    Atualiza progresso atual da tarefa.
    
    Args:
        progress: Porcentagem de progresso (0-100)
        task: Descrição da tarefa atual
    """
    with state_lock:
        system_state['progress'] = progress
        if task:
            system_state['current_task'] = task
    
    # Emite evento de progresso
    log_queue.put({
        'timestamp': datetime.now().strftime('%H:%M:%S'),
        'message': f'__PROGRESS__{progress}',
        'type': 'progress',
        'task': task
    })


# ============================================================================
# Rotas da API
# ============================================================================

@app.route('/')
def index():
    """Página principal com interface de controle."""
    return render_template('index.html')


@app.route('/api/config', methods=['GET'])
def get_config():
    """Retorna configurações atuais do sistema."""
    with state_lock:
        return jsonify(system_state['config'])


@app.route('/api/config', methods=['POST'])
def update_config():
    """Atualiza configurações do sistema."""
    new_config = request.json
    with state_lock:
        system_state['config'].update(new_config)
    emit_log(f'⚙️ Configurações atualizadas: {json.dumps(new_config)}', 'success')
    return jsonify({'status': 'ok', 'config': system_state['config']})


@app.route('/api/leagues')
def get_leagues():
    """Retorna lista de ligas disponíveis."""
    try:
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'clubes_sofascore.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return jsonify(data.get('competicoes', []))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/status')
def get_status():
    """Retorna status atual do sistema."""
    with state_lock:
        return jsonify({
            'is_running': system_state['is_running'],
            'current_task': system_state['current_task'],
            'progress': system_state['progress']
        })


@app.route('/api/logs/stream')
def stream_logs():
    """
    Endpoint SSE para streaming de logs em tempo real.
    
    Returns:
        Response: Stream de eventos SSE
    """
    def generate() -> Generator[str, None, None]:
        """Gerador de eventos SSE."""
        while True:
            try:
                # Aguarda nova mensagem (timeout 1s para manter conexão viva)
                log_entry = log_queue.get(timeout=1)
                yield f"data: {json.dumps(log_entry)}\n\n"
            except queue.Empty:
                # Envia heartbeat para manter conexão
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )


def get_current_season(league_name: str) -> str:
    """
    Determina a temporada atual baseada na liga.
    """
    european_leagues = [
        'Premier League', 'La Liga', 'Bundesliga', 
        'Serie A', 'Ligue 1', 'Serie A 25/26'
    ]
    
    if league_name in european_leagues:
        return "25/26"
    return "2025"

@app.route('/api/database/update', methods=['POST'])
def api_update_database():
    """Inicia atualização do banco de dados em background."""
    data = request.json or {}
    league_name = data.get('league_name')
    # Ignora o season_year enviado pelo front e calcula automaticamente
    season_year = get_current_season(league_name)
    
    if not league_name:
        return jsonify({'error': 'Nome da liga obrigatório'}), 400
    
    global current_task
    with state_lock:
        if system_state['is_running']: # Check system_state['is_running'] for any task
            return jsonify({'error': 'Já existe uma tarefa em execução'}), 400
        
        system_state['is_running'] = True
        system_state['current_task'] = f'Atualizando {league_name} ({season_year})'
        system_state['progress'] = 0
    
    def run_update():
        try:
            _update_database_task(league_name, season_year)
        finally:
            with state_lock:
                system_state['is_running'] = False
                system_state['current_task'] = None
                system_state['progress'] = 100
    
    thread = threading.Thread(target=run_update)
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'started'})


@app.route('/api/database/update_match', methods=['POST'])
def api_update_match():
    """Inicia atualização de jogo único em background."""
    data = request.json or {}
    url = data.get('url')
    
    if not url:
        return jsonify({'error': 'URL ou ID obrigatório'}), 400
        
    # Extract ID from URL if needed
    if "sofascore.com" in url:
        try:
            match_id = url.split("id:")[-1]
        except:
            return jsonify({'error': 'URL inválida. Use o formato do SofaScore.'}), 400
    else:
        match_id = url
        
    global current_task
    with state_lock:
        if system_state['is_running']:
            return jsonify({'error': 'Já existe uma tarefa em execução'}), 400
            
        system_state['is_running'] = True
        system_state['current_task'] = f'Atualizando jogo {match_id}'
        system_state['progress'] = 0
        
    def run_update_match():
        try:
            _update_single_match_task(match_id)
        finally:
            with state_lock:
                system_state['is_running'] = False
                system_state['current_task'] = None
                system_state['progress'] = 100
                
    thread = threading.Thread(target=run_update_match)
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'started', 'match_id': match_id})


@app.route('/api/model/train', methods=['POST'])
def api_train_model():
    """Inicia treinamento do modelo em background."""
    data = request.json or {}
    mode = data.get('mode', 'standard') # standard or optimized
    
    with state_lock:
        if system_state['is_running']:
            return jsonify({'error': 'Já existe uma tarefa em execução'}), 400
        system_state['is_running'] = True
        system_state['current_task'] = f'Treinando modelo ({mode})'
        system_state['progress'] = 0
    
    def run_train():
        try:
            _train_model_task(mode)
        finally:
            with state_lock:
                system_state['is_running'] = False
                system_state['current_task'] = None
                system_state['progress'] = 100
    
    thread = threading.Thread(target=run_train)
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'started'})


@app.route('/api/match/analyze', methods=['POST'])
def api_analyze_match():
    """Analisa uma partida específica."""
    data = request.json
    url = data.get('url', '')
    
    match_id_search = re.search(r'id:(\d+)', url)
    if not match_id_search:
        # Tenta extrair ID diretamente se for número
        if url.isdigit():
            match_id = url
        else:
            return jsonify({'error': 'ID do jogo não encontrado na URL'}), 400
    else:
        match_id = match_id_search.group(1)
    
    with state_lock:
        if system_state['is_running']:
            return jsonify({'error': 'Já existe uma tarefa em execução'}), 400
        system_state['is_running'] = True
        system_state['current_task'] = f'Analisando jogo {match_id}'
        system_state['progress'] = 0
    
    def run_analysis():
        try:
            result = _analyze_match_task(match_id)
            with state_lock:
                system_state['last_result'] = result
        finally:
            with state_lock:
                system_state['is_running'] = False
                system_state['current_task'] = None
                system_state['progress'] = 100
    
    thread = threading.Thread(target=run_analysis)
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'started', 'match_id': match_id})


@app.route('/api/match/result/<match_id>')
def get_match_result(match_id: str):
    """Retorna resultado da análise de uma partida."""
    db = DBManager()
    conn = db.connect()
    
    # Get Match Details
    match_query = "SELECT * FROM matches WHERE match_id = ?"
    match_info = pd.read_sql_query(match_query, conn, params=(match_id,))
    
    if match_info.empty:
        db.close()
        return jsonify({'error': 'Partida não encontrada'}), 404
    
    # Fetch ML Prediction
    query_ml = "SELECT predicted_value FROM predictions WHERE match_id = ? AND prediction_type = 'ML'"
    ml_pred = pd.read_sql_query(query_ml, conn, params=(match_id,))
    
    # Fetch Top 7
    query_top7 = """
        SELECT market_group, market, probability, odds, status 
        FROM predictions 
        WHERE match_id = ? AND category = 'Top7' 
        ORDER BY probability DESC
    """
    top7 = pd.read_sql_query(query_top7, conn, params=(match_id,))
    
    # Fetch Suggestions
    query_sugg = """
        SELECT category, market_group, market, probability, odds, status 
        FROM predictions 
        WHERE match_id = ? AND category LIKE 'Suggestion_%'
    """
    suggestions = pd.read_sql_query(query_sugg, conn, params=(match_id,))
    
    # Fetch Match Stats
    query_stats = "SELECT * FROM match_stats WHERE match_id = ?"
    stats = pd.read_sql_query(query_stats, conn, params=(match_id,))
    
    db.close()
    
    result = {
        'match': match_info.iloc[0].to_dict(),
        'ml_prediction': ml_pred.iloc[0]['predicted_value'] if not ml_pred.empty else None,
        'top7': top7.to_dict('records'),
        'suggestions': suggestions.to_dict('records'),
        'stats': stats.iloc[0].to_dict() if not stats.empty else None
    }
    
    return jsonify(result)


@app.route('/api/analyses')
def get_analyses():
    """Lista todas as análises salvas."""
    db = DBManager()
    conn = db.connect()
    
    query = """
        SELECT DISTINCT 
            m.match_id,
            m.home_team_name,
            m.away_team_name,
            m.home_score,
            m.away_score,
            m.status,
            m.start_timestamp,
            (SELECT predicted_value FROM predictions WHERE match_id = m.match_id AND prediction_type = 'ML' LIMIT 1) as ml_prediction,
            (SELECT COUNT(*) FROM predictions WHERE match_id = m.match_id AND category = 'Top7') as num_predictions
        FROM matches m
        WHERE EXISTS (SELECT 1 FROM predictions p WHERE p.match_id = m.match_id)
        ORDER BY m.start_timestamp DESC
        LIMIT 50
    """
    
    analyses = pd.read_sql_query(query, conn)
    db.close()
    
    # Convert int64 to int for JSON serialization
    records = analyses.to_dict('records')
    for record in records:
        for key, value in record.items():
            if hasattr(value, 'item'):  # numpy int64/float64
                record[key] = value.item()
    
    return jsonify(records)


@app.route('/api/stats')
def get_stats():
    """Retorna estatísticas do sistema."""
    db = DBManager()
    conn = db.connect()
    
    stats = {}
    
    # Total de jogos
    total_matches = pd.read_sql_query(
        "SELECT COUNT(*) as count FROM matches", conn
    ).iloc[0]['count']
    stats['total_matches'] = int(total_matches)
    
    # Total de previsões
    total_predictions = pd.read_sql_query(
        "SELECT COUNT(*) as count FROM predictions", conn
    ).iloc[0]['count']
    stats['total_predictions'] = int(total_predictions)
    
    # Previsões por status
    status_query = """
        SELECT status, COUNT(*) as count 
        FROM predictions 
        WHERE category = 'Top7'
        GROUP BY status
    """
    status_df = pd.read_sql_query(status_query, conn)
    stats['predictions_by_status'] = [
        {'status': row['status'], 'count': int(row['count'])} 
        for _, row in status_df.iterrows()
    ]
    
    # Taxa de acerto
    green_count = pd.read_sql_query(
        "SELECT COUNT(*) as count FROM predictions WHERE status = 'GREEN'", conn
    ).iloc[0]['count']
    
    total_resolved = pd.read_sql_query(
        "SELECT COUNT(*) as count FROM predictions WHERE status IN ('GREEN', 'RED')", conn
    ).iloc[0]['count']
    
    stats['accuracy'] = float(green_count / total_resolved * 100) if total_resolved > 0 else 0
    
    db.close()
    
    return jsonify(stats)


# ============================================================================
# Tarefas em Background (Adaptadas do main.py)
# ============================================================================

def _update_database_task(league_name: str, season_year: str) -> None:
    """Tarefa de atualização do banco de dados."""
    emit_log(f'🔄 Iniciando atualização: {league_name} ({season_year})...', 'info')
    update_progress(5, 'Inicializando...')
    
    db = DBManager()
    
    emit_log('📊 Verificando resultados de previsões anteriores...', 'info')
    db.check_predictions()
    update_progress(10, 'Feedback loop concluído')
    
    with state_lock:
        headless = system_state['config']['headless']
    
    scraper = SofaScoreScraper(headless=headless)
    
    try:
        scraper.start()
        update_progress(15, 'Navegador iniciado')
        
        # Get Tournament/Season IDs
        emit_log(f'🔍 Buscando torneio {league_name}...', 'info')
        t_id = scraper.get_tournament_id(league_name)
        if not t_id:
            emit_log('❌ Torneio não encontrado.', 'error')
            return
        
        s_id = scraper.get_season_id(t_id, season_year)
        if not s_id:
            emit_log('❌ Temporada não encontrada.', 'error')
            return
        
        emit_log(f'✅ ID Torneio: {t_id}, ID Temporada: {s_id}', 'success')
        update_progress(25, 'Torneio identificado')
        
        # Get Matches
        emit_log('📋 Obtendo lista de jogos...', 'info')
        matches = scraper.get_matches(t_id, s_id)
        emit_log(f'✅ Encontrados {len(matches)} jogos.', 'success')
        update_progress(30, f'Processando {len(matches)} jogos')
        
        # Process Matches
        finished_matches = [m for m in matches if m['status']['type'] == 'finished']
        total = len(finished_matches)
        
        for i, m in enumerate(finished_matches):
            progress = 30 + int((i / total) * 65)
            match_name = f"{m['homeTeam']['name']} vs {m['awayTeam']['name']}"
            
            emit_log(f'[{i+1}/{total}] Processando {match_name}...', 'info')
            update_progress(progress, f'Processando {i+1}/{total}')
            
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
        
        emit_log(f'✅ Banco de dados atualizado com {total} jogos!', 'success')
        update_progress(100, 'Concluído')
        
    except Exception as e:
        emit_log(f'❌ Erro: {str(e)}', 'error')
    finally:
        scraper.stop()
        db.close()


def _update_single_match_task(match_id: str) -> None:
    """Tarefa de atualização de jogo único."""
    emit_log(f'🔄 Atualizando jogo ID: {match_id}...', 'info')
    update_progress(10, 'Iniciando scraper...')
    
    with state_lock:
        headless = system_state['config']['headless']
        
    scraper = SofaScoreScraper(headless=headless)
    db = DBManager()
    
    try:
        scraper.start()
        update_progress(30, 'Buscando dados...')
        
        # 1. Get Match Details
        api_url = f"https://www.sofascore.com/api/v1/event/{match_id}"
        ev_data = scraper._fetch_api(api_url)
        
        if not ev_data or 'event' not in ev_data:
            emit_log('❌ Erro ao buscar dados do jogo.', 'error')
            return
            
        ev = ev_data['event']
        match_name = f"{ev['homeTeam']['name']} vs {ev['awayTeam']['name']}"
        status = ev.get('status', {}).get('type', 'unknown')
        
        emit_log(f'⚽ Jogo: {match_name} (Status: {status})', 'highlight')
        update_progress(50, 'Salvando dados...')
        
        # 2. Save Match Info
        match_data = {
            'id': match_id,
            'tournament': ev.get('tournament', {}).get('name', 'Unknown'),
            'season_id': ev.get('season', {}).get('id', 0),
            'round': ev.get('roundInfo', {}).get('round', 0),
            'status': status,
            'timestamp': ev.get('startTimestamp', 0),
            'home_id': ev['homeTeam']['id'],
            'home_name': ev['homeTeam']['name'],
            'away_id': ev['awayTeam']['id'],
            'away_name': ev['awayTeam']['name'],
            'home_score': ev.get('homeScore', {}).get('display', 0),
            'away_score': ev.get('awayScore', {}).get('display', 0)
        }
        db.save_match(match_data)
        
        # 3. Get & Save Stats (if finished)
        if status == 'finished':
            emit_log('📊 Coletando estatísticas finais...', 'info')
            stats = scraper.get_match_stats(match_id)
            db.save_stats(match_id, stats)
            
            emit_log('✅ Dados salvos. Verificando apostas...', 'success')
            update_progress(80, 'Validando apostas...')
            
            # 4. Trigger Feedback Loop
            db.check_predictions()
            emit_log('🔄 Feedback loop concluído!', 'success')
        else:
            emit_log('⚠️ Jogo não finalizado. Apenas dados básicos atualizados.', 'warning')
            
        update_progress(100, 'Concluído')
        
    except Exception as e:
        emit_log(f'❌ Erro: {str(e)}', 'error')
    finally:
        scraper.stop()
        db.close()


def _train_model_task(mode: str = 'standard') -> None:
    """Tarefa de treinamento do modelo."""
    emit_log(f'🤖 Iniciando treinamento do modelo (Modo: {mode})...', 'info')
    update_progress(10, 'Carregando dados...')
    
    db = DBManager()
    df = db.get_historical_data()
    db.close()
    
    if df.empty:
        emit_log('❌ Banco de dados vazio. Execute a atualização primeiro.', 'error')
        return
    
    emit_log(f'📊 Carregados {len(df)} registros para treino.', 'info')
    update_progress(30, 'Preparando features...')
    
    X, y, _ = prepare_training_data(df)
    
    emit_log(f'🔧 Features preparadas: {X.shape[1]} colunas, {len(y)} amostras', 'info')
    update_progress(50, 'Treinando modelo...')
    
    with state_lock:
        use_improved = system_state['config']['use_improved_model']
    
    if use_improved:
        try:
            from src.ml.model_improved import ImprovedCornerPredictor, prepare_improved_features
            emit_log('🚀 Usando modelo melhorado (LightGBM)...', 'highlight')
            
            # Prepara features melhoradas
            X_improved, y_improved, _ = prepare_improved_features(df)
            
            predictor = ImprovedCornerPredictor()
            
            if mode == 'optimized':
                emit_log('🚀 Executando Otimização (CV + GridSearch)...', 'highlight')
                best_params, best_score = predictor.train_with_optimization(X_improved, y_improved)
                emit_log(f'✅ Otimização concluída! Score: {best_score:.4f}', 'success')
            else:
                predictor.train(X_improved, y_improved)
            
            emit_log('✅ Modelo LightGBM treinado e salvo!', 'success')
        except ImportError:
            emit_log('⚠️ Modelo melhorado não disponível, usando Random Forest...', 'warning')
            predictor = CornerPredictor()
            predictor.train(X, y)
    else:
        predictor = CornerPredictor()
        predictor.train(X, y)
        emit_log('✅ Modelo Random Forest treinado e salvo!', 'success')
    
    update_progress(100, 'Treinamento concluído')


def _analyze_match_task(match_id: str) -> Dict[str, Any]:
    """
    Tarefa de análise de partida.
    
    Args:
        match_id: ID da partida no SofaScore
        
    Returns:
        Dict com resultados da análise
    """
    emit_log(f'🔍 Analisando jogo ID: {match_id}...', 'info')
    update_progress(10, 'Conectando ao SofaScore...')
    
    with state_lock:
        headless = system_state['config']['headless']
        use_improved = system_state['config']['use_improved_model']
    
    scraper = SofaScoreScraper(headless=headless)
    result = {}
    
    try:
        scraper.start()
        update_progress(20, 'Buscando dados da partida...')
        
        # Get Match Details
        api_url = f"https://www.sofascore.com/api/v1/event/{match_id}"
        ev_data = scraper._fetch_api(api_url)
        
        if not ev_data or 'event' not in ev_data:
            emit_log('❌ Erro ao buscar dados do jogo.', 'error')
            return {'error': 'Dados não encontrados'}
        
        ev = ev_data['event']
        home_id = ev['homeTeam']['id']
        away_id = ev['awayTeam']['id']
        match_name = f"{ev['homeTeam']['name']} vs {ev['awayTeam']['name']}"
        
        emit_log(f'⚽ Jogo: {match_name}', 'highlight')
        update_progress(30, 'Salvando informações...')
        
        # Save Match Info to DB
        db = DBManager()
        match_data = {
            'id': match_id,
            'tournament': ev.get('tournament', {}).get('name', 'Unknown'),
            'season_id': ev.get('season', {}).get('id', 0),
            'round': ev.get('roundInfo', {}).get('round', 0),
            'status': ev.get('status', {}).get('type', 'unknown'),
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
        
        result['match'] = match_data
        
        # Get Historical Data
        emit_log('📊 Coletando histórico recente...', 'info')
        update_progress(40, 'Analisando histórico...')
        
        db = DBManager()
        df = db.get_historical_data()
        db.close()
        
        if df.empty:
            emit_log('⚠️ Banco de dados vazio. Treine o modelo primeiro.', 'warning')
            return {'error': 'Banco vazio'}
        
        # Filter for Home/Away Team
        home_games = df[(df['home_team_id'] == home_id) | (df['away_team_id'] == home_id)].tail(5)
        away_games = df[(df['home_team_id'] == away_id) | (df['away_team_id'] == away_id)].tail(5)
        
        if len(home_games) < 3 or len(away_games) < 3:
            emit_log('⚠️ Dados insuficientes no histórico.', 'warning')
        
        # Calculate averages
        # Calculate detailed stats for ML
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
        
        h_avg_corners = h_stats_5['corners']
        a_avg_corners = a_stats_5['corners']
        
        emit_log(f'📈 Média Escanteios (Últimos 5): Casa {h_avg_corners:.1f} | Fora {a_avg_corners:.1f}', 'info')
        update_progress(60, 'Gerando previsão ML...')
        
        # Clear old predictions
        db = DBManager()
        db.delete_predictions(match_id)
        db.close()
        
        # ML Prediction
        ml_prediction = 0
        
        if use_improved:
            try:
                from src.ml.model_improved import ImprovedCornerPredictor
                predictor = ImprovedCornerPredictor()
                if predictor.load_model():
                    # Prepare features exactly as in model_improved.py
                    # ['home_avg_corners', 'home_avg_shots', 'home_avg_goals',
                    #  'away_avg_corners', 'away_avg_shots', 'away_avg_goals',
                    #  'home_avg_corners_ht', 'away_avg_corners_ht',
                    #  'total_corners_expected', 'corner_diff',
                    #  'home_trend', 'away_trend']
                    
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
                    pred = predictor.predict(X_new)
                    ml_prediction = pred[0]
                    emit_log(f'🤖 Previsão LightGBM: {ml_prediction:.2f} Escanteios', 'highlight')
            except Exception as e:
                emit_log(f'⚠️ Erro no modelo melhorado: {str(e)}', 'warning')
        
        if ml_prediction == 0:
            predictor = CornerPredictor()
            if predictor.load_model():
                # Fallback model uses fewer features
                # [home_avg_corners, home_avg_shots, home_avg_goals,
                #  away_avg_corners, away_avg_shots, away_avg_goals]
                X_new = [[
                    h_stats_5['corners'], h_stats_5['shots'], h_stats_5['goals'],
                    a_stats_5['corners'], a_stats_5['shots'], a_stats_5['goals']
                ]]
                pred = predictor.predict(X_new)
                ml_prediction = pred[0]
                emit_log(f'🤖 Previsão Random Forest: {ml_prediction:.2f} Escanteios', 'highlight')
        
        result['ml_prediction'] = ml_prediction
        
        # Save ML Prediction
        db = DBManager()
        db.save_prediction(match_id, 'ML', ml_prediction, f"Over {int(ml_prediction)}", 0.0)
        db.close()
        
        update_progress(70, 'Executando Monte Carlo...')
        
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
        
        emit_log('🎲 Executando 10.000 simulações Monte Carlo...', 'info')
        
        # Run Analysis
        top_picks, suggestions = analyzer.analyze_match(df_h_stats, df_a_stats, ml_prediction=ml_prediction, match_name=match_name)
        
        result['top7'] = top_picks
        
        update_progress(85, 'Salvando previsões...')
        
        # Save Predictions
        db = DBManager()
        
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
        
        # Save Suggestions
        # suggestions already generated by analyze_match using full list
        result['suggestions'] = suggestions
        
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
        
        db.close()
        
        emit_log('✅ Análise completa! Previsões salvas no banco.', 'success')
        update_progress(100, 'Concluído')
        
        return result
        
    except Exception as e:
        emit_log(f'❌ Erro na análise: {str(e)}', 'error')
        return {'error': str(e)}
    finally:
        scraper.stop()


# ============================================================================
# Inicialização
# ============================================================================

def run_server(host: str = '0.0.0.0', port: int = 5000, debug: bool = False) -> None:
    """
    Inicia o servidor web.
    
    Args:
        host: Endereço de bind (0.0.0.0 para todas interfaces)
        port: Porta do servidor
        debug: Modo debug do Flask
    """
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║        SISTEMA DE PREVISÃO DE ESCANTEIOS - WEB INTERFACE         ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  🌐 Servidor iniciado em: http://localhost:{port}                 ║
║                                                                  ║
║  📋 Funcionalidades:                                             ║
║     • Painel de controle visual                                  ║
║     • Logs em tempo real (SSE)                                   ║
║     • Configurações personalizáveis                              ║
║     • Histórico de análises                                      ║
║                                                                  ║
║  ⌨️  O terminal CLI continua funcional em src/main.py            ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Servidor Web do Sistema de Previsão')
    parser.add_argument('--host', default='0.0.0.0', help='Host do servidor')
    parser.add_argument('--port', type=int, default=5000, help='Porta do servidor')
    parser.add_argument('--debug', action='store_true', help='Modo debug')
    
    args = parser.parse_args()
    run_server(host=args.host, port=args.port, debug=args.debug)
