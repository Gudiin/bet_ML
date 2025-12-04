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
from src.ml.features_v2 import prepare_features_for_prediction, create_advanced_features
from src.ml.model_v2 import ProfessionalPredictor
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
        'model_type': 'lightgbm',  # random_forest, lightgbm, xgboost
        'use_improved_model': True,
        'confidence_threshold': 0.65
    },
    'scan_results': []  # Stores results from the latest scan
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
        'Serie A', 'Ligue 1', 'Serie A'
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
    if match_id_search:
        match_id = match_id_search.group(1)
    elif url.isdigit():
        match_id = url
    else:
        # Tenta extrair ID da URL (última parte numérica)
        # Ex: https://www.sofascore.com/.../1234567
        # Ex: https://www.sofascore.com/.../slug/1234567
        parts = url.split('/')
        last_part = parts[-1]
        if last_part.isdigit():
            match_id = last_part
        else:
            # Tenta fragmento #id:12345
            fragment_search = re.search(r'#id:(\d+)', url)
            if fragment_search:
                match_id = fragment_search.group(1)
            else:
                return jsonify({'error': 'ID do jogo não encontrado na URL'}), 400
    
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
    query_ml = "SELECT predicted_value FROM predictions WHERE match_id = ? AND prediction_type IN ('ML', 'ML_V2') ORDER BY id DESC LIMIT 1"
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


@app.route('/api/scanner/start', methods=['POST'])
def api_start_scanner():
    """Inicia o scanner de oportunidades em background."""
    data = request.json or {}
    date_mode = data.get('date_mode', 'today') # today, tomorrow, specific
    specific_date = data.get('specific_date')
    leagues_mode = data.get('leagues_mode', 'all') # all, top7
    
    with state_lock:
        if system_state['is_running']:
            return jsonify({'error': 'Já existe uma tarefa em execução'}), 400
        system_state['is_running'] = True
        system_state['current_task'] = 'Scanner de Oportunidades'
        system_state['progress'] = 0
        system_state['scan_results'] = [] # Limpa resultados anteriores
    
    def run_scanner():
        try:
            _scan_opportunities_task(date_mode, specific_date, leagues_mode)
        finally:
            with state_lock:
                system_state['is_running'] = False
                system_state['current_task'] = None
                system_state['progress'] = 100
    
    thread = threading.Thread(target=run_scanner)
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'started'})


@app.route('/api/scanner/results', methods=['GET'])
def api_get_scanner_results():
    """Retorna os resultados do último scan."""
    with state_lock:
        return jsonify(system_state.get('scan_results', []))


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
            (SELECT predicted_value FROM predictions WHERE match_id = m.match_id AND prediction_type IN ('ML', 'ML_V2') ORDER BY id DESC LIMIT 1) as ml_prediction,
            (SELECT COUNT(*) FROM predictions WHERE match_id = m.match_id AND category = 'Top7') as num_predictions
        FROM matches m
        WHERE EXISTS (SELECT 1 FROM predictions p WHERE p.match_id = m.match_id)
        ORDER BY m.start_timestamp DESC
        LIMIT 50
    """
    
    analyses = pd.read_sql_query(query, conn)
    db.close()
    
    # Convert int64 to int and NaN to None for JSON serialization
    records = analyses.to_dict('records')
    for record in records:
        for key, value in record.items():
            if hasattr(value, 'item'):  # numpy int64/float64
                record[key] = value.item()
            elif pd.isna(value):  # NaN to None
                record[key] = None
    
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
    
    # Calculate MAE and RMSE for ML predictions
    try:
        ml_predictions_query = """
            SELECT 
                p.predicted_value,
                (s.corners_home_ft + s.corners_away_ft) as actual_value
            FROM predictions p
            JOIN matches m ON p.match_id = m.match_id
            JOIN match_stats s ON m.match_id = s.match_id
            WHERE p.prediction_type IN ('ML', 'ML_V2') 
            AND m.status = 'finished'
            AND p.category = 'Professional'
        """
        ml_df = pd.read_sql_query(ml_predictions_query, conn)
        
        if not ml_df.empty and len(ml_df) > 0:
            errors = ml_df['actual_value'] - ml_df['predicted_value']
            mae = float(errors.abs().mean())
            rmse = float((errors ** 2).mean() ** 0.5)
            stats['mae'] = mae
            stats['rmse'] = rmse
        else:
            stats['mae'] = None
            stats['rmse'] = None
    except Exception as e:
        print(f"Erro ao calcular MAE/RMSE: {e}")
        stats['mae'] = None
        stats['rmse'] = None
    
    # Calculate ROI (assuming 1 unit bet per prediction, odds of 1.85)
    try:
        roi_query = """
            SELECT status, COUNT(*) as count
            FROM predictions
            WHERE category = 'Top7' AND status IN ('GREEN', 'RED')
            GROUP BY status
        """
        roi_df = pd.read_sql_query(roi_query, conn)
        
        if not roi_df.empty:
            green = roi_df[roi_df['status'] == 'GREEN']['count'].sum() if 'GREEN' in roi_df['status'].values else 0
            red = roi_df[roi_df['status'] == 'RED']['count'].sum() if 'RED' in roi_df['status'].values else 0
            total_bets = green + red
            
            if total_bets > 0:
                # Assuming average odds of 1.85 for wins
                profit = (green * 0.85) - red  # Win: +0.85 units, Loss: -1 unit
                roi_percentage = (profit / total_bets) * 100
                stats['roi'] = float(roi_percentage)
                stats['roi_units'] = float(profit)
            else:
                stats['roi'] = None
                stats['roi_units'] = None
        else:
            stats['roi'] = None
            stats['roi_units'] = None
    except Exception as e:
        print(f"Erro ao calcular ROI: {e}")
        stats['roi'] = None
        stats['roi_units'] = None
    
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
    """Tarefa de treinamento do modelo (Atualizado para V2)."""
    emit_log(f'🤖 Iniciando treinamento do modelo (Modo: {mode})...', 'info')
    update_progress(10, 'Carregando dados...')
    
    db = DBManager()
    df = db.get_historical_data()
    db.close()
    
    if df.empty:
        emit_log('❌ Banco de dados vazio. Execute a atualização primeiro.', 'error')
        return
    
    emit_log(f'📊 Carregados {len(df)} registros para treino.', 'info')
    update_progress(30, 'Gerando features avançadas (V2)...')
    
    try:
        # Garante colunas corretas
        if 'home_score' in df.columns and 'goals_ft_home' not in df.columns:
            df['goals_ft_home'] = df['home_score']
        if 'away_score' in df.columns and 'goals_ft_away' not in df.columns:
            df['goals_ft_away'] = df['away_score']

        # 1. Prepara features vetorizadas
        X, y, timestamps = create_advanced_features(df, window_short=3, window_long=5)
        
        emit_log(f'🔧 Features V2 geradas: {X.shape[1]} colunas, {len(y)} amostras', 'info')
        update_progress(50, 'Treinando Professional Predictor...')
        
        # 2. Treina Modelo
        predictor = ProfessionalPredictor()
        
        if mode == 'optimized':
            # Simulação de otimização (ou implementação real se houver método)
            emit_log('🚀 Treinando com validação temporal...', 'highlight')
            predictor.train_time_series_split(X, y, timestamps)
        else:
            predictor.train_time_series_split(X, y, timestamps)
            
        emit_log('✅ Modelo Professional V2 treinado e salvo!', 'success')
        
    except Exception as e:
        emit_log(f'❌ Erro no treinamento: {e}', 'error')
        import traceback
        traceback.print_exc()

    update_progress(100, 'Treinamento concluído')


def _process_match_prediction(match_data: Dict[str, Any], predictor: Any, df_history: pd.DataFrame, db: DBManager) -> Dict[str, Any]:
    """
    Lógica central de previsão e persistência (Atualizado para ML V2).
    Compartilhada entre 'Analisar Jogo' e 'Scanner'.
    """
    home_name = match_data['home_name']
    away_name = match_data['away_name']
    home_id = match_data['home_id']
    away_id = match_data['away_id']
    match_id = match_data['id']
    
    # 1. Salvar o jogo ANTES de tentar calcular features.
    try:
        db.save_match(match_data)
    except Exception as e:
        return {'error': f'Erro ao salvar dados básicos: {e}'}

    # 2. Preparar Features (V2)
    try:
        # Garante colunas corretas (compatibilidade)
        if 'home_score' in df_history.columns and 'goals_ft_home' not in df_history.columns:
            df_history['goals_ft_home'] = df_history['home_score']
        if 'away_score' in df_history.columns and 'goals_ft_away' not in df_history.columns:
            df_history['goals_ft_away'] = df_history['away_score']

        features_df = prepare_features_for_prediction(
            df_history=df_history,
            home_team_id=home_id,
            away_team_id=away_id,
            window_short=3,
            window_long=5
        )
    except Exception as e:
        return {'error': f'Erro ao gerar features V2: {e}'}
    
    if features_df is None or features_df.empty:
        return {'error': f'Histórico insuficiente para {home_name} ou {away_name}'}
    
    # Recupera médias para exibição (Simulado das features V2 ou calculado aqui)
    # Como features_v2 retorna um DF processado, vamos calcular médias simples do histórico para exibição
    try:
        home_games = df_history[(df_history['home_team_id'] == home_id) | (df_history['away_team_id'] == home_id)].tail(5)
        away_games = df_history[(df_history['home_team_id'] == away_id) | (df_history['away_team_id'] == away_id)].tail(5)
        
        h_corners = []
        for _, g in home_games.iterrows():
            h_corners.append(g['corners_home_ft'] if g['home_team_id'] == home_id else g['corners_away_ft'])
            
        a_corners = []
        for _, g in away_games.iterrows():
            a_corners.append(g['corners_home_ft'] if g['home_team_id'] == away_id else g['corners_away_ft'])
            
        h_avg = sum(h_corners)/len(h_corners) if h_corners else 0
        a_avg = sum(a_corners)/len(a_corners) if a_corners else 0
    except:
        h_avg, a_avg = 0, 0

    # 3. Previsão ML (Professional V2)
    try:
        pred_array = predictor.predict(features_df)
        ml_prediction = float(pred_array[0])
    except Exception as e:
        return {'error': f'Erro na inferência ML V2: {e}'}
    
    # 4. Confiança e Best Bet logic
    confidence = 0.60 # Base
    if ml_prediction > 10.5 or ml_prediction < 8.5: confidence += 0.15
    if confidence > 0.95: confidence = 0.95
    
    best_bet = 'Over 9.5' if ml_prediction > 10 else 'Under 10.5'
    
    # 5. Salvar Previsão ML
    db.save_prediction(
        match_id=match_id,
        pred_type='ML_V2',
        value=ml_prediction,
        market=best_bet,
        prob=confidence,
        odds=1.85, 
        category='Professional',
        market_group='Corners',
        verbose=False
    )

    # 6. Análise Estatística (Top 7 & Sugestões)
    try:
        analyzer = StatisticalAnalyzer()
        
        # Helper para preparar stats do histórico
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

        # Filtra jogos do histórico para cada time
        home_games = df_history[(df_history['home_team_id'] == home_id) | (df_history['away_team_id'] == home_id)].tail(5)
        away_games = df_history[(df_history['home_team_id'] == away_id) | (df_history['away_team_id'] == away_id)].tail(5)
        
        if not home_games.empty and not away_games.empty:
            df_h_stats = prepare_team_df(home_games, home_id)
            df_a_stats = prepare_team_df(away_games, away_id)

            # Executa análise estatística
            top_picks, suggestions = analyzer.analyze_match(df_h_stats, df_a_stats, ml_prediction=ml_prediction, match_name=f"{home_name} vs {away_name}")
            
            # Salva Top 7
            for pick in top_picks:
                db.save_prediction(match_id, 'Statistical', 0, pick['Seleção'], pick['Prob'], odds=pick['Odd'], category='Top7', market_group=pick['Mercado'])
                
            # Salva Sugestões
            for level, pick in suggestions.items():
                if pick:
                    db.save_prediction(match_id, 'Statistical', 0, pick['Seleção'], pick['Prob'], odds=pick['Odd'], category=f"Suggestion_{level}", market_group=pick['Mercado'])
    except Exception as e:
        # Não falha o processo todo se a estatística falhar, apenas loga
        print(f"Erro na análise estatística: {e}")
    
    return {
        'match_name': f"{home_name} vs {away_name}",
        'ml_prediction': round(ml_prediction, 1),
        'confidence': round(confidence * 100, 1),
        'best_bet': best_bet,
        'home_avg_corners': round(h_avg, 1),
        'away_avg_corners': round(a_avg, 1),
        'match_id': match_id
    }



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
        details = scraper.get_match_details(match_id)
        
        if not details:
            emit_log('❌ Erro ao buscar dados do jogo.', 'error')
            return {'error': 'Dados não encontrados'}
        
        match_name = f"{details['home_name']} vs {details['away_name']}"
        emit_log(f'⚽ Jogo: {match_name}', 'highlight')
        
        # 1. Carregar Modelo
        update_progress(40, 'Carregando modelo ML...')
        predictor = None
        model_loaded = False
        
        if use_improved:
            try:
                predictor = ProfessionalPredictor()
                if predictor.load_model():
                    model_loaded = True
                    emit_log('🤖 Modelo Professional V2 carregado.', 'info')
            except ImportError:
                pass
        
        if not model_loaded:
            # Fallback (tenta carregar mesmo se use_improved for false, ou se falhou)
            predictor = ProfessionalPredictor()
            if predictor.load_model():
                model_loaded = True
        
        if not model_loaded:
            emit_log('⚠️ Modelo Profissional não encontrado. Treine o sistema primeiro.', 'warning')
            return {'error': 'Modelo não treinado'}

        # 2. Buscar Histórico
        update_progress(60, 'Buscando histórico dos times...')
        db = DBManager()
        df_history = db.get_historical_data()
        
        if df_history.empty:
            db.close()
            return {'error': 'Banco de dados vazio'}
            
        # 3. Processar Previsão (Lógica Unificada)
        update_progress(80, 'Calculando previsão...')
        
        try:
            result = _process_match_prediction(details, predictor, df_history, db)
            
            if 'error' in result:
                emit_log(f"⚠️ {result['error']}", 'warning')
            else:
                emit_log(f"✅ Análise Concluída: {result['best_bet']} (Conf: {result['confidence']}%)", 'success')
                emit_log('💾 Jogo salvo no banco. Auto-refresh funcionará automaticamente.', 'success')
                
        finally:
            db.close()
        
        update_progress(100, 'Concluído')
        return result
        
    except Exception as e:
        emit_log(f'❌ Erro: {str(e)}', 'error')
        return {'error': str(e)}
    finally:
        scraper.stop()


def _scan_opportunities_task(date_mode: str, specific_date: str = None, leagues_mode: str = 'all') -> None:
    """
    Tarefa de background para o Scanner de Oportunidades.
    
    Regras de Negócio (Nível 3):
        1. Janela de Tempo: Usa horário de Brasília (UTC-3) para definir "Hoje" e "Amanhã".
        2. Filtro de Lixo: Pula silenciosamente times com menos de 3 jogos no histórico.
        3. API Economy: O scraper agora busca apenas a rodada atual (implementado no sofascore.py).
    """
    from datetime import datetime, timedelta, timezone
    import numpy as np
    import time
    
    # 1. Determina a data (Com Fuso Horário de Brasília - UTC-3)
    # Cria timezone UTC-3
    brt_tz = timezone(timedelta(hours=-3))
    now_brt = datetime.now(brt_tz)
    
    if date_mode == 'tomorrow':
        date_str = (now_brt + timedelta(days=1)).strftime('%Y-%m-%d')
        date_label = f"AMANHÃ ({date_str})"
    elif date_mode == 'specific' and specific_date:
        date_str = specific_date
        date_label = specific_date
    else: # today
        date_str = now_brt.strftime('%Y-%m-%d')
        date_label = f"HOJE ({date_str})"
        
    # 2. Determina ligas
    leagues_filter = None
    if leagues_mode == 'top7' or leagues_mode == 'all':
        # IDs: Brasileirão A (325), Série B (390), Premier (17), La Liga (8), 
        # Bundesliga (31), Serie A (35), Ligue 1 (34), Liga Profesional (23)
        # Nota: 'all' aqui refere-se a todas as ligas SUPORTADAS pelo sistema (Top 8)
        leagues_filter = [325, 390, 17, 8, 31, 35, 34, 23]
        
    emit_log(f'🔍 Iniciando Scanner para {date_label}...', 'info')
    update_progress(5, 'Inicializando scraper...')
    
    with state_lock:
        headless = system_state['config']['headless']
        
    scraper = SofaScoreScraper(headless=headless)
    db = DBManager()
    
    try:
        # Tenta carregar modelo ML
        try:
            predictor = ProfessionalPredictor()
            model_loaded = predictor.load_model()
            if model_loaded:
                emit_log('🤖 Modelo Professional V2 carregado com sucesso.', 'info')
            else:
                emit_log('⚠️ Modelo ML não encontrado. Usando simulação.', 'warning')
        except ImportError:
            emit_log('⚠️ Módulo ML não encontrado. Usando simulação.', 'warning')
            model_loaded = False
            predictor = None

        scraper.start()
        update_progress(10, 'Buscando jogos...')
        
        # Busca jogos usando o novo método otimizado (API Economy)
        matches = scraper.get_scheduled_matches(date_str, leagues_filter)
        
        if not matches:
            emit_log('❌ Nenhum jogo encontrado para esta data.', 'warning')
            return
            
        emit_log(f'📊 Encontrados {len(matches)} jogos.', 'success')
        update_progress(20, f'Analisando {len(matches)} jogos...')
        
        opportunities = []
        total = len(matches)
        
        # Carrega histórico para features
        df_history = db.get_historical_data()
        
        if df_history.empty:
            emit_log('⚠️ Banco de dados vazio! O Scanner precisa de histórico para funcionar.', 'warning')
            emit_log('💡 Dica: Execute "Atualizar Banco de Dados" primeiro.', 'info')
        
        for i, match in enumerate(matches):
            progress = 20 + int((i / total) * 75)
            match_name = f"{match['home_team']} vs {match['away_team']}"
            
            # Log de progresso explícito
            emit_log(f'[{i+1}/{total}] Analisando: {match_name}', 'info')
            update_progress(progress, f'Analisando {i+1}/{total}')
            
            try:
                # Se não tiver modelo ou histórico, pula
                if not model_loaded or df_history.empty:
                    emit_log('   ⚠️ Pulo: Modelo ou histórico indisponível.', 'warning')
                    continue
                
                # 1. Busca detalhes COMPLETOS do jogo (Necessário para análise correta)
                # Isso resolve o problema de "Analisar Jogo" vs "Scanner"
                match_id = match.get('match_id')
                if not match_id:
                    continue
                    
                details = scraper.get_match_details(match_id)
                if not details:
                    emit_log(f'   ⚠️ Erro ao buscar detalhes de {match_name}', 'warning')
                    continue
                
                # 2. Processa usando a lógica unificada
                result = _process_match_prediction(details, predictor, df_history, db)
                
                if 'error' in result:
                    emit_log(f"   ⚠️ {result['error']}", 'warning')
                else:
                    # Filtra apenas oportunidades com alta confiança para a UI
                    if result['confidence'] >= 60:
                        result['start_time'] = match['start_time'] # Mantém horário original
                        result['tournament'] = match['tournament']
                        opportunities.append(result)
                        emit_log(f"   ✅ Oportunidade: {result['best_bet']} (@1.85)", 'success')
                    else:
                        emit_log(f"   ℹ️ Rejeitado: Confiança baixa ({result['confidence']}%)", 'info')
                        
            except Exception as e:
                print(f"Erro ao analisar {match_name}: {e}")
                
        # Ordena e salva resultados para UI
        opportunities.sort(key=lambda x: x['confidence'], reverse=True)
        
        with state_lock:
            system_state['scan_results'] = opportunities
            
        emit_log(f'✅ Scanner finalizado! {len(opportunities)} oportunidades encontradas.', 'success')
        update_progress(100, 'Concluído')
        
    except Exception as e:
        emit_log(f'❌ Erro no scanner: {str(e)}', 'error')
    finally:
        scraper.stop()
        db.close()




def _update_pending_matches_task() -> None:
    """
    Tarefa para atualizar jogos pendentes (ao vivo ou finalizados recentemente).
    """
    emit_log('🔄 Verificando jogos pendentes...', 'info')
    update_progress(10, 'Buscando pendências...')
    
    db = DBManager()
    pending = db.get_pending_matches()
    
    if not pending:
        emit_log('✅ Nenhum jogo pendente para atualizar.', 'success')
        update_progress(100, 'Concluído')
        db.close()
        return
        
    emit_log(f'📋 Encontrados {len(pending)} jogos para atualizar.', 'info')
    
    with state_lock:
        headless = system_state['config']['headless']
        
    scraper = SofaScoreScraper(headless=headless)
    
    try:
        scraper.start()
        update_progress(20, 'Iniciando atualização...')
        
        total = len(pending)
        updated_count = 0
        
        for i, match in enumerate(pending):
            m_id = match['match_id']
            m_name = f"{match['home_team']} vs {match['away_team']}"
            
            emit_log(f'[{i+1}/{total}] Atualizando: {m_name}...', 'info')
            
            # 1. Busca detalhes
            details = scraper.get_match_details(m_id)
            if not details:
                emit_log(f'⚠️ Falha ao buscar dados de {m_name}', 'warning')
                continue
                
            new_status = details['status']
            
            # 2. Atualiza Match no DB
            db.save_match(details)
            
            # 3. Busca estatísticas para jogos finalizados OU em andamento (TEMPO REAL)
            if new_status == 'finished':
                stats = scraper.get_match_stats(m_id)
                db.save_stats(m_id, stats)
                emit_log(f'   ✅ Finalizado! Placar: {details["home_score"]}-{details["away_score"]}', 'success')
                updated_count += 1
            elif new_status == 'inprogress':
                # CORREÇÃO: Busca stats de jogos AO VIVO para exibição em tempo real
                stats = scraper.get_match_stats(m_id)
                db.save_stats(m_id, stats)
                emit_log(f'   ⚽ Ao Vivo: {details["home_score"]}-{details["away_score"]} | Stats atualizadas', 'highlight')
            else:
                emit_log(f'   ⏳ Status: {new_status}', 'info')
                
            time.sleep(1) # Delay para evitar bloqueio
            update_progress(20 + int((i/total)*70), f'Atualizando {i+1}/{total}')
            
        if updated_count > 0:
            emit_log(f'✅ {updated_count} jogos foram finalizados e atualizados.', 'success')
            # Trigger feedback loop
            db.check_predictions()
        else:
            emit_log('✅ Atualização concluída. Nenhum jogo novo finalizado.', 'success')
            
        update_progress(100, 'Concluído')
            
    except Exception as e:
        emit_log(f'❌ Erro na atualização: {str(e)}', 'error')
    finally:
        scraper.stop()
        db.close()

@app.route('/api/matches/update_pending', methods=['POST'])
def api_update_pending():
    """Endpoint para forçar atualização de jogos pendentes."""
    with state_lock:
        if system_state['is_running']:
            return jsonify({'error': 'Já existe uma tarefa em execução'}), 400
        system_state['is_running'] = True
        system_state['current_task'] = 'Atualizando Pendentes'
        system_state['progress'] = 0
    
    def run_task():
        try:
            _update_pending_matches_task()
        finally:
            with state_lock:
                system_state['is_running'] = False
                system_state['current_task'] = None
                system_state['progress'] = 100
    
    thread = threading.Thread(target=run_task)
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'started'})

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
    
    # DESABILITADO: Auto-scheduler estava causando duplicação de dados
    # O scanner agora só roda quando o usuário clicar manualmente
    # _start_background_scheduler()
    
    app.run(host=host, port=port, debug=debug, threaded=True)


def _start_background_scheduler() -> None:
    """
    Inicia scheduler de background para atualização automática de dados.
    
    Regra de Negócio:
        - Atualiza jogos pendentes (agendados e ao vivo) a cada 5 minutos
        - Roda em thread daemon para não bloquear o servidor
        - Só executa se não houver outra tarefa rodando (evita conflitos)
    
    Contexto:
        Criado para resolver o problema de dados desatualizados.
        Antes, o usuário precisava clicar manualmente em "Atualizar Jogo".
        Agora o sistema mantém os dados frescos automaticamente.
    """
    def scheduler_loop():
        """Loop infinito que executa atualização a cada 5 minutos."""
        import time
        
        while True:
            try:
                # Aguarda 60 segundos (1 minuto) para atualização mais rápida
                time.sleep(60)
                
                # Verifica se não há tarefa rodando
                with state_lock:
                    is_busy = system_state['is_running']
                
                if not is_busy:
                    print("⏰ [Auto-Update] Verificando jogos pendentes...")
                    
                    # Executa atualização em thread separada para não bloquear
                    def run_update():
                        try:
                            with state_lock:
                                system_state['is_running'] = True
                                system_state['current_task'] = 'Auto-Update'
                            
                            _update_pending_matches_task()
                        finally:
                            with state_lock:
                                system_state['is_running'] = False
                                system_state['current_task'] = None
                    
                    update_thread = threading.Thread(target=run_update, daemon=True)
                    update_thread.start()
                else:
                    print("⏰ [Auto-Update] Sistema ocupado, pulando ciclo.")
                    
            except Exception as e:
                print(f"❌ [Auto-Update] Erro no scheduler: {e}")
    
    # Inicia thread do scheduler
    scheduler_thread = threading.Thread(target=scheduler_loop, daemon=True)
    scheduler_thread.start()
    print("✅ Scheduler de atualização automática iniciado (intervalo: 60 segundos)")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Servidor Web do Sistema de Previsão')
    parser.add_argument('--host', default='127.0.0.1', help='Host do servidor')
    parser.add_argument('--port', type=int, default=5000, help='Porta do servidor')
    parser.add_argument('--debug', action='store_true', help='Modo debug')
    
    args = parser.parse_args()
    run_server(host=args.host, port=args.port, debug=args.debug)
