"""
Módulo de Web Scraping para SofaScore.
"""

import time
import random
from playwright.sync_api import sync_playwright

class SofaScoreScraper:
    def __init__(self, headless: bool = True):
        self.headless = headless
        self.playwright = None
        self.browser = None
        self.page = None

    def start(self) -> None:
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=self.headless)
        self.page = self.browser.new_page()
        self.page.set_extra_http_headers({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        })
        self.page.goto("https://www.sofascore.com")

    def stop(self) -> None:
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()

    def _fetch_api(self, url: str) -> dict | None:
        time.sleep(random.uniform(0.5, 1.5))
        script = f"""
            async () => {{
                try {{
                    const r = await fetch('{url}');
                    if (r.status !== 200) return null;
                    return await r.json();
                }} catch {{ return null; }}
            }}
        """
        return self.page.evaluate(script)

    def get_tournament_id(self, query: str = "Brasileirão") -> int | None:
        tournament_mapping = {
            "Brasileirão Série A": "Brasileirão",
            "Brasileirão Série B": "Brasileirão Série B",
            "Serie A 25/26": "Serie A",
            "Serie A (Itália)": "Serie A",
            "La Liga": "LaLiga",
            "Ligue 1": "Ligue 1",
            "Bundesliga": "Bundesliga",
            "Premier League": "Premier League",
            "Liga Profesional (Argentina)": "Liga Profesional"
        }
        search_query = tournament_mapping.get(query, query)
        url = f"https://www.sofascore.com/api/v1/search/{search_query}"
        print(f"Buscando torneio: {query}...")
        
        data = self._fetch_api(url)
        if data and 'results' in data:
            for item in data['results']:
                if item['type'] == 'uniqueTournament':
                    entity = item['entity']
                    print(f"Encontrado: {entity['name']} (ID: {entity['id']})")
                    if search_query.lower() in entity['name'].lower() or entity['name'].lower() in search_query.lower():
                        return entity['id']
        return None

    def get_season_id(self, tournament_id: int, year: str = "2024") -> int | None:
        url = f"https://www.sofascore.com/api/v1/unique-tournament/{tournament_id}/seasons"
        data = self._fetch_api(url)
        if data and 'seasons' in data:
            for s in data['seasons']:
                if s['year'] == year:
                    return s['id']
                # Check for "24/25" format if year is "2025"
                if year == "2025" and s['year'] == "24/25":
                    return s['id']
                if year == "2024" and s['year'] == "23/24":
                    return s['id']
            
            try:
                year_int = int(year)
                prev_year = year_int - 1
                euro_format = f"{str(prev_year)[-2:]}/{str(year_int)[-2:]}"
                for s in data['seasons']:
                    if s['year'] == euro_format:
                        print(f"Temporada encontrada (formato europeu): {euro_format}")
                        return s['id']
            except ValueError:
                pass
        return None

    def get_current_round(self, tournament_id: int, season_id: int) -> int | None:
        """
        Obtém o número da rodada atual de um torneio.
        
        Regra de Negócio:
            - Economia de API: Em vez de baixar todas as rodadas, perguntamos à API
              qual é a rodada atual (currentRound) para baixar apenas ela.
              
        Args:
            tournament_id: ID do torneio.
            season_id: ID da temporada.
            
        Returns:
            int: Número da rodada atual ou None se não encontrar.
        """
        url = f"https://www.sofascore.com/api/v1/unique-tournament/{tournament_id}/season/{season_id}/rounds"
        print(f"Buscando rodada atual (Torneio {tournament_id})...")
        
        data = self._fetch_api(url)
        if data and 'currentRound' in data:
            return data['currentRound']['round']
        return None

    def get_scheduled_matches(self, date_str: str, league_ids: list = None) -> list:
        """
        Busca jogos agendados para uma data específica (Scanner).
        
        Regra de Negócio:
            - Scanner de Oportunidades: Busca jogos de hoje/amanhã para análise.
            - API Economy: Para cada liga, busca a rodada atual E adjacentes (+1, -1).
              Isso resolve o problema de jogos adiados/adiantados que não estão na "currentRound".
            - Se league_ids for None, usa uma lista padrão de ligas principais.
            
        Args:
            date_str: Data no formato 'YYYY-MM-DD'.
            league_ids: Lista de IDs de torneios para escanear.
            
        Returns:
            list: Lista de dicionários com dados simplificados dos jogos.
        """
        matches = []
        
        # Lista padrão se nenhuma for fornecida (Top 7 + Brasileirão)
        if not league_ids:
            league_ids = [325, 390, 17, 8, 31, 35, 34, 23]
            
        print(f"Iniciando Scanner para {date_str} em {len(league_ids)} ligas...")
        
        processed_matches = set() # Evita duplicatas
        
        for t_id in league_ids:
            # 1. Descobrir Season ID (assumindo ano atual/recente)
            s_id = self.get_season_id(t_id, "2025")
            if not s_id:
                s_id = self.get_season_id(t_id, "25/26")
            if not s_id:
                s_id = self.get_season_id(t_id, "2024") # Fallback
            if not s_id:
                s_id = self.get_season_id(t_id, "24/25") # Fallback extra
                
            if not s_id:
                continue
                
            # 2. Descobrir Rodada Atual (API Economy)
            current_round = self.get_current_round(t_id, s_id)
            if not current_round:
                continue
                
            # 3. Baixar Jogos da Rodada Atual e Adjacentes
            # Verifica current, current+1, current-1 para garantir cobertura
            rounds_to_check = {current_round, current_round + 1, current_round - 1}
            # Remove rodadas inválidas (<1)
            rounds_to_check = {r for r in rounds_to_check if r > 0}
            
            print(f"   Liga {t_id}: Verificando rodadas {rounds_to_check}...")
            
            for r in rounds_to_check:
                url = f"https://www.sofascore.com/api/v1/unique-tournament/{t_id}/season/{s_id}/events/round/{r}"
                data = self._fetch_api(url)
                
                if data and 'events' in data:
                    for event in data['events']:
                        # Filtra pela data
                        evt_ts = event.get('startTimestamp')
                        if not evt_ts:
                            continue
                            
                        # Converte timestamp para data string (UTC-3 fixo)
                        import datetime
                        tz_offset = datetime.timezone(datetime.timedelta(hours=-3))
                        evt_date = datetime.datetime.fromtimestamp(evt_ts, tz=tz_offset).strftime('%Y-%m-%d')
                        
                        if evt_date == date_str:
                            # Evita duplicatas (mesmo jogo em rodadas diferentes? Improvável, mas seguro)
                            if event['id'] in processed_matches:
                                continue
                                
                            processed_matches.add(event['id'])
                            
                            # Extrai dados relevantes
                            match_info = {
                                'match_id': event['id'],
                                'tournament': event['tournament']['name'],
                                'home_team': event['homeTeam']['name'],
                                'away_team': event['awayTeam']['name'],
                                'start_time': datetime.datetime.fromtimestamp(evt_ts, tz=tz_offset).strftime('%Y-%m-%d %H:%M'),
                                'status': event['status']['type']
                            }
                            matches.append(match_info)
                        
        print(f"Scanner finalizado. {len(matches)} jogos encontrados para {date_str}.")
        return matches

    def get_matches(self, tournament_id: int, season_id: int, start_round: int = 1) -> list:
        """
        Coleta partidas com suporte a início customizado (start_round).
        """
        matches = []
        round_num = start_round
        max_rounds = 50 
        empty_rounds_limit = 3 
        empty_rounds = 0
        
        print(f"Iniciando coleta de partidas (Torneio {tournament_id}, Season {season_id})")
        print(f"🚀 Começando da Rodada {start_round}...")
        
        while round_num <= max_rounds:
            print(f"Coletando rodada {round_num}...", end='\r')
            url = f"https://www.sofascore.com/api/v1/unique-tournament/{tournament_id}/season/{season_id}/events/round/{round_num}"
            data = self._fetch_api(url)
            
            if data and 'events' in data and len(data['events']) > 0:
                matches.extend(data['events'])
                empty_rounds = 0
            else:
                empty_rounds += 1
                if empty_rounds >= empty_rounds_limit:
                    print(f"\nSem eventos por {empty_rounds_limit} rodadas consecutivas. Parando na rodada {round_num}.")
                    break
            
            round_num += 1
            
        print(f"\nTotal de partidas coletadas nesta execução: {len(matches)}")
        return matches

    def get_match_stats(self, match_id: int) -> dict:
        url = f"https://www.sofascore.com/api/v1/event/{match_id}/statistics"
        data = self._fetch_api(url)
        
        stats = {
            'corners_home_ft': 0, 'corners_away_ft': 0,
            'corners_home_ht': 0, 'corners_away_ht': 0,
            'shots_ot_home_ft': 0, 'shots_ot_away_ft': 0,
            'shots_ot_home_ht': 0, 'shots_ot_away_ht': 0,
            'possession_home': 0, 'possession_away': 0,
            'total_shots_home': 0, 'total_shots_away': 0,
            'fouls_home': 0, 'fouls_away': 0,
            'yellow_cards_home': 0, 'yellow_cards_away': 0,
            'red_cards_home': 0, 'red_cards_away': 0,
            'big_chances_home': 0, 'big_chances_away': 0,
            'expected_goals_home': 0.0, 'expected_goals_away': 0.0
        }

        if not data or 'statistics' not in data:
            return stats

        def extract_val(groups: list, keywords: list, is_home: bool, return_float: bool = False):
            if not groups:
                return 0.0 if return_float else 0
            for g in groups:
                if 'statisticsItems' not in g:
                    continue
                for item in g['statisticsItems']:
                    item_name_lower = item.get('name', '').lower()
                    for keyword in keywords:
                        if keyword.lower() in item_name_lower:
                            try:
                                value = item.get('home' if is_home else 'away')
                                if value is not None:
                                    if isinstance(value, str) and '%' in value:
                                        return int(value.replace('%', ''))
                                    if return_float:
                                        return float(value)
                                    else:
                                        return int(value)
                            except (ValueError, TypeError):
                                return 0.0 if return_float else 0
            return 0.0 if return_float else 0

        all_stats = next((p['groups'] for p in data['statistics'] if p['period'] == 'ALL'), [])
        
        stats['corners_home_ft'] = extract_val(all_stats, ['corner kicks'], True)
        stats['corners_away_ft'] = extract_val(all_stats, ['corner kicks'], False)
        stats['shots_ot_home_ft'] = extract_val(all_stats, ['shots on target'], True)
        stats['shots_ot_away_ft'] = extract_val(all_stats, ['shots on target'], False)
        stats['total_shots_home'] = extract_val(all_stats, ['total shots'], True)
        stats['total_shots_away'] = extract_val(all_stats, ['total shots'], False)
        stats['possession_home'] = extract_val(all_stats, ['ball possession'], True)
        stats['possession_away'] = extract_val(all_stats, ['ball possession'], False)
        stats['fouls_home'] = extract_val(all_stats, ['fouls'], True)
        stats['fouls_away'] = extract_val(all_stats, ['fouls'], False)
        stats['yellow_cards_home'] = extract_val(all_stats, ['yellow cards'], True)
        stats['yellow_cards_away'] = extract_val(all_stats, ['yellow cards'], False)
        stats['red_cards_home'] = extract_val(all_stats, ['red cards'], True)
        stats['red_cards_away'] = extract_val(all_stats, ['red cards'], False)
        stats['big_chances_home'] = extract_val(all_stats, ['big chances'], True)
        stats['big_chances_away'] = extract_val(all_stats, ['big chances'], False)
        stats['expected_goals_home'] = extract_val(all_stats, ['expected goals'], True, return_float=True)
        stats['expected_goals_away'] = extract_val(all_stats, ['expected goals'], False, return_float=True)

        ht_stats = next((p['groups'] for p in data['statistics'] if p['period'] == '1ST'), [])
        stats['corners_home_ht'] = extract_val(ht_stats, ['corner kicks'], True)
        stats['corners_away_ht'] = extract_val(ht_stats, ['corner kicks'], False)
        stats['shots_ot_home_ht'] = extract_val(ht_stats, ['shots on target'], True)
        stats['shots_ot_away_ht'] = extract_val(ht_stats, ['shots on target'], False)

        return stats

    def get_match_details(self, match_id: int) -> dict:
        """Busca detalhes completos de uma partida incluindo minuto do jogo."""
        url = f"https://www.sofascore.com/api/v1/event/{match_id}"
        data = self._fetch_api(url)
        
        if not data or 'event' not in data:
            return None
            
        ev = data['event']
        
        # Extract match minute for live matches
        status_info = ev.get('status', {})
        match_minute = None
        status_description = status_info.get('description', '')
        
        # Get minute from status description (e.g., "45+2", "HT", "78")
        if status_info.get('type') == 'inprogress':
            match_minute = status_description if status_description else None
        
        return {
            'id': ev['id'],
            'tournament': ev.get('tournament', {}).get('name', 'Unknown'),
            'tournament_id': ev.get('tournament', {}).get('uniqueTournament', {}).get('id', 0),
            'season_id': ev.get('season', {}).get('id', 0),
            'round': ev.get('roundInfo', {}).get('round', 0),
            'status': status_info.get('type', 'unknown'),
            'status_description': status_description,
            'match_minute': match_minute,
            'timestamp': ev.get('startTimestamp', 0),
            'home_id': ev['homeTeam']['id'],
            'home_name': ev['homeTeam']['name'],
            'away_id': ev['awayTeam']['id'],
            'away_name': ev['awayTeam']['name'],
            'home_score': ev.get('homeScore', {}).get('display', 0),
            'away_score': ev.get('awayScore', {}).get('display', 0)
        }