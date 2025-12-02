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

    # --- MÉTODO MODIFICADO ---
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
    # --------------------------

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