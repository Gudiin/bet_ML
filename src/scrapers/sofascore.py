"""
Módulo de Web Scraping para SofaScore.

Este módulo é o "Olheiro" do sistema. Ele navega na internet como se fosse um humano
para coletar os dados que alimentam nossa inteligência.

Conceitos Principais:
---------------------
1. **Web Scraping**:
   Técnica de extrair dados de sites. Como o SofaScore não tem um botão "Baixar Excel",
   criamos um robô que entra página por página e copia os números.

2. **Playwright**:
   A ferramenta que controla o navegador. É como um "piloto automático" para o Chrome.

3. **Rate Limiting (Limitação de Velocidade)**:
   Se o robô for muito rápido, o site percebe e bloqueia. Por isso, colocamos "pausas"
   aleatórias (como um humano pensando) entre cada clique.

Regras de Negócio:
------------------
- O robô finge ser um navegador real (User-Agent) para não ser detectado.
- Coletamos dados de torneios, temporadas e estatísticas detalhadas de cada jogo.
"""

import time
import random
from playwright.sync_api import sync_playwright


class SofaScoreScraper:
    """
    Classe responsável por realizar web scraping da API do SofaScore.
    
    Esta classe gerencia a conexão com o navegador automatizado e fornece
    métodos para coletar dados de torneios, temporadas e estatísticas de partidas.
    
    Regras de Negócio:
        - Utiliza Chromium headless por padrão para melhor performance
        - Implementa delays aleatórios entre requisições (0.5-1.5s) para evitar bloqueios
        - Simula User-Agent de navegador real para evitar detecção de bot
    
    Attributes:
        headless (bool): Se True, executa o navegador sem interface gráfica.
        playwright: Instância do Playwright.
        browser: Instância do navegador Chromium.
        page: Página ativa do navegador.
    
    Example:
        >>> scraper = SofaScoreScraper(headless=True)
        >>> scraper.start()
        >>> t_id = scraper.get_tournament_id("Brasileirão")
        >>> scraper.stop()
    """
    
    def __init__(self, headless: bool = True):
        """
        Inicializa o scraper com configurações padrão.
        
        Args:
            headless: Se True, executa navegador sem interface gráfica.
                     Use False para debug visual.
        
        Lógica:
            - Inicializa atributos de conexão como None
            - headless=True é recomendado para produção (mais rápido)
            - headless=False útil para debug e visualizar o comportamento
        """
        self.headless = headless
        self.playwright = None
        self.browser = None
        self.page = None

    def start(self) -> None:
        """
        Inicia o navegador automatizado e configura a sessão.
        
        Este método deve ser chamado antes de qualquer operação de scraping.
        Configura o Playwright, lança o Chromium e prepara a página para requisições.
        
        Lógica:
            1. Inicia o Playwright em modo síncrono
            2. Lança navegador Chromium com configuração headless
            3. Cria nova página e configura headers HTTP
            4. Navega para sofascore.com para inicializar cookies/sessão
        
        Regras de Negócio:
            - User-Agent simula Chrome 119 no Windows 10
            - Navegação inicial estabelece sessão válida com o servidor
        
        Raises:
            PlaywrightError: Se houver falha ao iniciar o navegador.
        """
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=self.headless)
        self.page = self.browser.new_page()
        self.page.set_extra_http_headers({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        })
        # Navega para página inicial para inicializar sessão
        self.page.goto("https://www.sofascore.com")

    def stop(self) -> None:
        """
        Encerra o navegador e libera recursos do Playwright.
        
        Este método deve ser chamado ao final das operações para evitar
        vazamento de memória e processos órfãos.
        
        Lógica:
            1. Fecha o navegador se estiver aberto
            2. Para o Playwright e libera recursos
        
        Regras de Negócio:
            - Sempre chamar este método em bloco finally
            - Seguro para chamar múltiplas vezes
        """
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()

    def _fetch_api(self, url: str) -> dict | None:
        """
        Executa requisição à API do SofaScore através do contexto do navegador.
        
        Utiliza fetch() no contexto da página para herdar cookies e sessão,
        evitando bloqueios por CORS ou detecção de bot.
        
        Args:
            url: URL completa da API do SofaScore.
        
        Returns:
            dict: Dados JSON da resposta se sucesso.
            None: Se a requisição falhar ou retornar status != 200.
        
        Lógica:
            1. Aplica delay aleatório (0.5-1.5s) para rate limiting
            2. Executa JavaScript fetch() no contexto da página
            3. Retorna JSON parseado ou None em caso de erro
        
        Regras de Negócio:
            - Rate limiting previne bloqueio por excesso de requisições
            - Execução via page.evaluate() herda autenticação do navegador
            - Retorna None silenciosamente em caso de erro (fail-safe)
        
        Example:
            >>> data = scraper._fetch_api("https://www.sofascore.com/api/v1/search/Brasileirão")
        """
        time.sleep(random.uniform(0.5, 1.5))  # Rate limiting
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
        """
        Busca o ID único de um torneio pelo nome.
        
        Utiliza a API de busca do SofaScore para encontrar torneios
        que correspondam à query fornecida.
        
        Args:
            query: Nome ou parte do nome do torneio a buscar.
                  Default: "Brasileirão" para o Campeonato Brasileiro.
        
        Returns:
            int: ID único do torneio encontrado.
            None: Se nenhum torneio correspondente for encontrado.
        
        Lógica:
            1. Faz requisição à API de busca com a query
            2. Filtra resultados pelo tipo 'uniqueTournament'
            3. Verifica match case-insensitive entre query e nome do torneio
            4. Retorna o primeiro ID que corresponder
        
        Regras de Negócio:
            - Busca é case-insensitive
            - Retorna primeiro match encontrado
            - IDs de torneios são únicos e permanentes no SofaScore
        
        Example:
            >>> t_id = scraper.get_tournament_id("Brasileirão")  # Retorna 325
            >>> t_id = scraper.get_tournament_id("Premier League")
        """
        # Mapeamento de nomes customizados para nomes oficiais do SofaScore
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
        
        # Usa nome mapeado se existir, senão usa o nome original
        search_query = tournament_mapping.get(query, query)
        
        url = f"https://www.sofascore.com/api/v1/search/{search_query}"
        print(f"Buscando torneio: {query}...")
        if search_query != query:
            print(f"  → Mapeado para: {search_query}")
        
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
        """
        Obtém o ID de uma temporada específica de um torneio.
        
        Tenta encontrar a temporada pelo ano exato ou formatos alternativos
        (ex: "2024" -> "23/24" para ligas europeias).
        
        Args:
            tournament_id: ID único do torneio.
            year: Ano da temporada (ex: "2024").
        
        Returns:
            int: ID da temporada ou None se não encontrar.
        """
        url = f"https://www.sofascore.com/api/v1/unique-tournament/{tournament_id}/seasons"
        data = self._fetch_api(url)
        
        if data and 'seasons' in data:
            # 1. Tenta match exato
            for s in data['seasons']:
                if s['year'] == year:
                    return s['id']
            
            # 2. Tenta formato europeu (ex: 2024 -> 23/24)
            try:
                year_int = int(year)
                prev_year = year_int - 1
                euro_format = f"{str(prev_year)[-2:]}/{str(year_int)[-2:]}" # Ex: 23/24
                
                for s in data['seasons']:
                    if s['year'] == euro_format:
                        print(f"Temporada encontrada (formato europeu): {euro_format}")
                        return s['id']
            except ValueError:
                pass
                
        return None

    def get_matches(self, tournament_id: int, season_id: int) -> list:
        """
        Coleta todas as partidas de uma temporada do torneio.
        
        Itera dinamicamente sobre as rodadas até não encontrar mais eventos.
        Isso permite suportar ligas com diferentes números de rodadas.
        
        Args:
            tournament_id: ID do torneio.
            season_id: ID da temporada.
            
        Returns:
            list: Lista de partidas encontradas.
        """
        matches = []
        round_num = 1
        max_rounds = 50 # Safety break
        empty_rounds_limit = 3 # Stop after 3 empty rounds (some leagues have gaps)
        empty_rounds = 0
        
        print(f"Iniciando coleta de partidas (Torneio {tournament_id}, Season {season_id})...")
        
        while round_num <= max_rounds:
            print(f"Coletando rodada {round_num}...", end='\r')
            url = f"https://www.sofascore.com/api/v1/unique-tournament/{tournament_id}/season/{season_id}/events/round/{round_num}"
            data = self._fetch_api(url)
            
            if data and 'events' in data and len(data['events']) > 0:
                matches.extend(data['events'])
                empty_rounds = 0 # Reset counter
            else:
                empty_rounds += 1
                if empty_rounds >= empty_rounds_limit:
                    print(f"\nSem eventos por {empty_rounds_limit} rodadas consecutivas. Parando na rodada {round_num}.")
                    break
            
            round_num += 1
            
        print(f"\nTotal de partidas coletadas: {len(matches)}")
        return matches

    def get_match_stats(self, match_id: int) -> dict:
        """
        Coleta estatísticas detalhadas de uma partida específica.
        
        Busca dados de escanteios e chutes no gol para ambos os times,
        separados por tempo de jogo (1º tempo / jogo completo).
        
        Args:
            match_id: ID único da partida no SofaScore.
        
        Returns:
            dict: Dicionário com estatísticas da partida:
                - corners_home_ft: Escanteios mandante (jogo completo)
                - corners_away_ft: Escanteios visitante (jogo completo)
                - corners_home_ht: Escanteios mandante (1º tempo)
                - corners_away_ht: Escanteios visitante (1º tempo)
                - shots_ot_home_ft: Chutes no gol mandante (jogo completo)
                - shots_ot_away_ft: Chutes no gol visitante (jogo completo)
                - shots_ot_home_ht: Chutes no gol mandante (1º tempo)
                - shots_ot_away_ht: Chutes no gol visitante (1º tempo)
        
        Lógica:
            1. Busca dados da API de estatísticas do evento
            2. Separa estatísticas por período (ALL, 1ST)
            3. Extrai valores usando keywords bilíngues (PT/EN)
            4. Retorna zeros se dados não disponíveis
        
        Regras de Negócio:
            - Keywords bilíngues cobrem API em português e inglês
            - Estatísticas de 2º tempo são calculadas: FT - HT
            - Partidas sem estatísticas retornam dict com zeros
            - Valores inválidos são tratados como 0 (fail-safe)
        
        Example:
            >>> stats = scraper.get_match_stats(13472605)
            >>> print(stats['corners_home_ft'])  # Ex: 6
        """
        url = f"https://www.sofascore.com/api/v1/event/{match_id}/statistics"
        data = self._fetch_api(url)
        
        stats = {
            'corners_home_ft': 0, 'corners_away_ft': 0,
            'corners_home_ht': 0, 'corners_away_ht': 0,
            'shots_ot_home_ft': 0, 'shots_ot_away_ft': 0,
            'shots_ot_home_ht': 0, 'shots_ot_away_ht': 0,
            # Novas Estatísticas
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
            """
            Extrai valor numérico de estatística dos grupos de dados.
            
            Args:
                groups: Lista de grupos de estatísticas da API.
                keywords: Lista de palavras-chave para buscar (bilíngue).
                is_home: True para mandante, False para visitante.
                return_float: Se True, retorna float; se False, retorna int.
            
            Returns:
                float ou int: Valor da estatística ou 0 se não encontrado.
            
            Lógica:
                1. Itera sobre grupos de estatísticas
                2. Busca item cujo nome contenha alguma keyword
                3. Retorna valor 'home' ou 'away' conforme is_home
            """
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
                                    # Handle percentage strings (e.g., "61%" -> 61)
                                    if isinstance(value, str) and '%' in value:
                                        return int(value.replace('%', ''))
                                    # Return float or int based on parameter
                                    if return_float:
                                        return float(value)
                                    else:
                                        return int(value)
                            except (ValueError, TypeError):
                                return 0.0 if return_float else 0
            return 0.0 if return_float else 0

        # Extração de Dados (ALL) - usando nomes EXATOS da API
        all_stats = next((p['groups'] for p in data['statistics'] if p['period'] == 'ALL'), [])
        
        # Escanteios - campo é "corner kicks" não "corner"
        stats['corners_home_ft'] = extract_val(all_stats, ['corner kicks'], True)
        stats['corners_away_ft'] = extract_val(all_stats, ['corner kicks'], False)
        
        # Chutes no gol
        stats['shots_ot_home_ft'] = extract_val(all_stats, ['shots on target'], True)
        stats['shots_ot_away_ft'] = extract_val(all_stats, ['shots on target'], False)
        
        # Novas Estatísticas - usando nomes exatos da API
        stats['total_shots_home'] = extract_val(all_stats, ['total shots'], True)
        stats['total_shots_away'] = extract_val(all_stats, ['total shots'], False)
        
        stats['possession_home'] = extract_val(all_stats, ['ball possession'], True)
        stats['possession_away'] = extract_val(all_stats, ['ball possession'], False)
        
        stats['fouls_home'] = extract_val(all_stats, ['fouls'], True)
        stats['fouls_away'] = extract_val(all_stats, ['fouls'], False)
        
        stats['yellow_cards_home'] = extract_val(all_stats, ['yellow cards'], True)
        stats['yellow_cards_away'] = extract_val(all_stats, ['yellow cards'], False)
        
        # Cartões Vermelhos (Level 2 Improvement)
        stats['red_cards_home'] = extract_val(all_stats, ['red cards'], True)
        stats['red_cards_away'] = extract_val(all_stats, ['red cards'], False)
        
        # Big Chances (Level 2 Improvement - Métrica de Qualidade Ofensiva)
        stats['big_chances_home'] = extract_val(all_stats, ['big chances'], True)
        stats['big_chances_away'] = extract_val(all_stats, ['big chances'], False)
        
        # Expected Goals - xG (Level 2 Improvement - Qualidade das Finalizações)
        stats['expected_goals_home'] = extract_val(all_stats, ['expected goals'], True, return_float=True)
        stats['expected_goals_away'] = extract_val(all_stats, ['expected goals'], False, return_float=True)

        # Extração de Dados (1ST)
        ht_stats = next((p['groups'] for p in data['statistics'] if p['period'] == '1ST'), [])
        stats['corners_home_ht'] = extract_val(ht_stats, ['corner kicks'], True)
        stats['corners_away_ht'] = extract_val(ht_stats, ['corner kicks'], False)
        stats['shots_ot_home_ht'] = extract_val(ht_stats, ['shots on target'], True)
        stats['shots_ot_away_ht'] = extract_val(ht_stats, ['shots on target'], False)

        return stats
