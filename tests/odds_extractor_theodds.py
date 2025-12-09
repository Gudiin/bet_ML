"""
=============================================================================
  THE ODDS API - EXTRATOR DE ODDS DE ESCANTEIOS
  
  📌 Plano Free:
     - 500 créditos/mês
     - Suporta odds em tempo real
     - Mercados de escanteios disponíveis
     
  📖 Documentação: https://the-odds-api.com/liveapi/guides/v4/
  🔑 Obter API Key: https://the-odds-api.com/ (grátis!)
=============================================================================
"""

import requests
import os
from datetime import datetime

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


class TheOddsApiExtractor:
    """
    Extrator de Odds usando The Odds API.
    Focado em mercados de Escanteios (Total Corners).
    
    Mercados disponíveis para escanteios:
    - alternate_totals_corners: Over/Under Escanteios (Total do Jogo)
    - alternate_spreads_corners: Handicap de Escanteios
    """
    
    BASE_URL = "https://api.the-odds-api.com/v4"
    
    # IDs das ligas de futebol
    SPORTS = {
        'soccer_brazil_campeonato': 'Brasileirão Série A',
        'soccer_brazil_serie_b': 'Brasileirão Série B',
        'soccer_epl': 'Premier League',
        'soccer_spain_la_liga': 'La Liga',
        'soccer_italy_serie_a': 'Serie A (Itália)',
        'soccer_germany_bundesliga': 'Bundesliga',
        'soccer_france_ligue_one': 'Ligue 1',
        'soccer_portugal_primeira_liga': 'Primeira Liga',
    }
    
    # Mercados de escanteios
    CORNER_MARKETS = [
        'alternate_totals_corners',   # Over/Under Total Corners
        'alternate_spreads_corners',  # Handicap Corners
    ]
    
    # Bookmakers preferidos
    PREFERRED_BOOKMAKERS = [
        'bet365',
        'betfair',
        'pinnacle',
        '1xbet',
        'williamhill',
        'unibet',
    ]
    
    def __init__(self, api_key: str = None):
        """
        Inicializa o extrator.
        
        Args:
            api_key: API Key do The Odds API. Se não fornecida, tenta carregar de .env
        """
        if not api_key:
            if load_dotenv:
                load_dotenv()
            api_key = os.getenv("THE_ODDS_API_KEY")
        
        if not api_key:
            raise ValueError(
                "❌ API Key não encontrada!\n"
                "   1. Obtenha grátis em: https://the-odds-api.com/\n"
                "   2. Passe no construtor: TheOddsApiExtractor('sua_key')\n"
                "   3. Ou crie .env com: THE_ODDS_API_KEY=sua_key"
            )
        
        self.api_key = api_key
        self.remaining_requests = None
        self.used_requests = None
    
    def check_quota(self) -> dict:
        """Verifica a cota restante de requisições."""
        # A quota é retornada nos headers de qualquer requisição
        # Vamos fazer uma requisição leve para checar
        try:
            response = requests.get(
                f"{self.BASE_URL}/sports",
                params={'apiKey': self.api_key}
            )
            
            self.remaining_requests = response.headers.get('x-requests-remaining')
            self.used_requests = response.headers.get('x-requests-used')
            
            return {
                'success': response.status_code == 200,
                'remaining': self.remaining_requests,
                'used': self.used_requests,
                'status_code': response.status_code
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_available_sports(self) -> list:
        """Lista esportes disponíveis (útil para descobrir IDs de ligas)."""
        try:
            response = requests.get(
                f"{self.BASE_URL}/sports",
                params={'apiKey': self.api_key}
            )
            
            if response.status_code != 200:
                print(f"❌ Erro: {response.status_code}")
                return []
            
            sports = response.json()
            # Filtra apenas futebol
            soccer = [s for s in sports if 'soccer' in s['key']]
            return soccer
            
        except Exception as e:
            print(f"❌ Erro: {e}")
            return []
    
    def get_games(self, sport: str = 'soccer_brazil_campeonato') -> list:
        """
        Busca jogos disponíveis para um esporte.
        
        Args:
            sport: ID do esporte (ex: 'soccer_brazil_campeonato')
            
        Returns:
            Lista de jogos com id, times, horário
        """
        try:
            print(f"🔄 Buscando jogos para {self.SPORTS.get(sport, sport)}...")
            
            response = requests.get(
                f"{self.BASE_URL}/sports/{sport}/odds",
                params={
                    'apiKey': self.api_key,
                    'regions': 'eu,uk',  # Regiões das casas de apostas
                    'markets': 'h2h',     # Mercado básico para listar jogos
                    'oddsFormat': 'decimal'
                }
            )
            
            # Atualiza quota
            self.remaining_requests = response.headers.get('x-requests-remaining')
            
            if response.status_code == 404:
                print(f"⚠️ Esporte '{sport}' não encontrado ou sem jogos.")
                return []
            
            if response.status_code != 200:
                print(f"❌ Erro HTTP: {response.status_code}")
                error_data = response.json() if response.text else {}
                print(f"   Detalhes: {error_data}")
                return []
            
            games = response.json()
            print(f"✅ Encontrados: {len(games)} jogos")
            print(f"📊 Requisições restantes: {self.remaining_requests}")
            
            return games
            
        except Exception as e:
            print(f"❌ Erro: {e}")
            return []
    
    def get_corner_odds(self, sport: str = 'soccer_brazil_campeonato') -> list:
        """
        Busca odds de escanteios para jogos de um esporte.
        
        Args:
            sport: ID do esporte
            
        Returns:
            Lista de jogos com odds de escanteios
        """
        try:
            print(f"🔄 Buscando odds de ESCANTEIOS para {self.SPORTS.get(sport, sport)}...")
            
            # Busca especificamente mercados de escanteios
            response = requests.get(
                f"{self.BASE_URL}/sports/{sport}/odds",
                params={
                    'apiKey': self.api_key,
                    'regions': 'eu,uk,us',
                    'markets': ','.join(self.CORNER_MARKETS),
                    'oddsFormat': 'decimal'
                }
            )
            
            self.remaining_requests = response.headers.get('x-requests-remaining')
            
            if response.status_code == 404:
                print(f"⚠️ Esporte '{sport}' não encontrado.")
                return []
            
            if response.status_code == 422:
                print(f"⚠️ Mercado de escanteios não disponível para este esporte.")
                # Tenta listar mercados disponíveis
                return []
            
            if response.status_code != 200:
                print(f"❌ Erro HTTP: {response.status_code}")
                return []
            
            games = response.json()
            
            # Filtra apenas jogos que têm odds de escanteios
            games_with_corners = []
            for game in games:
                if game.get('bookmakers'):
                    games_with_corners.append(self._parse_corner_odds(game))
            
            print(f"✅ Jogos com odds de escanteios: {len(games_with_corners)}")
            print(f"📊 Requisições restantes: {self.remaining_requests}")
            
            return games_with_corners
            
        except Exception as e:
            print(f"❌ Erro: {e}")
            return []
    
    def _parse_corner_odds(self, game: dict) -> dict:
        """Processa e estrutura as odds de escanteios de um jogo."""
        result = {
            'id': game['id'],
            'home_team': game['home_team'],
            'away_team': game['away_team'],
            'commence_time': game['commence_time'],
            'sport': game['sport_key'],
            'corner_odds': {}
        }
        
        for bookmaker in game.get('bookmakers', []):
            bk_name = bookmaker['key']
            
            # Prioriza bookmakers preferidos
            if bk_name not in self.PREFERRED_BOOKMAKERS and result['corner_odds']:
                continue
            
            for market in bookmaker.get('markets', []):
                market_key = market['key']
                
                if market_key == 'alternate_totals_corners':
                    # Over/Under Corners
                    lines = {}
                    
                    for outcome in market.get('outcomes', []):
                        # outcome['name'] = 'Over' ou 'Under'
                        # outcome['point'] = linha (ex: 9.5)
                        # outcome['price'] = odd
                        
                        bet_type = outcome['name']  # Over/Under
                        line = str(outcome.get('point', ''))
                        odd = outcome['price']
                        
                        if line:
                            if line not in lines:
                                lines[line] = {}
                            lines[line][bet_type] = odd
                    
                    if lines:
                        result['corner_odds'][bk_name] = {
                            'market': 'Total Corners',
                            'lines': lines
                        }
        
        return result


# =============================================================================
#                           EXECUÇÃO DE TESTE
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  THE ODDS API - EXTRATOR DE ESCANTEIOS")
    print("  Plano: FREE (500 créditos/mês)")
    print("=" * 60)
    
    # 🔑 COLOQUE SUA API KEY AQUI
    # Obtenha grátis em: https://the-odds-api.com/
    API_KEY = "48ee48ee8de68057d1c1343acfb6a866"
    
    try:
        extractor = TheOddsApiExtractor(API_KEY)
        
        # 1️⃣ Verifica status da conta
        print("\n📊 VERIFICANDO QUOTA...")
        quota = extractor.check_quota()
        
        if quota['success']:
            print(f"   ✅ Requisições restantes: {quota['remaining']}")
            print(f"   📈 Requisições usadas: {quota['used']}")
        else:
            print(f"   ❌ Erro: {quota.get('error', 'Desconhecido')}")
        
        # 2️⃣ Lista esportes de futebol disponíveis
        print("\n⚽ ESPORTES DE FUTEBOL DISPONÍVEIS:")
        sports = extractor.get_available_sports()
        
        for s in sports[:10]:  # Limita a 10
            active = "🟢" if s.get('active') else "🔴"
            print(f"   {active} {s['key']}: {s['title']}")
        
        # 3️⃣ Busca jogos do Brasileirão ou Premier League
        print("\n⚽ BUSCANDO JOGOS...")
        
        # Tenta Brasileirão primeiro
        games = extractor.get_games('soccer_brazil_campeonato')
        sport_key = 'soccer_brazil_campeonato'
        
        if not games:
            print("   ℹ️ Sem jogos no Brasileirão, tentando Premier League...")
            games = extractor.get_games('soccer_epl')
            sport_key = 'soccer_epl'
        
        if games:
            for game in games[:3]:
                print(f"\n🏟️  {game['home_team']} vs {game['away_team']}")
                print(f"   📅 {game['commence_time']}")
        
        # 4️⃣ Busca odds de escanteios
        print("\n🎯 BUSCANDO ODDS DE ESCANTEIOS...")
        corners = extractor.get_corner_odds(sport_key)
        
        if corners:
            for game in corners[:2]:
                print(f"\n🏟️  {game['home_team']} vs {game['away_team']}")
                
                for bk_name, data in game['corner_odds'].items():
                    print(f"   🏦 {bk_name} - {data['market']}")
                    
                    for line, odds in sorted(data['lines'].items(), key=lambda x: float(x[0])):
                        over = odds.get('Over', '-')
                        under = odds.get('Under', '-')
                        print(f"      • Linha {line}: Over @ {over} | Under @ {under}")
        else:
            print("   ⚠️ Nenhuma odd de escanteios encontrada.")
            print("   💡 Tente Premier League: 'soccer_epl'")
        
        print("\n" + "=" * 60)
        print(f"📊 Requisições restantes: {extractor.remaining_requests}")
        print("=" * 60)
        
    except ValueError as e:
        print(f"\n{e}")
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
        import traceback
        traceback.print_exc()
