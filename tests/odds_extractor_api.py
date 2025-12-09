"""
=============================================================================
  API-FOOTBALL ODDS EXTRACTOR (FREE TIER COMPATIBLE)
  Versão otimizada para o plano gratuito da API-Football
  
  📌 Limites do Plano Free:
     - 100 requisições/dia
     - 10 requisições/minuto
     - Odds pré-jogo disponíveis
     
  📖 Documentação: https://www.api-football.com/documentation-v3
=============================================================================
"""

import requests
import os
import time
from datetime import datetime, timedelta

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


class ApiFootballOdds:
    """
    Extrator de Odds compatível com o plano FREE da API-Football.
    Focado em mercados de Escanteios (Total, Mandante, Visitante).
    """
    
    # 📚 Ligas populares com jogos frequentes
    POPULAR_LEAGUES = {
        39: "Premier League (Inglaterra)",
        140: "La Liga (Espanha)", 
        135: "Serie A (Itália)",
        78: "Bundesliga (Alemanha)",
        61: "Ligue 1 (França)",
        71: "Brasileirão Série A",
        94: "Primeira Liga (Portugal)",
        88: "Eredivisie (Holanda)",
        144: "Jupiler Pro League (Bélgica)",
        203: "Super Lig (Turquia)",
    }
    
    # 📊 IDs dos mercados de escanteios na API-Football
    CORNER_MARKETS = {
        45: "Corners Total (FT)",          # Over/Under Jogo Completo
        40: "Corners Home (Mandante)",     # Over/Under Casa
        41: "Corners Away (Visitante)",    # Over/Under Fora
    }
    
    # 🏦 Casas de apostas preferidas (por ordem de prioridade)
    BOOKMAKERS = {
        1: "Bet365",
        8: "Bet365", 
        6: "Bwin",
        5: "1xBet",
        11: "Betfair",
        3: "Unibet",
    }
    
    def __init__(self, api_key: str = None):
        """
        Inicializa o extrator.
        
        Args:
            api_key: Chave da API. Se não fornecida, tenta carregar de .env (API_FOOTBALL_KEY)
        """
        if not api_key:
            if load_dotenv:
                load_dotenv()
            api_key = os.getenv("API_FOOTBALL_KEY")
        
        if not api_key:
            raise ValueError(
                "❌ API Key não encontrada!\n"
                "   Opção 1: Passe a chave no construtor: ApiFootballOdds('sua_key')\n"
                "   Opção 2: Crie um arquivo .env com: API_FOOTBALL_KEY=sua_key"
            )
        
        self.api_key = api_key
        self.base_url = "https://v3.football.api-sports.io"
        self.headers = {
            'x-rapidapi-host': "v3.football.api-sports.io",
            'x-rapidapi-key': api_key
        }
        self.request_count = 0
        
    def check_account_status(self) -> dict:
        """
        Verifica o status da conta e requisições restantes.
        IMPORTANTE: Esta chamada também conta como 1 requisição!
        """
        try:
            response = requests.get(
                f"{self.base_url}/status",
                headers=self.headers
            )
            data = response.json()
            
            if 'errors' in data and data['errors']:
                return {'success': False, 'error': str(data['errors'])}
            
            account = data.get('response', {}).get('account', {})
            subscription = data.get('response', {}).get('subscription', {})
            requests_info = data.get('response', {}).get('requests', {})
            
            return {
                'success': True,
                'plan': subscription.get('plan', 'Unknown'),
                'is_active': subscription.get('active', False),
                'requests_today': requests_info.get('current', 0),
                'requests_limit': requests_info.get('limit_day', 100),
                'remaining': requests_info.get('limit_day', 100) - requests_info.get('current', 0)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_fixtures(self, date: str = None, league_id: int = None, 
                     status: str = "NS") -> list:
        """
        Busca jogos agendados.
        
        Args:
            date: Data no formato YYYY-MM-DD (padrão: hoje)
            league_id: ID da liga (opcional - economiza requisições)
            status: NS=Não iniciado, LIVE=Ao vivo, FT=Finalizado
            
        Returns:
            Lista de jogos com id, home, away, league, datetime
        """
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")
        
        endpoint = f"{self.base_url}/fixtures"
        params = {'date': date}
        
        if league_id:
            params['league'] = league_id
            params['season'] = 2025  # Brasileirão 2025 (temporada atual)
        
        try:
            self._rate_limit()
            print(f"🔄 Buscando jogos para {date}...")
            
            response = requests.get(endpoint, headers=self.headers, params=params)
            self.request_count += 1
            
            if response.status_code == 429:
                print("⛔ Limite de requisições atingido! Aguarde reset diário.")
                return []
            
            if response.status_code != 200:
                print(f"❌ Erro HTTP: {response.status_code}")
                return []
                
            data = response.json()
            
            # Debug: mostra erros da API
            if data.get('errors') and len(data['errors']) > 0:
                print(f"⚠️ Aviso da API: {data['errors']}")
                return []
            
            if not data.get('response'):
                print("ℹ️ Nenhum jogo encontrado para esta data/liga.")
                return []
            
            fixtures = []
            for item in data['response']:
                fixture = item['fixture']
                teams = item['teams']
                league = item['league']
                
                fixtures.append({
                    'id': fixture['id'],
                    'datetime': fixture['date'],
                    'status': fixture['status']['short'],
                    'home': teams['home']['name'],
                    'away': teams['away']['name'],
                    'league': league['name'],
                    'league_id': league['id'],
                    'country': league.get('country', 'N/A')
                })
            
            # Filtra apenas jogos não iniciados se solicitado
            if status:
                fixtures = [f for f in fixtures if f['status'] == status]
            
            print(f"✅ Encontrados: {len(fixtures)} jogos")
            return fixtures
            
        except Exception as e:
            print(f"❌ Erro: {e}")
            return []
    
    def get_odds(self, fixture_id: int) -> dict:
        """
        Busca odds de escanteios para uma partida específica.
        
        Args:
            fixture_id: ID da partida (obtido via get_fixtures)
            
        Returns:
            Dict com bookmaker e mercados de escanteios
        """
        endpoint = f"{self.base_url}/odds"
        params = {'fixture': fixture_id}
        
        try:
            self._rate_limit()
            response = requests.get(endpoint, headers=self.headers, params=params)
            self.request_count += 1
            
            data = response.json()
            
            if not data.get('response'):
                return None
            
            # A API retorna uma lista, pegamos o primeiro
            odds_data = data['response'][0]
            bookmakers = odds_data.get('bookmakers', [])
            
            if not bookmakers:
                return None
            
            # Seleciona casa de aposta preferida
            selected_bk = None
            for bk in bookmakers:
                if bk['id'] in self.BOOKMAKERS:
                    selected_bk = bk
                    break
            
            if not selected_bk:
                selected_bk = bookmakers[0]
            
            result = {
                'fixture_id': fixture_id,
                'bookmaker': selected_bk['name'],
                'bookmaker_id': selected_bk['id'],
                'markets': {},
                'all_markets': []  # Lista de todos os mercados disponíveis
            }
            
            # Processa cada mercado (bet)
            for market in selected_bk.get('bets', []):
                market_id = market['id']
                market_name = market['name']
                
                # Debug: lista todos os mercados disponíveis
                result['all_markets'].append({
                    'id': market_id,
                    'name': market_name
                })
                
                # Filtra apenas mercados de escanteios
                if market_id in self.CORNER_MARKETS:
                    friendly_name = self.CORNER_MARKETS[market_id]
                    lines = {}
                    
                    for val in market.get('values', []):
                        selection = val['value']  # Ex: "Over 9.5"
                        odd = float(val['odd'])
                        
                        if "Over" in selection or "Under" in selection:
                            parts = selection.split(" ")
                            if len(parts) >= 2:
                                bet_type = parts[0]  # Over/Under
                                line = parts[1]      # 9.5
                                
                                if line not in lines:
                                    lines[line] = {}
                                lines[line][bet_type] = odd
                    
                    if lines:
                        result['markets'][friendly_name] = lines
            
            return result
            
        except Exception as e:
            print(f"❌ Erro ao buscar odds: {e}")
            return None
    
    def find_games_with_odds(self, max_games: int = 5) -> list:
        """
        Busca jogos com odds disponíveis em múltiplas ligas.
        Útil quando não há jogos em uma liga específica.
        
        Args:
            max_games: Máximo de jogos a retornar (economiza requisições)
            
        Returns:
            Lista de jogos com odds de escanteios
        """
        all_games = []
        today = datetime.now().strftime("%Y-%m-%d")
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        
        print("🔍 Buscando jogos com odds em ligas populares...")
        
        # Primeiro tenta jogos de hoje
        games_today = self.get_fixtures(date=today)
        
        if not games_today:
            print("ℹ️ Sem jogos hoje, tentando amanhã...")
            games_today = self.get_fixtures(date=tomorrow)
        
        if not games_today:
            print("❌ Nenhum jogo encontrado para hoje ou amanhã.")
            return []
        
        # Limita a quantidade para economizar requisições
        for game in games_today[:max_games]:
            odds = self.get_odds(game['id'])
            
            if odds and odds['markets']:
                game['odds'] = odds
                all_games.append(game)
                print(f"   ✅ {game['home']} vs {game['away']} - Odds encontradas!")
            else:
                print(f"   ⚠️ {game['home']} vs {game['away']} - Sem odds de escanteios")
        
        return all_games
    
    def _rate_limit(self):
        """Implementa rate limiting para respeitar 10 req/min."""
        time.sleep(0.15)  # ~6.6 req/seg max, bem abaixo do limite


# =============================================================================
#                           EXECUÇÃO DE TESTE
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  API-FOOTBALL ODDS EXTRACTOR - TESTE")
    print("  Plano: FREE (100 req/dia, 10 req/min)")
    print("=" * 60)
    
    # 🔑 COLOQUE SUA API KEY AQUI (ou use .env)
    API_KEY = "5624f9012c325692a729e0c2d7a46254"
    
    try:
        extractor = ApiFootballOdds(API_KEY)
        
        # 1️⃣ Verifica status da conta
        print("\n📊 VERIFICANDO STATUS DA CONTA...")
        status = extractor.check_account_status()
        
        if status['success']:
            print(f"   ✅ Plano: {status['plan']}")
            print(f"   📈 Requisições hoje: {status['requests_today']}/{status['requests_limit']}")
            print(f"   🔋 Restantes: {status['remaining']}")
            
            if status['remaining'] < 10:
                print("   ⚠️ ATENÇÃO: Poucas requisições restantes!")
        else:
            print(f"   ❌ Erro ao verificar conta: {status['error']}")
            print("   ⚠️ Continuando mesmo assim...")
        
        # 2️⃣ Busca jogos com odds
        print("\n🎯 BUSCANDO JOGOS COM ODDS DE ESCANTEIOS...")
        
        # Tenta Premier League primeiro (geralmente tem jogos)
        games = extractor.get_fixtures(league_id=39)  # Premier League
        
        if not games:
            # Se não houver Premier League, busca qualquer jogo
            games = extractor.find_games_with_odds(max_games=3)
        
        if not games:
            print("\n❌ Nenhum jogo com odds encontrado.")
            print("   Possíveis causas:")
            print("   - Não há jogos agendados para hoje")
            print("   - Mercado de escanteios não disponível")
            print("   - Limite de requisições atingido")
        else:
            print(f"\n📋 ODDS DE ESCANTEIOS ENCONTRADAS:")
            print("-" * 50)
            
            for game in games[:3]:  # Limita a 3 para não poluir o output
                print(f"\n🏟️  {game['home']} vs {game['away']}")
                print(f"   📍 {game['league']} ({game.get('country', 'N/A')})")
                print(f"   📅 {game['datetime']}")
                
                # Busca odds se ainda não tiver
                odds = game.get('odds') or extractor.get_odds(game['id'])
                
                if odds and odds['markets']:
                    print(f"   🏦 Casa: {odds['bookmaker']}")
                    
                    for market_name, lines in odds['markets'].items():
                        print(f"\n   � {market_name}:")
                        sorted_lines = sorted(lines.keys(), key=float)
                        
                        for line in sorted_lines:
                            over = lines[line].get('Over', '-')
                            under = lines[line].get('Under', '-')
                            print(f"      • Linha {line}: Over @ {over} | Under @ {under}")
                    
                    # Debug: mostra outros mercados disponíveis
                    if odds.get('all_markets'):
                        corner_ids = list(ApiFootballOdds.CORNER_MARKETS.keys())
                        other_markets = [m for m in odds['all_markets'] if m['id'] not in corner_ids]
                        if other_markets:
                            print(f"\n   📋 Outros mercados disponíveis: {len(other_markets)}")
                            for m in other_markets[:5]:
                                print(f"      - ID {m['id']}: {m['name']}")
                else:
                    print("   ⚠️ Mercado de escanteios não disponível")
        
        print("\n" + "=" * 60)
        print(f"📊 Total de requisições usadas neste teste: {extractor.request_count}")
        print("=" * 60)
        
    except ValueError as e:
        print(f"\n❌ {e}")
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
        import traceback
        traceback.print_exc()