"""
Módulo de Análise Estatística para Previsão de Escanteios.

Este módulo implementa análise estatística avançada utilizando distribuições
probabilísticas (Poisson e Binomial Negativa) e simulações de Monte Carlo
para calcular probabilidades de mercados de escanteios.

Regras de Negócio:
    - Utiliza distribuição de Poisson quando variância ≤ média
    - Utiliza Binomial Negativa quando variância > média (overdispersion)
    - Monte Carlo com 10.000 simulações para precisão estatística
    - Gera sugestões categorizadas por nível de risco (Easy/Medium/Hard)
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson, nbinom
from tabulate import tabulate


class Colors:
    """
    Constantes ANSI para colorização de output no terminal.
    
    Permite destacar visualmente diferentes tipos de informação:
    - GREEN: Apostas Over, vitórias
    - RED: Alertas, erros
    - CYAN: Apostas Under
    - YELLOW: Destaques importantes
    """
    RESET = "\033[0m"
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"


class StatisticalAnalyzer:
    """
    Analisador estatístico para previsão de escanteios em partidas de futebol.
    
    Utiliza modelos probabilísticos e simulação Monte Carlo para calcular
    probabilidades de diferentes mercados de escanteios (Over/Under).
    
    Modelos Probabilísticos:
        - Distribuição de Poisson: Usada quando variância ≈ média (equidispersão)
          Ideal para eventos raros e independentes como escanteios.
          
        - Binomial Negativa: Usada quando variância > média (overdispersion)
          Mais flexível, captura variabilidade extra em jogos atípicos.
    
    Simulação Monte Carlo:
        Gera 10.000 cenários aleatórios baseados na distribuição escolhida,
        permitindo estimar probabilidades de qualquer mercado.
    
    Mercados Analisados:
        - JOGO COMPLETO: Total de escanteios (8.5 a 12.5)
        - MANDANTE/VISITANTE: Escanteios por time (3.5 a 6.5)
        - 1º/2º TEMPO: Escanteios por período (3.5 a 5.5)
        - MANDANTE/VISITANTE por tempo: Linhas mais baixas (1.5 a 3.5)
    
    Cálculos Principais:
        1. Lambda (λ): Taxa média de escanteios esperados
           λ = 0.6 * média_10_jogos + 0.4 * média_5_jogos
        
        2. Odd Justa: Conversão de probabilidade em odd
           Odd = 1 / Probabilidade
        
        3. Score: Ranking de oportunidades
           Score = Probabilidade * (1 - CV * fator)
           onde CV = Coeficiente de Variação (σ/μ)
    
    Attributes:
        Nenhum atributo persistente - stateless por design.
    
    Example:
        >>> analyzer = StatisticalAnalyzer()
        >>> top_picks = analyzer.analyze_match(df_home, df_away)
    """
    
    def __init__(self):
        """
        Inicializa o analisador estatístico.
        
        Classe é stateless - nenhuma inicialização necessária.
        """
        self.n_simulations = 10000
        
        # Pesos padrão (serão substituídos por Bayesianos se houver histórico)
        self.default_weights = {
            'IA': 0.40,
            'Specific': 0.25,
            'Defense': 0.15,
            'H2H': 0.10,
            'Momentum': 0.10
        }
    
    def calculate_bayesian_weights(
        self,
        historical_errors: dict = None
    ) -> dict:
        """
        Calcula pesos dinamicamente usando inverso do erro quadrático médio.
        
        Permite que fontes mais precisas recebam mais peso automaticamente.
        
        Args:
            historical_errors: Dict com listas de erros por fonte.
                              Ex: {'IA': [1.2, 0.8, ...], 'Specific': [0.5, 1.1, ...]}
        
        Returns:
            dict: Pesos normalizados que somam 1.0
        
        Fórmula Bayesiana (aproximada):
            w_i = (1 / MSE_i) / Σ(1 / MSE_j)
            
        Onde MSE_i é o erro quadrático médio da fonte i.
        """
        if not historical_errors or len(historical_errors) == 0:
            return self.default_weights.copy()
        
        weights = {}
        total_precision = 0
        
        for source, errors in historical_errors.items():
            if len(errors) == 0:
                weights[source] = 1.0
                total_precision += 1.0
                continue
                
            mse = np.mean(np.array(errors) ** 2) + 1e-6  # Evita divisão por zero
            precision = 1.0 / mse
            weights[source] = precision
            total_precision += precision
        
        # Normaliza para somar 1
        if total_precision > 0:
            for source in weights:
                weights[source] /= total_precision
        
        return weights

    @staticmethod
    def calculate_ev(probability: float, odds: float) -> float:
        """
        Calcula o Valor Esperado (+EV) de uma aposta.
        
        Args:
            probability: Probabilidade estimada pelo modelo (0.0 a 1.0)
            odds: Odd oferecida pela casa
            
        Returns:
            float: Valor Esperado em % (ex: 5.0 para 5% de valor)
        """
        if odds <= 1.0:
            return -100.0
        
        # Fórmula EV: (Prob * Odds) - 1
        ev = (probability * odds) - 1
        return ev * 100

    @staticmethod
    def calculate_kelly(probability: float, odds: float, fraction: float = 0.25) -> float:
        """
        Calcula a gestão de banca sugerida pelo Critério de Kelly (Fracionário).
        
        Args:
            probability: Probabilidade estimada (0.0 a 1.0)
            odds: Odd Decimal
            fraction: Fração do Kelly (padrão 1/4 Kelly para segurança)
            
        Returns:
            float: Porcentagem da banca a apostar (0.0 a 100.0)
        """
        if odds <= 1.0:
            return 0.0
            
        b = odds - 1
        q = 1 - probability
        p = probability
        
        # Kelly: (bp - q) / b
        f = (b * p - q) / b
        
        # Ajuste fracionário e proibe negativos
        recommendation = max(0, f) * fraction
        
        # Cap de segurança (ex: nunca apostar mais de 5% da banca)
        return min(recommendation * 100, 5.0)

    def calculate_hybrid_lambda(
        self,
        ia_prediction: float,
        avg_corners_home_when_home: float,
        avg_corners_away_when_away: float,
        avg_corners_conceded_by_home: float,
        avg_corners_conceded_by_away: float,
        avg_corners_h2h_home: float = None,
        avg_corners_h2h_away: float = None,
        momentum_home: float = None,
        momentum_away: float = None
    ) -> tuple:
        """
        Calcula lambdas híbridos combinando previsão da IA com métricas avançadas.
        
        Esta função integra o melhor de dois mundos:
        1. A inteligência da IA (padrões complexos aprendidos)
        2. As métricas específicas de contexto (Casa/Fora, H2H, Defesa)
        
        Args:
            ia_prediction: Previsão total da IA (ex: 9.7 escanteios)
            avg_corners_home_when_home: Média do mandante jogando em casa
            avg_corners_away_when_away: Média do visitante jogando fora
            avg_corners_conceded_by_home: Escanteios cedidos pelo mandante em casa
            avg_corners_conceded_by_away: Escanteios cedidos pelo visitante fora
            avg_corners_h2h_home: Média H2H do mandante (opcional)
            avg_corners_h2h_away: Média H2H do visitante (opcional)
            momentum_home: Média geral recente do mandante (opcional)
            momentum_away: Média geral recente do visitante (opcional)
            
        Returns:
            tuple: (lambda_home, lambda_away) para uso nas simulações Monte Carlo
            
        Fórmula:
            λ_home = W_IA * (IA * proporção_home) + 
                     W_SPECIFIC * avg_corners_home_when_home +
                     W_DEFENSE * avg_corners_conceded_by_away +
                     W_H2H * avg_corners_h2h_home +
                     W_MOMENTUM * momentum_home
                     
            Onde W_* são pesos que somam 1.0
            
        Regra de Negócio:
            Os pesos foram calibrados para priorizar:
            1. A previsão da IA (40%) - Captura padrões complexos
            2. Performance específica Home/Away (25%) - Contexto do mando
            3. Fraqueza defensiva do oponente (15%) - Oportunidade ofensiva
            4. Histórico H2H (10%) - Padrão do confronto
            5. Momentum geral (10%) - Forma atual do time
        """
        # Pesos para cada componente
        W_IA = 0.40
        W_SPECIFIC = 0.25
        W_DEFENSE = 0.15
        W_H2H = 0.10
        W_MOMENTUM = 0.10
        
        # Proporção histórica para dividir a previsão da IA
        total_specific = avg_corners_home_when_home + avg_corners_away_when_away
        if total_specific > 0:
            prop_home = avg_corners_home_when_home / total_specific
        else:
            prop_home = 0.5  # Fallback: divisão igual
            
        # Componente 1: IA (ajustada pela proporção)
        ia_home = ia_prediction * prop_home
        ia_away = ia_prediction * (1 - prop_home)
        
        # Componente 2: Específico (Home when Home, Away when Away)
        specific_home = avg_corners_home_when_home
        specific_away = avg_corners_away_when_away
        
        # Componente 3: Defesa (Oportunidade ofensiva = Fraqueza defensiva do oponente)
        defense_home = avg_corners_conceded_by_away  # Mandante ataca fraqueza do visitante
        defense_away = avg_corners_conceded_by_home  # Visitante ataca fraqueza do mandante
        
        # Componente 4: H2H (usa específico como fallback se não tiver H2H)
        h2h_home = avg_corners_h2h_home if avg_corners_h2h_home is not None else specific_home
        h2h_away = avg_corners_h2h_away if avg_corners_h2h_away is not None else specific_away
        
        # Componente 5: Momentum (usa específico como fallback)
        mom_home = momentum_home if momentum_home is not None else specific_home
        mom_away = momentum_away if momentum_away is not None else specific_away
        
        # Cálculo final do Lambda Híbrido
        lambda_home = (
            W_IA * ia_home +
            W_SPECIFIC * specific_home +
            W_DEFENSE * defense_home +
            W_H2H * h2h_home +
            W_MOMENTUM * mom_home
        )
        
        lambda_away = (
            W_IA * ia_away +
            W_SPECIFIC * specific_away +
            W_DEFENSE * defense_away +
            W_H2H * h2h_away +
            W_MOMENTUM * mom_away
        )
        
        # Log detalhado para transparência
        # Mostra cada componente do cálculo para facilitar a compreensão
        print(f"\n{Colors.YELLOW}{'='*70}")
        print(f"🧮 LAMBDA HÍBRIDO (IA + Métricas Avançadas)")
        print(f"{'='*70}{Colors.RESET}")
        print(f"📊 Previsão IA Total: {ia_prediction:.2f} escanteios")
        print(f"")
        
        # Mandante detalhado
        print(f"{Colors.GREEN}🏠 MANDANTE (λ = {lambda_home:.2f}){Colors.RESET}")
        print(f"   ├─ IA ({int(W_IA*100)}%):       {ia_home:.2f}  ← Previsão da IA para o mandante")
        print(f"   ├─ Casa ({int(W_SPECIFIC*100)}%):    {specific_home:.2f}  ← Média de escanteios jogando EM CASA")
        print(f"   ├─ Def. Adv ({int(W_DEFENSE*100)}%): {defense_home:.2f}  ← Escanteios que o visitante CEDE fora")
        print(f"   ├─ H2H ({int(W_H2H*100)}%):      {h2h_home:.2f}  ← Média nos confrontos diretos")
        print(f"   └─ Momentum ({int(W_MOMENTUM*100)}%): {mom_home:.2f}  ← Forma recente geral")
        print(f"")
        
        # Visitante detalhado
        print(f"{Colors.CYAN}✈️ VISITANTE (λ = {lambda_away:.2f}){Colors.RESET}")
        print(f"   ├─ IA ({int(W_IA*100)}%):       {ia_away:.2f}  ← Previsão da IA para o visitante")
        print(f"   ├─ Fora ({int(W_SPECIFIC*100)}%):    {specific_away:.2f}  ← Média de escanteios jogando FORA")
        print(f"   ├─ Def. Adv ({int(W_DEFENSE*100)}%): {defense_away:.2f}  ← Escanteios que o mandante CEDE em casa")
        print(f"   ├─ H2H ({int(W_H2H*100)}%):      {h2h_away:.2f}  ← Média nos confrontos diretos")
        print(f"   └─ Momentum ({int(W_MOMENTUM*100)}%): {mom_away:.2f}  ← Forma recente geral")
        print(f"")
        print(f"{Colors.BOLD}🎯 TOTAL ESPERADO: {lambda_home + lambda_away:.2f} escanteios{Colors.RESET}")
        print(f"{Colors.YELLOW}{'='*70}{Colors.RESET}")
        
        return lambda_home, lambda_away

    def _get_distribution_params(self, data: pd.Series) -> tuple:
        """
        Calcula parâmetros da distribuição para uma série de dados.
        
        Args:
            data: Série temporal de dados (ex: escanteios nos últimos jogos).
            
        Returns:
            tuple: (tipo_distribuicao, media, variancia)
        """
        if len(data) == 0:
            return 'poisson', 0, 0
            
        mean = data.mean()
        var = data.var() if len(data) > 1 else 0
        
        # Se variância for zero ou NaN, assume Poisson com a média
        if pd.isna(var) or var == 0:
            return 'poisson', mean, 0
            
        dist_type = 'nbinom' if var > mean else 'poisson'
        return dist_type, mean, var

    def simulate_match_event(self, avg_home: float, avg_away: float, 
                           var_home: float = 0, var_away: float = 0) -> np.ndarray:
        """
        Simula um evento de partida (ex: Total Escanteios) combinando mandante e visitante.
        
        Args:
            avg_home: Média do mandante.
            avg_away: Média do visitante.
            var_home: Variância do mandante.
            var_away: Variância do visitante.
            
        Returns:
            np.ndarray: Array com a soma das simulações (Home + Away).
        """
        sim_home = self.monte_carlo_simulation(avg_home, var_home)
        sim_away = self.monte_carlo_simulation(avg_away, var_away)
        return sim_home + sim_away

    def monte_carlo_simulation(self, lambda_val: float, var_val: float, 
                               n_sims: int = 10000) -> np.ndarray:
        """
        Executa simulação de Monte Carlo para estimar distribuição de escanteios.
        
        Gera N cenários aleatórios seguindo a distribuição apropriada
        (Poisson ou Binomial Negativa) baseada na relação variância/média.
        
        Args:
            lambda_val: Taxa média esperada de escanteios (λ).
            var_val: Variância observada nos dados históricos.
            n_sims: Número de simulações (default: 10.000).
        
        Returns:
            np.ndarray: Array com n_sims valores simulados de escanteios.
        
        Lógica:
            1. Compara variância com média (lambda)
            2. Se variância > lambda: usa Binomial Negativa (overdispersion)
            3. Se variância ≤ lambda: usa Poisson (equidispersion)
            4. Gera n_sims amostras da distribuição escolhida
        
        Fórmulas:
            Poisson:
                P(X=k) = (λ^k * e^(-λ)) / k!
                Onde λ = média esperada
            
            Binomial Negativa (parametrização alternativa):
                p = λ / σ²  (probabilidade de sucesso)
                n = λ² / (σ² - λ)  (número de sucessos)
        
        Regras de Negócio:
            - 10.000 simulações fornece precisão de ~1% nas probabilidades
            - Overdispersion é comum em futebol (jogos imprevisíveis)
            - Monte Carlo captura toda a distribuição, não apenas a média
        
        Example:
            >>> sims = analyzer.monte_carlo_simulation(10.5, 15.0)
            >>> prob_over_9 = (sims > 9.5).mean()  # ~65%
        """
        if var_val > lambda_val:
            # Overdispersion: usa Binomial Negativa
            p = lambda_val / var_val
            n = (lambda_val ** 2) / (var_val - lambda_val)
            sims = nbinom.rvs(n, p, size=n_sims)
        else:
            # Equidispersion: usa Poisson
            sims = poisson.rvs(lambda_val, size=n_sims)
        return sims

    def generate_suggestions(self, opportunities: list, 
                            ml_prediction: float = None) -> dict:
        """
        Gera sugestões de apostas categorizadas por nível de risco.
        
        Analisa as oportunidades encontradas e seleciona as melhores
        para cada nível de risco, alinhando com a previsão do modelo ML.
        
        Args:
            opportunities: Lista de dicionários com oportunidades.
                          Cada dict tem: Mercado, Seleção, Prob, Odd, Score, Tipo
            ml_prediction: Previsão do modelo ML (ex: 10.5 escanteios).
                          Usada para alinhar sugestões estatísticas.
        
        Returns:
            dict: Sugestões por nível de risco:
                - Easy: Alta probabilidade (>70%), odds baixas (1.25-1.60)
                - Medium: Média probabilidade (50-75%), odds médias (1.60-2.20)
                - Hard: Baixa probabilidade (30-55%), odds altas (>2.20)
        
        Lógica:
            1. Ordena oportunidades por probabilidade (decrescente)
            2. Para cada nível, busca primeira oportunidade que:
               a) Atenda critérios de probabilidade e odd
               b) Esteja alinhada com previsão ML
            3. Retorna dict com melhor opção por nível
        
        Alinhamento com ML:
            - Se ML prevê >10.5 escanteios: favorece Overs
            - Se ML prevê <9.5 escanteios: favorece Unders
            - Se ML entre 9.5-10.5: aceita ambos (zona neutra)
        
        Regras de Negócio:
            - Easy: Para apostadores conservadores, green frequente
            - Medium: Equilibrio risco/retorno, ROI melhor
            - Hard: Value bets de alto risco, odds atrativas
        
        Example:
            >>> suggestions = analyzer.generate_suggestions(opportunities, ml_prediction=11.2)
            >>> print(suggestions['Easy'])  # {'Mercado': 'JOGO COMPLETO', 'Seleção': 'Over 9.5', ...}
        """
        suggestions = {
            "Easy": None,
            "Medium": None,
            "Hard": None
        }
        
        # Ordena por probabilidade (decrescente)
        sorted_ops = sorted(opportunities, key=lambda x: x['Prob'], reverse=True)
        
        def aligns_with_ml(op: dict) -> bool:
            """
            Verifica se a oportunidade está alinhada com a previsão ML.
            
            Args:
                op: Dicionário da oportunidade.
            
            Returns:
                bool: True se alinhada ou ML não disponível.
            """
            if ml_prediction is None:
                return True
            # ML alto (>10.5): favorece Overs
            if "Over" in op['Seleção'] and ml_prediction > 10.5:
                return True
            # ML baixo (<9.5): favorece Unders
            if "Under" in op['Seleção'] and ml_prediction < 9.5:
                return True
            # ML neutro (9.5-10.5): aceita ambos
            if 9.5 <= ml_prediction <= 10.5:
                return True
            return False

        # Easy: Alta probabilidade (>70%), odds baixas (1.25-1.60)
        for op in sorted_ops:
            if op['Prob'] >= 0.70 and 1.25 <= op['Odd'] <= 1.60:
                if aligns_with_ml(op):
                    suggestions["Easy"] = op
                    break
        
        # Medium: Média probabilidade (50-75%), odds médias (1.60-2.20)
        for op in sorted_ops:
            if 0.50 <= op['Prob'] < 0.75 and 1.60 <= op['Odd'] <= 2.20:
                if aligns_with_ml(op):
                    suggestions["Medium"] = op
                    break
                
        # Hard: Probabilidade moderada (30-55%), odds altas (>2.20) - Value Bet
        for op in sorted_ops:
            if 0.30 <= op['Prob'] < 0.55 and op['Odd'] > 2.20:
                if aligns_with_ml(op):
                    suggestions["Hard"] = op
                    break
                
        return suggestions

    def analyze_match(self, df_home: pd.DataFrame, df_away: pd.DataFrame, 
                     ml_prediction: float = None, match_name: str = None,
                     advanced_metrics: dict = None, scraped_odds: dict = None) -> tuple:
        """
        Executa análise estatística completa de uma partida.
        
        Calcula probabilidades para múltiplos mercados de escanteios
        usando Monte Carlo e gera ranking de melhores oportunidades.
        
        Args:
            df_home: DataFrame com histórico do mandante.
            df_away: DataFrame com histórico do visitante.
            ml_prediction: Previsão do modelo ML para alinhamento.
            match_name: Nome da partida.
            advanced_metrics: Métricas avançadas da IA.
            scraped_odds: Dict com odds raspadas do bookmaker (ex: {'Over 9.5': 1.85}).
        """
        # 1. Extração de Estatísticas Básicas
        # Calculamos médias e variâncias para alimentar as simulações
        
        # Total FT (Full Time)
        h_corners_ft = df_home['corners_ft']
        a_corners_ft = df_away['corners_ft']
        
        # Total HT (Half Time)
        h_corners_ht = df_home['corners_ht']
        a_corners_ht = df_away['corners_ht']
        
        # Simulações (O "Coração" do Monte Carlo)
        # ---------------------------------------
        
        # Simula Jogo Completo (FT)
        dist_h, mean_h, var_h = self._get_distribution_params(h_corners_ft)
        dist_a, mean_a, var_a = self._get_distribution_params(a_corners_ft)
        
        # Lógica de Integração IA + Estatística (NÍVEL 2 - HÍBRIDO)
        # ------------------------------------------------------------
        
        if advanced_metrics is not None and ml_prediction is not None and ml_prediction > 0:
            # 🚀 MODO HÍBRIDO: Usa as métricas avançadas da feature engineering
            mean_h, mean_a = self.calculate_hybrid_lambda(
                ia_prediction=ml_prediction,
                avg_corners_home_when_home=advanced_metrics.get('home_avg_corners_home', mean_h),
                avg_corners_away_when_away=advanced_metrics.get('away_avg_corners_away', mean_a),
                avg_corners_conceded_by_home=advanced_metrics.get('home_avg_corners_conceded_home', mean_h),
                avg_corners_conceded_by_away=advanced_metrics.get('away_avg_corners_conceded_away', mean_a),
                avg_corners_h2h_home=advanced_metrics.get('home_avg_corners_h2h'),
                avg_corners_h2h_away=advanced_metrics.get('away_avg_corners_h2h'),
                momentum_home=advanced_metrics.get('home_avg_corners_general'),
                momentum_away=advanced_metrics.get('away_avg_corners_general')
            )
            
        elif ml_prediction is not None and ml_prediction > 0:
            # 🤖 MODO LEGADO: Apenas IA, sem métricas avançadas
            historical_avg = mean_h + mean_a
            if historical_avg > 0:
                prop_h = mean_h / historical_avg
                mean_h = ml_prediction * prop_h
                mean_a = ml_prediction * (1 - prop_h)
            else:
                mean_h = ml_prediction / 2
                mean_a = ml_prediction / 2
        
        sim_total = self.simulate_match_event(mean_h, mean_a, var_h, var_a)
        
        # Simula Primeiro Tempo (HT)
        dist_h_ht, mean_h_ht, var_h_ht = self._get_distribution_params(h_corners_ht)
        dist_a_ht, mean_a_ht, var_a_ht = self._get_distribution_params(a_corners_ht)
        
        sim_ht = self.simulate_match_event(mean_h_ht, mean_a_ht, var_h_ht, var_a_ht)
        
        # Simula Totais Individuais
        sim_home_total = self.monte_carlo_simulation(mean_h, var_h)
        sim_away_total = self.monte_carlo_simulation(mean_a, var_a)
        
        # Análise de Mercados
        markets = []
        
        # Função auxiliar para adicionar mercado analisado
        def add_market(name, simulations, line, type_='Over'):
            count = np.sum(simulations > line) if type_ == 'Over' else np.sum(simulations < line)
            prob = count / self.n_simulations # 10000 simulações
            
            if prob > 0.01: 
                fair_odd = 1 / prob
                
                # Busca odd do bookmaker se disponível
                selection_key = f"{type_} {line}" # Ex: "Over 9.5"
                bookmaker_odd = 0.0
                ev = 0.0
                kelly = 0.0
                
                if scraped_odds and selection_key in scraped_odds:
                    bookmaker_odd = scraped_odds[selection_key]
                    ev = self.calculate_ev(prob, bookmaker_odd)
                    kelly = self.calculate_kelly(prob, bookmaker_odd)
                
                markets.append({
                    'Mercado': name,
                    'Seleção': selection_key,
                    'Prob': prob,
                    'FairOdd': fair_odd,
                    'Odd': bookmaker_odd if bookmaker_odd > 0 else fair_odd, # Usa Fair se não tiver Book
                    'IsBookmaker': bookmaker_odd > 0,
                    'EV': ev,
                    'Kelly': kelly
                })

        # Define as linhas padrão a serem analisadas
        lines_ft = [8.5, 9.5, 10.5, 11.5, 12.5]
        lines_ht = [3.5, 4.5, 5.5]
        lines_team = [3.5, 4.5, 5.5, 6.5]

        # Analisa Over/Under para cada linha
        for line in lines_ft:
            add_market('JOGO COMPLETO', sim_total, line, 'Over')
            add_market('JOGO COMPLETO', sim_total, line, 'Under') 

        for line in lines_ht:
            add_market('1º TEMPO (HT)', sim_ht, line, 'Over')
            add_market('1º TEMPO (HT)', sim_ht, line, 'Under')

        for line in lines_ht:
            add_market('2º TEMPO (FT)', sim_ht, line, 'Over')
            add_market('2º TEMPO (FT)', sim_ht, line, 'Under')

        for line in lines_team:
            add_market('TOTAL MANDANTE', sim_home_total, line, 'Over')
            add_market('TOTAL VISITANTE', sim_away_total, line, 'Over')
            add_market('TOTAL MANDANTE', sim_home_total, line, 'Under')
            add_market('TOTAL VISITANTE', sim_away_total, line, 'Under')

        # Seleção das Melhores Oportunidades (BALANCEADA)
        # ----------------------------------
        # Separamos Over e Under para garantir diversidade
        over_markets = [m for m in markets if 'Over' in m['Seleção'] and m['Prob'] > 0.50]
        under_markets = [m for m in markets if 'Under' in m['Seleção'] and m['Prob'] > 0.50]
        
        # Ordena cada grupo por probabilidade
        over_markets = sorted(over_markets, key=lambda x: x['Prob'], reverse=True)
        under_markets = sorted(under_markets, key=lambda x: x['Prob'], reverse=True)
        
        # Estratégia balanceada: 
        # - Pega top 3 Under (geralmente mais conservadores)
        # - Pega top 2 Over (oportunidades de value)
        # - Pega mais 2 das melhores restantes (qualquer tipo)
        top_picks = []
        top_picks.extend(under_markets[:3])  # Top 3 Under
        top_picks.extend(over_markets[:2])   # Top 2 Over
        
        # Adiciona as 2 melhores restantes (pode ser Over ou Under)
        remaining = [m for m in markets if m not in top_picks and m['Prob'] > 0.50]
        remaining = sorted(remaining, key=lambda x: x['Prob'], reverse=True)
        top_picks.extend(remaining[:2])
        
        # Reordena o Top 7 final por probabilidade para exibição
        top_picks = sorted(top_picks, key=lambda x: x['Prob'], reverse=True)[:7]
                         
        # Gera sugestões categorizadas (Easy/Medium/Hard) usando TODOS os mercados analisados
        suggestions = self.generate_suggestions(markets, ml_prediction)

        # Exibição no Terminal (apenas se executado via CLI)
        if match_name:
            print(f"\n▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓")
            print(f" 🧠 {Colors.BOLD}CÉREBRO ESTATÍSTICO (Monte Carlo){Colors.RESET}")
            print(f"▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓")
            
            print(f"\n🏆 {Colors.BOLD}TOP 7 OPORTUNIDADES (DATA DRIVEN){Colors.RESET}")
            
            tabela_display = []
            for pick in top_picks:
                prob = pick['Prob']
                tipo = "OVER" if "Over" in pick['Seleção'] else "UNDER"
                cor = Colors.GREEN if tipo == "OVER" else Colors.CYAN
                seta = "▲" if tipo == "OVER" else "▼"
                
                linha_fmt = f"{cor}{pick['Seleção']}{Colors.RESET}"
                prob_fmt = f"{prob * 100:.1f}%"
                odd_fmt = f"{Colors.BOLD}@{pick['Odd']:.2f}{Colors.RESET}"
                direcao_fmt = f"{cor}{seta} {tipo}{Colors.RESET}"
                
                tabela_display.append([pick['Mercado'], linha_fmt, prob_fmt, odd_fmt, direcao_fmt])
                
            headers = ["MERCADO", "LINHA", "PROB.", "ODD JUSTA", "TIPO"]
            print(tabulate(tabela_display, headers=headers, tablefmt="fancy_grid", stralign="center"))

            print(f"\n🎯 {Colors.BOLD}SUGESTÕES DA IA:{Colors.RESET}")
            for level, pick in suggestions.items():
                if pick:
                    color = Colors.GREEN if level == 'Easy' else (Colors.YELLOW if level == 'Medium' else Colors.RED)
                    print(f"[{color}{level.upper()}{Colors.RESET}] {pick['Mercado']} - {pick['Seleção']} (@{pick['Odd']:.2f}) | Prob: {pick['Prob']*100:.1f}%")
                else:
                    print(f"[{level.upper()}] Nenhuma oportunidade encontrada.")

        return top_picks, suggestions
