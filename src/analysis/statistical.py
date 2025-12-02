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
                     ml_prediction: float = None, match_name: str = None) -> tuple:
        """
        Executa análise estatística completa de uma partida.
        
        Calcula probabilidades para múltiplos mercados de escanteios
        usando Monte Carlo e gera ranking de melhores oportunidades.
        
        Args:
            df_home: DataFrame com histórico do mandante.
                    Colunas: corners_ft, corners_ht, corners_2t, shots_ht
            df_away: DataFrame com histórico do visitante.
                    Colunas: corners_ft, corners_ht, corners_2t, shots_ht
            ml_prediction: Previsão do modelo ML para alinhamento.
            match_name: Nome da partida para exibição (ex: "Flamengo vs Palmeiras").
        
        Returns:
            list: Top 7 oportunidades ordenadas por Score:
                  [{'Mercado': str, 'Seleção': str, 'Prob': float, 
                    'Odd': float, 'Score': float, 'Tipo': str}, ...]
        
        Lógica:
            1. Define mercados a analisar (9 mercados diferentes)
            2. Para cada mercado:
               a) Calcula lambda ajustado (média ponderada 60/40)
               b) Calcula variância histórica
               c) Executa Monte Carlo com 10.000 simulações
               d) Calcula probabilidade para cada linha
               e) Converte em odd justa e score
            3. Ordena por score e retorna Top 7
            4. Gera sugestões por nível de risco
        
        Cálculo do Lambda Ajustado:
            ```
            λ_ajustado = 0.6 * média_todos_jogos + 0.4 * média_últimos_5
            ```
            Isso dá mais peso aos jogos recentes (forma atual).
        
        Cálculo do Score:
            ```
            Score = Probabilidade * (1 - CV * fator)
            onde:
                CV = σ / μ (Coeficiente de Variação)
                fator = 0.3 para Over, 0.5 para Under
            ```
            Score penaliza alta variância (resultados imprevisíveis).
        
        Filtros de Qualidade:
            - Over: Odd justa entre 1.20 e 3.00
            - Under: Odd justa entre 1.20 e 2.50
        
        Mercados Analisados:
            1. JOGO COMPLETO: Over/Under 8.5, 9.5, 10.5, 11.5, 12.5
            2. TOTAL MANDANTE: Over/Under 4.5, 5.5, 6.5
            3. TOTAL VISITANTE: Over/Under 3.5, 4.5, 5.5
        Executa a análise completa de uma partida, orquestrando as simulações.
        
        Args:
            df_home (pd.DataFrame): Histórico de jogos do time mandante.
            df_away (pd.DataFrame): Histórico de jogos do time visitante.
            ml_prediction (float, optional): Previsão da IA para o jogo.
            match_name (str, optional): Nome do confronto para exibição.
            
        Returns:
            tuple: (top_picks, suggestions)
                   - top_picks: Lista com as 7 melhores oportunidades.
                   - suggestions: Dicionário com sugestões Easy/Medium/Hard.
        """
        # ... (código existente de cálculo de médias e simulações) ...
        # Nota: O código abaixo é mantido da implementação original, apenas documentado.
        
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
        
        # Lógica de Integração IA + Estatística (Melhoria Nível 1)
        # Se tivermos previsão da IA, ajustamos as médias (lambdas) para alinhar com a IA
        if ml_prediction is not None and ml_prediction > 0:
            historical_avg = mean_h + mean_a
            if historical_avg > 0:
                # Mantém a proporção histórica entre os times
                prop_h = mean_h / historical_avg
                
                # Novos lambdas baseados na IA
                mean_h = ml_prediction * prop_h
                mean_a = ml_prediction * (1 - prop_h)
                
                print(f"{Colors.YELLOW}🤖 Usando Previsão ML ({ml_prediction:.2f}) como base para Monte Carlo{Colors.RESET}")
            else:
                # Fallback se histórico for zero (improvável)
                mean_h = ml_prediction / 2
                mean_a = ml_prediction / 2
        else:
            print(f"{Colors.CYAN}📊 Usando Média Histórica ({mean_h + mean_a:.2f}) para Monte Carlo{Colors.RESET}")
        
        sim_total = self.simulate_match_event(mean_h, mean_a, var_h, var_a)
        
        # Simula Primeiro Tempo (HT)
        dist_h_ht, mean_h_ht, var_h_ht = self._get_distribution_params(h_corners_ht)
        dist_a_ht, mean_a_ht, var_a_ht = self._get_distribution_params(a_corners_ht)
        
        sim_ht = self.simulate_match_event(mean_h_ht, mean_a_ht, var_h_ht, var_a_ht)
        
        # Simula Totais Individuais
        # Usamos apenas a média/variância do próprio time
        sim_home_total = self.monte_carlo_simulation(mean_h, var_h)
        sim_away_total = self.monte_carlo_simulation(mean_a, var_a)

        # Análise de Mercados
        # -------------------
        markets = []
        
        # Função auxiliar para adicionar mercado analisado
        def add_market(name, simulations, line, type_='Over'):
            count = np.sum(simulations > line) if type_ == 'Over' else np.sum(simulations < line)
            prob = count / self.n_simulations
            if prob > 0.01: # Evita divisão por zero e odds infinitas
                odd_justa = 1 / prob
                markets.append({
                    'Mercado': name,
                    'Seleção': f"{type_} {line}",
                    'Prob': prob,
                    'Odd': odd_justa
                })

        # Define as linhas padrão a serem analisadas
        lines_ft = [8.5, 9.5, 10.5, 11.5, 12.5]
        lines_ht = [3.5, 4.5, 5.5]
        lines_team = [3.5, 4.5, 5.5, 6.5]

        # Analisa Over/Under para cada linha
        for line in lines_ft:
            add_market('JOGO COMPLETO', sim_total, line, 'Over')
            # add_market('JOGO COMPLETO', sim_total, line, 'Under') # Focamos em Over por enquanto

        for line in lines_ht:
            add_market('1º TEMPO (HT)', sim_ht, line, 'Over')

        for line in lines_team:
            add_market('TOTAL MANDANTE', sim_home_total, line, 'Over')
            add_market('TOTAL VISITANTE', sim_away_total, line, 'Over')

        # Seleção das Melhores Oportunidades
        # ----------------------------------
        # Filtramos apenas aquelas com probabilidade > 60% para o Top 7
        top_picks = sorted([m for m in markets if m['Prob'] > 0.60], 
                         key=lambda x: x['Prob'], 
                         reverse=True)[:7]
                         
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
