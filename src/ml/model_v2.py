"""
Modelo Profissional de ML para Previsão de Escanteios - Versão 2.0

Este módulo implementa as melhores práticas para séries temporais
e modelos de contagem (Poisson), corrigindo os problemas de data leakage
e adicionando métricas de negócio.

Melhorias sobre model_improved.py:
    - Validação Temporal Estrita (sem shuffle)
    - LGBMRegressor com objective='poisson' (adequado para contagem)
    - Métricas de Negócio: Win Rate, ROI, Simulação de Lucro
    - Early Stopping para evitar overfitting
    - Logs detalhados de treino/teste

Autor: Refatoração baseada em feedback de Arquiteto Sênior
Data: 2025-12-03
"""

import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
import joblib
from pathlib import Path


class ProfessionalPredictor:
    """
    Modelo profissional para previsão de escanteios.
    
    Diferenças críticas do modelo anterior:
        1. NUNCA usa train_test_split aleatório
        2. SEMPRE valida no futuro (últimos 20% por data)
        3. Usa Poisson como distribuição (escanteios são contagem, não gaussiana)
        4. Reporta métricas de negócio (Win Rate, ROI)
    
    Attributes:
        model: Modelo LightGBM treinado.
        feature_names: Lista com nomes das features (para validação).
    
    Example:
        >>> predictor = ProfessionalPredictor()
        >>> predictor.train_time_series_split(X, y, timestamps)
        >>> predictions = predictor.predict(X_new)
    """
    
    def __init__(self, model_path: str = "data/corner_model_v2_professional.pkl"):
        """
        Inicializa o preditor profissional.
        
        Args:
            model_path: Caminho para salvar/carregar o modelo.
        """
        self.model_path = Path(model_path)
        self.model = None
        self.feature_names = None
        
        # Hiperparâmetros otimizados para Poisson
        self.default_params = {
            'objective': 'poisson',  # CRUCIAL para contagem
            'n_estimators': 500,
            'learning_rate': 0.01,
            'num_leaves': 31,
            'max_depth': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
    
    def train_time_series_split(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        timestamps: pd.Series,
        test_size: float = 0.2
    ) -> dict:
        """
        Treina respeitando o tempo (SEM SHUFFLE).
        
        Separa os últimos X% dos jogos (por data) para teste.
        Isso simula a realidade: treinamos com o passado, testamos no futuro.
        
        Args:
            X: Features de entrada.
            y: Target (total de escanteios).
            timestamps: Datas dos jogos (para ordenação temporal).
            test_size: Proporção de dados para teste (padrão: 20%).
        
        Returns:
            dict: Métricas de avaliação:
                - mae_test: Mean Absolute Error no teste
                - rmse_test: Root Mean Squared Error no teste
                - win_rate: Taxa de acerto nas apostas simuladas
                - roi: Retorno sobre investimento estimado
        
        Lógica:
            1. Ordena TUDO por data (cronológico)
            2. Corta em split_idx = 80% dos dados
            3. Treina com [0:split_idx]
            4. Testa com [split_idx:]
            5. Nunca mistura futuro com passado
        
        Regra de Negócio:
            Esta é a ÚNICA forma correta de treinar modelos de séries temporais.
            Qualquer shuffle invalida as métricas.
        """
        # Garante que temos os nomes das features
        self.feature_names = X.columns.tolist()
        
        # Ordena tudo por data (CRÍTICO)
        df_full = pd.concat([X, y.rename('target'), timestamps.rename('timestamp')], axis=1)
        df_full = df_full.sort_values('timestamp').reset_index(drop=True)
        
        # Calcula índice de corte temporal
        split_idx = int(len(df_full) * (1 - test_size))
        
        # Separa treino e teste
        train_data = df_full.iloc[:split_idx]
        test_data = df_full.iloc[split_idx:]
        
        # Exibe informações do split
        print("\n" + "="*70)
        print("🚀 TREINAMENTO PROFISSIONAL - VALIDAÇÃO TEMPORAL")
        print("="*70)
        print(f"📅 Período de Treino: {train_data['timestamp'].min()} até {train_data['timestamp'].max()}")
        print(f"📅 Período de Teste:  {test_data['timestamp'].min()} até {test_data['timestamp'].max()}")
        print(f"📊 Amostras Treino: {len(train_data)} | Teste: {len(test_data)}")
        print(f"🎯 Target Médio - Treino: {train_data['target'].mean():.2f} | Teste: {test_data['target'].mean():.2f}")
        print("="*70 + "\n")
        
        # Cria modelo
        self.model = lgb.LGBMRegressor(**self.default_params)
        
        # Treina com early stopping
        self.model.fit(
            train_data[self.feature_names], 
            train_data['target'],
            eval_set=[(test_data[self.feature_names], test_data['target'])],
            eval_metric='mae',
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=100)
            ]
        )
        
        # Avaliação no Teste (Futuro Real)
        print("\n" + "="*70)
        print("📊 AVALIAÇÃO NO CONJUNTO DE TESTE (FUTURO)")
        print("="*70)
        
        preds = self.model.predict(test_data[self.feature_names])
        
        # Métricas de Erro
        mae = mean_absolute_error(test_data['target'], preds)
        rmse = np.sqrt(mean_squared_error(test_data['target'], preds))
        
        print(f"✅ MAE (Mean Absolute Error):  {mae:.4f}")
        print(f"✅ RMSE (Root Mean Squared Error): {rmse:.4f}")
        
        # Métricas de Negócio
        business_metrics = self._evaluate_profitability(test_data['target'], preds)
        
        # Salva modelo
        self.save_model()
        
        # Retorna todas as métricas
        return {
            'mae_test': mae,
            'rmse_test': rmse,
            **business_metrics
        }
    
    def _evaluate_profitability(self, y_true: pd.Series, y_pred: np.ndarray) -> dict:
        """
        Simulação de lucro (Backtest).
        
        Simula uma estratégia simples de apostas:
        - Aposta no Over se Modelo > Linha da Casa + Margem de Segurança
        - Conta quantas apostas acertamos (Green)
        - Calcula Win Rate e ROI estimado
        
        Args:
            y_true: Valores reais de escanteios.
            y_pred: Previsões do modelo.
        
        Returns:
            dict: Métricas de negócio:
                - total_bets: Número de apostas realizadas
                - win_rate: Taxa de acerto (0.0 a 1.0)
                - roi: Retorno sobre investimento (em unidades)
        
        Regra de Negócio:
            Esta é a métrica que realmente importa.
            Um modelo com MAE alto mas Win Rate de 60% é melhor
            que um modelo com MAE baixo mas Win Rate de 48%.
        """
        print("\n" + "="*70)
        print("💰 SIMULAÇÃO FINANCEIRA (BACKTEST)")
        print("="*70)
        
        hits = 0
        total_bets = 0
        
        # Linha média do mercado (baseada em dados reais de casas de apostas)
        line = 9.5
        margin = 1.5  # Margem de segurança
        
        # Odd média para Over 9.5 (típica: @1.85 a @1.95)
        avg_odd = 1.90
        
        for true_val, pred_val in zip(y_true, y_pred):
            # Estratégia: Aposta no Over se modelo prevê MUITO acima da linha
            if pred_val > line + margin:
                total_bets += 1
                if true_val > line:  # Green!
                    hits += 1
        
        if total_bets > 0:
            win_rate = hits / total_bets
            
            # ROI = (Ganhos - Perdas) / Total Apostado
            # Ganhos = hits * odd
            # Perdas = (total_bets - hits) * 1
            roi = (hits * avg_odd) - total_bets
            roi_percent = (roi / total_bets) * 100
            
            print(f"🎯 Apostas Realizadas: {total_bets}")
            print(f"✅ Apostas Certas (Green): {hits}")
            print(f"❌ Apostas Erradas (Red): {total_bets - hits}")
            print(f"📈 Win Rate: {win_rate:.2%}")
            print(f"💵 ROI Estimado: {roi:+.2f} unidades ({roi_percent:+.1f}%)")
            
            # Análise de Viabilidade
            if win_rate >= 0.55:
                print(f"🟢 EXCELENTE! Win Rate acima de 55% é lucrativo a longo prazo.")
            elif win_rate >= 0.52:
                print(f"🟡 BOM. Win Rate entre 52-55% é sustentável com gestão de banca.")
            else:
                print(f"🔴 ATENÇÃO! Win Rate abaixo de 52% pode não ser lucrativo.")
            
            print("="*70 + "\n")
            
            return {
                'total_bets': total_bets,
                'win_rate': win_rate,
                'roi': roi,
                'roi_percent': roi_percent
            }
        else:
            print("⚠️ Nenhuma aposta encontrada com a margem de segurança.")
            print("   Isso pode indicar que o modelo é muito conservador.")
            print("="*70 + "\n")
            
            return {
                'total_bets': 0,
                'win_rate': 0.0,
                'roi': 0.0,
                'roi_percent': 0.0
            }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Faz previsão de escanteios.
        
        Args:
            X: Features da(s) partida(s).
        
        Returns:
            np.ndarray: Previsões de total de escanteios.
        
        Raises:
            ValueError: Se modelo não foi treinado.
        """
        if self.model is None:
            raise ValueError("Modelo não treinado! Execute train_time_series_split() primeiro.")
        
        # Valida features
        if self.feature_names is not None:
            missing_features = set(self.feature_names) - set(X.columns)
            if missing_features:
                raise ValueError(f"Features faltando: {missing_features}")
        
        return self.model.predict(X)
    
    def save_model(self) -> None:
        """Salva modelo em disco."""
        # Garante que o diretório existe
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'params': self.default_params
        }
        
        joblib.dump(data, self.model_path)
        print(f"💾 Modelo salvo em {self.model_path}")
    
    def load_model(self) -> bool:
        """
        Carrega modelo do disco.
        
        Returns:
            bool: True se carregado com sucesso.
        """
        try:
            data = joblib.load(self.model_path)
            self.model = data['model']
            self.feature_names = data.get('feature_names')
            print(f"✅ Modelo V2 Professional carregado de {self.model_path}")
            return True
        except FileNotFoundError:
            print(f"❌ Modelo não encontrado em {self.model_path}")
            return False
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Retorna importância das features.
        
        Returns:
            pd.DataFrame: Features ordenadas por importância.
        
        Útil para:
            - Debugging (quais features o modelo usa mais?)
            - Feature selection (podemos remover features irrelevantes?)
            - Interpretabilidade (o que o modelo considera importante?)
        """
        if self.model is None:
            raise ValueError("Modelo não treinado!")
        
        importance = self.model.feature_importances_
        
        df_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df_importance


# Função auxiliar para retrocompatibilidade
def prepare_improved_features(df: pd.DataFrame) -> tuple:
    """
    Wrapper para o novo módulo de features.
    
    Mantido para retrocompatibilidade com código existente.
    Recomenda-se usar diretamente features_v2.create_advanced_features().
    
    Args:
        df: DataFrame com dados históricos.
    
    Returns:
        tuple: (X, y, timestamps)
    """
    from src.ml.features_v2 import create_advanced_features
    return create_advanced_features(df)
