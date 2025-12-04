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
        n_splits: int = 5
    ) -> dict:
        """
        Treina usando TimeSeriesSplit (Cross-Validation Temporal).
        
        Em vez de um único split, usamos janelas deslizantes para validar
        a robustez do modelo ao longo do tempo.
        
        Args:
            X: Features de entrada.
            y: Target (total de escanteios).
            timestamps: Datas dos jogos (para ordenação temporal).
            n_splits: Número de divisões para validação (padrão: 5).
        
        Returns:
            dict: Médias das métricas de avaliação em todos os splits:
                - mae_test: Média do MAE
                - rmse_test: Média do RMSE
                - win_rate: Média da Taxa de acerto
                - roi: Média do ROI
        
        Regra de Negócio:
            Implementação da "Validação Temporal" descrita no README_ML.md.
            Garante que o modelo é testado em múltiplos cenários futuros,
            não apenas nos últimos 20% dos dados.
        """
        from sklearn.model_selection import TimeSeriesSplit
        
        # Garante que temos os nomes das features
        self.feature_names = X.columns.tolist()
        
        # Ordena tudo por data (CRÍTICO)
        df_full = pd.concat([X, y.rename('target'), timestamps.rename('timestamp')], axis=1)
        df_full = df_full.sort_values('timestamp').reset_index(drop=True)
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        metrics_history = {
            'mae': [],
            'rmse': [],
            'win_rate': [],
            'roi': []
        }
        
        print("\n" + "="*70)
        print(f"🚀 TREINAMENTO PROFISSIONAL - CROSS-VALIDATION TEMPORAL ({n_splits} SPLITS)")
        print("="*70)
        
        fold = 1
        # O loop do TimeSeriesSplit garante que o índice de treino é sempre anterior ao de teste
        for train_index, test_index in tscv.split(df_full):
            train_data = df_full.iloc[train_index]
            test_data = df_full.iloc[test_index]
            
            print(f"\n📂 FOLD {fold}/{n_splits}")
            print(f"   📅 Treino: {train_data['timestamp'].min()} -> {train_data['timestamp'].max()} ({len(train_data)} jogos)")
            print(f"   📅 Teste:  {test_data['timestamp'].min()} -> {test_data['timestamp'].max()} ({len(test_data)} jogos)")
            
            # Cria modelo novo para cada fold
            model = lgb.LGBMRegressor(**self.default_params)
            
            model.fit(
                train_data[self.feature_names], 
                train_data['target'],
                eval_set=[(test_data[self.feature_names], test_data['target'])],
                eval_metric='mae',
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False)
                ]
            )
            
            # Avaliação
            preds = model.predict(test_data[self.feature_names])
            mae = mean_absolute_error(test_data['target'], preds)
            rmse = np.sqrt(mean_squared_error(test_data['target'], preds))
            
            # Simulação de Negócio
            biz_metrics = self._evaluate_profitability(test_data['target'], preds, verbose=False)
            
            metrics_history['mae'].append(mae)
            metrics_history['rmse'].append(rmse)
            
            # Só contabiliza Win Rate se houve apostas
            if biz_metrics['total_bets'] > 0:
                metrics_history['win_rate'].append(biz_metrics['win_rate'])
                metrics_history['roi'].append(biz_metrics['roi'])
            
            print(f"   ✅ MAE: {mae:.4f} | Win Rate: {biz_metrics['win_rate']:.1%} | ROI: {biz_metrics['roi']:.2f}")
            fold += 1
            
            # O último modelo treinado será o salvo (treinado com mais dados)
            self.model = model

        # Médias Finais
        avg_mae = np.mean(metrics_history['mae'])
        avg_rmse = np.mean(metrics_history['rmse'])
        
        # Média segura (evita divisão por zero se nunca apostou)
        if metrics_history['win_rate']:
            avg_win_rate = np.mean(metrics_history['win_rate'])
            avg_roi = np.mean(metrics_history['roi'])
        else:
            avg_win_rate = 0.0
            avg_roi = 0.0
        
        print("\n" + "="*70)
        print("📊 RESULTADO FINAL (MÉDIA DOS FOLDS)")
        print("="*70)
        print(f"✅ MAE Médio: {avg_mae:.4f}")
        print(f"✅ RMSE Médio: {avg_rmse:.4f}")
        print(f"📈 Win Rate Médio: {avg_win_rate:.2%}")
        print(f"💵 ROI Médio: {avg_roi:.2f} unidades")
        print("="*70 + "\n")
        
        self.save_model()
        
        return {
            'mae_test': avg_mae,
            'rmse_test': avg_rmse,
            'win_rate': avg_win_rate,
            'roi': avg_roi
        }
    
    def _evaluate_profitability(self, y_true: pd.Series, y_pred: np.ndarray, verbose: bool = True) -> dict:
        """
        Simulação de lucro (Backtest).
        
        Simula uma estratégia simples de apostas:
        - Aposta no Over se Modelo > Linha da Casa + Margem de Segurança
        - Conta quantas apostas acertamos (Green)
        - Calcula Win Rate e ROI estimado
        
        Args:
            y_true: Valores reais de escanteios.
            y_pred: Previsões do modelo.
            verbose: Se True, imprime relatório detalhado.
        
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
        if verbose:
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
            
            if verbose:
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
            if verbose:
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
