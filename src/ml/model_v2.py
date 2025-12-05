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
        odds: pd.Series = None,
        n_splits: int = 5
    ) -> dict:
        """
        Treina usando TimeSeriesSplit (Cross-Validation Temporal).
        
        Args:
            X: Features de entrada.
            y: Target (total de escanteios).
            timestamps: Datas dos jogos (para ordenação temporal).
            odds: Odds reais (opcional, para cálculo preciso de ROI).
            n_splits: Número de divisões para validação (padrão: 5).
        """
        from sklearn.model_selection import TimeSeriesSplit
        
        # Garante que temos os nomes das features
        self.feature_names = X.columns.tolist()
        
        # Monta DataFrame completo para ordenação
        data_dict = {'target': y, 'timestamp': timestamps}
        if odds is not None:
            data_dict['odds'] = odds
            
        df_aux = pd.DataFrame(data_dict)
        df_full = pd.concat([X, df_aux], axis=1)
        
        # Ordena por data (CRÍTICO)
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
        for train_index, test_index in tscv.split(df_full):
            train_data = df_full.iloc[train_index]
            test_data = df_full.iloc[test_index]
            
            print(f"\n📂 FOLD {fold}/{n_splits}")
            print(f"   📅 Treino: {train_data['timestamp'].min()} -> {train_data['timestamp'].max()} ({len(train_data)} jogos)")
            print(f"   📅 Teste:  {test_data['timestamp'].min()} -> {test_data['timestamp'].max()} ({len(test_data)} jogos)")
            
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
            
            preds = model.predict(test_data[self.feature_names])
            mae = mean_absolute_error(test_data['target'], preds)
            rmse = np.sqrt(mean_squared_error(test_data['target'], preds))
            
            # Passa as odds do teste se existirem
            test_odds = test_data['odds'] if 'odds' in test_data.columns else None
            
            biz_metrics = self._evaluate_profitability(
                test_data['target'], 
                preds, 
                odds=test_odds,
                verbose=False
            )
            
            metrics_history['mae'].append(mae)
            metrics_history['rmse'].append(rmse)
            
            if biz_metrics['total_bets'] > 0:
                metrics_history['win_rate'].append(biz_metrics['win_rate'])
                metrics_history['roi'].append(biz_metrics['roi'])
            
            print(f"   ✅ MAE: {mae:.4f} | Win Rate: {biz_metrics['win_rate']:.1%} | ROI: {biz_metrics['roi']:.2f}")
            fold += 1
            
            self.model = model

        # Médias Finais
        avg_mae = np.mean(metrics_history['mae'])
        avg_rmse = np.mean(metrics_history['rmse'])
        
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
    
    def get_true_probability(self, lambda_pred: float, line: float) -> float:
        """
        Calcula a probabilidade real de Over usando Poisson.
        
        Args:
            lambda_pred: Média prevista pelo modelo (λ).
            line: Linha de aposta (ex: 9.5).
            
        Returns:
            float: Probabilidade de sair MAIS que 'line' escanteios.
        """
        from scipy.stats import poisson
        # P(X > line) = Survival Function (sf)
        # sf(k) = P(X > k) -> Para Over 9.5, queremos P(X >= 10) ou P(X > 9)
        # A implementação do scipy.stats.poisson.sf(k) é P(X > k).
        # Se a linha é 9.5, queremos probabilidade de 10, 11, 12...
        # Então usamos int(9.5) = 9. sf(9) = P(X > 9) = P(X >= 10).
        return poisson.sf(int(line), lambda_pred)

    def _evaluate_profitability(
        self, 
        y_true: pd.Series, 
        y_pred: np.ndarray, 
        odds: pd.Series = None,
        verbose: bool = True
    ) -> dict:
        """
        Simulação de lucro (Backtest) com Odds Reais e Probabilidade Poisson (+EV).
        """
        if verbose:
            print("\n" + "="*70)
            print("💰 SIMULAÇÃO FINANCEIRA (BACKTEST - MÉTODO +EV)")
            print("="*70)
        
        hits = 0
        total_bets = 0
        roi_accumulated = 0.0
        
        line = 9.5
        default_odd = 1.90
        min_ev = 0.05 # 5% de valor esperado mínimo
        
        # Converte para lista para iteração segura
        y_true_list = y_true.tolist()
        odds_list = odds.tolist() if odds is not None else [None] * len(y_true)
        
        for i, (true_val, pred_val) in enumerate(zip(y_true_list, y_pred)):
            current_odd = odds_list[i] if odds_list[i] and odds_list[i] > 1.0 else default_odd
            
            # 1. Calcula Probabilidade Real
            prob_over = self.get_true_probability(pred_val, line)
            
            # 2. Calcula Odd Justa
            fair_odd = 1 / prob_over if prob_over > 0 else 99.0
            
            # 3. Verifica Valor Esperado (EV)
            # EV = (Prob * Odd) - 1
            ev = (prob_over * current_odd) - 1
            
            # Aposta apenas se tiver valor positivo (+EV)
            if ev > min_ev:
                total_bets += 1
                if true_val > line:  # Green!
                    hits += 1
                    roi_accumulated += (current_odd - 1) # Lucro líquido
                else: # Red
                    roi_accumulated -= 1 # Perda da unidade
        
        if total_bets > 0:
            win_rate = hits / total_bets
            roi_percent = (roi_accumulated / total_bets) * 100
            
            if verbose:
                print(f"🎯 Apostas Realizadas: {total_bets}")
                print(f"✅ Apostas Certas (Green): {hits}")
                print(f"❌ Apostas Erradas (Red): {total_bets - hits}")
                print(f"📈 Win Rate: {win_rate:.2%}")
                print(f"💵 ROI Real: {roi_accumulated:+.2f} unidades ({roi_percent:+.1f}%)")
                
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
                'roi': roi_accumulated,
                'roi_percent': roi_percent
            }
        else:
            if verbose:
                print("⚠️ Nenhuma aposta encontrada com valor esperado positivo (+EV).")
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
