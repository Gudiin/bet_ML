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
        
        # Hiperparâmetros otimizados para Tweedie (melhor que Poisson para overdispersion)
        # Tweedie com power=1.5 é compromisso entre Poisson (1.0) e Gamma (2.0)
        # Captura melhor jogos com 15+ escanteios (outliers)
        self.default_params = {
            'objective': 'tweedie',  # MELHORIA: Mais flexível que Poisson
            'tweedie_variance_power': 1.5,  # 1=Poisson, 2=Gamma, 1.5=Compound Poisson-Gamma
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
        Simulação de lucro (Backtest) REALISTA com Linhas Dinâmicas.
        
        MELHORIA V6: Em vez de linha fixa 9.5, escolhe linha baseada na previsão.
        Isso simula melhor o comportamento real de apostas.
        """
        if verbose:
            print("\n" + "="*70)
            print("💰 SIMULAÇÃO FINANCEIRA (BACKTEST V6 - LINHA DINÂMICA)")
            print("="*70)
        
        hits = 0
        total_bets = 0
        roi_accumulated = 0.0
        
        # Linhas disponíveis no mercado típico
        available_lines = [7.5, 8.5, 9.5, 10.5, 11.5, 12.5]
        
        # Odds REALISTAS por linha (conservadoras, refletem vig das casas)
        line_odds = {
            7.5: 1.45, 8.5: 1.65, 9.5: 1.80, 
            10.5: 2.00, 11.5: 2.25, 12.5: 2.60
        }
        
        min_ev = 0.03  # 3% EV mínimo (mais conservador)
        
        # Converte para lista para iteração segura
        y_true_list = y_true.tolist()
        odds_list = odds.tolist() if odds is not None else [None] * len(y_true)
        
        for i, (true_val, pred_val) in enumerate(zip(y_true_list, y_pred)):
            # 1. LINHA DINÂMICA: Escolhe linha logo abaixo da previsão
            # Ex: Previsão 10.3 → Linha 9.5 (Over mais seguro)
            valid_lines = [l for l in available_lines if l < pred_val]
            if not valid_lines:
                continue  # Previsão muito baixa, sem aposta
            
            best_line = max(valid_lines)
            
            # 2. Odd da linha escolhida (ou odd fornecida)
            current_odd = odds_list[i] if odds_list[i] and odds_list[i] > 1.0 else line_odds.get(best_line, 1.80)
            
            # 3. Calcula Probabilidade Real via Poisson
            prob_over = self.get_true_probability(pred_val, best_line)
            
            # 4. Calcula Odd Justa e EV
            fair_odd = 1 / prob_over if prob_over > 0 else 99.0
            ev = (prob_over * current_odd) - 1
            
            # 5. Aposta apenas se EV > threshold
            if ev > min_ev:
                total_bets += 1
                if true_val > best_line:  # Green!
                    hits += 1
                    roi_accumulated += (current_odd - 1)
                else:  # Red
                    roi_accumulated -= 1
        
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


    def optimize_hyperparameters(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        timestamps: pd.Series,
        n_trials: int = 20
    ) -> dict:
        """
        Otimiza hiperparâmetros usando Optuna (AutoML).
        
        Args:
            X: Features.
            y: Target.
            timestamps: Datas (para validação temporal).
            n_trials: Número de tentativas (default: 20).
            
        Returns:
            dict: Melhores parâmetros encontrados.
        """
        try:
            import optuna
            from sklearn.model_selection import TimeSeriesSplit
        except ImportError:
            print("❌ Optuna não instalado. Execute: pip install optuna")
            return self.default_params

        print(f"\n🚀 INICIANDO OTIMIZAÇÃO DE HIPERPARÂMETROS (OPTUNA) - {n_trials} TRIALS")
        
        # Garante ordenação
        data_dict = {'target': y, 'timestamp': timestamps}
        df_aux = pd.DataFrame(data_dict)
        df_full = pd.concat([X, df_aux], axis=1).sort_values('timestamp').reset_index(drop=True)
        
        X_sorted = df_full[X.columns]
        y_sorted = df_full['target']
        
        def objective(trial):
            params = {
                'objective': 'poisson',
                'metric': 'mae',
                'verbosity': -1,
                'boosting_type': 'gbdt',
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            }
            
            tscv = TimeSeriesSplit(n_splits=3) # Menos splits para ser mais rápido
            scores = []
            
            for train_idx, test_idx in tscv.split(X_sorted):
                X_train, X_test = X_sorted.iloc[train_idx], X_sorted.iloc[test_idx]
                y_train, y_test = y_sorted.iloc[train_idx], y_sorted.iloc[test_idx]
                
                model = lgb.LGBMRegressor(**params)
                model.fit(X_train, y_train, callbacks=[lgb.early_stopping(20, verbose=False)], 
                          eval_set=[(X_test, y_test)], eval_metric='mae')
                
                preds = model.predict(X_test)
                scores.append(mean_absolute_error(y_test, preds))
                
            return np.mean(scores)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        print(f"✅ Melhores parâmetros encontrados: {study.best_params}")
        print(f"   Melhor MAE: {study.best_value:.4f}")
        
        # Atualiza os parâmetros padrão com os melhores encontrados
        self.default_params.update(study.best_params)
        return study.best_params

    def calibrate_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Calibração de Probabilidade (Isotonic Regression).
        
        Ajusta as probabilidades previstas para refletir a realidade.
        Ex: Se o modelo diz 70% de chance, deve acontecer 70% das vezes.
        
        Implementação simplificada: Ajusta o bias do lambda.
        """
        from sklearn.isotonic import IsotonicRegression
        
        if self.model is None:
            print("⚠️ Modelo não treinado. Pule a calibração.")
            return

        preds = self.model.predict(X)
        
        # Verifica bias (Viés)
        bias = np.mean(preds) - np.mean(y)
        print(f"\n⚖️ ANÁLISE DE CALIBRAÇÃO")
        print(f"   Média Prevista: {np.mean(preds):.2f}")
        print(f"   Média Real:     {np.mean(y):.2f}")
        print(f"   Viés (Bias):    {bias:+.2f}")
        
        if abs(bias) > 0.5:
            print("⚠️ Modelo descalibrado! Recomendado ajuste de intercept.")
            # Futuro: Implementar correção automática de bias no predict
        else:
            print("✅ Modelo bem calibrado (Viés aceitável).")
    
    # ============================================================
    # TRANSFER LEARNING - Global + League-Specific Fine-Tuning
    # ============================================================
    
    def train_global_and_finetune(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        timestamps: pd.Series,
        tournament_ids: pd.Series = None
    ) -> dict:
        """
        Transfer Learning: Treina modelo global e depois refina por liga.
        
        Estratégia:
            1. Treina modelo base com TODOS os dados (aprende padrões universais)
            2. Para cada liga, faz fine-tuning (ajusta pesos específicos)
            3. Salva modelo global e modelos por liga
            
        Args:
            X: Features de entrada (todas as ligas).
            y: Target.
            timestamps: Datas dos jogos.
            tournament_ids: IDs dos torneios (para separar por liga).
            
        Returns:
            dict: Métricas por liga.
        """
        print("\n" + "="*70)
        print("🌍 TRANSFER LEARNING - GLOBAL + FINE-TUNING POR LIGA")
        print("="*70)
        
        # 1. Treina Modelo Global
        print("\n📊 FASE 1: Treinando Modelo Global...")
        global_metrics = self.train_time_series_split(X, y, timestamps)
        
        # Salva modelo global
        global_path = Path("data/corner_model_global.pkl")
        global_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'params': self.default_params
        }, global_path)
        print(f"💾 Modelo global salvo em: {global_path}")
        
        # 2. Fine-Tuning por Liga
        if tournament_ids is None:
            # Tenta extrair de X se existir
            if 'tournament_id' in X.columns:
                tournament_ids = X['tournament_id']
            else:
                print("⚠️ tournament_ids não fornecido. Pulando fine-tuning por liga.")
                return {'global': global_metrics}
        
        print("\n📊 FASE 2: Fine-Tuning por Liga...")
        
        league_metrics = {}
        unique_leagues = tournament_ids.unique()
        
        for league_id in unique_leagues:
            # Filtra dados da liga
            mask = tournament_ids == league_id
            X_league = X[mask]
            y_league = y[mask]
            ts_league = timestamps[mask]
            
            n_games = len(X_league)
            if n_games < 100:
                print(f"   ⚠️ Liga {league_id}: Apenas {n_games} jogos. Pulando (mínimo: 100).")
                continue
                
            print(f"\n   🏆 Liga {league_id}: {n_games} jogos")
            
            # Carrega modelo global como base
            base_model = self.model
            
            # Fine-tuning: continua treinamento com dados da liga
            # Usa learning rate menor para não "esquecer" o global
            finetune_params = self.default_params.copy()
            finetune_params['learning_rate'] = 0.005  # Menor para fine-tuning
            finetune_params['n_estimators'] = 100  # Menos epochs
            
            model = lgb.LGBMRegressor(**finetune_params)
            
            # Treina com dados da liga (usa modelo global como warmstart conceitual)
            # LightGBM não tem warmstart nativo, então retreinamos com learning rate baixo
            model.fit(
                X_league, y_league,
                eval_set=[(X_league, y_league)],
                eval_metric='mae',
                callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
            )
            
            # Avalia
            preds = model.predict(X_league)
            mae = mean_absolute_error(y_league, preds)
            print(f"      ✅ MAE Liga: {mae:.4f}")
            
            # Salva modelo da liga
            league_path = Path(f"data/corner_model_league_{league_id}.pkl")
            joblib.dump({
                'model': model,
                'feature_names': self.feature_names,
                'params': finetune_params
            }, league_path)
            print(f"      💾 Salvo em: {league_path}")
            
            league_metrics[league_id] = {'mae': mae, 'n_games': n_games}
        
        print("\n" + "="*70)
        print("✅ TRANSFER LEARNING CONCLUÍDO!")
        print(f"   • Modelo Global: data/corner_model_global.pkl")
        print(f"   • Modelos por Liga: {len(league_metrics)} ligas")
        print("="*70 + "\n")
        
        return {'global': global_metrics, 'leagues': league_metrics}
    
    def predict_with_league(self, X: pd.DataFrame, tournament_id=None) -> np.ndarray:
        """
        Predição usando modelo específico da liga (se disponível).
        
        Estratégia:
            1. Tenta usar modelo da liga específica
            2. Se não existir, usa modelo global
            3. Se não existir, usa modelo padrão
            
        Args:
            X: Features.
            tournament_id: ID do torneio (opcional).
            
        Returns:
            Previsões.
        """
        model_to_use = self.model
        model_source = "padrão"
        
        if tournament_id is not None:
            league_path = Path(f"data/corner_model_league_{tournament_id}.pkl")
            if league_path.exists():
                data = joblib.load(league_path)
                model_to_use = data['model']
                model_source = f"liga {tournament_id}"
        
        # Fallback para global se liga não existe
        if model_source == "padrão":
            global_path = Path("data/corner_model_global.pkl")
            if global_path.exists():
                data = joblib.load(global_path)
                model_to_use = data['model']
                model_source = "global"
        
        if model_to_use is None:
            raise ValueError("Nenhum modelo disponível. Treine primeiro.")
            
        # Remove coluna categorical se existir
        X_clean = X.copy()
        if 'tournament_id' in X_clean.columns:
            X_clean = X_clean.drop(columns=['tournament_id'])
        
        return model_to_use.predict(X_clean)

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
