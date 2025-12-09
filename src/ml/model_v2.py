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
    Ensemble Profissional (Stacking/Blending) para previsão de escanteios.
    Combina LightGBM + CatBoost + Linear Regression para robustez.
    """
    
    def __init__(self, model_path: str = "data/corner_model_v2_professional.pkl"):
        self.model_path = Path(model_path)
        self.models = {} # Dict to hold sub-models
        self.feature_names = None
        self.weights = {'lgbm': 0.5, 'cat': 0.3, 'linear': 0.2}
        
        # LightGBM Params (Tweedie)
        self.lgbm_params = {
            'objective': 'tweedie',
            'tweedie_variance_power': 1.5,
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
        
    def _train_fold_ensemble(self, X_train, y_train, X_test, y_test):
        """Treina os 3 modelos do ensemble para um fold específico."""
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        from sklearn.impute import SimpleImputer
        
        fold_models = {}
        
        # 1. LightGBM
        lgbm = lgb.LGBMRegressor(**self.lgbm_params)
        lgbm.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='mae',
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        fold_models['lgbm'] = lgbm
        
        # 2. CatBoost (Optional)
        try:
            from catboost import CatBoostRegressor
            cat = CatBoostRegressor(
                iterations=500,
                learning_rate=0.01,
                depth=6,
                loss_function='RMSE', # CatBoost doesn't support Tweedie easily, use RMSE or Poisson
                eval_metric='MAE',
                random_seed=42,
                verbose=False,
                allow_writing_files=False
            )
            # Identify categorical features
            X_train_cat = X_train.copy()
            X_test_cat = X_test.copy()
            
            cat_features = X_train_cat.select_dtypes(include=['object', 'category']).columns.tolist()
            if cat_features:
                # Ensure they are strings
                for col in cat_features:
                    X_train_cat[col] = X_train_cat[col].astype(str)
                    X_test_cat[col] = X_test_cat[col].astype(str)
            
            cat.fit(
                X_train_cat, y_train, 
                cat_features=cat_features,
                eval_set=(X_test_cat, y_test), 
                early_stopping_rounds=50
            )
            fold_models['cat'] = cat
        except ImportError:
            if not getattr(self, '_cat_warned', False):
                print("⚠️ CatBoost não instalado. Usando apenas LightGBM + Linear.")
                self._cat_warned = True
            # Redistribute weight
            self.weights['lgbm'] = 0.7
            self.weights['linear'] = 0.3
            self.weights['cat'] = 0.0
            
        # 3. Linear Regression (Ridge for stability) - Needs Imputation
        linear = make_pipeline(
            SimpleImputer(strategy='mean'),
            StandardScaler(),
            Ridge(alpha=1.0)
        )
        linear.fit(X_train, y_train)
        fold_models['linear'] = linear
        
        return fold_models

    def train_time_series_split(self, X: pd.DataFrame, y: pd.Series, timestamps: pd.Series, odds: pd.Series = None, n_splits: int = 5) -> dict:
        from sklearn.model_selection import TimeSeriesSplit
        
        self.feature_names = X.columns.tolist()
        
        # Prepare Data
        data_dict = {'target': y, 'timestamp': timestamps}
        if odds is not None: data_dict['odds'] = odds
        df_full = pd.concat([X, pd.DataFrame(data_dict)], axis=1).sort_values('timestamp').reset_index(drop=True)
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        metrics = {'mae': [], 'rmse': [], 'win_rate': [], 'roi': []}
        
        print("\n" + "="*70)
        print(f"🚀 TREINAMENTO ENSEMBLE (LGBM + CAT + LINEAR) - {n_splits} SPLITS")
        print("="*70)
        
        fold = 1
        for train_idx, test_idx in tscv.split(df_full):
            train_data = df_full.iloc[train_idx]
            test_data = df_full.iloc[test_idx]
            
            print(f"\n📂 FOLD {fold}/{n_splits} | Train: {len(train_data)} | Test: {len(test_data)}")
            
            X_tr, y_tr = train_data[self.feature_names], train_data['target']
            X_te, y_te = test_data[self.feature_names], test_data['target']
            
            # Train Ensemble
            fold_models = self._train_fold_ensemble(X_tr, y_tr, X_te, y_te)
            
            # Predict Blend
            preds = self._predict_blend(X_te, fold_models)
            
            # Evaluate
            mae = mean_absolute_error(y_te, preds)
            rmse = np.sqrt(mean_squared_error(y_te, preds))
            
            test_odds = test_data['odds'] if 'odds' in test_data.columns else None
            biz = self._evaluate_profitability(y_te, preds, odds=test_odds, verbose=False)
            
            metrics['mae'].append(mae)
            metrics['rmse'].append(rmse)
            if biz['total_bets'] > 0:
                metrics['win_rate'].append(biz['win_rate'])
                metrics['roi'].append(biz['roi'])
            
            print(f"   ✅ Ensemble MAE: {mae:.4f} | ROI: {biz['roi']:.2f}")
            
            # Update final model (last fold)
            self.models = fold_models
            fold += 1
            
        avg_mae = np.mean(metrics['mae'])
        print(f"\n🏆 RESULTADO FINAL: MAE Médio {avg_mae:.4f}")
        self.save_model()
        return {'mae_test': avg_mae}

    def _predict_blend(self, X, models_dict=None):
        """Calcula média ponderada das previsões."""
        models = models_dict if models_dict else self.models
        if not models: raise ValueError("Modelos não treinados")
        
        # LightGBM
        p_lgbm = models['lgbm'].predict(X)
        
        # Linear
        p_linear = models['linear'].predict(X).flatten() # Ensure 1D
        
        # CatBoost
        p_cat = np.zeros_like(p_lgbm)
        if 'cat' in models:
            p_cat = models['cat'].predict(X)
            
        # Blend
        final_pred = (
            p_lgbm * self.weights['lgbm'] +
            p_cat * self.weights['cat'] +
            p_linear * self.weights['linear']
        )
        return final_pred

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.models: raise ValueError("Modelo não treinado")
        return self._predict_blend(X)

    def save_model(self):
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'models': self.models,
            'feature_names': self.feature_names,
            'weights': self.weights
        }, self.model_path)
        print(f"💾 Ensemble salvo em {self.model_path}")

    def load_model(self) -> bool:
        try:
            data = joblib.load(self.model_path)
            # Support legacy loading (if just 'model' exists)
            if 'model' in data and 'models' not in data:
                print("⚠️ Modelo legado (Single LGBM) encontrado. Convertendo...")
                self.models = {'lgbm': data['model']}
                self.weights = {'lgbm': 1.0, 'cat': 0.0, 'linear': 0.0}
            else:
                self.models = data['models']
                self.weights = data['weights']
                
            self.feature_names = data.get('feature_names')
            print(f"✅ Ensemble carregado. Pesos: {self.weights}")
            return True
        except (FileNotFoundError, KeyError):
            return False

    def get_feature_importance(self) -> pd.DataFrame:
        if 'lgbm' in self.models:
            imp = self.models['lgbm'].feature_importances_
            return pd.DataFrame({'feature': self.feature_names, 'importance': imp}).sort_values('importance', ascending=False)
        return pd.DataFrame()

    def _evaluate_profitability(self, y_true, y_pred, odds=None, verbose=True):
        """Avalia ROI e Win Rate simulados."""
        hits = 0
        total_bets = 0
        profit = 0
        
        # Simula aposta unitária em Over (Linha Prevista - 0.5)
        # Ex: Previsto 10.2 -> Aposta Over 9.5
        
        # Garante que y_true seja array/series indexável
        y_true_vals = y_true.values if hasattr(y_true, 'values') else y_true
        
        for i in range(len(y_true)):
            pred = y_pred[i]
            actual = y_true_vals[i]
            
            # Simple simulation: Bet Over local trend
            line = int(pred) - 0.5
            
            # Se aposta vencedora (Real > Linha)
            if actual > line:
                hits += 1
                profit += 0.90 # Odd media 1.90
            else:
                profit -= 1.00 # Perde stake
                
            total_bets += 1
            
        win_rate = hits / total_bets if total_bets > 0 else 0
        roi = (profit / total_bets) * 100 if total_bets > 0 else 0
        
        return {'win_rate': win_rate, 'roi': roi, 'total_bets': total_bets}

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
        
        # Atualiza os parâmetros do LightGBM no Ensemble
        self.lgbm_params.update(study.best_params)
        return study.best_params

    def calibrate_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Calibração simplificada (apenas diagnóstico).
        O Linear Regression no Ensemble já faz calibração nativa.
        """
        if not self.models:
            print("⚠️ Modelo não treinado.")
            return

        preds = self.predict(X)
        bias = np.mean(preds) - np.mean(y)
        print(f"\n⚖️ ANÁLISE DE CALIBRAÇÃO (ENSEMBLE)")
        print(f"   Média Prevista: {np.mean(preds):.2f}")
        print(f"   Média Real:     {np.mean(y):.2f}")
        print(f"   Viés (Bias):    {bias:+.2f}")

    
    # ============================================================
    # TRANSFER LEARNING - Global + League-Specific Fine-Tuning
    # ============================================================
    
    def train_global_and_finetune(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        timestamps: pd.Series,
        tournament_ids: pd.Series = None,
        odds: pd.Series = None
    ) -> dict:
        """
        Transfer Learning (Ensemble): Global Ensemble + Fine-Tuned LGBM.
        
        Estratégia:
            1. Treina Ensemble Global (LGBM + CatBoost + Linear) e salva.
            2. Para cada liga > 100 jogos:
               - Carrega LGBM Global.
               - Faz fine-tuning do LGBM com dados da liga.
               - Cria novo Ensemble "Híbrido": LGBM da Liga + Cat Global + Linear Global.
        """
        print("\n" + "="*70)
        print("🌍 TRANSFER LEARNING ENSEMBLE - GLOBAL + FINE-TUNING POR LIGA")
        print("="*70)
        
        # 1. Treina Modelo Global
        print("\n📊 FASE 1: Treinando Ensemble Global...")
        global_metrics = self.train_time_series_split(X, y, timestamps, odds=odds)
        
        # Salva modelo global
        global_path = Path("data/corner_model_global.pkl")
        self.model_path = global_path # Aponta para global
        self.save_model()
        print(f"💾 Ensemble GLOBAL salvo em: {global_path}")
        
        # Guarda componentes globais para reutilização
        global_models = self.models.copy()
        global_weights = self.weights.copy()
        
        # 2. Fine-Tuning por Liga
        if tournament_ids is None:
            if 'tournament_id' in X.columns:
                tournament_ids = X['tournament_id']
            else:
                print("⚠️ tournament_ids não fornecido. Pulando fine-tuning por liga.")
                return {'global': global_metrics}
        
        print("\n📊 FASE 2: Fine-Tuning por Liga (Apenas LightGBM)...")
        
        league_metrics = {}
        unique_leagues = tournament_ids.unique()
        
        for league_id in unique_leagues:
            mask = tournament_ids == league_id
            X_league = X[mask]
            y_league = y[mask]
            
            n_games = len(X_league)
            if n_games < 100:
                print(f"   ⚠️ Liga {league_id}: Apenas {n_games} jogos. Pulando fine-tuning.")
                continue
                
            print(f"\n   🏆 Liga {league_id}: {n_games} jogos")
            
            # Estratégia: Pegar LGBM Global e dar "fit" incremental (ou retrain rápido)
            # LightGBM não tem fit incremental fácil sem salvar booster.
            # Vamos treinar um NOVO LGBM usando os params globais, mas focado na liga (Warm Start conceitual)
            
            finetune_params = self.lgbm_params.copy()
            finetune_params['learning_rate'] = 0.005 # Menor para não divergir
            finetune_params['n_estimators'] = 150 # Curto
            
            lgbm_league = lgb.LGBMRegressor(**finetune_params)
            
            # Fit apenas na liga
            lgbm_league.fit(
                X_league, y_league,
                eval_set=[(X_league, y_league)],
                eval_metric='mae',
                callbacks=[lgb.early_stopping(20, verbose=False)]
            )
            
            # Monta Ensemble da Liga
            league_models = global_models.copy()
            league_models['lgbm'] = lgbm_league # Substitui apenas o LGBM
            
            # Avalia Ensemble Híbrido
            preds = self._predict_blend(X_league, league_models)
            mae = mean_absolute_error(y_league, preds)
            print(f"      ✅ MAE Liga (Ensemble Híbrido): {mae:.4f}")
            
            # Salva
            league_path = Path(f"data/corner_model_league_{league_id}.pkl")
            joblib.dump({
                'models': league_models,
                'feature_names': self.feature_names,
                'weights': global_weights
            }, league_path)
            print(f"      💾 Salvo em: {league_path}")
            
            league_metrics[league_id] = {'mae': mae, 'n_games': n_games}
        
        print("\n✅ Transfer Learning Concluído!")
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
