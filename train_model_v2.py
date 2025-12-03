"""
Script de Treinamento - Modelo Profissional V2

Este script demonstra como treinar o novo modelo profissional
com as melhorias implementadas:
    1. Features vetorizadas (features_v2.py)
    2. Validação temporal estrita (model_v2.py)
    3. Métricas de negócio (Win Rate, ROI)

Uso:
    python train_model_v2.py

Autor: Refatoração baseada em feedback de Arquiteto Sênior
Data: 2025-12-03
"""

import pandas as pd
import sys
from pathlib import Path

# Adiciona src ao path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.ml.features_v2 import create_advanced_features
from src.ml.model_v2 import ProfessionalPredictor


def load_historical_data() -> pd.DataFrame:
    """
    Carrega dados históricos de partidas do banco SQLite.
    
    Returns:
        pd.DataFrame: Dados históricos prontos para processamento.
    """
    print("📂 Carregando dados históricos do banco SQLite...")
    
    # Importa o gerenciador de banco
    sys.path.insert(0, str(Path(__file__).parent / 'src'))
    from src.database.db_manager import DBManager
    
    # Conecta ao banco
    db = DBManager()
    
    # Carrega dados históricos (apenas jogos finalizados)
    df = db.get_historical_data()
    
    if df.empty:
        raise ValueError(
            "Nenhum dado histórico encontrado no banco! "
            "Execute o scraper primeiro para coletar dados."
        )
    
    # Converte timestamp Unix para datetime
    df['start_timestamp'] = pd.to_datetime(df['start_timestamp'], unit='s')
    
    # Renomeia colunas para compatibilidade com features_v2.py
    # O banco usa 'home_score'/'away_score', mas features_v2 espera 'goals_ft_home'/'goals_ft_away'
    if 'home_score' in df.columns and 'goals_ft_home' not in df.columns:
        df['goals_ft_home'] = df['home_score']
        df['goals_ft_away'] = df['away_score']
    
    print(f"✅ Carregados {len(df)} jogos finalizados")
    print(f"   Período: {df['start_timestamp'].min()} até {df['start_timestamp'].max()}")
    print(f"   Média de escanteios: {(df['corners_home_ft'] + df['corners_away_ft']).mean():.2f}")
    
    db.close()
    
    return df


def main():
    """Função principal de treinamento."""
    print("\n" + "="*70)
    print("🚀 TREINAMENTO DO MODELO PROFISSIONAL V2")
    print("="*70 + "\n")
    
    # 1. Carrega dados
    df = load_historical_data()
    
    # 2. Cria features vetorizadas
    print("\n🔧 Criando features vetorizadas...")
    X, y, timestamps = create_advanced_features(df, window_short=3, window_long=5)
    
    print(f"✅ Features criadas:")
    print(f"   Shape: {X.shape}")
    print(f"   Colunas: {list(X.columns)}")
    print(f"   Target médio: {y.mean():.2f} escanteios")
    
    # 3. Treina modelo com validação temporal
    print("\n🎯 Iniciando treinamento com validação temporal...")
    
    predictor = ProfessionalPredictor()
    metrics = predictor.train_time_series_split(X, y, timestamps, test_size=0.2)
    
    # 4. Exibe resumo final
    print("\n" + "="*70)
    print("📊 RESUMO FINAL DO TREINAMENTO")
    print("="*70)
    print(f"MAE (Teste):     {metrics['mae_test']:.4f}")
    print(f"RMSE (Teste):    {metrics['rmse_test']:.4f}")
    print(f"Win Rate:        {metrics['win_rate']:.2%}")
    print(f"ROI:             {metrics['roi']:+.2f} unidades ({metrics['roi_percent']:+.1f}%)")
    print(f"Total de Apostas: {metrics['total_bets']}")
    print("="*70)
    
    # 5. Exibe importância das features
    print("\n🔍 TOP 10 FEATURES MAIS IMPORTANTES:")
    print("="*70)
    importance = predictor.get_feature_importance()
    print(importance.head(10).to_string(index=False))
    print("="*70 + "\n")
    
    print("✅ Treinamento concluído com sucesso!")
    print(f"💾 Modelo salvo em: {predictor.model_path}")
    
    # 6. Análise de Viabilidade
    print("\n" + "="*70)
    print("💡 ANÁLISE DE VIABILIDADE")
    print("="*70)
    
    if metrics['win_rate'] >= 0.55:
        print("🟢 EXCELENTE! Este modelo tem potencial para ser lucrativo.")
        print("   Recomendação: Testar em ambiente de paper trading por 2-4 semanas.")
    elif metrics['win_rate'] >= 0.52:
        print("🟡 BOM. Este modelo pode ser lucrativo com gestão de banca adequada.")
        print("   Recomendação: Usar critério Kelly para dimensionar apostas.")
    else:
        print("🔴 ATENÇÃO! Win Rate abaixo do ideal.")
        print("   Recomendação: Coletar mais dados ou ajustar features antes de usar.")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
