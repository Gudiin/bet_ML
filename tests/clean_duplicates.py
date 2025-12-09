"""
Script para Limpar Duplicatas do Banco de Dados

Execute este script para remover previsões duplicadas.
"""

import sqlite3
import sys

def clean_duplicates(match_id=None):
    """
    Remove previsões duplicadas do banco de dados.
    
    Args:
        match_id: Se fornecido, limpa apenas este jogo. Caso contrário, limpa tudo.
    """
    conn = sqlite3.connect('data/football_data.db')
    cursor = conn.cursor()
    
    try:
        # Backup automático
        print("📦 Criando backup...")
        import os
        if os.path.exists('data/football_data_backup.db'):
            os.remove('data/football_data_backup.db')
        cursor.execute("VACUUM INTO 'data/football_data_backup.db'")
        print("✅ Backup criado: data/football_data_backup.db")
        
        # Conta duplicatas antes
        if match_id:
            cursor.execute('''
                SELECT COUNT(*) FROM predictions WHERE match_id = ?
            ''', (match_id,))
            before = cursor.fetchone()[0]
            print(f"\n📊 Jogo {match_id}: {before} previsões antes da limpeza")
        else:
            cursor.execute('SELECT COUNT(*) FROM predictions')
            before = cursor.fetchone()[0]
            print(f"\n📊 Total: {before} previsões antes da limpeza")
        
        # Remove duplicatas (mantém apenas a mais recente de cada tipo)
        if match_id:
            cursor.execute('''
                DELETE FROM predictions 
                WHERE match_id = ? AND rowid NOT IN (
                    SELECT MAX(rowid)
                    FROM predictions
                    WHERE match_id = ?
                    GROUP BY match_id, prediction_type, category, market
                )
            ''', (match_id, match_id))
        else:
            cursor.execute('''
                DELETE FROM predictions 
                WHERE rowid NOT IN (
                    SELECT MAX(rowid)
                    FROM predictions
                    GROUP BY match_id, prediction_type, category, market
                )
            ''')
        
        deleted = cursor.rowcount
        conn.commit()
        
        # Conta após limpeza
        if match_id:
            cursor.execute('''
                SELECT COUNT(*) FROM predictions WHERE match_id = ?
            ''', (match_id,))
            after = cursor.fetchone()[0]
            print(f"✅ Jogo {match_id}: {after} previsões após limpeza")
        else:
            cursor.execute('SELECT COUNT(*) FROM predictions')
            after = cursor.fetchone()[0]
            print(f"✅ Total: {after} previsões após limpeza")
        
        print(f"🗑️  Removidas: {deleted} duplicatas")
        
        # Otimiza o banco
        print("\n🔧 Otimizando banco de dados...")
        cursor.execute("VACUUM")
        print("✅ Banco otimizado!")
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        match_id = int(sys.argv[1])
        print(f"🎯 Limpando duplicatas do jogo {match_id}...")
        clean_duplicates(match_id)
    else:
        print("🌐 Limpando duplicatas de TODOS os jogos...")
        response = input("⚠️  Tem certeza? (s/n): ")
        if response.lower() == 's':
            clean_duplicates()
        else:
            print("❌ Cancelado.")
