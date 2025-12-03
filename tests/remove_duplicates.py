"""
Script para remover previsões duplicadas do banco de dados.

Remove duplicatas mantendo apenas a entrada mais antiga (menor ID)
baseado em: match_id, market e market_group.
"""

import sqlite3

DB_PATH = "data/football_data.db"

def remove_duplicates():
    """Remove previsões duplicadas do banco de dados."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # 1. Conta quantas duplicatas existem
        cursor.execute("""
            SELECT COUNT(*) - COUNT(DISTINCT match_id || market || market_group)
            FROM predictions
        """)
        num_duplicates = cursor.fetchone()[0]
        
        print(f"🔍 Duplicatas encontradas: {num_duplicates}")
        
        if num_duplicates == 0:
            print("✅ Nenhuma duplicata encontrada!")
            return
        
        # 2. Mostra exemplos de duplicatas
        cursor.execute("""
            SELECT match_id, market, market_group, COUNT(*) as count
            FROM predictions
            GROUP BY match_id, market, market_group
            HAVING COUNT(*) > 1
            LIMIT 5
        """)
        
        print("\n📋 Exemplos de duplicatas:")
        for row in cursor.fetchall():
            print(f"   Match ID: {row[0]}, Market: {row[1]}, Group: {row[2]}, Count: {row[3]}")
        
        # 3. Confirma com o usuário
        print(f"\n⚠️  Isso irá remover {num_duplicates} previsões duplicadas.")
        confirm = input("Deseja continuar? (s/n): ")
        
        if confirm.lower() != 's':
            print("❌ Operação cancelada.")
            return
        
        # 4. Remove duplicatas (mantém a mais antiga - menor ID)
        cursor.execute("""
            DELETE FROM predictions 
            WHERE id NOT IN (
                SELECT MIN(id) 
                FROM predictions 
                GROUP BY match_id, market, market_group
            )
        """)
        
        deleted = cursor.rowcount
        conn.commit()
        
        print(f"✅ {deleted} previsões duplicadas removidas com sucesso!")
        
        # 5. Verifica se ainda há duplicatas
        cursor.execute("""
            SELECT COUNT(*) - COUNT(DISTINCT match_id || market || market_group)
            FROM predictions
        """)
        remaining = cursor.fetchone()[0]
        
        if remaining == 0:
            print("✅ Banco de dados limpo! Nenhuma duplicata restante.")
        else:
            print(f"⚠️  Ainda existem {remaining} duplicatas (pode ser necessário executar novamente)")
            
    except Exception as e:
        print(f"❌ Erro: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    print("=" * 60)
    print("🧹 LIMPEZA DE PREVISÕES DUPLICADAS")
    print("=" * 60)
    remove_duplicates()
