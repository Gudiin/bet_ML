import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.data.external.manager import ExternalDataManager

def main():
    manager = ExternalDataManager()
    leagues = ['BRA', 'E0', 'SP1', 'D1', 'I1', 'F1']
    
    print("🌍 INICIANDO DOWNLOAD GLOBAL DE DADOS EXTERNOS...")
    for l in leagues:
        manager.download_data(l)
        
    print("\n✅ Download concluído.")

if __name__ == "__main__":
    main()
