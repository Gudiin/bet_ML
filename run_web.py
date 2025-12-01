"""
Script de inicialização do servidor web.

Execute este arquivo para iniciar a interface web do sistema de previsão.
O terminal CLI continua funcional em src/main.py

Usage:
    python run_web.py              # Inicia na porta 5000
    python run_web.py --port 8080  # Inicia na porta 8080
    python run_web.py --debug      # Modo debug
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.web.server import run_server

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Servidor Web - Sistema de Previsão de Escanteios')
    parser.add_argument('--host', default='127.0.0.1', help='Host do servidor (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=5000, help='Porta do servidor (default: 5000)')
    parser.add_argument('--debug', action='store_true', help='Modo debug')
    
    args = parser.parse_args()
    
    run_server(host=args.host, port=args.port, debug=args.debug)
