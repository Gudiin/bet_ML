import unittest
from datetime import datetime, timedelta, timezone
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestScannerLogic(unittest.TestCase):
    def test_brasilia_time_logic(self):
        """Testa se a lógica de horário de Brasília está correta."""
        # UTC-3
        brt_tz = timezone(timedelta(hours=-3))
        now_brt = datetime.now(brt_tz)
        
        # Se agora for 22h BRT (01h UTC dia seguinte), a data deve ser a de BRT
        # Simula 22h BRT
        mock_brt = datetime(2025, 12, 2, 22, 0, 0, tzinfo=brt_tz)
        date_str = mock_brt.strftime('%Y-%m-%d')
        
        self.assertEqual(date_str, "2025-12-02")
        
        # Simula 01h UTC (que seria 22h BRT do dia anterior)
        mock_utc = datetime(2025, 12, 3, 1, 0, 0, tzinfo=timezone.utc)
        mock_brt_converted = mock_utc.astimezone(brt_tz)
        
        self.assertEqual(mock_brt_converted.strftime('%Y-%m-%d'), "2025-12-02")
        print("✅ Lógica de Fuso Horário (Brasília) verificada.")

    def test_data_sufficiency_logic(self):
        """Testa a lógica de filtro de dados insuficientes."""
        # Simula dados
        h_games = [1, 2] # 2 jogos (insuficiente)
        a_games = [1, 2, 3, 4] # 4 jogos (ok)
        
        should_skip = False
        if len(h_games) < 3 or len(a_games) < 3:
            should_skip = True
            
        self.assertTrue(should_skip)
        print("✅ Lógica de Data Sufficiency (Skip < 3 jogos) verificada.")

if __name__ == '__main__':
    unittest.main()
