import sys
import os
import builtins

# Fix Windows Unicode Output
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Mock input before importing main
input_values = iter(["3", "2025-12-12"]) # Option 3 (Specific Date), The Date
def mock_input(prompt=""):
    print(f"{prompt}{next(input_values)}")
    return input_values.__next__()

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.main import scan_opportunities

# Monkey patch input in the module scope if needed, but imported function uses builtins.input
builtins.input = lambda prompt="": next(input_values)

if __name__ == "__main__":
    # We need to supply 2 inputs: "3" and "2025-12-12"
    # Wait, scan_opportunities asks:
    # 1. "Escolha: " -> 3
    # 2. "Digite a data (AAAA-MM-DD): " -> 2025-12-12
    
    input_values = iter(["3", "2025-12-12"])
    
    try:
        scan_opportunities()
    except StopIteration:
        pass
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
