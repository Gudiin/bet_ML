import sys
import os
import inspect

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.web.server import app, get_leagues, api_update_database, api_train_model
    
    print("✅ Imports successful")
    
    # Check routes
    routes = [str(p) for p in app.url_map.iter_rules()]
    if '/api/leagues' in routes:
        print("✅ Route /api/leagues present")
    else:
        print("❌ Route /api/leagues missing")
        
    # Check function signatures/logic (basic check)
    if inspect.isfunction(get_leagues):
        print("✅ get_leagues function present")
        
    # Check index.html content
    index_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'web', 'templates', 'index.html')
    with open(index_path, 'r', encoding='utf-8') as f:
        content = f.read()
        if 'id="updateModal"' in content:
            print("✅ Update Modal present in index.html")
        else:
            print("❌ Update Modal missing in index.html")
            
        if 'id="trainModal"' in content:
            print("✅ Train Modal present in index.html")
        else:
            print("❌ Train Modal missing in index.html")
            
        if 'loadLeagues()' in content:
            print("✅ loadLeagues() call present in index.html")
        else:
            print("❌ loadLeagues() call missing in index.html")

except ImportError as e:
    print(f"❌ ImportError: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
