import json
import os
import difflib
import unicodedata

class TeamNameMapper:
    """
    Responsável por mapear nomes de times de fontes externas (Football-Data)
    para os nomes internos (SofaScore / DB).
    """
    
    MAPPING_FILE = os.path.join(os.path.dirname(__file__), 'mappings', 'team_map.json')
    
    def __init__(self):
        self.mapping = self._load_mapping()
        
    def _load_mapping(self):
        if os.path.exists(self.MAPPING_FILE):
            try:
                with open(self.MAPPING_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Erro ao carregar mapping: {e}. Iniciando vazio.")
                return {}
        return {}
        
    def save_mapping(self):
        os.makedirs(os.path.dirname(self.MAPPING_FILE), exist_ok=True)
        with open(self.MAPPING_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.mapping, f, indent=4, ensure_ascii=False)
            
    def normalize_name(self, name):
        """Normaliza string para comparação (minuscula, sem acentos)."""
        if not isinstance(name, str): return ""
        n = name.lower().strip()
        n = ''.join(c for c in unicodedata.normalize('NFD', n) if unicodedata.category(c) != 'Mn')
        # Remove sufixos comuns irrelevantes para matching
        replacements = [' fc', ' ec', ' sc', ' club', ' cf']
        for r in replacements:
            n = n.replace(r, '')
        return n.strip()
        
    def find_match(self, external_name, internal_candidates, threshold=0.6):
        """
        Encontra o melhor match para um nome externo dado uma lista de candidatos internos.
        Usa difflib (embutido no Python).
        """
        if external_name in self.mapping:
            return self.mapping[external_name]
            
        norm_ext = self.normalize_name(external_name)
        
        # 1. Tentativa Exata Normalizada
        for candidate in internal_candidates:
            if norm_ext == self.normalize_name(candidate):
                self.mapping[external_name] = candidate # Cache it automatically? Maybe strict verify first.
                return candidate
                
        # 2. Fuzzy Match
        matches = difflib.get_close_matches(external_name, internal_candidates, n=1, cutoff=threshold)
        if matches:
            return matches[0]
            
        return None

    def auto_map_league(self, external_names, internal_names, league_code):
        """
        Gera mapeamento automático para uma lista de times.
        """
        print(f"🔄 Mapeando times para {league_code} ({len(external_names)} externos vs {len(internal_names)} internos)...")
        
        mapped_count = 0
        for ext in external_names:
            if ext in self.mapping:
                continue
                
            match = self.find_match(ext, internal_names, threshold=0.7) # Strict threshold
            if match:
                self.mapping[ext] = match
                mapped_count += 1
            else:
                # Tenta normalizado manual se falhar
                pass
                
        self.save_mapping()
        print(f"   ✅ {mapped_count} novos mapeamentos automáticos salvos.")

