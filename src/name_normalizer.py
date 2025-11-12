import re
from collections import defaultdict

class NameNormalizer:
    """
    Kelas khusus untuk normalisasi dan pengelompokan nama karakter
    """
    
    def __init__(self):
        self.name_mapping = {}  # Mapping variant -> canonical name
        
    def normalize_and_group(self, names_with_counts):
        """
        Normalisasi dan kelompokkan variasi nama
        
        Args:
            names_with_counts: dict {name: count}
        
        Returns:
            dict {canonical_name: total_count}
        """
        # Step 1: Bersihkan possessive
        cleaned = {}
        for name, count in names_with_counts.items():
            clean_name = self._clean_name(name)
            if clean_name:
                if clean_name in cleaned:
                    cleaned[clean_name] += count
                else:
                    cleaned[clean_name] = count
        
        # Step 2: Group similar names
        grouped = self._group_similar_names(cleaned)
        
        return grouped
    
    def _clean_name(self, name):
        """
        Bersihkan nama dari possessive dan karakter aneh
        """
        if not name:
            return None
        
        # Remove possessive
        name = re.sub(r"'s$|s'$", "", name)
        
        # Remove extra whitespace
        name = ' '.join(name.split())
        
        # Capitalize properly
        name = ' '.join(word.capitalize() for word in name.split())
        
        return name.strip()
    
    def _group_similar_names(self, names_with_counts):
        """
        Kelompokkan nama yang mirip
        """
        grouped = {}
        processed = set()
        
        # Sort by count (descending)
        sorted_names = sorted(names_with_counts.items(), key=lambda x: x[1], reverse=True)
        
        for name, count in sorted_names:
            if name in processed:
                continue
            
            # Ini akan jadi canonical name
            canonical = name
            total_count = count
            processed.add(name)
            self.name_mapping[name] = canonical
            
            # Cari variants
            for other_name, other_count in sorted_names:
                if other_name in processed:
                    continue
                
                if self._are_variants(canonical, other_name):
                    total_count += other_count
                    processed.add(other_name)
                    self.name_mapping[other_name] = canonical
            
            grouped[canonical] = total_count
        
        return grouped
    
    def _are_variants(self, name1, name2):
        """
        Cek apakah dua nama adalah variant
        """
        n1 = name1.lower()
        n2 = name2.lower()
        
        # Exact match
        if n1 == n2:
            return True
        
        # Substring match (min 4 chars)
        if len(n1) >= 4 and len(n2) >= 4:
            if n1 in n2 or n2 in n1:
                return True
        
        # First name match
        parts1 = n1.split()
        parts2 = n2.split()
        
        if len(parts1) > 0 and len(parts2) > 0:
            if parts1[0] == parts2[0] and len(parts1[0]) >= 3:
                return True
        
        return False
    
    def get_canonical_name(self, name):
        """
        Dapatkan canonical name dari variant
        """
        clean = self._clean_name(name)
        return self.name_mapping.get(clean, clean)