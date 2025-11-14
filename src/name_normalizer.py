import re
from collections import defaultdict

class NameNormalizer:
    """
    Kelas khusus untuk normalisasi dan pengelompokan nama karakter Indonesia
    """
    
    def __init__(self):
        self.name_mapping = {}  # Mapping variant -> canonical name
        
        # Indonesian honorifics & titles
        self.indonesian_honorifics = {
            'pak', 'bu', 'bapak', 'ibu', 'mas', 'mbak', 'bang', 'kang',
            'tante', 'om', 'kakak', 'adik', 'mbah', 'eyang',
            'haji', 'hajjah', 'ustadz', 'ustadzah', 'kyai',
            'raden', 'gusti', 'sultan', 'pangeran', 'putri',
            'dokter', 'dr', 'prof', 'profesor', 'drs', 'ir',
            'tuan', 'nyonya', 'nona'
        }
        
    def normalize_and_group(self, names_with_counts):
        """
        Normalisasi dan kelompokkan variasi nama
        
        Args:
            names_with_counts: dict {name: count}
        
        Returns:
            dict {canonical_name: total_count}
        """
        # Step 1: Bersihkan possessive dan normalize
        cleaned = {}
        for name, count in names_with_counts.items():
            clean_name = self._clean_name(name)
            if clean_name:
                if clean_name in cleaned:
                    cleaned[clean_name] += count
                else:
                    cleaned[clean_name] = count
        
        # Step 2: Group similar names (dengan honorifics)
        grouped = self._group_similar_names(cleaned)
        
        return grouped
    
    def _clean_name(self, name):
        """
        Bersihkan nama dari possessive dan karakter aneh
        Support Indonesian patterns
        """
        if not name:
            return None
        
        # Remove possessive suffix (-nya, -mu, -ku)
        name = re.sub(r'(nya|mu|ku)$', '', name, flags=re.IGNORECASE)
        
        # Remove English possessive ('s)
        name = re.sub(r"'s$|s'$", "", name)
        
        # Remove extra whitespace
        name = ' '.join(name.split())
        
        # Capitalize properly (penting untuk nama Indonesia)
        # Tapi pertahankan lowercase untuk honorifics tertentu
        words = name.split()
        cleaned_words = []
        for word in words:
            if word.lower() in self.indonesian_honorifics:
                # Honorifics capitalize first letter only
                cleaned_words.append(word.capitalize())
            else:
                # Regular names
                cleaned_words.append(word.capitalize())
        
        name = ' '.join(cleaned_words)
        
        return name.strip()
    
    def _group_similar_names(self, names_with_counts):
        """
        Kelompokkan nama yang mirip
        Handle Indonesian name patterns:
        - "Pak Hasan" vs "Hasan"
        - "Bima" vs "Bimo"
        - "Siti Nurbaya" vs "Siti"
        """
        grouped = {}
        processed = set()
        
        # Sort by count (descending) untuk prioritas nama yang sering muncul
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
        Special handling untuk nama Indonesia
        """
        n1 = name1.lower()
        n2 = name2.lower()
        
        # Exact match
        if n1 == n2:
            return True
        
        # Remove honorifics untuk comparison
        n1_no_honor = self._remove_honorifics(n1)
        n2_no_honor = self._remove_honorifics(n2)
        
        # Jika sama setelah honorifics dihapus
        if n1_no_honor == n2_no_honor and len(n1_no_honor) >= 3:
            return True
        
        # Substring match (min 4 chars) - hati-hati dengan nama pendek
        if len(n1_no_honor) >= 4 and len(n2_no_honor) >= 4:
            if n1_no_honor in n2_no_honor or n2_no_honor in n1_no_honor:
                return True
        
        # First name match untuk nama lengkap
        parts1 = n1_no_honor.split()
        parts2 = n2_no_honor.split()
        
        if len(parts1) > 0 and len(parts2) > 0:
            # Jika first name sama DAN cukup panjang (min 4 huruf)
            if parts1[0] == parts2[0] and len(parts1[0]) >= 4:
                return True
        
        # Check untuk variasi ejaan (Bima vs Bimo, Siti vs Sity)
        if self._is_spelling_variant(n1_no_honor, n2_no_honor):
            return True
        
        return False
    
    def _remove_honorifics(self, name):
        """
        Hapus honorifics dari nama
        """
        words = name.split()
        filtered = [w for w in words if w.lower() not in self.indonesian_honorifics]
        return ' '.join(filtered) if filtered else name
    
    def _is_spelling_variant(self, name1, name2):
        """
        Cek variasi ejaan (Bima/Bimo, Siti/Sity, dsb)
        """
        # Harus panjang minimal sama
        if abs(len(name1) - len(name2)) > 1:
            return False
        
        # Hitung berapa karakter yang berbeda
        if len(name1) == len(name2):
            diff_count = sum(c1 != c2 for c1, c2 in zip(name1, name2))
            # Max 1 karakter berbeda untuk nama pendek, 2 untuk panjang
            max_diff = 1 if len(name1) <= 5 else 2
            return diff_count <= max_diff
        
        return False
    
    def get_canonical_name(self, name):
        """
        Dapatkan canonical name dari variant
        """
        clean = self._clean_name(name)
        return self.name_mapping.get(clean, clean)
    
    def extract_honorific(self, name):
        """
        Ekstrak honorific dari nama (jika ada)
        Returns: (honorific, base_name)
        """
        words = name.lower().split()
        if len(words) > 1 and words[0] in self.indonesian_honorifics:
            return (words[0].capitalize(), ' '.join(words[1:]).title())
        return (None, name)