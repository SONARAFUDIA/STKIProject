import spacy
from collections import Counter, defaultdict
import re

class CharacterExtractor:
    def __init__(self, model='en_core_web_lg'):
        """
        Inisialisasi dengan model spaCy
        """
        self.nlp = spacy.load(model)
        
        # Expanded blacklist
        self.non_character_words = {
            # Pronouns
            'he', 'she', 'they', 'we', 'i', 'you', 'it', 'his', 'her', 'their',
            'our', 'my', 'your', 'its', 'him', 'them', 'us', 'me',
            
            # Common words
            'the', 'and', 'but', 'or', 'for', 'with', 'at', 'from', 'by',
            'about', 'as', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'to', 'of', 'in', 'on', 'off', 'over', 'under',
            
            # Temporal words
            'christmas', 'today', 'tomorrow', 'yesterday', 'now', 'then',
            'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 
            'sunday', 'january', 'february', 'march', 'april', 'may', 'june', 
            'july', 'august', 'september', 'october', 'november', 'december',
            
            # Demonstratives & Articles
            'this', 'that', 'these', 'those', 'a', 'an',
            
            # Quantifiers
            'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
            'nine', 'ten', 'eleven', 'twelve', 'twenty', 'thirty', 'hundred',
            'thousand', 'some', 'many', 'few', 'all', 'each', 'every',
            
            # Common sentence starters
            'when', 'where', 'why', 'how', 'what', 'which', 'who', 'whose',
            'if', 'so', 'because', 'although', 'while', 'since',
            
            # Titles (not names themselves)
            'mr', 'mrs', 'miss', 'ms', 'dr', 'doctor', 'sir', 'madam', 'madame',
            'lord', 'lady', 'king', 'queen', 'prince', 'princess', 'mme',
            'majesty', 'majestys', 'highness',
            
            # Religious/Mythological
            'god', 'lord', 'christ', 'jesus',
            
            # Story elements
            'magi', 'there', 'here'
        }
        
        # Words that are likely part of names but not standalone characters
        self.name_parts_only = {
            'young', 'jr', 'sr', 'iii', 'iv', 'von', 'van', 'de', 'la', 'le'
        }
        
    def extract_characters(self, text, sentences, min_mentions=2):
        """
        Ekstraksi karakter dengan prioritas nama lengkap
        """
        print("\n[Character Extraction] Starting hybrid extraction...")
        
        # Method 1: spaCy NER
        ner_characters = self._extract_via_ner(text)
        print(f"  ✓ NER found {len(ner_characters)} character mentions")
        
        # Method 2: Pattern Matching
        pattern_characters = self._extract_via_patterns_strict(text, sentences)
        print(f"  ✓ Pattern matching found {len(pattern_characters)} character mentions")
        
        # Gabungkan
        all_mentions = ner_characters + pattern_characters
        
        # Normalisasi
        normalized_mentions = []
        for name in all_mentions:
            normalized = self._normalize_name(name)
            if self._is_valid_name(normalized):
                normalized_mentions.append(normalized)
        
        print(f"  ✓ After normalization: {len(normalized_mentions)} mentions")
        
        # Count frequency
        raw_freq = Counter(normalized_mentions)
        
        # Filter blacklist dan standalone name parts
        filtered_freq = {}
        for name, count in raw_freq.items():
            name_lower = name.lower()
            
            # Skip blacklist
            if name_lower in self.non_character_words:
                continue
            
            # Skip standalone name parts (kecuali frekuensi tinggi)
            if name_lower in self.name_parts_only and count < 5:
                continue
            
            # Skip common words
            if not self._is_common_word(name):
                filtered_freq[name] = count
        
        print(f"  ✓ After filtering: {len(filtered_freq)} unique characters")
        
        # CRITICAL: Merge dengan prioritas nama lengkap
        merged_freq = self._merge_with_full_name_priority(filtered_freq)
        print(f"  ✓ After merging: {len(merged_freq)} final characters")
        
        # Filter by min_mentions
        main_characters = {char: count for char, count in merged_freq.items()
                          if count >= min_mentions}
        
        # Get contexts
        characters_with_context = self._add_context(sentences, main_characters)
        
        return {
            'all_entities': all_mentions,
            'raw_frequency': dict(raw_freq),
            'filtered_frequency': dict(filtered_freq),
            'main_characters': main_characters,
            'characters_with_context': characters_with_context
        }
    
    def _extract_via_ner(self, text):
        """
        Ekstraksi menggunakan spaCy NER
        """
        doc = self.nlp(text)
        return [ent.text for ent in doc.ents if ent.label_ == 'PERSON']
    
    def _extract_via_patterns_strict(self, text, sentences):
        """
        Pattern matching dengan filter ketat
        """
        characters = []
        
        for sentence in sentences:
            words = sentence.split()
            
            for i, word in enumerate(words):
                # Skip first word
                if i == 0:
                    continue
                
                if not word or not word[0].isupper():
                    continue
                
                clean_word = re.sub(r'[^\w\s]', '', word)
                
                if not clean_word or len(clean_word) < 3:
                    continue
                
                if clean_word.isupper():
                    continue
                
                if self._is_likely_name(clean_word):
                    characters.append(clean_word)
        
        return characters
    
    def _is_likely_name(self, word):
        """
        Heuristic untuk nama
        """
        if len(word) < 3:
            return False
        
        if word.isupper():
            return False
        
        if not word[0].isupper():
            return False
        
        if any(char.isdigit() for char in word):
            return False
        
        if not re.match(r'^[A-Z][a-z]+(?:-[A-Z][a-z]+)?$', word):
            return False
        
        return True
    
    def _is_valid_name(self, name):
        """
        Validasi nama
        """
        if not name or len(name) < 2:
            return False
        
        if not any(c.isalpha() for c in name):
            return False
        
        if name.islower():
            return False
        
        return True
    
    def _is_common_word(self, word):
        """
        Cek common word
        """
        word_lower = word.lower()
        
        if word_lower in self.non_character_words:
            return True
        
        common_patterns = [
            r'^and$', r'^but$', r'^or$', r'^the$', r'^so$',
            r'^for$', r'^nor$', r'^yet$', r'^a$', r'^an$'
        ]
        
        for pattern in common_patterns:
            if re.match(pattern, word_lower):
                return True
        
        return False
    
    def _normalize_name(self, name):
        """
        Normalisasi nama
        """
        if not name:
            return ""
        
        # Remove possessive
        name = re.sub(r"'s$", "", name)
        
        # Remove punctuation at end
        name = re.sub(r'[.,!?;:]$', '', name)
        
        # Remove extra whitespace
        name = ' '.join(name.split())
        
        # Capitalize properly
        name = ' '.join(word.capitalize() for word in name.split())
        
        return name.strip()
    
    def _merge_with_full_name_priority(self, character_freq):
        """
        Merge dengan PRIORITAS pada nama lengkap
        Nama lengkap (multi-word) diprioritaskan daripada single word
        """
        merged = {}
        processed = set()
        
        # Separate full names vs single names
        full_names = {name: count for name, count in character_freq.items() 
                      if len(name.split()) > 1}
        single_names = {name: count for name, count in character_freq.items() 
                       if len(name.split()) == 1}
        
        # STEP 1: Process full names first (they have priority)
        for full_name, count in sorted(full_names.items(), key=lambda x: x[1], reverse=True):
            if full_name in processed:
                continue
            
            canonical = full_name
            total_count = count
            processed.add(full_name)
            
            # Find single names that are part of this full name
            full_name_parts = set(full_name.lower().split())
            
            for single_name, single_count in single_names.items():
                if single_name in processed:
                    continue
                
                single_lower = single_name.lower()
                
                # If single name is part of full name, merge it
                if single_lower in full_name_parts:
                    total_count += single_count
                    processed.add(single_name)
                    print(f"    → Merging '{single_name}' into '{canonical}'")
            
            # Check for variants of full name
            for other_full, other_count in full_names.items():
                if other_full in processed:
                    continue
                
                if self._is_name_variant(canonical, other_full):
                    total_count += other_count
                    processed.add(other_full)
                    print(f"    → Merging '{other_full}' into '{canonical}'")
            
            merged[canonical] = total_count
        
        # STEP 2: Process remaining single names
        for single_name, count in sorted(single_names.items(), key=lambda x: x[1], reverse=True):
            if single_name in processed:
                continue
            
            canonical = single_name
            total_count = count
            processed.add(single_name)
            
            # Find variants of this single name
            for other_single, other_count in single_names.items():
                if other_single in processed:
                    continue
                
                if self._is_name_variant(canonical, other_single):
                    total_count += other_count
                    processed.add(other_single)
                    print(f"    → Merging '{other_single}' into '{canonical}'")
            
            merged[canonical] = total_count
        
        return merged
    
    def _is_name_variant(self, name1, name2):
        """
        Check if two names are variants
        """
        n1_lower = name1.lower()
        n2_lower = name2.lower()
        
        if n1_lower == n2_lower:
            return True
        
        # Possessive/plural
        if n1_lower + 's' == n2_lower or n2_lower + 's' == n1_lower:
            return True
        
        # Substring match (min 4 chars, reasonable length difference)
        if len(n1_lower) >= 4 and len(n2_lower) >= 4:
            if n1_lower in n2_lower or n2_lower in n1_lower:
                if abs(len(n1_lower) - len(n2_lower)) <= 3:
                    return True
        
        # First name match for multi-word names
        parts1 = n1_lower.split()
        parts2 = n2_lower.split()
        
        if len(parts1) > 0 and len(parts2) > 0:
            if parts1[0] == parts2[0] and len(parts1[0]) >= 4:
                return True
        
        return False
    
    def _add_context(self, sentences, characters):
        """
        Add context
        """
        character_contexts = defaultdict(list)
        
        for idx, sentence in enumerate(sentences):
            sent_lower = sentence.lower()
            
            for character in characters.keys():
                char_lower = character.lower()
                
                # For multi-word names, check if all words present
                if ' ' in char_lower:
                    # Check if all parts of name appear in sentence
                    parts = char_lower.split()
                    if all(re.search(r'\b' + re.escape(part) + r'\b', sent_lower) for part in parts):
                        character_contexts[character].append({
                            'sentence_id': idx,
                            'sentence': sentence
                        })
                        continue
                
                # For single names
                pattern = r'\b' + re.escape(char_lower) + r'\b'
                if re.search(pattern, sent_lower):
                    character_contexts[character].append({
                        'sentence_id': idx,
                        'sentence': sentence
                    })
        
        return dict(character_contexts)
    
    def get_character_statistics(self, extraction_result):
        """
        Get statistics
        """
        main_chars = extraction_result['main_characters']
        
        if not main_chars:
            return {
                'total_characters': 0,
                'most_mentioned': None,
                'average_mentions': 0,
                'character_list': []
            }
        
        return {
            'total_characters': len(main_chars),
            'most_mentioned': max(main_chars.items(), key=lambda x: x[1]),
            'average_mentions': sum(main_chars.values()) / len(main_chars),
            'character_list': sorted(main_chars.keys())
        }