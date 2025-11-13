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
            
            # Story elements & places
            'magi', 'there', 'here',
            
            # ADDED: Geographic/Political terms
            'alabama', 'federal', 'southern', 'northern', 'south', 'north',
            'east', 'west', 'yankee', 'yanks', 'confederate',
            
            # ADDED: Common false positives
            'owl', 'creek', 'bridge', 'dear', 'ill', 'ive', 'im', 'id', 
            'youre', 'hes', 'shes', 'theyre', 'weve', 'ive',
            'cousin', 'mother', 'father', 'brother', 'sister',
            
            # ADDED: Historical/mythological references
            'sheba', 'solomon', 'magi', 'wise men'
        }
        
        # Name parts that shouldn't be standalone
        self.name_parts_only = {
            'young', 'jr', 'sr', 'iii', 'iv', 'von', 'van', 'de', 'la', 'le',
            'dear'  # "John dear" -> just "John"
        }
        
    def extract_characters(self, text, sentences, min_mentions=2, detect_narrator=True):
        """
        Ekstraksi karakter dengan ALWAYS detect narrator untuk cerita orang pertama
        """
        print("\n[Character Extraction] Starting hybrid extraction...")
        
        # Standard extraction
        ner_characters = self._extract_via_ner(text)
        print(f"  ✓ NER found {len(ner_characters)} character mentions")
        
        pattern_characters = self._extract_via_patterns_strict(text, sentences)
        print(f"  ✓ Pattern matching found {len(pattern_characters)} character mentions")
        
        # Combine
        all_mentions = ner_characters + pattern_characters
        
        # Normalize
        normalized_mentions = []
        for name in all_mentions:
            normalized = self._normalize_name(name)
            if self._is_valid_name(normalized):
                normalized_mentions.append(normalized)
        
        print(f"  ✓ After normalization: {len(normalized_mentions)} mentions")
        
        # Count frequency
        raw_freq = Counter(normalized_mentions)
        
        # Filter
        filtered_freq = {}
        for name, count in raw_freq.items():
            name_lower = name.lower()
            
            if name_lower in self.non_character_words:
                continue
            
            if name_lower in self.name_parts_only and count < 5:
                continue
            
            if not self._is_common_word(name):
                filtered_freq[name] = count
        
        print(f"  ✓ After filtering: {len(filtered_freq)} unique characters")
        
        # Merge variants
        merged_freq = self._merge_with_full_name_priority(filtered_freq)
        print(f"  ✓ After merging: {len(merged_freq)} final characters")
        
        # Filter by min_mentions
        main_characters = {char: count for char, count in merged_freq.items()
                        if count >= min_mentions}
        
        # ALWAYS detect narrator for first-person narratives (for consistency)
        if detect_narrator:
            narrator_data = self._detect_narrator(text, sentences)
            if narrator_data:
                narrator_count = narrator_data.get('Narrator (I)', 0)
                
                # If significant first-person usage (20+ mentions), add narrator
                if narrator_count >= 20:
                    if 'Narrator (I)' not in main_characters:
                        main_characters['Narrator (I)'] = narrator_count
                        print(f"  ✓ First-person narrator detected: {narrator_count} mentions of 'I'")
                    
                    # Also add other role-based characters if found
                    for role, count in narrator_data.items():
                        if role != 'Narrator (I)' and count >= min_mentions:
                            if role not in main_characters:
                                main_characters[role] = count
                                print(f"  ✓ Role-based character detected: {role} ({count} mentions)")
        
        # Get contexts
        characters_with_context = self._add_context(sentences, main_characters)
        
        return {
            'all_entities': all_mentions,
            'raw_frequency': dict(raw_freq),
            'filtered_frequency': dict(filtered_freq),
            'main_characters': main_characters,
            'characters_with_context': characters_with_context
        }

    def _detect_narrator(self, text, sentences):
        """
        Detect unnamed first-person narrator dan karakter lain berdasarkan role
        """
        text_lower = text.lower()
        
        # Count first-person pronouns
        first_person_count = len(re.findall(r'\bi\b', text_lower))
        
        if first_person_count < 20:
            return {}
        
        # Detect narrator and other role-based characters
        characters = {}
        
        # Narrator (first person)
        characters['Narrator (I)'] = first_person_count
        
        # Look for role descriptions
        role_patterns = {
            'The Old Man': [r'\bold man\b', r'\bthe old man\b'],
            'The Officers': [r'\bofficers?\b', r'\bpolice\b'],
            'The Neighbor': [r'\bneighbou?rs?\b']
        }
        
        for role, patterns in role_patterns.items():
            count = 0
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                count += len(matches)
            
            if count >= 3:
                characters[role] = count
        
        return characters if len(characters) > 0 else {}
    
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
        Merge dengan PRIORITAS pada nama lengkap, dengan cleaning tambahan
        """
        # STEP 0: Pre-clean names (remove common suffixes)
        cleaned_freq = {}
        for name, count in character_freq.items():
            # Remove "dear" suffix
            cleaned_name = re.sub(r'\s+dear$', '', name, flags=re.IGNORECASE).strip()
            
            # If name becomes empty or too short, skip
            if len(cleaned_name) < 2:
                continue
            
            # Accumulate counts if same name after cleaning
            if cleaned_name in cleaned_freq:
                cleaned_freq[cleaned_name] += count
            else:
                cleaned_freq[cleaned_name] = count
        
        # Separate full names vs single names
        full_names = {name: count for name, count in cleaned_freq.items() 
                    if len(name.split()) > 1}
        single_names = {name: count for name, count in cleaned_freq.items() 
                    if len(name.split()) == 1}
        
        merged = {}
        processed = set()
        
        # STEP 1: Process full names first
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
            
            # Find variants (including possessives)
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
        Check if two names are variants - IMPROVED
        """
        n1_lower = name1.lower()
        n2_lower = name2.lower()
        
        # Exact match
        if n1_lower == n2_lower:
            return True
        
        # Possessive/plural: "John" vs "Johns"
        if n1_lower + 's' == n2_lower or n2_lower + 's' == n1_lower:
            return True
        
        # Possessive with apostrophe removed: "john" vs "johns"
        if n1_lower == n2_lower.rstrip('s') or n2_lower == n1_lower.rstrip('s'):
            # Only if difference is exactly 1 's'
            if abs(len(n1_lower) - len(n2_lower)) == 1:
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
            # First name match (min 4 chars to avoid false positives)
            if parts1[0] == parts2[0] and len(parts1[0]) >= 4:
                return True
        
        return False
        
    def _add_context(self, sentences, characters):
        """
        Add context for each character - ENHANCED untuk handle special names
        """
        character_contexts = defaultdict(list)
        
        for idx, sentence in enumerate(sentences):
            sent_lower = sentence.lower()
            
            for character in characters.keys():
                char_lower = character.lower()
                
                # SPECIAL: Handle "Narrator (I)" - look for "I" pronoun
                if char_lower == "narrator (i)":
                    # Count "I" as standalone word
                    pattern = r'\bi\b'
                    if re.search(pattern, sent_lower):
                        character_contexts[character].append({
                            'sentence_id': idx,
                            'sentence': sentence
                        })
                    continue
                
                # SPECIAL: Handle role-based characters like "The Old Man"
                if char_lower.startswith('the '):
                    # Remove "the" and search for the rest
                    role_name = char_lower.replace('the ', '')
                    pattern = r'\b' + re.escape(role_name) + r'\b'
                    if re.search(pattern, sent_lower):
                        character_contexts[character].append({
                            'sentence_id': idx,
                            'sentence': sentence
                        })
                    continue
                
                # Regular character names - direct mention
                # For multi-word names, check if all words present
                if ' ' in char_lower:
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