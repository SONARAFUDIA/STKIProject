"""
Character Extraction using Stanza NLP - Indonesian Version
COMPLETE REWRITE dari spaCy ke Stanza
"""

import stanza
from collections import Counter, defaultdict
import re

class CharacterExtractor:
    def __init__(self):
        """
        Inisialisasi dengan Stanza Indonesian model
        """
        print("🔧 Initializing Stanza Indonesian NLP...")
        
        try:
            # Initialize Stanza pipeline (NO NER - not available for Indonesian)
            # Use POS tagging instead for better name detection
            self.nlp = stanza.Pipeline(
                'id',  # Indonesian
                processors='tokenize,pos',  # Only tokenize and POS
                verbose=False,
                download_method=None
            )
            print("✅ Stanza Indonesian model loaded (POS tagging)")
            
        except Exception as e:
            print(f"❌ Error loading Stanza model: {e}")
            print("⚠️  Please run: python install_stanza.py")
            raise
        
        # Comprehensive Indonesian blacklist
        self.non_character_words = {
            # Pronouns
            'dia', 'ia', 'mereka', 'kami', 'kita', 'saya', 'aku', 'kamu', 'anda',
            'ku', 'mu', 'nya', 'kau', 'engkau',
            
            # Conjunctions & Prepositions
            'dan', 'atau', 'tetapi', 'tapi', 'namun', 'karena', 'sebab',
            'untuk', 'bagi', 'kepada', 'pada', 'di', 'ke', 'dari', 'oleh', 'dengan',
            'dalam', 'luar', 'atas', 'bawah', 'antara', 'hingga', 'sampai',
            'yang', 'ini', 'itu', 'tersebut', 'begitu', 'begini',
            
            # Common particles
            'adalah', 'ialah', 'merupakan', 'yaitu', 'yakni',
            'ada', 'tidak', 'bukan', 'belum', 'sudah', 'telah', 'akan', 'sedang',
            'juga', 'pun', 'lah', 'kah', 'hanya', 'saja', 'cuma',
            'sangat', 'amat', 'sekali', 'lebih', 'paling', 'terlalu',
            'masih', 'lagi', 'selalu', 'sering', 'kadang', 'jarang',
            
            # Question words
            'apa', 'siapa', 'kapan', 'dimana', 'kemana', 'mengapa', 'kenapa', 'bagaimana',
            'mana', 'berapa',
            
            # Time & Numbers
            'hari', 'minggu', 'bulan', 'tahun', 'jam', 'menit', 'detik',
            'pagi', 'siang', 'sore', 'malam', 'subuh', 'maghrib',
            'senin', 'selasa', 'rabu', 'kamis', 'jumat', 'sabtu', 'minggu',
            'januari', 'februari', 'maret', 'april', 'mei', 'juni',
            'juli', 'agustus', 'september', 'oktober', 'november', 'desember',
            'kemarin', 'sekarang', 'besok', 'lusa', 'dulu', 'nanti',
            'satu', 'dua', 'tiga', 'empat', 'lima', 'enam', 'tujuh', 'delapan',
            'sembilan', 'sepuluh', 'ratus', 'ribu', 'juta', 'miliar',
            
            # Common nouns (NOT names)
            'rumah', 'gedung', 'kantor', 'sekolah', 'tempat', 'ruang',
            'warung', 'kafe', 'restoran', 'toko', 'pasar', 'mall',
            'jalan', 'gang', 'lorong', 'pintu', 'jendela', 'dinding',
            'meja', 'kursi', 'lantai', 'atap', 'lemari',
            
            # Body parts (common false positives)
            'wajah', 'mata', 'hidung', 'mulut', 'tangan', 'kaki',
            'kepala', 'badan', 'tubuh', 'kulit', 'rambut',
            
            # Adjectives that appear capitalized
            'besar', 'kecil', 'tinggi', 'rendah', 'panjang', 'pendek',
            'baik', 'buruk', 'bagus', 'jelek', 'cantik', 'tampan',
            
            # Common verbs (stem forms)
            'ada', 'datang', 'pergi', 'pulang', 'tiba', 'sampai',
            'lihat', 'dengar', 'rasa', 'cium', 'sentuh',
            'duduk', 'berdiri', 'jalan', 'lari', 'tidur', 'bangun',
            
            # Sentence starters (often capitalized)
            'setelah', 'sebelum', 'ketika', 'saat', 'waktu',
            'kemudian', 'lalu', 'lantas', 'selanjutnya',
            'namun', 'tetapi', 'tapi', 'akan', 'jadi', 'maka',
            
            # Geographic (not person names)
            'jakarta', 'bandung', 'surabaya', 'yogyakarta', 'bali',
            'indonesia', 'jawa', 'sumatera', 'kalimantan', 'sulawesi',
            
            # Religious/mythological
            'tuhan', 'allah', 'yesus', 'nabi', 'rasul', 'malaikat',
            
            # Other common false positives
            'soal', 'masalah', 'hal', 'perkara', 'urusan',
            'kali', 'kalinya', 'sekali', 'dua kali',
            'maaf', 'tolong', 'silakan', 'terima kasih',
        }
        
        # Indonesian honorifics (titles, not standalone names)
        self.indonesian_honorifics = {
            'pak', 'bu', 'bapak', 'ibu', 'mas', 'mbak', 'bang', 'kang',
            'tante', 'om', 'kakak', 'adik', 'mbah', 'eyang', 'nek', 'kek',
            'haji', 'hajjah', 'ustadz', 'ustadzah', 'kyai',
            'raden', 'gusti', 'sultan', 'pangeran', 'putri',
            'dokter', 'dr', 'prof', 'profesor', 'drs', 'ir',
            'tuan', 'nyonya', 'nona', 'neng', 'dik',
        }
        
        # Common non-name POS tags
        self.non_name_pos = {
            'VERB', 'ADV', 'ADP', 'DET', 'CCONJ', 'SCONJ',
            'PRON', 'NUM', 'PART', 'INTJ'
        }
        
        # CRITICAL: Common verbs/nouns that appear at start of false positive names
        self.common_prefix_words = {
            # Verbs commonly before names
            'wajah', 'kata', 'ucap', 'jawab', 'tanya', 'sergah', 'sahut',
            'balas', 'timpal', 'tutur', 'beritahu', 'bilang',
            
            # Action verbs
            'lihat', 'dengar', 'rasa', 'pikir', 'ingat', 'lupa',
            
            # Possessives
            'milik', 'punya', 'kepunyaan',
            
            # Other common starters
            'soal', 'masalah', 'hal', 'perkara', 'urusan',
            'ide', 'gagasan', 'rencana', 'konsep',
        }
        
        # CRITICAL: Common nouns that should NEVER be names
        self.common_nouns_blacklist = {
            # Physical objects
            'cermin', 'jendela', 'pintu', 'meja', 'kursi', 'lemari',
            'gelas', 'piring', 'sendok', 'garpu', 'pisau',
            
            # Abstract nouns
            'ide', 'gagasan', 'pikiran', 'perasaan', 'emosi',
            'senyum', 'tawa', 'tangis', 'isak', 'sedu',
            'dering', 'bunyi', 'suara', 'bisik', 'desah',
            'hening', 'sunyi', 'senyap', 'diam',
            
            # Emotions/States
            'senang', 'sedih', 'marah', 'takut', 'bingung',
            'gembira', 'kecewa', 'lelah', 'capek',
            
            # Time/Events
            'revisi', 'final', 'meeting', 'rapat', 'pertemuan',
            'acara', 'kegiatan', 'agenda',
            
            # Titles/Descriptions
            'sang', 'si', 'para', 'kaum',
            
            # Body parts
            'mata', 'hidung', 'mulut', 'telinga', 'tangan', 'kaki',
            
            # Places (not person names)
            'warung', 'kafe', 'toko', 'pasar', 'rumah', 'gedung',
            
            # Roles (use with caution)
            'klien', 'client', 'customer', 'pelanggan',
            'donatur', 'donor', 'sponsor',
        }
        
        # CRITICAL: Nickname patterns (3-char names that might be nicknames)
        self.known_nicknames = {
            'rin': 'rina',
            'san': 'sandi',
            'bim': 'bima',
            'ani': 'anita',
            'adi': 'aditya',
            'rio': 'mario',
        }
    
    def extract_characters(self, text, sentences, min_mentions=2, detect_narrator=True):
        """
        Ekstraksi karakter menggunakan Stanza NER + Pattern Matching
        """
        print("\n[Character Extraction] Memulai Stanza-based extraction...")
        
        # Step 1: POS-based extraction (using Stanza POS tags)
        pos_based_characters = self._extract_via_pos_tagging(text)
        print(f"  ✓ POS-based extraction menemukan {len(pos_based_characters)} sebutan karakter")
        
        # Step 2: Pattern-based extraction (backup)
        pattern_characters = self._extract_via_patterns_indonesian(sentences)
        print(f"  ✓ Pattern matching menemukan {len(pattern_characters)} sebutan karakter")
        
        # Combine
        all_mentions = pos_based_characters + pattern_characters
        
        # Normalize
        normalized_mentions = []
        for name in all_mentions:
            normalized = self._normalize_name_indonesian(name)
            if self._is_valid_name_indonesian(normalized):
                normalized_mentions.append(normalized)
        
        print(f"  ✓ Setelah normalisasi: {len(normalized_mentions)} sebutan")
        
        # Count frequency
        raw_freq = Counter(normalized_mentions)
        
        # CRITICAL: Filter with quality checks
        filtered_freq = self._filter_with_quality_checks(raw_freq)
        print(f"  ✓ Setelah filtering: {len(filtered_freq)} karakter unik")
        
        # Merge variants (with strict rules)
        merged_freq = self._merge_indonesian_names_strict(filtered_freq)
        print(f"  ✓ Setelah merging: {len(merged_freq)} karakter final")
        
        # Filter by min_mentions
        main_characters = {char: count for char, count in merged_freq.items()
                          if count >= min_mentions}
        
        # Detect narrator for first-person narratives
        if detect_narrator:
            narrator_data = self._detect_narrator_indonesian(text, sentences)
            if narrator_data:
                narrator_count = narrator_data.get('Narator (Aku)', 0)
                if narrator_count >= 15:
                    if 'Narator (Aku)' not in main_characters:
                        main_characters['Narator (Aku)'] = narrator_count
                        print(f"  ✓ Narator orang pertama terdeteksi: {narrator_count} sebutan 'aku'")
        
        # Get contexts
        characters_with_context = self._add_context_indonesian(sentences, main_characters)
        
        return {
            'all_entities': all_mentions,
            'raw_frequency': dict(raw_freq),
            'filtered_frequency': dict(filtered_freq),
            'main_characters': main_characters,
            'characters_with_context': characters_with_context
        }
    
    def _extract_via_pos_tagging(self, text):
        """
        Extract names using POS tagging (PROPN = Proper Noun = Names)
        This is MORE ACCURATE than NER for Indonesian!
        """
        characters = []
        
        try:
            # Process dengan Stanza
            doc = self.nlp(text)
            
            # Extract PROPN (Proper Nouns) = potential names
            for sentence in doc.sentences:
                current_name = []
                
                for word in sentence.words:
                    # PROPN = Proper Noun (names, places, etc)
                    if word.upos == 'PROPN':
                        current_name.append(word.text)
                    else:
                        # End of name sequence
                        if current_name:
                            full_name = ' '.join(current_name)
                            
                            # Quality check
                            if self._is_quality_name(full_name):
                                characters.append(full_name)
                            
                            current_name = []
                
                # Handle name at end of sentence
                if current_name:
                    full_name = ' '.join(current_name)
                    if self._is_quality_name(full_name):
                        characters.append(full_name)
            
        except Exception as e:
            print(f"  ⚠️  Stanza POS error: {e}")
        
        return characters
    
    def _is_quality_name(self, name):
        """
        Quality check untuk nama yang dideteksi dari POS
        """
        if not name or len(name) < 2:
            return False
        
        # Max 4 words
        if len(name.split()) > 4:
            return False
        
        # Not in blacklist
        if name.lower() in self.non_character_words:
            return False
        
        # Not pure honorific
        if name.lower() in self.indonesian_honorifics:
            return False
        
        # Check if contains blacklisted words
        words = name.lower().split()
        for word in words:
            if word in ['yang', 'ini', 'itu', 'untuk', 'dengan', 'adalah']:
                return False
        
        return True
    
    def _extract_via_patterns_indonesian(self, sentences):
        """
        Pattern matching untuk backup Stanza NER
        """
        characters = []
        
        for sentence in sentences:
            words = sentence.split()
            
            # Pattern 1: Capitalized words di tengah kalimat (bukan awal)
            for i in range(1, len(words)):  # Skip first word
                word = words[i]
                
                # Basic checks
                if not word or len(word) < 3:
                    continue
                
                if not word[0].isupper():
                    continue
                
                clean_word = re.sub(r'[^\w\s-]', '', word)
                
                if not clean_word or clean_word.isupper():
                    continue
                
                # Check if likely a name
                if self._is_likely_indonesian_name(clean_word):
                    characters.append(clean_word)
            
            # Pattern 2: Honorific + Name (Pak Suroto, Bu Ani)
            for i in range(len(words) - 1):
                word1_clean = re.sub(r'[^\w\s]', '', words[i]).lower()
                word2_clean = re.sub(r'[^\w\s]', '', words[i + 1])
                
                if (word1_clean in self.indonesian_honorifics and 
                    word2_clean and 
                    word2_clean[0].isupper() and 
                    len(word2_clean) >= 3):
                    
                    # CRITICAL: Check if word2 is NOT a verb/common word
                    if not self._is_common_word_indonesian(word2_clean):
                        full_name = f"{words[i].capitalize()} {word2_clean}"
                        characters.append(full_name)
        
        return characters
    
    def _filter_with_quality_checks(self, raw_freq):
        """
        CRITICAL: Filter dengan quality checks yang ketat
        """
        filtered = {}
        
        for name, count in raw_freq.items():
            name_lower = name.lower()
            
            # Check 1: Blacklist
            if name_lower in self.non_character_words:
                continue
            
            # Check 2: Standalone honorifics
            if name_lower in self.indonesian_honorifics:
                if count < 5:  # Only keep if very frequent
                    continue
            
            # Check 3: Max word length (names rarely >4 words)
            if len(name.split()) > 4:
                continue
            
            # Check 4: Contains lowercase words in middle (likely not a name)
            words = name.split()
            if len(words) > 1:
                # Check middle words (not first, not last)
                middle_words = words[1:-1] if len(words) > 2 else []
                has_lowercase_middle = any(
                    w.islower() and w not in self.indonesian_honorifics 
                    for w in middle_words
                )
                if has_lowercase_middle:
                    # Exception: "dan" might appear in some names
                    if 'dan' not in name_lower:
                        continue
            
            # Check 5: Contains blacklisted words
            contains_blacklist = any(
                blacklisted in name_lower 
                for blacklisted in ['yang', 'ini', 'itu', 'adalah', 'untuk', 'dengan']
            )
            if contains_blacklist:
                continue
            
            # Check 6: Starts with common verbs/particles
            first_word = words[0].lower()
            if first_word in ['setelah', 'kemudian', 'lalu', 'ketika', 'saat', 'jika']:
                continue
            
            # Passed all checks
            filtered[name] = count
        
        return filtered
    
    def _merge_indonesian_names_strict(self, character_freq):
        """
        Merge dengan STRICT rules untuk avoid false positives
        """
        # Separate by type
        with_honorifics = {}
        full_names = {}
        single_names = {}
        
        for name, count in character_freq.items():
            words = name.split()
            
            if len(words) > 1 and words[0].lower() in self.indonesian_honorifics:
                with_honorifics[name] = count
            elif len(words) > 1:
                full_names[name] = count
            else:
                single_names[name] = count
        
        merged = {}
        processed = set()
        
        # STEP 1: Process names dengan honorifics FIRST (highest priority)
        for name_with_honor, count in sorted(with_honorifics.items(), 
                                            key=lambda x: x[1], reverse=True):
            if name_with_honor in processed:
                continue
            
            canonical = name_with_honor
            total_count = count
            processed.add(name_with_honor)
            
            # Extract base name
            words = name_with_honor.split()
            base_name = ' '.join(words[1:])
            base_lower = base_name.lower()
            
            # Merge with exact matching single names only
            for single_name, single_count in single_names.items():
                if single_name in processed:
                    continue
                
                if single_name.lower() == base_lower:
                    total_count += single_count
                    processed.add(single_name)
                    print(f"    → Merging '{single_name}' ke '{canonical}'")
            
            merged[canonical] = total_count
        
        # STEP 2: Process full names (no honorifics)
        for full_name, count in sorted(full_names.items(), 
                                       key=lambda x: x[1], reverse=True):
            if full_name in processed:
                continue
            
            canonical = full_name
            total_count = count
            processed.add(full_name)
            
            # STRICT: Only merge if single name is FIRST or LAST word of full name
            words = full_name.split()
            first_word = words[0].lower()
            last_word = words[-1].lower() if len(words) > 1 else ""
            
            for single_name, single_count in single_names.items():
                if single_name in processed:
                    continue
                
                single_lower = single_name.lower()
                
                # Only merge if exact match with first or last word
                if single_lower == first_word or single_lower == last_word:
                    total_count += single_count
                    processed.add(single_name)
                    print(f"    → Merging '{single_name}' ke '{canonical}'")
            
            merged[canonical] = total_count
        
        # STEP 3: Remaining single names
        for single_name, count in sorted(single_names.items(), 
                                        key=lambda x: x[1], reverse=True):
            if single_name in processed:
                continue
            
            merged[single_name] = count
            processed.add(single_name)
        
        return merged
    
    def _detect_narrator_indonesian(self, text, sentences):
        """
        Detect first-person narrator
        """
        text_lower = text.lower()
        
        aku_count = len(re.findall(r'\baku\b', text_lower))
        saya_count = len(re.findall(r'\bsaya\b', text_lower))
        
        first_person_count = aku_count + saya_count
        
        if first_person_count < 15:
            return {}
        
        return {'Narator (Aku)': first_person_count}
    
    def _is_likely_indonesian_name(self, word):
        """
        Check if word is likely an Indonesian name
        """
        if len(word) < 3:
            return False
        
        if word.isupper():
            return False
        
        if not word[0].isupper():
            return False
        
        if any(char.isdigit() for char in word):
            return False
        
        # Allow hyphens
        if not re.match(r'^[A-Z][a-z]+(?:-[A-Z][a-z]+)?$', word):
            return False
        
        # Check blacklist
        if word.lower() in self.non_character_words:
            return False
        
        return True
    
    def _is_valid_name_indonesian(self, name):
        """
        Validate name
        """
        if not name or len(name) < 2:
            return False
        
        if not any(c.isalpha() for c in name):
            return False
        
        if name.islower():
            return False
        
        return True
    
    def _is_common_word_indonesian(self, word):
        """
        Check if word is common (not a name)
        """
        return word.lower() in self.non_character_words
    
    def _normalize_name_indonesian(self, name):
        """
        Normalize Indonesian name
        """
        if not name:
            return ""
        
        # Remove possessive
        name = re.sub(r'(nya|mu|ku)$', '', name, flags=re.IGNORECASE)
        name = re.sub(r"'s$", "", name)
        
        # Remove punctuation at end
        name = re.sub(r'[.,!?;:]$', '', name)
        
        # Remove extra whitespace
        name = ' '.join(name.split())
        
        # Capitalize properly
        words = name.split()
        capitalized = []
        for word in words:
            if word.lower() in self.indonesian_honorifics:
                capitalized.append(word.capitalize())
            else:
                capitalized.append(word.capitalize())
        
        return ' '.join(capitalized).strip()
    
    def _add_context_indonesian(self, sentences, characters):
        """
        Add context for each character
        """
        character_contexts = defaultdict(list)
        
        for idx, sentence in enumerate(sentences):
            sent_lower = sentence.lower()
            
            for character in characters.keys():
                char_lower = character.lower()
                
                # Handle "Narator (Aku)"
                if char_lower == "narator (aku)":
                    if re.search(r'\b(aku|saya)\b', sent_lower):
                        character_contexts[character].append({
                            'sentence_id': idx,
                            'sentence': sentence
                        })
                    continue
                
                # Handle names with honorifics
                if ' ' in char_lower:
                    words = char_lower.split()
                    if all(re.search(r'\b' + re.escape(word) + r'\b', sent_lower) 
                          for word in words):
                        character_contexts[character].append({
                            'sentence_id': idx,
                            'sentence': sentence
                        })
                    continue
                
                # Single name
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