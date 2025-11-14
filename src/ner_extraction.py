import spacy
from collections import Counter, defaultdict
import re

class CharacterExtractor:
    def __init__(self, model='id_core_news_md'):
        """
        Inisialisasi dengan model spaCy Indonesia
        Fallback ke model English jika Indonesian tidak tersedia
        """
        try:
            self.nlp = spacy.load(model)
            print(f"✓ Model spaCy loaded: {model}")
        except OSError:
            print(f"⚠️  Model '{model}' tidak ditemukan. Mencoba 'id_core_news_sm'...")
            try:
                self.nlp = spacy.load('id_core_news_sm')
                print("✓ Model spaCy loaded: id_core_news_sm")
            except OSError:
                print("⚠️  Model Indonesia tidak tersedia. Menggunakan 'en_core_web_sm'")
                print("   Jalankan: python -m spacy download id_core_news_md")
                self.nlp = spacy.load('en_core_web_sm')
        
        # Expanded blacklist untuk bahasa Indonesia
        self.non_character_words = {
            # Kata ganti
            'dia', 'ia', 'mereka', 'kami', 'kita', 'saya', 'aku', 'kamu', 'anda',
            'ku', 'mu', 'nya', 'kau',
            
            # Kata sambung & preposisi
            'dan', 'atau', 'tetapi', 'tapi', 'namun', 'karena', 'sebab',
            'untuk', 'bagi', 'kepada', 'pada', 'di', 'ke', 'dari', 'oleh', 'dengan',
            'dalam', 'luar', 'atas', 'bawah', 'antara', 'hingga', 'sampai',
            'yang', 'ini', 'itu', 'tersebut',
            
            # Kata kerja bantu
            'adalah', 'ialah', 'merupakan', 'yaitu', 'yakni',
            'ada', 'tidak', 'bukan', 'belum', 'sudah', 'telah', 'akan', 'sedang',
            
            # Kata keterangan waktu
            'hari', 'minggu', 'bulan', 'tahun', 'pagi', 'siang', 'sore', 'malam',
            'senin', 'selasa', 'rabu', 'kamis', 'jumat', 'sabtu', 'minggu',
            'januari', 'februari', 'maret', 'april', 'mei', 'juni',
            'juli', 'agustus', 'september', 'oktober', 'november', 'desember',
            'kemarin', 'sekarang', 'besok', 'lusa', 'dulu', 'nanti',
            
            # Kata tanya
            'apa', 'siapa', 'kapan', 'dimana', 'kemana', 'mengapa', 'kenapa', 'bagaimana',
            
            # Angka
            'satu', 'dua', 'tiga', 'empat', 'lima', 'enam', 'tujuh', 'delapan',
            'sembilan', 'sepuluh', 'ratus', 'ribu', 'juta', 'miliar',
            
            # Tempat umum (bukan nama orang)
            'jakarta', 'bandung', 'surabaya', 'yogyakarta', 'bali',
            'indonesia', 'jawa', 'sumatera', 'kalimantan',
            'warung', 'kafe', 'restoran', 'toko', 'pasar', 'mall',
            
            # Kata umum lainnya
            'sebuah', 'suatu', 'para', 'semua', 'setiap', 'masing', 'tiap',
            'lah', 'kah', 'pun', 'saja', 'hanya', 'cuma',
            'sangat', 'amat', 'sekali', 'lebih', 'paling',
            
            # False positives umum
            'tuhan', 'allah', 'tuan', 'nyonya',
        }
        
        # Indonesian honorifics (bukan nama standalone)
        self.indonesian_honorifics = {
            'pak', 'bu', 'bapak', 'ibu', 'mas', 'mbak', 'bang', 'kang',
            'tante', 'om', 'kakak', 'adik', 'mbah', 'eyang', 'nek', 'kek',
            'haji', 'hajjah', 'ustadz', 'ustadzah', 'kyai',
            'raden', 'gusti', 'sultan', 'pangeran', 'putri',
            'dokter', 'dr', 'prof', 'profesor', 'drs', 'ir',
        }
        
    def extract_characters(self, text, sentences, min_mentions=2, detect_narrator=True):
        """
        Ekstraksi karakter dengan support bahasa Indonesia
        """
        print("\n[Character Extraction] Memulai hybrid extraction...")
        
        # Standard extraction
        ner_characters = self._extract_via_ner(text)
        print(f"  ✓ NER menemukan {len(ner_characters)} sebutan karakter")
        
        pattern_characters = self._extract_via_patterns_indonesian(text, sentences)
        print(f"  ✓ Pattern matching menemukan {len(pattern_characters)} sebutan karakter")
        
        # Combine
        all_mentions = ner_characters + pattern_characters
        
        # Normalize
        normalized_mentions = []
        for name in all_mentions:
            normalized = self._normalize_name_indonesian(name)
            if self._is_valid_name_indonesian(normalized):
                normalized_mentions.append(normalized)
        
        print(f"  ✓ Setelah normalisasi: {len(normalized_mentions)} sebutan")
        
        # Count frequency
        raw_freq = Counter(normalized_mentions)
        
        # Filter
        filtered_freq = {}
        for name, count in raw_freq.items():
            name_lower = name.lower()
            
            # Skip blacklisted words
            if name_lower in self.non_character_words:
                continue
            
            # Skip standalone honorifics
            if name_lower in self.indonesian_honorifics and count < 5:
                continue
            
            # Skip very common words
            if not self._is_common_word_indonesian(name):
                filtered_freq[name] = count
        
        print(f"  ✓ Setelah filtering: {len(filtered_freq)} karakter unik")
        
        # Merge variants (Indonesian-aware)
        merged_freq = self._merge_indonesian_names(filtered_freq)
        print(f"  ✓ Setelah merging: {len(merged_freq)} karakter final")
        
        # Filter by min_mentions
        main_characters = {char: count for char, count in merged_freq.items()
                          if count >= min_mentions}
        
        # Detect narrator untuk cerita orang pertama
        if detect_narrator:
            narrator_data = self._detect_narrator_indonesian(text, sentences)
            if narrator_data:
                narrator_count = narrator_data.get('Narator (Aku)', 0)
                
                # Jika penggunaan orang pertama signifikan
                if narrator_count >= 15:  # Threshold lebih rendah untuk Indonesia
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

    def _detect_narrator_indonesian(self, text, sentences):
        """
        Detect unnamed first-person narrator (orang pertama)
        """
        text_lower = text.lower()
        
        # Count first-person pronouns
        aku_count = len(re.findall(r'\baku\b', text_lower))
        saya_count = len(re.findall(r'\bsaya\b', text_lower))
        
        first_person_count = aku_count + saya_count
        
        if first_person_count < 15:  # Threshold untuk Indonesia
            return {}
        
        # Narator terdeteksi
        characters = {
            'Narator (Aku)': first_person_count
        }
        
        return characters
    
    def _extract_via_ner(self, text):
        """
        Ekstraksi menggunakan spaCy NER
        """
        doc = self.nlp(text)
        return [ent.text for ent in doc.ents if ent.label_ == 'PERSON']
    
    def _extract_via_patterns_indonesian(self, text, sentences):
        """
        Pattern matching dengan filter untuk nama Indonesia
        """
        characters = []
        
        for sentence in sentences:
            words = sentence.split()
            
            for i, word in enumerate(words):
                # Skip first word (biasanya bukan nama)
                if i == 0:
                    continue
                
                if not word or not word[0].isupper():
                    continue
                
                clean_word = re.sub(r'[^\w\s]', '', word)
                
                if not clean_word or len(clean_word) < 2:  # Min 2 huruf untuk Indonesia
                    continue
                
                if clean_word.isupper():  # Skip ALL CAPS
                    continue
                
                if self._is_likely_indonesian_name(clean_word):
                    characters.append(clean_word)
            
            # Check untuk nama dengan honorifics (Pak Suroto, Bu Ani)
            for i in range(len(words) - 1):
                word1 = re.sub(r'[^\w\s]', '', words[i]).lower()
                word2 = re.sub(r'[^\w\s]', '', words[i + 1])
                
                if word1 in self.indonesian_honorifics and word2 and word2[0].isupper():
                    if len(word2) >= 3:  # Min 3 huruf untuk nama sesudah gelar
                        full_name = f"{words[i].capitalize()} {word2}"
                        characters.append(full_name)
        
        return characters
    
    def _is_likely_indonesian_name(self, word):
        """
        Heuristic untuk nama Indonesia
        """
        if len(word) < 2:  # Min 2 huruf
            return False
        
        if word.isupper():
            return False
        
        if not word[0].isupper():
            return False
        
        if any(char.isdigit() for char in word):
            return False
        
        # Allow hyphenated names
        if not re.match(r'^[A-Z][a-z]+(?:-[A-Z][a-z]+)?$', word):
            return False
        
        # Skip jika terlalu pendek dan bukan nama umum
        if len(word) <= 2 and word.lower() not in ['lia', 'ari', 'ani', 'adi']:
            return False
        
        return True
    
    def _is_valid_name_indonesian(self, name):
        """
        Validasi nama Indonesia
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
        Cek common word Indonesia
        """
        word_lower = word.lower()
        
        if word_lower in self.non_character_words:
            return True
        
        # Common patterns
        common_patterns = [
            r'^dan$', r'^atau$', r'^yang$', r'^ini$', r'^itu$',
            r'^untuk$', r'^dengan$', r'^pada$', r'^dari$'
        ]
        
        for pattern in common_patterns:
            if re.match(pattern, word_lower):
                return True
        
        return False
    
    def _normalize_name_indonesian(self, name):
        """
        Normalisasi nama Indonesia
        """
        if not name:
            return ""
        
        # Remove possessive suffix (-nya, -mu, -ku)
        name = re.sub(r'(nya|mu|ku)$', '', name, flags=re.IGNORECASE)
        
        # Remove English possessive ('s)
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
        
        name = ' '.join(capitalized)
        
        return name.strip()
    
    def _merge_indonesian_names(self, character_freq):
        """
        Merge dengan awareness untuk pola nama Indonesia
        Prioritas: Nama lengkap dengan gelar > Nama lengkap > Nama tunggal
        """
        # Separate berdasarkan jenis
        with_honorifics = {}
        full_names = {}
        single_names = {}
        
        for name, count in character_freq.items():
            name_lower = name.lower()
            words = name.split()
            
            # Has honorific?
            if len(words) > 1 and words[0].lower() in self.indonesian_honorifics:
                with_honorifics[name] = count
            elif len(words) > 1:
                full_names[name] = count
            else:
                single_names[name] = count
        
        merged = {}
        processed = set()
        
        # STEP 1: Process names dengan honorifics first (highest priority)
        for name_with_honor, count in sorted(with_honorifics.items(), key=lambda x: x[1], reverse=True):
            if name_with_honor in processed:
                continue
            
            canonical = name_with_honor
            total_count = count
            processed.add(name_with_honor)
            
            # Extract base name (tanpa honorific)
            words = name_with_honor.split()
            base_name = ' '.join(words[1:]) if len(words) > 1 else name_with_honor
            base_lower = base_name.lower()
            
            # Merge dengan single names yang match
            for single_name, single_count in single_names.items():
                if single_name in processed:
                    continue
                
                if single_name.lower() == base_lower:
                    total_count += single_count
                    processed.add(single_name)
                    print(f"    → Merging '{single_name}' ke '{canonical}'")
            
            # Merge dengan full names yang match
            for full_name, full_count in full_names.items():
                if full_name in processed:
                    continue
                
                if full_name.lower() == base_lower:
                    total_count += full_count
                    processed.add(full_name)
                    print(f"    → Merging '{full_name}' ke '{canonical}'")
            
            merged[canonical] = total_count
        
        # STEP 2: Process full names tanpa honorifics
        for full_name, count in sorted(full_names.items(), key=lambda x: x[1], reverse=True):
            if full_name in processed:
                continue
            
            canonical = full_name
            total_count = count
            processed.add(full_name)
            
            # Find matching single names
            words = full_name.lower().split()
            
            for single_name, single_count in single_names.items():
                if single_name in processed:
                    continue
                
                single_lower = single_name.lower()
                
                # If single name adalah bagian dari full name
                if single_lower in words:
                    total_count += single_count
                    processed.add(single_name)
                    print(f"    → Merging '{single_name}' ke '{canonical}'")
            
            merged[canonical] = total_count
        
        # STEP 3: Process remaining single names
        for single_name, count in sorted(single_names.items(), key=lambda x: x[1], reverse=True):
            if single_name in processed:
                continue
            
            canonical = single_name
            total_count = count
            processed.add(single_name)
            
            # Check for variants
            for other_single, other_count in single_names.items():
                if other_single in processed:
                    continue
                
                if self._are_indonesian_name_variants(canonical, other_single):
                    total_count += other_count
                    processed.add(other_single)
                    print(f"    → Merging '{other_single}' ke '{canonical}'")
            
            merged[canonical] = total_count
        
        return merged

    def _are_indonesian_name_variants(self, name1, name2):
        """
        Check if two Indonesian names are variants
        """
        n1_lower = name1.lower()
        n2_lower = name2.lower()
        
        # Exact match
        if n1_lower == n2_lower:
            return True
        
        # Possessive variants: "Rina" vs "Rinanya"
        if n1_lower + 'nya' == n2_lower or n2_lower + 'nya' == n1_lower:
            return True
        
        if n1_lower + 'mu' == n2_lower or n2_lower + 'mu' == n1_lower:
            return True
        
        if n1_lower + 'ku' == n2_lower or n2_lower + 'ku' == n1_lower:
            return True
        
        # Substring match (min 4 chars)
        if len(n1_lower) >= 4 and len(n2_lower) >= 4:
            if n1_lower in n2_lower or n2_lower in n1_lower:
                if abs(len(n1_lower) - len(n2_lower)) <= 3:
                    return True
        
        return False
        
    def _add_context_indonesian(self, sentences, characters):
        """
        Add context for each character - Indonesian version
        """
        character_contexts = defaultdict(list)
        
        for idx, sentence in enumerate(sentences):
            sent_lower = sentence.lower()
            
            for character in characters.keys():
                char_lower = character.lower()
                
                # SPECIAL: Handle "Narator (Aku)"
                if char_lower == "narator (aku)":
                    # Look for "aku" or "saya"
                    pattern = r'\b(aku|saya)\b'
                    if re.search(pattern, sent_lower):
                        character_contexts[character].append({
                            'sentence_id': idx,
                            'sentence': sentence
                        })
                    continue
                
                # Handle names with honorifics
                if ' ' in char_lower:
                    words = char_lower.split()
                    # Check if all words present in sentence
                    if all(re.search(r'\b' + re.escape(word) + r'\b', sent_lower) for word in words):
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