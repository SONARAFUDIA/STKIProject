import spacy
import re
from collections import Counter

class TraitExtractor:
    def __init__(self):
        """
        Inisialisasi dengan model spaCy Indonesia
        """
        try:
            self.nlp = spacy.load('id_core_news_md')
        except OSError:
            try:
                self.nlp = spacy.load('id_core_news_sm')
            except OSError:
                print("⚠️  Model Indonesia tidak tersedia, menggunakan English fallback")
                self.nlp = spacy.load('en_core_web_sm')
        
        # EXPANDED Indonesian trait keywords
        self.trait_keywords = {
            'positive': [
                # Karakter baik
                'baik', 'ramah', 'sopan', 'santun', 'hormat', 'mulia', 'bijak',
                'jujur', 'setia', 'loyal', 'tulus', 'ikhlas', 'rendah hati',
                'dermawan', 'murah hati', 'pemurah', 'royal',
                
                # Kemampuan positif
                'cerdas', 'pintar', 'pandai', 'genius', 'jenius', 'brilian',
                'mahir', 'terampil', 'cekatan', 'tangkas', 'gesit',
                'rajin', 'tekun', 'ulet', 'gigih', 'pekerja keras',
                
                # Sifat positif
                'sabar', 'tenang', 'kalem', 'damai', 'teduh',
                'peduli', 'perhatian', 'penyayang', 'kasih sayang', 'lembut',
                'kuat', 'berani', 'pemberani', 'gagah', 'tangguh',
                'percaya diri', 'yakin', 'optimis', 'positif',
                
                # Penampilan positif
                'cantik', 'tampan', 'ganteng', 'rupawan', 'elok',
                'menarik', 'menawan', 'anggun', 'gagah', 'perkasa',
                'rapi', 'bersih', 'necis', 'stylish',
            ],
            'negative': [
                # Karakter buruk
                'jahat', 'kejam', 'keji', 'biadab', 'zalim', 'bengis',
                'licik', 'curang', 'tidak jujur', 'bohong', 'pembohong',
                'munafik', 'munafik', 'dua muka', 'penipu',
                'egois', 'serakah', 'tamak', 'rakus', 'loba',
                
                # Sifat negatif
                'malas', 'pemalas', 'jorok', 'kotor', 'kumuh',
                'kasar', 'kurang ajar', 'tidak sopan', 'biadab',
                'sombong', 'angkuh', 'congkak', 'tinggi hati', 'arogan',
                'bodoh', 'tolol', 'dungu', 'bego', 'idiot', 'goblok',
                
                # Perilaku buruk
                'pemarah', 'temperamental', 'galak', 'garang',
                'penakut', 'cemen', 'pengecut', 'takut',
                'lemah', 'payah', 'loyo', 'lemas',
                
                # Penampilan negatif
                'jelek', 'buruk rupa', 'hodoh', 'suram',
                'lusuh', 'compang-camping', 'dekil',
            ],
            'emotional': [
                # Emosi positif
                'gembira', 'senang', 'bahagia', 'girang', 'riang',
                'ceria', 'sumringah', 'suka cita', 'excited',
                'terharu', 'bangga', 'puas', 'lega', 'tenang',
                
                # Emosi negatif
                'sedih', 'duka', 'murung', 'muram', 'suram', 'melankolis',
                'marah', 'geram', 'berang', 'murka', 'jengkel', 'kesal',
                'takut', 'cemas', 'khawatir', 'gelisah', 'resah', 'was-was',
                'bingung', 'galau', 'kalut', 'bimbang', 'ragu',
                'kecewa', 'frustasi', 'putus asa', 'hopeless',
                
                # Ekspresi emosi
                'menangis', 'menitikkan air mata', 'berlinang', 'terisak',
                'tertawa', 'tersenyum', 'sumringah', 'berseri',
                'meratap', 'mengeluh', 'merengek',
            ],
            'physical': [
                # Ukuran & bentuk tubuh
                'tinggi', 'pendek', 'jangkung', 'cebol',
                'besar', 'kecil', 'gemuk', 'gendut', 'obesitas',
                'kurus', 'langsing', 'ramping', 'singset',
                'berotot', 'kekar', 'tegap', 'atletis',
                
                # Fitur fisik
                'tua', 'uzur', 'sepuh', 'lanjut usia', 'renta',
                'muda', 'belia', 'remaja', 'bocah',
                'tampan', 'cantik', 'rupawan', 'ganteng',
                'jelek', 'buruk rupa', 'hodoh',
                
                # Kondisi fisik
                'sehat', 'bugar', 'fit', 'segar',
                'sakit', 'lemah', 'pucat', 'pasi', 'lesu',
                'lelah', 'capek', 'letih', 'penat', 'payah',
                'kuat', 'tangguh', 'kokoh', 'perkasa',
                
                # Ciri khusus
                'berjenggot', 'berkumis', 'botak', 'gundul',
                'berkacamata', 'rabun', 'buta', 'tuli',
            ],
            'behavioral': [
                # Gaya hidup
                'aktif', 'energik', 'lincah', 'dinamis',
                'pasif', 'lesu', 'diam', 'pendiam',
                'rajin', 'tekun', 'giat', 'ulet',
                'malas', 'pemalas', 'santai', 'cuek',
                
                # Interaksi sosial
                'ramah', 'supel', 'friendly', 'hangat',
                'dingin', 'jutek', 'judes', 'kasar',
                'pemalu', 'malu-malu', 'introvert', 'tertutup',
                'berani', 'pemberani', 'nekat', 'gegabah',
                'hati-hati', 'waspada', 'teliti', 'cermat',
                
                # Kebiasaan
                'rapi', 'teratur', 'disiplin', 'tertib',
                'berantakan', 'kacau', 'amburadul', 'semrawut',
                'pelit', 'kikir', 'bakhil', 'perhitungan',
                'boros', 'konsumtif', 'pemborosan',
            ]
        }
        
        # Action verbs yang mengindikasikan trait (Indonesian)
        self.trait_indicating_actions = {
            'positive': [
                'menolong', 'membantu', 'menyelamatkan', 'melindungi',
                'mencintai', 'menyayangi', 'merawat', 'menjaga',
                'memberikan', 'memberi', 'menyumbang', 'berkorban',
                'bekerja keras', 'berjuang', 'belajar', 'berlatih',
            ],
            'negative': [
                'menyakiti', 'melukai', 'membunuh', 'menganiaya',
                'membenci', 'memusuhi', 'menghianati', 'mengkhianati',
                'mencuri', 'merampok', 'menjarah', 'korupsi',
                'membohongi', 'menipu', 'mengecoh', 'memperdaya',
                'memukul', 'menendang', 'menampar', 'menyiksa',
            ],
            'emotional': [
                'menangis', 'menitikkan air mata', 'terisak', 'berlinang',
                'tertawa', 'tersenyum', 'terkekeh', 'cekikikan',
                'berteriak', 'menjerit', 'memekik', 'bersorak',
                'meratap', 'mengeluh', 'menggerutu', 'protes',
            ]
        }
        
        # Indonesian sentiment words untuk context analysis
        self.sentiment_words = {
            'positive': [
                'baik', 'bagus', 'hebat', 'luar biasa', 'menakjubkan',
                'indah', 'cantik', 'sempurna', 'mantap', 'keren',
            ],
            'negative': [
                'buruk', 'jelek', 'mengerikan', 'menyeramkan',
                'menakutkan', 'menjijikkan', 'menyedihkan',
            ]
        }
    
    def extract_traits(self, character_name, character_contexts):
        """
        Ekstraksi watak karakter dari konteks - INDONESIAN VERSION
        """
        all_traits = []
        trait_sentences = []
        
        # Check if narrator
        is_narrator = 'narator' in character_name.lower() and 'aku' in character_name.lower()
        
        for context in character_contexts:
            sentence = context['sentence']
            doc = self.nlp(sentence)
            
            if is_narrator:
                # Extract traits dari perspektif orang pertama
                traits_found = self._extract_narrator_traits_indonesian(doc, sentence)
            else:
                # Extract traits untuk karakter reguler
                # Method 1: Adjektiva berdekatan
                traits_found = self._find_adjacent_adjectives_indonesian(doc, character_name, sentence)
                
                # Method 2: Pattern matching Indonesia
                pattern_traits = self._pattern_matching_indonesian(sentence, character_name)
                traits_found.extend(pattern_traits)
                
                # Method 3: Deskripsi possessive (rambutnya, matanya, dll)
                possessive_traits = self._possessive_descriptions_indonesian(sentence, character_name)
                traits_found.extend(possessive_traits)
                
                # Method 4: Action-based trait inference
                action_traits = self._action_based_traits_indonesian(doc, character_name, sentence)
                traits_found.extend(action_traits)
                
                # Method 5: Descriptive phrases Indonesian
                descriptive_traits = self._descriptive_phrases_indonesian(sentence, character_name)
                traits_found.extend(descriptive_traits)
            
            # Method 6: Sentiment analysis sederhana
            sentiment_trait = self._analyze_sentiment_indonesian(sentence)
            
            combined_traits = traits_found
            if sentiment_trait:
                combined_traits.append(sentiment_trait)
            
            if combined_traits:
                all_traits.extend(combined_traits)
                trait_sentences.append({
                    'sentence': sentence,
                    'traits': combined_traits
                })
        
        # Aggregate and classify
        trait_summary = self._classify_traits(all_traits)
        
        return {
            'character': character_name,
            'raw_traits': all_traits,
            'trait_frequency': dict(Counter(all_traits)),
            'classified_traits': trait_summary,
            'evidence_sentences': trait_sentences
        }

    def _extract_narrator_traits_indonesian(self, doc, sentence):
        """
        Extract traits untuk narator orang pertama (aku/saya)
        """
        traits = []
        sent_lower = sentence.lower()
        
        # Pattern untuk "aku VERB ADJECTIVE" atau "aku adalah/merasa ADJECTIVE"
        patterns = [
            r'\baku\s+(?:adalah|ialah|merupakan)\s+(\w+)',
            r'\baku\s+(?:merasa|merasakan|terasa)\s+(\w+)',
            r'\baku\s+(?:menjadi|jadi)\s+(\w+)',
            r'\baku\s+(?:tampak|terlihat|kelihatan)\s+(\w+)',
            r'\baku\s+(?:sangat|amat|sekali)\s+(\w+)',
            
            # "saya" variants
            r'\bsaya\s+(?:adalah|ialah|merupakan)\s+(\w+)',
            r'\bsaya\s+(?:merasa|merasakan|terasa)\s+(\w+)',
            r'\bsaya\s+(?:menjadi|jadi)\s+(\w+)',
            r'\bsaya\s+(?:sangat|amat|sekali)\s+(\w+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, sent_lower)
            for match in matches:
                # Verify it's an adjective
                if self._is_indonesian_adjective(match):
                    traits.append(match)
        
        # Pattern untuk "aku yang ADJECTIVE"
        pattern_yang = r'\baku\s+yang\s+(\w+)'
        matches = re.findall(pattern_yang, sent_lower)
        for match in matches:
            if self._is_indonesian_adjective(match):
                traits.append(match)
        
        return traits
    
    def _find_adjacent_adjectives_indonesian(self, doc, character_name, sentence):
        """
        Cari adjektiva di sekitar nama karakter (Indonesian patterns)
        """
        traits = []
        sent_lower = sentence.lower()
        
        # Get base name (without honorifics)
        char_parts = character_name.lower().split()
        char_base = char_parts[-1] if len(char_parts) > 1 else character_name.lower()
        
        # Find character position in sentence
        char_pattern = r'\b' + re.escape(char_base) + r'\b'
        char_match = re.search(char_pattern, sent_lower)
        
        if not char_match:
            return traits
        
        char_pos = char_match.start()
        
        # Get words in window (±50 chars around character name)
        window_start = max(0, char_pos - 50)
        window_end = min(len(sent_lower), char_pos + 50)
        window_text = sent_lower[window_start:window_end]
        
        # Extract adjectives from window
        words = window_text.split()
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            if self._is_indonesian_adjective(clean_word):
                traits.append(clean_word)
        
        return traits
    
    def _pattern_matching_indonesian(self, sentence, character_name):
        """
        Pattern matching untuk struktur kalimat Indonesia
        """
        traits = []
        sent_lower = sentence.lower()
        
        # Get base name
        char_parts = character_name.lower().split()
        char_base = char_parts[-1] if len(char_parts) > 1 else character_name.lower()
        char_escaped = re.escape(char_base)
        
        # Indonesian patterns
        patterns = [
            # "KARAKTER adalah/ialah SIFAT"
            rf'\b{char_escaped}\s+(?:adalah|ialah|merupakan)\s+(?:seorang\s+)?(\w+)',
            
            # "KARAKTER yang SIFAT"
            rf'\b{char_escaped}\s+yang\s+(\w+)',
            
            # "KARAKTER terlihat/tampak/kelihatan SIFAT"
            rf'\b{char_escaped}\s+(?:terlihat|tampak|kelihatan|nampak)\s+(\w+)',
            
            # "KARAKTER sangat/amat/sekali SIFAT"
            rf'\b{char_escaped}\s+(?:sangat|amat|sekali|terlalu)\s+(\w+)',
            
            # "SIFAT si KARAKTER" atau "SIFAT KARAKTER"
            rf'(\w+)\s+(?:si\s+)?{char_escaped}',
            
            # "KARAKTER merasa/terasa SIFAT"
            rf'\b{char_escaped}\s+(?:merasa|merasakan|terasa)\s+(\w+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, sent_lower)
            for match in matches:
                potential_trait = match if isinstance(match, str) else match[-1]
                
                # Verify it's an adjective
                if self._is_indonesian_adjective(potential_trait):
                    traits.append(potential_trait)
        
        return traits
    
    def _possessive_descriptions_indonesian(self, sentence, character_name):
        """
        Extract traits dari deskripsi possessive Indonesia
        e.g., "matanya yang indah", "wajahnya pucat", "rambutnya hitam"
        """
        traits = []
        sent_lower = sentence.lower()
        
        char_parts = character_name.lower().split()
        char_base = char_parts[-1] if len(char_parts) > 1 else character_name.lower()
        char_escaped = re.escape(char_base)
        
        # Pattern: "KARAKTER dengan BODYPART yang SIFAT"
        body_parts = [
            'wajah', 'muka', 'mata', 'hidung', 'mulut', 'bibir',
            'rambut', 'kulit', 'tangan', 'kaki', 'tubuh', 'badan',
            'suara', 'tatapan', 'senyum', 'ekspresi', 'penampilan'
        ]
        
        for body_part in body_parts:
            # Pattern 1: "matanya yang SIFAT"
            pattern1 = rf'{body_part}(?:nya|mu|ku)?\s+yang\s+(\w+)'
            matches = re.findall(pattern1, sent_lower)
            for match in matches:
                if self._is_indonesian_adjective(match):
                    traits.append(match)
            
            # Pattern 2: "matanya SIFAT"
            pattern2 = rf'{body_part}(?:nya|mu|ku)?\s+(\w+)'
            matches = re.findall(pattern2, sent_lower)
            for match in matches:
                if self._is_indonesian_adjective(match):
                    traits.append(match)
        
        return traits
    
    def _action_based_traits_indonesian(self, doc, character_name, sentence):
        """
        Infer traits dari action verbs (Indonesian)
        """
        traits = []
        sent_lower = sentence.lower()
        
        char_parts = character_name.lower().split()
        char_base = char_parts[-1] if len(char_parts) > 1 else character_name.lower()
        
        # Check if character is in sentence
        if char_base not in sent_lower:
            return traits
        
        # Check for action verbs
        for category, verbs in self.trait_indicating_actions.items():
            for verb in verbs:
                if verb in sent_lower:
                    # Check proximity to character name
                    verb_pos = sent_lower.find(verb)
                    char_pos = sent_lower.find(char_base)
                    
                    if abs(verb_pos - char_pos) < 50:  # Within 50 chars
                        traits.append(f"{category}_aksi")
        
        return traits
    
    def _descriptive_phrases_indonesian(self, sentence, character_name):
        """
        Extract traits dari frasa deskriptif Indonesia
        """
        traits = []
        sent_lower = sentence.lower()
        
        char_parts = character_name.lower().split()
        char_base = char_parts[-1] if len(char_parts) > 1 else character_name.lower()
        
        # Common descriptive patterns in Indonesian
        descriptive_patterns = {
            'tua': r'(?:orang\s+)?tua',
            'muda': r'(?:anak\s+)?muda',
            'cantik': r'(?:sangat\s+)?cantik',
            'tampan': r'(?:sangat\s+)?tampan',
            'baik': r'(?:sangat\s+)?baik',
            'ramah': r'(?:sangat\s+)?ramah',
            'sopan': r'(?:sangat\s+)?sopan',
        }
        
        for trait, pattern in descriptive_patterns.items():
            if re.search(pattern, sent_lower) and char_base in sent_lower:
                traits.append(trait)
        
        return traits
    
    def _analyze_sentiment_indonesian(self, sentence):
        """
        Analisis sentimen sederhana untuk bahasa Indonesia
        """
        sent_lower = sentence.lower()
        
        positive_count = sum(1 for word in self.sentiment_words['positive'] if word in sent_lower)
        negative_count = sum(1 for word in self.sentiment_words['negative'] if word in sent_lower)
        
        if positive_count > negative_count and positive_count >= 1:
            return 'konteks_positif'
        elif negative_count > positive_count and negative_count >= 1:
            return 'konteks_negatif'
        
        return None
    
    def _is_indonesian_adjective(self, word):
        """
        Check if word is likely an Indonesian adjective
        """
        if not word or len(word) < 3:
            return False
        
        # Check in our trait keywords
        for category, traits in self.trait_keywords.items():
            if word in traits:
                return True
        
        # Check common adjective patterns in Indonesian
        # Many adjectives don't have special markers, so we check common ones
        common_adjectives = [
            'baik', 'buruk', 'besar', 'kecil', 'tinggi', 'rendah',
            'panjang', 'pendek', 'lebar', 'sempit', 'tebal', 'tipis',
            'kuat', 'lemah', 'cepat', 'lambat', 'mudah', 'sulit',
            'mahal', 'murah', 'baru', 'lama', 'muda', 'tua',
            'cantik', 'jelek', 'indah', 'hodoh', 'tampan', 'rupawan',
            'pintar', 'bodoh', 'rajin', 'malas', 'ramah', 'kasar',
        ]
        
        if word in common_adjectives:
            return True
        
        # NOT adjectives (common false positives)
        non_adjectives = [
            'yang', 'dan', 'atau', 'untuk', 'dengan', 'dari', 'pada',
            'adalah', 'akan', 'sudah', 'telah', 'sedang', 'masih',
            'juga', 'hanya', 'saja', 'pun', 'lah', 'kah',
        ]
        
        if word in non_adjectives:
            return False
        
        return False
    
    def _classify_traits(self, traits):
        """
        Klasifikasi traits ke kategori
        """
        classified = {
            'positive': [],
            'negative': [],
            'emotional': [],
            'physical': [],
            'behavioral': [],
            'other': []
        }
        
        for trait in traits:
            categorized = False
            
            # Check each category
            for category, keywords in self.trait_keywords.items():
                if trait in keywords:
                    classified[category].append(trait)
                    categorized = True
                    break
            
            # Check action-based traits
            if '_aksi' in trait or '_context' in trait or 'konteks_' in trait:
                category = trait.split('_')[0]
                if category in classified:
                    classified[category].append(trait)
                    categorized = True
            
            if not categorized:
                classified['other'].append(trait)
        
        return classified