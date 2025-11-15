"""
INDONESIAN VERSION: Relation Extraction with Indonesian Patterns
Complete rewrite untuk bahasa Indonesia
"""

import spacy
from collections import defaultdict
import itertools
import re

class RelationExtractor:
    def __init__(self):
        """
        Inisialisasi dengan pattern bahasa Indonesia
        """
        try:
            self.nlp = spacy.load('id_core_news_md')
        except OSError:
            try:
                self.nlp = spacy.load('id_core_news_sm')
            except OSError:
                self.nlp = spacy.load('en_core_web_sm')
        
        # COMPLETE REWRITE: Indonesian relation patterns
        self.specific_relation_patterns = {
            # KELUARGA - Family Relations
            'orang-tua-anak': [
                r'\b(?:ibu|bapak|ayah|mama|papa|bunda|ayahanda|ibunda)\b',
                r'\b(?:anak|putra|putri|putera|puteri)\b',
                r'\bibunya\b', r'\bayahnya\b', r'\borang tuanya\b',
                r'\banak(?:nya|mu|ku)\b',
                r'\bmelahirkan\b', r'\bmengandung\b', r'\bmembesarkan\b',
            ],
            'kakak-adik': [
                r'\bkakak\b', r'\badik\b', r'\babang\b', r'\bayang\b',
                r'\bkakaknya\b', r'\badiknya\b',
                r'\bkakak (?:laki-laki|perempuan)\b',
                r'\badik (?:laki-laki|perempuan)\b',
                r'\bsaudara kandung\b', r'\bsaudara\b',
            ],
            'suami-istri': [
                r'\bsuami\b', r'\bistri\b', r'\bsuaminya\b', r'\bistrinya\b',
                r'\bpasangan\b', r'\bpasangan hidup\b',
                r'\bmenikah dengan\b', r'\bnikah dengan\b',
                r'\bpernikahan\b', r'\bperkawinan\b',
                r'\bmempelai\b', r'\bpengantin\b',
            ],
            'keluarga-besar': [
                r'\bnenek\b', r'\bkakek\b', r'\bmbah\b', r'\beyang\b',
                r'\bcucu\b', r'\bpaman\b', r'\bbibi\b', r'\btante\b', r'\bom\b',
                r'\bkeponakan\b', r'\bsepupu\b', r'\bkemenakan\b',
            ],
            
            # ROMANTIS - Romantic Relations
            'kekasih': [
                r'\bpacar\b', r'\bpacarnya\b', r'\bkekasih\b', r'\bkekasihnya\b',
                r'\bcinta\b', r'\bmencintai\b', r'\bdicintai\b',
                r'\bsayang\b', r'\bmenyayangi\b', r'\bdisayangi\b',
                r'\byang tersayang\b', r'\byang tercinta\b',
                r'\bberpacaran\b', r'\bpacaran\b',
            ],
            'tertarik-romantis': [
                r'\bsuka\b', r'\bmenyukai\b', r'\bdisukai\b',
                r'\bterpesona\b', r'\bterpukau\b', r'\btertarik\b',
                r'\bmengagumi\b', r'\bmengidolakan\b',
                r'\bnaksir\b', r'\bcrush\b', r'\bgebetan\b',
            ],
            
            # PERTEMANAN - Friendship
            'sahabat': [
                r'\bsahabat\b', r'\bsahabatnya\b', r'\bsahabat karib\b',
                r'\bsahabat dekat\b', r'\bsahabat baik\b',
                r'\bbersahabat\b', r'\bpersahabatan\b',
            ],
            'teman': [
                r'\bteman\b', r'\btemannya\b', r'\bteman dekat\b',
                r'\bteman akrab\b', r'\bteman baik\b',
                r'\bberteman\b', r'\bpertemanan\b',
                r'\bkawan\b', r'\bsohib\b', r'\bbestie\b',
            ],
            'kenalan': [
                r'\bkenalan\b', r'\bberkenalan\b',
                r'\bkenal\b', r'\bmengenal\b', r'\bdikenal\b',
                r'\bbertemu\b', r'\bpertemuan\b',
            ],
            
            # PROFESIONAL - Professional Relations
            'atasan-bawahan': [
                r'\batasan\b', r'\bbos\b', r'\bpimpinan\b', r'\bmanajer\b',
                r'\bbawahan\b', r'\bkaryawan\b', r'\bpegawai\b', r'\bstaf\b',
                r'\bmemimpin\b', r'\bdipimpin\b',
                r'\bmengawasi\b', r'\bdiawasi\b',
                r'\bbekerja untuk\b', r'\bbekerja di bawah\b',
            ],
            'rekan-kerja': [
                r'\brekan kerja\b', r'\brekan\b', r'\bkolega\b',
                r'\bbekerja bersama\b', r'\bsatu kantor\b',
                r'\bsatu tim\b', r'\bsatu divisi\b',
                r'\bpartner kerja\b', r'\bmitra kerja\b',
            ],
            'partner-bisnis': [
                r'\bpartner bisnis\b', r'\bmitra bisnis\b',
                r'\bpartner usaha\b', r'\bsekutu bisnis\b',
                r'\bkerjasama bisnis\b', r'\bkolaborasi\b',
            ],
            'pelanggan-penjual': [
                r'\bpelanggan\b', r'\bpembeli\b', r'\bkonsumen\b',
                r'\bpenjual\b', r'\bpedagang\b', r'\btukang\b',
                r'\bmembeli dari\b', r'\bmenjual kepada\b',
                r'\btransaksi\b', r'\bberdagang\b',
            ],
            
            # PERMUSUHAN - Antagonistic Relations
            'musuh': [
                r'\bmusuh\b', r'\bmusuhnya\b', r'\blawan\b',
                r'\bmemusuhi\b', r'\bdibenci\b', r'\bmembenci\b',
                r'\bpermusuhan\b', r'\bdendam\b', r'\bberdendam\b',
            ],
            'rival': [
                r'\brival\b', r'\bpesaing\b', r'\bsaingan\b',
                r'\bbersaing\b', r'\bpersaingan\b', r'\bkompetisi\b',
                r'\bberlomba\b', r'\bmenantang\b',
            ],
            'korban-pelaku': [
                r'\bkorban\b', r'\bpelaku\b', r'\bpenjahat\b',
                r'\bmenyakiti\b', r'\bdisakiti\b',
                r'\bmembunuh\b', r'\bdibunuh\b',
                r'\bmenganiaya\b', r'\bdianiaya\b',
                r'\bmelukai\b', r'\bdilukai\b',
            ],
            'bertentangan': [
                r'\bbertentangan\b', r'\bmelawan\b', r'\bdilawan\b',
                r'\bmenentang\b', r'\bditentang\b',
                r'\bkonflik\b', r'\bpertengkaran\b', r'\bbertengkar\b',
            ],
            
            # SOSIAL LAINNYA - Other Social Relations
            'tetangga': [
                r'\btetangga\b', r'\btetangganya\b',
                r'\bbertetangga\b', r'\bsebelah rumah\b',
                r'\bdi sebelah\b', r'\brumah sebelah\b',
            ],
            'guru-murid': [
                r'\bguru\b', r'\bpengajar\b', r'\bdosen\b',
                r'\bmurid\b', r'\bsiswa\b', r'\bmahasiswa\b', r'\bpelajar\b',
                r'\bmengajar\b', r'\bdiajar\b', r'\bbelajar dari\b',
                r'\bmentori\b', r'\bbimbingan\b',
            ],
            'tuan-pembantu': [
                r'\bmajikan\b', r'\btuan rumah\b', r'\bnyonya rumah\b',
                r'\bpembantu\b', r'\basisten\b', r'\bpelayan\b',
                r'\bmelayani\b', r'\bdilayani\b',
            ],
        }
        
        # Possessive pronouns mapping Indonesia
        self.possessive_mapping = {
            'nya': 'third_person',
            'mu': 'second_person',
            'ku': 'first_person',
        }
        
        # Non-story characters (mythological, religious, etc)
        self.non_story_characters = {
            'tuhan', 'allah', 'yesus', 'nabi', 'rasul',
            'malaikat', 'iblis', 'setan', 'jin',
            'dewa', 'dewi', 'bidadari',
        }
    
    def extract_relations(self, characters, sentences):
        """
        Extract relations dengan Indonesian pattern detection
        """
        story_characters = {char: count for char, count in characters.items()
                           if char.lower() not in self.non_story_characters}
        
        if len(story_characters) < 2:
            print("  ⚠️  Tidak cukup tokoh untuk ekstraksi relasi")
            return self._empty_result()
        
        print(f"\n  [Enhanced Relation Extraction] Menganalisis {len(story_characters)} tokoh...")
        
        # Build character mention index
        char_mentions = self._index_character_mentions(story_characters, sentences)
        
        # Step 1: Co-occurrence (kalimat yang sama)
        cooccurrence = self._detect_cooccurrence(story_characters, sentences)
        print(f"    ✓ Pasangan co-occurrence: {len(cooccurrence)}")
        
        # Step 2: Proximity detection (dalam N kalimat)
        proximity_pairs = self._detect_proximity_pairs(char_mentions, window=10)
        print(f"    ✓ Pasangan proximity: {len(proximity_pairs)}")
        
        # Step 3: Detect relations in proximity contexts
        specific_relations = self._detect_relations_in_proximity(
            proximity_pairs, sentences, char_mentions
        )
        print(f"    ✓ Relasi spesifik terdeteksi: {len(specific_relations)}")
        
        # Step 4: Possessive pronoun inference (nya, mu, ku)
        possessive_relations = self._detect_possessive_relations_indonesian(
            story_characters, sentences, char_mentions
        )
        print(f"    ✓ Relasi possessive: {len(possessive_relations)}")
        
        # Merge all
        all_relations = specific_relations + possessive_relations
        merged_relations = self._merge_and_rank_relations(
            all_relations, cooccurrence, proximity_pairs
        )
        
        return {
            'cooccurrence': cooccurrence,
            'proximity_pairs': proximity_pairs,
            'specific_relations': specific_relations,
            'possessive_relations': possessive_relations,
            'merged_relations': merged_relations,
            'relation_graph': self._build_detailed_graph(merged_relations),
            # Backward compatibility
            'direct_cooccurrence': cooccurrence,
            'proximity': proximity_pairs,
            'semantic': all_relations,
            'rulebased': all_relations
        }
    
    def _index_character_mentions(self, characters, sentences):
        """
        Build index dimana setiap karakter disebut
        """
        mentions = defaultdict(list)
        
        for sent_id, sentence in enumerate(sentences):
            sent_lower = sentence.lower()
            
            for char in characters.keys():
                if self._is_character_in_sentence_indonesian(char, sent_lower):
                    mentions[char].append(sent_id)
        
        return dict(mentions)
    
    def _detect_proximity_pairs(self, char_mentions, window=10):
        """
        Detect pasangan karakter yang muncul dalam N kalimat berdekatan
        """
        proximity = defaultdict(lambda: {'count': 0, 'sentence_pairs': []})
        
        chars = list(char_mentions.keys())
        
        for char1, char2 in itertools.combinations(chars, 2):
            mentions1 = char_mentions.get(char1, [])
            mentions2 = char_mentions.get(char2, [])
            
            # Find close mentions
            for sent1 in mentions1:
                for sent2 in mentions2:
                    distance = abs(sent1 - sent2)
                    if distance <= window:
                        pair = tuple(sorted([char1, char2]))
                        proximity[pair]['count'] += 1
                        proximity[pair]['sentence_pairs'].append((sent1, sent2))
        
        return dict(proximity)
    
    def _detect_relations_in_proximity(self, proximity_pairs, sentences, char_mentions):
        """
        Detect relations dalam kalimat yang dekat dengan kemunculan karakter
        """
        relations = []
        
        for pair, data in proximity_pairs.items():
            char1, char2 = pair
            
            # Get all sentences in proximity
            sent_ids = set()
            for sent1, sent2 in data['sentence_pairs']:
                # Include sentences in between dan sekitarnya
                start = min(sent1, sent2) - 2
                end = max(sent1, sent2) + 3
                sent_ids.update(range(max(0, start), min(len(sentences), end)))
            
            # Check untuk relation patterns
            relation_evidence = defaultdict(list)
            
            for sent_id in sent_ids:
                sentence = sentences[sent_id]
                sent_lower = sentence.lower()
                
                # Check apakah salah satu karakter ada di kalimat
                char1_base = self._get_base_name(char1).lower()
                char2_base = self._get_base_name(char2).lower()
                
                has_char1 = char1_base in sent_lower or char1.lower() in sent_lower
                has_char2 = char2_base in sent_lower or char2.lower() in sent_lower
                
                # Check jika minimal satu karakter ada
                if has_char1 or has_char2:
                    for relation_type, patterns in self.specific_relation_patterns.items():
                        for pattern in patterns:
                            if re.search(pattern, sent_lower):
                                relation_evidence[relation_type].append({
                                    'sentence_id': sent_id,
                                    'sentence': sentence,
                                    'pattern': pattern,
                                    'has_char1': has_char1,
                                    'has_char2': has_char2
                                })
            
            # Create relation entries
            for relation_type, evidence in relation_evidence.items():
                confidence = self._calculate_proximity_confidence(
                    len(evidence), 
                    data['count'],
                    any(e['has_char1'] and e['has_char2'] for e in evidence)
                )
                
                relations.append({
                    'character1': char1,
                    'character2': char2,
                    'relation_type': relation_type,
                    'evidence_count': len(evidence),
                    'evidence': evidence[:3],
                    'confidence': confidence,
                    'source': 'proximity'
                })
        
        return relations
    
    def _detect_possessive_relations_indonesian(self, characters, sentences, char_mentions):
        """
        Detect relations dari possessive pronouns Indonesia (-nya, -mu, -ku)
        """
        relations = []
        
        # Possessive patterns untuk relasi keluarga & romantis
        possessive_patterns = {
            'suami-istri': [
                (r'\bsuaminya\b', 'istri_suami'),
                (r'\bistrinya\b', 'suami_istri'),
            ],
            'orang-tua-anak': [
                (r'\bibunya\b', 'anak_ibu'),
                (r'\bayahnya\b', 'anak_ayah'),
                (r'\borang tuanya\b', 'anak_ortu'),
                (r'\banaknya\b', 'ortu_anak'),
            ],
            'kekasih': [
                (r'\bpacarnya\b', 'pacar'),
                (r'\bkekasihnya\b', 'kekasih'),
                (r'\bsayangnya\b', 'kesayangan'),
            ],
            'sahabat': [
                (r'\bsahabatnya\b', 'sahabat'),
                (r'\btemannya\b', 'teman'),
            ],
        }
        
        for char in characters.keys():
            char_lower = char.lower()
            char_base = self._get_base_name(char).lower()
            
            # Get sentences where this character appears
            sent_ids = char_mentions.get(char, [])
            
            for sent_id in sent_ids:
                sentence = sentences[sent_id]
                sent_lower = sentence.lower()
                
                # Check for possessive patterns
                for relation_type, patterns in possessive_patterns.items():
                    for pattern, pattern_name in patterns:
                        matches = list(re.finditer(pattern, sent_lower))
                        
                        if matches:
                            # Find which other character might be referenced
                            for other_char in characters.keys():
                                if other_char == char:
                                    continue
                                
                                other_lower = other_char.lower()
                                other_base = self._get_base_name(other_char).lower()
                                
                                # Check if other character nearby (±5 kalimat)
                                nearby_sent_ids = range(
                                    max(0, sent_id - 5),
                                    min(len(sentences), sent_id + 6)
                                )
                                
                                other_nearby = any(
                                    other_lower in sentences[sid].lower() or 
                                    other_base in sentences[sid].lower()
                                    for sid in nearby_sent_ids
                                )
                                
                                if other_nearby:
                                    pair = tuple(sorted([char, other_char]))
                                    relations.append({
                                        'character1': pair[0],
                                        'character2': pair[1],
                                        'relation_type': relation_type,
                                        'evidence_count': len(matches),
                                        'evidence': [{'sentence': sentence, 'pattern': pattern_name}],
                                        'confidence': 0.75,
                                        'source': 'possessive'
                                    })
        
        return relations
    
    def _detect_cooccurrence(self, characters, sentences):
        """
        Detect exact co-occurrence (kedua tokoh di kalimat yang sama)
        """
        cooccurrence = defaultdict(lambda: {
            'count': 0, 'sentences': [], 'contexts': []
        })
        
        char_list = list(characters.keys())
        
        for sent_id, sentence in enumerate(sentences):
            sent_lower = sentence.lower()
            present = []
            
            for char in char_list:
                if self._is_character_in_sentence_indonesian(char, sent_lower):
                    present.append(char)
            
            for char1, char2 in itertools.combinations(present, 2):
                pair = tuple(sorted([char1, char2]))
                cooccurrence[pair]['count'] += 1
                cooccurrence[pair]['sentences'].append(sent_id)
                cooccurrence[pair]['contexts'].append(sentence)
        
        return dict(cooccurrence)
    
    def _merge_and_rank_relations(self, all_relations, cooccurrence, proximity_pairs):
        """
        Merge and rank relations - PRIORITASKAN hubungan personal
        """
        relations_by_pair = defaultdict(lambda: {
            'relations': [], 
            'cooccurrence_count': 0,
            'proximity_count': 0
        })
        
        # Relation type priority (Indonesia-specific)
        relation_priority = {
            # HIGHEST: Violence & Crime
            'korban-pelaku': 12,
            
            # HIGH: Close personal relationships
            'suami-istri': 11,
            'orang-tua-anak': 11,
            'kekasih': 10,
            'kakak-adik': 10,
            'sahabat': 9,
            'keluarga-besar': 8,
            
            # MEDIUM: Social relationships
            'tertarik-romantis': 7,
            'teman': 7,
            'guru-murid': 6,
            'tetangga': 5,
            
            # LOW: Professional (biasanya tidak emotional)
            'rekan-kerja': 4,
            'atasan-bawahan': 4,
            'partner-bisnis': 3,
            
            # VERY LOW: Transactional
            'pelanggan-penjual': 2,
            'kenalan': 2,
            
            # Antagonistic
            'musuh': 8,
            'rival': 6,
            'bertentangan': 5,
        }
        
        for rel in all_relations:
            pair = tuple(sorted([rel['character1'], rel['character2']]))
            
            # Add priority boost
            priority = relation_priority.get(rel['relation_type'], 3)
            adjusted_confidence = min(rel['confidence'] + (priority * 0.02), 0.98)
            
            relations_by_pair[pair]['relations'].append({
                'type': rel['relation_type'],
                'confidence': adjusted_confidence,
                'original_confidence': rel['confidence'],
                'evidence_count': rel['evidence_count'],
                'source': rel['source'],
                'priority': priority
            })
        
        # Add counts
        for pair, data in cooccurrence.items():
            if pair in relations_by_pair:
                relations_by_pair[pair]['cooccurrence_count'] = data['count']
        
        for pair, data in proximity_pairs.items():
            if pair in relations_by_pair:
                relations_by_pair[pair]['proximity_count'] = data['count']
        
        # Build final list
        merged = []
        for pair, data in relations_by_pair.items():
            char1, char2 = pair
            
            # Sort by priority FIRST, then confidence
            relations = sorted(
                data['relations'], 
                key=lambda x: (x['priority'], x['confidence'], x['evidence_count']), 
                reverse=True
            )
            
            if not relations:
                continue
            
            primary = relations[0]
            all_types = list(set(r['type'] for r in relations))
            
            strength = self._calculate_overall_strength(
                primary['confidence'],
                data['cooccurrence_count'],
                data['proximity_count'],
                len(relations)
            )
            
            merged.append({
                'character1': char1,
                'character2': char2,
                'primary_relation': primary['type'],
                'all_relations': all_types,
                'relation_types': all_types,
                'confidence': primary['confidence'],
                'cooccurrence_count': data['cooccurrence_count'],
                'proximity_count': data['proximity_count'],
                'strength': strength,
                'source': primary['source']
            })
        
        merged.sort(key=lambda x: x['strength'], reverse=True)
        return merged
    
    def _get_base_name(self, full_name):
        """
        Get base name (tanpa gelar) dari full name
        """
        # Remove Indonesian honorifics
        honorifics = ['pak', 'bu', 'bapak', 'ibu', 'mas', 'mbak', 'bang', 'kang']
        words = full_name.split()
        
        if len(words) > 1 and words[0].lower() in honorifics:
            return ' '.join(words[1:])
        
        return full_name
    
    def _is_character_in_sentence_indonesian(self, character, sentence_lower):
        """
        Check apakah karakter ada di kalimat (Indonesian-aware)
        """
        char_lower = character.lower()
        
        # SPECIAL: Handle "Narator (Aku)"
        if char_lower == "narator (aku)":
            return bool(re.search(r'\b(aku|saya)\b', sentence_lower))
        
        # Handle names dengan gelar
        if ' ' in char_lower:
            # Check if all words present
            words = char_lower.split()
            if all(word in sentence_lower for word in words):
                return True
            
            # Check base name only (tanpa gelar)
            base_name = self._get_base_name(character).lower()
            if base_name in sentence_lower:
                return True
        
        # Single name - check dengan word boundary
        pattern = r'\b' + re.escape(char_lower) + r'\b'
        if re.search(pattern, sentence_lower):
            return True
        
        return False
    
    def _calculate_proximity_confidence(self, evidence_count, proximity_count, has_both):
        """
        Calculate confidence untuk proximity-based detection
        """
        base = min(evidence_count / 2.0, 0.85)
        
        # Boost jika kedua karakter di kalimat yang sama
        if has_both:
            base = min(base + 0.1, 0.95)
        
        # Boost berdasarkan proximity frequency
        if proximity_count > 5:
            base = min(base + 0.05, 0.95)
        
        return base
    
    def _calculate_overall_strength(self, confidence, cooccurrence, proximity, relation_count):
        """
        Calculate overall strength
        """
        confidence_score = confidence * 0.4
        cooccurrence_score = min(cooccurrence / 10.0, 0.2)
        proximity_score = min(proximity / 20.0, 0.2)
        diversity_score = min(relation_count / 5.0, 0.2)
        
        return min(confidence_score + cooccurrence_score + proximity_score + diversity_score, 1.0)
    
    def _build_detailed_graph(self, relations):
        """
        Build graph
        """
        graph = {'nodes': [], 'edges': []}
        
        if not relations:
            return graph
        
        characters = set()
        for rel in relations:
            characters.add(rel['character1'])
            characters.add(rel['character2'])
        
        graph['nodes'] = [{'id': char, 'label': char} for char in characters]
        
        for rel in relations:
            graph['edges'].append({
                'source': rel['character1'],
                'target': rel['character2'],
                'relation': rel['primary_relation'],
                'all_relations': rel['all_relations'],
                'types': rel['all_relations'],
                'confidence': rel['confidence'],
                'strength': rel['strength'],
                'weight': rel['strength'],
                'cooccurrence': rel['cooccurrence_count'],
                'proximity': rel['proximity_count'],
                'count': rel['cooccurrence_count'],
                'source_type': rel['source']
            })
        
        return graph
    
    def _empty_result(self):
        """
        Return empty result
        """
        return {
            'cooccurrence': {},
            'proximity_pairs': {},
            'specific_relations': [],
            'possessive_relations': [],
            'merged_relations': [],
            'relation_graph': {'nodes': [], 'edges': []},
            'direct_cooccurrence': {},
            'proximity': {},
            'semantic': [],
            'rulebased': []
        }