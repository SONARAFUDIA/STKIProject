import spacy
from collections import defaultdict
import itertools

class RelationExtractor:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_lg')
        
        # Kata kunci relasi
        self.relation_patterns = {
            'family': ['mother', 'father', 'sister', 'brother', 'son', 'daughter', 
                      'wife', 'husband', 'parent', 'child', 'family'],
            'romantic': ['love', 'lover', 'beloved', 'romance', 'marry', 'married',
                        'kiss', 'embrace', 'affection'],
            'friendship': ['friend', 'companion', 'ally', 'partner', 'buddy'],
            'conflict': ['enemy', 'rival', 'oppose', 'fight', 'conflict', 'hate',
                        'against', 'battle', 'war'],
            'professional': ['colleague', 'boss', 'employee', 'coworker', 'work'],
            'social': ['neighbor', 'acquaintance', 'know', 'meet']
        }
    
    def extract_relations(self, characters, sentences):
        """
        Ekstraksi hubungan antar karakter
        """
        # Method 1: Co-occurrence
        cooccurrence_relations = self._cooccurrence_analysis(characters, sentences)
        
        # Method 2: Rule-based extraction
        rulebased_relations = self._rulebased_extraction(characters, sentences)
        
        # Gabungkan hasil
        all_relations = self._merge_relations(cooccurrence_relations, rulebased_relations)
        
        return {
            'cooccurrence': cooccurrence_relations,
            'rulebased': rulebased_relations,
            'merged_relations': all_relations,
            'relation_graph': self._build_graph(all_relations)
        }
    
    def _cooccurrence_analysis(self, characters, sentences):
        """
        Analisis co-occurrence: karakter yang muncul di kalimat sama
        """
        cooccurrence = defaultdict(lambda: {'count': 0, 'sentences': []})
        
        char_list = list(characters.keys())
        
        for sent_id, sentence in enumerate(sentences):
            doc = self.nlp(sentence)
            persons_in_sent = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']
            
            # Filter hanya karakter utama
            present_characters = [p for p in persons_in_sent if p in char_list]
            
            # Hitung semua pasangan
            for char1, char2 in itertools.combinations(present_characters, 2):
                pair = tuple(sorted([char1, char2]))
                cooccurrence[pair]['count'] += 1
                cooccurrence[pair]['sentences'].append({
                    'id': sent_id,
                    'text': sentence
                })
        
        return dict(cooccurrence)
    
    def _rulebased_extraction(self, characters, sentences):
        """
        Rule-based extraction menggunakan pattern kata kunci
        """
        relations = []
        char_list = list(characters.keys())
        
        for sent_id, sentence in enumerate(sentences):
            doc = self.nlp(sentence)
            persons_in_sent = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']
            present_characters = [p for p in persons_in_sent if p in char_list]
            
            # Cek apakah ada 2+ karakter dan kata kunci relasi
            if len(present_characters) >= 2:
                relation_type = self._identify_relation_type(sentence.lower())
                
                if relation_type:
                    for char1, char2 in itertools.combinations(present_characters, 2):
                        relations.append({
                            'character1': char1,
                            'character2': char2,
                            'relation_type': relation_type,
                            'sentence_id': sent_id,
                            'sentence': sentence
                        })
        
        return relations
    
    def _identify_relation_type(self, sentence):
        """
        Identifikasi tipe relasi dari kalimat
        """
        for rel_type, keywords in self.relation_patterns.items():
            for keyword in keywords:
                if keyword in sentence:
                    return rel_type
        return 'unknown'
    
    def _merge_relations(self, cooccurrence, rulebased):
        """
        Menggabungkan hasil co-occurrence dan rule-based
        """
        merged = []
        
        # Dari co-occurrence
        for pair, data in cooccurrence.items():
            char1, char2 = pair
            relation_types = []
            
            # Cari tipe relasi dari rule-based
            for rule_rel in rulebased:
                if (rule_rel['character1'] == char1 and rule_rel['character2'] == char2) or \
                   (rule_rel['character1'] == char2 and rule_rel['character2'] == char1):
                    relation_types.append(rule_rel['relation_type'])
            
            merged.append({
                'character1': char1,
                'character2': char2,
                'cooccurrence_count': data['count'],
                'relation_types': list(set(relation_types)) if relation_types else ['cooccurrence'],
                'strength': self._calculate_strength(data['count'], relation_types)
            })
        
        return merged
    
    def _calculate_strength(self, cooccurrence_count, relation_types):
        """
        Hitung kekuatan relasi (0-1)
        """
        base_strength = min(cooccurrence_count / 10, 0.5)  # Max 0.5 dari co-occurrence
        type_bonus = len(relation_types) * 0.1  # 0.1 per tipe relasi
        return min(base_strength + type_bonus, 1.0)
    
    def _build_graph(self, relations):
        """
        Build graph representation untuk visualisasi
        """
        graph = {
            'nodes': [],
            'edges': []
        }
        
        characters = set()
        for rel in relations:
            characters.add(rel['character1'])
            characters.add(rel['character2'])
        
        graph['nodes'] = [{'id': char, 'label': char} for char in characters]
        
        for rel in relations:
            graph['edges'].append({
                'source': rel['character1'],
                'target': rel['character2'],
                'weight': rel['strength'],
                'types': rel['relation_types']
            })
        
        return graph