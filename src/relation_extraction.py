import spacy
from collections import defaultdict
import itertools
import re

class RelationExtractor:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_lg')
        
        # Relation patterns
        self.relation_patterns = {
            'family': [
                'mother', 'father', 'sister', 'brother', 'son', 'daughter', 
                'wife', 'husband', 'parent', 'child', 'family', 'married',
                'uncle', 'aunt', 'cousin', 'grandfather', 'grandmother'
            ],
            'romantic': [
                'love', 'lover', 'beloved', 'romance', 'marry', 'married',
                'kiss', 'embrace', 'affection', 'darling', 'dear', 'sweetheart',
                'adore', 'passion', 'heart', 'valentine', 'treasure'
            ],
            'friendship': [
                'friend', 'companion', 'ally', 'partner', 'buddy', 'pal',
                'comrade', 'mate', 'acquaintance'
            ],
            'conflict': [
                'enemy', 'rival', 'oppose', 'fight', 'conflict', 'hate',
                'against', 'battle', 'war', 'quarrel', 'argue', 'dispute'
            ],
            'professional': [
                'colleague', 'boss', 'employee', 'coworker', 'work', 'business',
                'client', 'customer', 'merchant', 'seller', 'buyer', 'sold', 'bought'
            ],
            'social': [
                'neighbor', 'acquaintance', 'know', 'meet', 'visit', 'see'
            ]
        }
        
        self.non_story_characters = {
            'sheba', 'solomon', 'magi', 'wise men', 'king solomon',
            'queen of sheba', 'god', 'jesus', 'moses'
        }
    
    def extract_relations(self, characters, sentences):
        """
        Ekstraksi hubungan menggunakan multiple methods
        """
        # Filter non-story characters
        story_characters = {char: count for char, count in characters.items()
                           if char.lower() not in self.non_story_characters}
        
        if len(story_characters) < 2:
            print("  ⚠️  Not enough characters for relation extraction")
            return self._empty_result()
        
        print(f"\n  [Relation Extraction] Analyzing {len(story_characters)} characters...")
        
        # Method 1: Direct co-occurrence (same sentence)
        direct_cooccurrence = self._direct_cooccurrence(story_characters, sentences)
        print(f"    ✓ Direct co-occurrence: {len(direct_cooccurrence)} pairs")
        
        # Method 2: Proximity-based (within window)
        proximity_relations = self._proximity_based(story_characters, sentences, window=5)
        print(f"    ✓ Proximity relations: {len(proximity_relations)} pairs")
        
        # Method 3: Semantic relation detection
        semantic_relations = self._semantic_relations(story_characters, sentences)
        print(f"    ✓ Semantic relations: {len(semantic_relations)} detected")
        
        # Method 4: Pronoun resolution hints
        pronoun_relations = self._pronoun_based_relations(story_characters, sentences)
        print(f"    ✓ Pronoun-based: {len(pronoun_relations)} hints")
        
        # Merge all methods
        all_relations = self._merge_all_methods(
            direct_cooccurrence, 
            proximity_relations, 
            semantic_relations,
            pronoun_relations
        )
        
        return {
            'direct_cooccurrence': direct_cooccurrence,
            'proximity': proximity_relations,
            'semantic': semantic_relations,
            'cooccurrence': direct_cooccurrence,  # For backward compatibility
            'rulebased': semantic_relations,       # For backward compatibility
            'merged_relations': all_relations,
            'relation_graph': self._build_graph(all_relations)
        }
    
    def _direct_cooccurrence(self, characters, sentences):
        """
        Direct co-occurrence dengan special handling untuk special characters
        """
        cooccurrence = defaultdict(lambda: {'count': 0, 'sentences': []})
        char_list = list(characters.keys())
        
        for sent_id, sentence in enumerate(sentences):
            sent_lower = sentence.lower()
            present = set()
            
            for char in char_list:
                char_lower = char.lower()
                
                # SPECIAL: Narrator (I)
                if char_lower == "narrator (i)":
                    if re.search(r'\bi\b', sent_lower):
                        present.add(char)
                # SPECIAL: Role names like "The Old Man"
                elif char_lower.startswith('the '):
                    role_name = char_lower.replace('the ', '')
                    if role_name in sent_lower:
                        present.add(char)
                # Regular names
                elif char_lower in sent_lower:
                    present.add(char)
                elif ' ' in char_lower:
                    first_name = char_lower.split()[0]
                    if first_name in sent_lower and len(first_name) >= 4:
                        present.add(char)
            
            # Count pairs
            for char1, char2 in itertools.combinations(present, 2):
                pair = tuple(sorted([char1, char2]))
                cooccurrence[pair]['count'] += 1
                cooccurrence[pair]['sentences'].append(sent_id)
        
        return dict(cooccurrence)
    
    def _proximity_based(self, characters, sentences, window=5):
        """
        Proximity-based dengan special handling untuk Narrator (I)
        """
        proximity = defaultdict(lambda: {'count': 0, 'method': 'proximity'})
        char_list = list(characters.keys())
        
        # Track last appearance of each character
        last_appearance = {char: [] for char in char_list}
        
        for sent_id, sentence in enumerate(sentences):
            sent_lower = sentence.lower()
            
            for char in char_list:
                char_lower = char.lower()
                found = False
                
                # SPECIAL: Handle "Narrator (I)"
                if char_lower == "narrator (i)":
                    # Look for first-person pronoun "I"
                    if re.search(r'\bi\b', sent_lower):
                        last_appearance[char].append(sent_id)
                        found = True
                # SPECIAL: Handle "The X" role names
                elif char_lower.startswith('the '):
                    role_name = char_lower.replace('the ', '')
                    if role_name in sent_lower:
                        last_appearance[char].append(sent_id)
                        found = True
                # Regular names
                else:
                    if char_lower in sent_lower:
                        last_appearance[char].append(sent_id)
                        found = True
                    elif ' ' in char_lower:
                        # For multi-word names, check first name
                        first_name = char_lower.split()[0]
                        if first_name in sent_lower and len(first_name) >= 4:
                            last_appearance[char].append(sent_id)
                            found = True
        
        # Find characters that appear within window of each other
        for char1, char2 in itertools.combinations(char_list, 2):
            appearances1 = last_appearance[char1]
            appearances2 = last_appearance[char2]
            
            proximity_count = 0
            for sent1 in appearances1:
                for sent2 in appearances2:
                    if abs(sent1 - sent2) <= window:
                        proximity_count += 1
            
            if proximity_count > 0:
                pair = tuple(sorted([char1, char2]))
                proximity[pair]['count'] = proximity_count
        
        return dict(proximity)
    
    def _semantic_relations(self, characters, sentences):
        """
        Detect semantic relations dari keywords
        """
        relations = []
        char_list = list(characters.keys())
        
        for sent_id, sentence in enumerate(sentences):
            sent_lower = sentence.lower()
            
            # Find which characters are mentioned (or nearby)
            relevant_chars = []
            for char in char_list:
                char_lower = char.lower()
                if char_lower in sent_lower:
                    relevant_chars.append(char)
                elif ' ' in char_lower:
                    first = char_lower.split()[0]
                    if first in sent_lower and len(first) >= 4:
                        relevant_chars.append(char)
            
            # Check for relation keywords
            relation_type = self._identify_relation_type(sent_lower)
            
            if relation_type and len(relevant_chars) >= 1:
                # Even if only one character mentioned, relation keyword suggests relationship
                # We'll infer relationship between main characters
                for char in relevant_chars:
                    relations.append({
                        'character': char,
                        'relation_type': relation_type,
                        'sentence_id': sent_id,
                        'sentence': sentence
                    })
        
        return relations
    
    def _pronoun_based_relations(self, characters, sentences):
        """
        Detect relations from pronoun usage and possessives
        """
        hints = defaultdict(list)
        char_list = list(characters.keys())
        
        # Track pronoun patterns
        pronoun_patterns = {
            'romantic/family': [
                r'\bhis\s+(?:wife|love|darling|dear|treasure)',
                r'\bher\s+(?:husband|love|darling|dear|treasure)',
                r'\btheir\s+(?:love|marriage|relationship|home)',
            ]
        }
        
        for sent_id, sentence in enumerate(sentences):
            sent_lower = sentence.lower()
            
            # Find characters in nearby sentences (context window)
            window_start = max(0, sent_id - 3)
            window_end = min(len(sentences), sent_id + 4)
            context_chars = set()
            
            for i in range(window_start, window_end):
                ctx_sent = sentences[i].lower()
                for char in char_list:
                    char_lower = char.lower()
                    if char_lower in ctx_sent:
                        context_chars.add(char)
            
            # Check for pronoun patterns
            for rel_type, patterns in pronoun_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, sent_lower):
                        for char in context_chars:
                            hints[tuple(sorted(context_chars))].append({
                                'type': rel_type,
                                'sentence_id': sent_id
                            })
        
        return dict(hints)
    
    def _merge_all_methods(self, direct, proximity, semantic, pronoun):
        """
        Merge semua metode deteksi
        """
        merged = {}
        
        # Start with proximity (lebih reliable untuk cerita seperti Gift of Magi)
        for pair, data in proximity.items():
            char1, char2 = pair
            
            merged[pair] = {
                'character1': char1,
                'character2': char2,
                'proximity_count': data['count'],
                'direct_cooccurrence': direct.get(pair, {}).get('count', 0),
                'relation_types': [],
                'evidence': []
            }
        
        # Add direct co-occurrence pairs not in proximity
        for pair, data in direct.items():
            if pair not in merged:
                char1, char2 = pair
                merged[pair] = {
                    'character1': char1,
                    'character2': char2,
                    'proximity_count': 0,
                    'direct_cooccurrence': data['count'],
                    'relation_types': [],
                    'evidence': []
                }
        
        # Add semantic relations
        for rel_data in semantic:
            char = rel_data['character']
            rel_type = rel_data['relation_type']
            
            # Add this relation type to all pairs involving this character
            for pair in merged.keys():
                if char in pair:
                    if rel_type not in merged[pair]['relation_types']:
                        merged[pair]['relation_types'].append(rel_type)
                        merged[pair]['evidence'].append({
                            'type': 'semantic',
                            'sentence_id': rel_data['sentence_id']
                        })
        
        # Calculate final strength and format
        result = []
        for pair, data in merged.items():
            # Calculate strength
            proximity_score = min(data['proximity_count'] / 30, 0.4)
            direct_score = min(data['direct_cooccurrence'] / 5, 0.3)
            semantic_score = len(data['relation_types']) * 0.15
            
            strength = min(proximity_score + direct_score + semantic_score, 1.0)
            
            if not data['relation_types']:
                data['relation_types'] = ['cooccurrence']
            
            result.append({
                'character1': data['character1'],
                'character2': data['character2'],
                'cooccurrence_count': data['proximity_count'] + data['direct_cooccurrence'],
                'proximity_count': data['proximity_count'],
                'direct_count': data['direct_cooccurrence'],
                'relation_types': data['relation_types'],
                'strength': strength
            })
        
        # Sort by strength
        result.sort(key=lambda x: x['strength'], reverse=True)
        
        return result
    
    def _identify_relation_type(self, sentence):
        """
        Identify relation type from sentence
        """
        for rel_type, keywords in self.relation_patterns.items():
            for keyword in keywords:
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, sentence):
                    return rel_type
        return None
    
    def _build_graph(self, relations):
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
                'weight': rel['strength'],
                'types': rel['relation_types'],
                'count': rel['cooccurrence_count']
            })
        
        return graph
    
    def _empty_result(self):
        """
        Return empty result structure
        """
        return {
            'cooccurrence': {},
            'rulebased': [],
            'merged_relations': [],
            'relation_graph': {'nodes': [], 'edges': []}
        }