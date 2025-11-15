"""
FINAL FIX: Proximity-based relation detection
Replace src/relation_extraction.py with this code
"""

import spacy
from collections import defaultdict
import itertools
import re

class RelationExtractor:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_lg')
        
        # Specific relation patterns
        self.specific_relation_patterns = {
            'parent-child': [
                r'\b(?:mother|father|parent|mom|dad|mama|papa)\s+(?:of|to)\b',
                r'\b(?:son|daughter|child|children)\s+(?:of|to)\b',
                r'\bmy\s+(?:mother|father|parent|son|daughter|child)\b',
                r'\bher\s+(?:mother|father|parent|son|daughter|child)\b',
                r'\bhis\s+(?:mother|father|parent|son|daughter|child)\b',
                r'\bgave birth to\b', r'\braised\b', r'\bborn to\b'
            ],
            'siblings': [
                r'\bmy\s+(?:brother|sister|sibling)\b',
                r'\bher\s+(?:brother|sister|sibling)\b',
                r'\bhis\s+(?:brother|sister|sibling)\b',
                r'\btwin\b', r'\bolder brother\b', r'\byounger sister\b'
            ],
            'spouse': [
                r'\b(?:husband|wife|spouse)\b',
                r'\bmarried to\b', r'\bwedding\b', r'\bbride|groom\b',
                r'\bhis wife\b', r'\bher husband\b',
                r'\bmy wife\b', r'\bmy husband\b'
            ],
            'extended-family': [
                r'\b(?:uncle|aunt|cousin|nephew|niece)\b',
                r'\b(?:grandfather|grandmother|grandparent)\b',
                r'\b(?:grandson|granddaughter|grandchild)\b'
            ],
            'married-couple': [
                r'\bmarried\b', r'\bhusband and wife\b', r'\bspouse\b',
                r'\bwedding\b', r'\bmarriage\b',
                r'\bhis wife\b', r'\bher husband\b',
                r'\btheir home\b', r'\btheir marriage\b',
                r'\bmarried life\b', r'\bmarried couple\b'
            ],
            'lovers': [
                r'\blover\b', r'\bboyfriend|girlfriend\b', r'\bsweetheart\b',
                r'\bdating\b', r'\bin love with\b',
                r'\bdarling\b', r'\bhoney\b', r'\bbeloved\b',
                r'\bmy love\b', r'\bI love\b', r'\bshe loves\b', r'\bhe loves\b'
            ],
            'romantic-interest': [
                r'\bcrush\b', r'\badmire[sd]?\b',
                r'\baffection for\b', r'\bfond of\b',
                r'\btreasure\b', r'\bprecious\b',
                r'\badore[sd]?\b', r'\bcherish\b'
            ],
            'close-friends': [
                r'\bbest friend\b', r'\bclose friend\b',
                r'\binseparable\b', r'\bconfidant\b'
            ],
            'acquaintances': [
                r'\bacquaintance\b', r'\bknow each other\b', r'\bmet\b'
            ],
            'companions': [
                r'\bcompanion\b', r'\bcomrade\b', r'\bally\b', r'\bpartner\b'
            ],
            'employer-employee': [
                r'\bboss\b', r'\bemployer\b', r'\bemployee\b',
                r'\bworks for\b', r'\bhired\b'
            ],
            'colleagues': [
                r'\bcolleague\b', r'\bcoworker\b',
                r'\bwork together\b', r'\bteammate\b'
            ],
            'business-partners': [
                r'\bbusiness partner\b', r'\bpartnership\b', r'\bjoint venture\b'
            ],
            'customer-merchant': [
                r'\b(?:customer|client)\b', r'\b(?:merchant|seller|vendor)\b',
                r'\bbought from\b', r'\bsold to\b', 
                r'\bpurchased from\b', r'\bsold (?:it|them) to\b'
            ],
            'enemies': [
                r'\benemy\b', r'\bhate[sd]?\b', r'\bfoe\b', r'\badversary\b'
            ],
            'rivals': [
                r'\brival\b', r'\bcompete\b', r'\bcompetition\b', r'\bcontest\b'
            ],
            'victim-perpetrator': [
                r'\bvictim\b', r'\battacker\b', r'\bkilled\b',
                r'\bmurdered\b', r'\bhurt\b', r'\bharmed\b'
            ],
            'opposing-sides': [
                r'\bagainst\b', r'\boppose[sd]?\b',
                r'\bconfronted\b', r'\bfight\b'
            ],
            'neighbors': [
                r'\bneighbor\b', r'\bnext door\b', r'\blive nearby\b'
            ],
            'teacher-student': [
                r'\bteacher\b', r'\bstudent\b', r'\bpupil\b',
                r'\bmentor\b', r'\btaught\b', r'\blearned from\b'
            ],
            'master-servant': [
                r'\bmaster\b', r'\bservant\b', r'\bserve[sd]?\b', r'\bslave\b'
            ]
        }
        
        # Possessive pronouns mapping
        self.possessive_mapping = {
            'his': 'male',
            'her': 'female',
            'their': 'plural'
        }
        
        self.non_story_characters = {
            'sheba', 'solomon', 'magi', 'wise men', 'king solomon',
            'queen of sheba', 'god', 'jesus', 'moses'
        }
    
    def extract_relations(self, characters, sentences):
        """Extract relations with PROXIMITY detection"""
        story_characters = {char: count for char, count in characters.items()
                           if char.lower() not in self.non_story_characters}
        
        if len(story_characters) < 2:
            print("  ⚠️  Not enough characters for relation extraction")
            return self._empty_result()
        
        print(f"\n  [Enhanced Relation Extraction] Analyzing {len(story_characters)} characters...")
        
        # Build character mention index
        char_mentions = self._index_character_mentions(story_characters, sentences)
        
        # Step 1: Co-occurrence (exact same sentence)
        cooccurrence = self._detect_cooccurrence(story_characters, sentences)
        print(f"    ✓ Co-occurrence pairs: {len(cooccurrence)}")
        
        # Step 2: Proximity detection (within N sentences)
        proximity_pairs = self._detect_proximity_pairs(char_mentions, window=10)
        print(f"    ✓ Proximity pairs: {len(proximity_pairs)}")
        
        # Step 3: Detect relations in proximity contexts
        specific_relations = self._detect_relations_in_proximity(
            proximity_pairs, sentences, char_mentions
        )
        print(f"    ✓ Specific relations detected: {len(specific_relations)}")
        
        # Step 4: Possessive pronoun inference
        possessive_relations = self._detect_possessive_relations(
            story_characters, sentences, char_mentions
        )
        print(f"    ✓ Possessive relations: {len(possessive_relations)}")
        
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
        """Build index of where each character is mentioned"""
        mentions = defaultdict(list)
        
        for sent_id, sentence in enumerate(sentences):
            sent_lower = sentence.lower()
            
            for char in characters.keys():
                if self._is_character_in_sentence(char, sent_lower):
                    mentions[char].append(sent_id)
        
        return dict(mentions)
    
    def _detect_proximity_pairs(self, char_mentions, window=10):
        """Detect character pairs that appear within N sentences of each other"""
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
        """Detect relations in sentences near where characters appear"""
        relations = []
        
        for pair, data in proximity_pairs.items():
            char1, char2 = pair
            
            # Get all sentences in proximity
            sent_ids = set()
            for sent1, sent2 in data['sentence_pairs']:
                # Include sentences in between and around
                start = min(sent1, sent2) - 2
                end = max(sent1, sent2) + 3
                sent_ids.update(range(max(0, start), min(len(sentences), end)))
            
            # Check these sentences for relation patterns
            relation_evidence = defaultdict(list)
            
            for sent_id in sent_ids:
                sentence = sentences[sent_id]
                sent_lower = sentence.lower()
                
                # Check if either character is mentioned
                has_char1 = char1.lower() in sent_lower or self._get_first_name(char1).lower() in sent_lower
                has_char2 = char2.lower() in sent_lower or self._get_first_name(char2).lower() in sent_lower
                
                # Only check if at least one character is in this sentence
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
    
    def _detect_possessive_relations(self, characters, sentences, char_mentions):
        """Detect relations from possessive pronouns (his wife, her husband, etc.)"""
        relations = []
        
        # Possessive patterns
        possessive_patterns = {
            'spouse': [
                (r'\bhis\s+wife\b', 'male_spouse'),
                (r'\bher\s+husband\b', 'female_spouse'),
            ],
            'romantic': [
                (r'\bhis\s+(?:love|darling|treasure|precious)\b', 'male_romantic'),
                (r'\bher\s+(?:love|darling|treasure|precious)\b', 'female_romantic'),
            ]
        }
        
        for char in characters.keys():
            char_lower = char.lower()
            char_first = self._get_first_name(char).lower()
            
            # Get sentences where this character appears
            sent_ids = char_mentions.get(char, [])
            
            for sent_id in sent_ids:
                sentence = sentences[sent_id]
                sent_lower = sentence.lower()
                
                # Check for possessive patterns NEAR this character
                for relation_type, patterns in possessive_patterns.items():
                    for pattern, pattern_name in patterns:
                        matches = list(re.finditer(pattern, sent_lower))
                        
                        if matches:
                            # Find which other character might be referenced
                            for other_char in characters.keys():
                                if other_char == char:
                                    continue
                                
                                other_lower = other_char.lower()
                                other_first = self._get_first_name(other_char).lower()
                                
                                # Check if other character is nearby
                                nearby_sent_ids = range(
                                    max(0, sent_id - 5),
                                    min(len(sentences), sent_id + 6)
                                )
                                
                                other_nearby = any(
                                    other_lower in sentences[sid].lower() or 
                                    other_first in sentences[sid].lower()
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
                                        'confidence': 0.80,
                                        'source': 'possessive'
                                    })
        
        return relations
    
    def _detect_cooccurrence(self, characters, sentences):
        """Detect exact co-occurrence (both in same sentence)"""
        cooccurrence = defaultdict(lambda: {
            'count': 0, 'sentences': [], 'contexts': []
        })
        
        char_list = list(characters.keys())
        
        for sent_id, sentence in enumerate(sentences):
            sent_lower = sentence.lower()
            present = []
            
            for char in char_list:
                if self._is_character_in_sentence(char, sent_lower):
                    present.append(char)
            
            for char1, char2 in itertools.combinations(present, 2):
                pair = tuple(sorted([char1, char2]))
                cooccurrence[pair]['count'] += 1
                cooccurrence[pair]['sentences'].append(sent_id)
                cooccurrence[pair]['contexts'].append(sentence)
        
        return dict(cooccurrence)
    
    def _merge_and_rank_relations(self, all_relations, cooccurrence, proximity_pairs):
        """Merge and rank relations - PRIORITIZE personal relationships"""
        relations_by_pair = defaultdict(lambda: {
            'relations': [], 
            'cooccurrence_count': 0,
            'proximity_count': 0
        })
        
        # Relation type priority (higher = more important)
        relation_priority = {
            'victim-perpetrator': 11,  # HIGHEST for crime/violence stories
            'married-couple': 10,
            'spouse': 10,
            'lovers': 9,
            'romantic-interest': 8,
            'parent-child': 9,
            'siblings': 8,
            'close-friends': 7,
            'extended-family': 6,
            'companions': 5,
            'acquaintances': 4,
            'colleagues': 3,
            'customer-merchant': 1,  # LOW priority
            'neighbors': 3,
            'enemies': 6,
            'rivals': 5,
            'opposing-sides': 5
        }
        
        for rel in all_relations:
            pair = tuple(sorted([rel['character1'], rel['character2']]))
            
            # Add priority boost to confidence for important relations
            priority = relation_priority.get(rel['relation_type'], 2)
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
    
    def _get_first_name(self, full_name):
        """Get first name from full name"""
        return full_name.split()[0] if ' ' in full_name else full_name
    
    def _is_character_in_sentence(self, character, sentence_lower):
        """Check if character in sentence"""
        char_lower = character.lower()
        
        if char_lower == "narrator (i)":
            return bool(re.search(r'\bi\b', sentence_lower))
        
        if char_lower.startswith('the '):
            role = char_lower.replace('the ', '')
            return role in sentence_lower
        
        if char_lower in sentence_lower:
            return True
        
        if ' ' in char_lower:
            first_name = char_lower.split()[0]
            if len(first_name) >= 3:
                return first_name in sentence_lower
        
        return False
    
    def _calculate_proximity_confidence(self, evidence_count, proximity_count, has_both):
        """Calculate confidence for proximity-based detection"""
        base = min(evidence_count / 2.0, 0.85)
        
        # Boost if both characters in same evidence sentence
        if has_both:
            base = min(base + 0.1, 0.95)
        
        # Boost based on proximity frequency
        if proximity_count > 5:
            base = min(base + 0.05, 0.95)
        
        return base
    
    def _calculate_overall_strength(self, confidence, cooccurrence, proximity, relation_count):
        """Calculate overall strength - PRIORITIZE romantic/family over transactional"""
        confidence_score = confidence * 0.4
        cooccurrence_score = min(cooccurrence / 10.0, 0.2)
        proximity_score = min(proximity / 20.0, 0.2)
        diversity_score = min(relation_count / 5.0, 0.2)
        
        return min(confidence_score + cooccurrence_score + proximity_score + diversity_score, 1.0)
    
    def _build_detailed_graph(self, relations):
        """Build graph"""
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
        """Return empty result"""
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