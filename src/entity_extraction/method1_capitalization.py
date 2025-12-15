"""
Method 1: Capitalization-Based Entity Extraction
File: src/entity_extraction/method1_capitalization.py
"""

from typing import Dict, List, Any
from collections import Counter
import re
from .base_extractor import BaseEntityExtractor

class CapitalizationExtractor(BaseEntityExtractor):
    """
    Extract entities based on capitalization patterns and frequency
    
    Features:
    - Consistent capitalization detection
    - Position-based weighting (mid-sentence vs sentence-start)
    - Pattern-based enhancement (Mr./Mrs./The)
    - Blacklist filtering
    """
    
    def get_default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'min_mentions': 3,
            'consistency_threshold': 0.65,
            'position_weights': {
                'sentence_start': 0.5,
                'mid_sentence': 1.0,
                'after_punctuation': 0.7
            },
            'blacklist': self._get_default_blacklist(),
            'title_patterns': [
                r'^(Mr|Mrs|Miss|Ms|Dr|Sir|Madam|Mme)\.?\s+\w+',
                r'^The\s+[A-Z]\w+(\s+[A-Z]\w+)*$'
            ]
        }
    
    def _get_default_blacklist(self) -> set:
        """Default blacklist of non-character entities"""
        return {
            # Temporal
            'christmas', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
            'saturday', 'sunday', 'january', 'february', 'march', 'april', 'may',
            'june', 'july', 'august', 'september', 'october', 'november', 'december',
            'today', 'tomorrow', 'yesterday',
            
            # Religious/Mythological
            'god', 'lord', 'jesus', 'christ', 'moses', 'magi', 'wise men',
            
            # Geographic (common false positives)
            'america', 'american', 'alabama', 'england', 'english',
            
            # Political/Military
            'federal', 'yankee', 'yanks', 'confederate', 'union',
            
            # Common false positives
            'dear', 'young', 'old',
            
            # Story-specific references (mythological)
            'sheba', 'solomon', 'king solomon', 'queen of sheba',
            
            # Pronouns (just in case)
            'i', 'he', 'she', 'they', 'we', 'you'
        }
    
    def get_method_name(self) -> str:
        return "Method1_Capitalization"
    
    def extract(self, preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract entities based on capitalization patterns
        
        Pipeline:
        1. Get capitalization patterns from preprocessing
        2. Apply frequency filtering
        3. Calculate weighted scores
        4. Apply pattern enhancements
        5. Filter by blacklist
        6. Return scored candidates
        """
        self.validate_input(preprocessed_data)
        self.logger.info("Starting capitalization-based extraction...")
        
        cap_patterns = preprocessed_data['capitalization_patterns']
        sentences = preprocessed_data['sentences']
        ngrams_data = preprocessed_data['ngrams']
        
        # Step 1: Frequency filtering
        candidates = self._filter_by_frequency(
            cap_patterns, 
            min_mentions=self.config['min_mentions']
        )
        self.logger.info(f"  After frequency filter: {len(candidates)} candidates")
        
        # Step 2: Calculate weighted scores
        scored_candidates = []
        for candidate, data in candidates.items():
            score = self._calculate_weighted_score(candidate, data)
            
            scored_candidates.append({
                'name': candidate,
                'score': score,
                'mentions': data['total_mentions'],
                'metadata': {
                    'consistency': data['consistency_score'],
                    'mid_sentence_ratio': data['mid_sentence_count'] / data['total_mentions'],
                    'capitalization_ratio': data['capitalized_mentions'] / data['total_mentions']
                }
            })
        
        self.logger.info(f"  Scored {len(scored_candidates)} candidates")
        
        # Step 3: Pattern enhancements
        enhanced_candidates = self._apply_pattern_enhancements(
            scored_candidates, 
            sentences
        )
        
        # Step 4: Merge n-grams
        merged_candidates = self._merge_ngrams(
            enhanced_candidates,
            ngrams_data
        )
        
        # Step 5: Blacklist filtering
        filtered_candidates = self._filter_blacklist(merged_candidates)
        self.logger.info(f"  After blacklist: {len(filtered_candidates)} candidates")
        
        # Sort by score
        final_candidates = self.sort_by_score(filtered_candidates)
        
        return {
            'candidates': final_candidates,
            'method_name': self.get_method_name(),
            'statistics': self.get_statistics({'candidates': final_candidates})
        }
    
    def _filter_by_frequency(self, cap_patterns: Dict, min_mentions: int) -> Dict:
        """Filter candidates by minimum mentions"""
        return {
            candidate: data 
            for candidate, data in cap_patterns.items()
            if data['total_mentions'] >= min_mentions
        }
    
    def _calculate_weighted_score(self, candidate: str, data: Dict) -> float:
        """
        Calculate weighted score based on:
        - Consistency of capitalization
        - Position in sentences
        - Frequency
        """
        # Base: consistency score
        consistency = data['consistency_score']
        
        # Frequency component (normalized)
        frequency_score = min(data['total_mentions'] / 50.0, 1.0)
        
        # Position weighting
        total = data['total_mentions']
        if total > 0:
            mid_ratio = data['mid_sentence_count'] / total
            position_score = mid_ratio * self.config['position_weights']['mid_sentence']
        else:
            position_score = 0.0
        
        # Weighted combination
        final_score = (
            consistency * 0.4 +
            frequency_score * 0.3 +
            position_score * 0.3
        )
        
        return min(final_score, 1.0)
    
    def _apply_pattern_enhancements(self, candidates: List[Dict], sentences: List[str]) -> List[Dict]:
        """
        Enhance detection for specific patterns (Mr./Mrs./The)
        """
        enhanced = []
        
        for candidate_dict in candidates:
            candidate = candidate_dict['name']
            
            # Check if matches title patterns
            for pattern in self.config['title_patterns']:
                if re.match(pattern, candidate):
                    # Boost score
                    candidate_dict['score'] = min(candidate_dict['score'] + 0.1, 1.0)
                    candidate_dict['metadata']['has_title'] = True
                    break
            
            enhanced.append(candidate_dict)
        
        return enhanced
    
    def _merge_ngrams(self, candidates: List[Dict], ngrams_data: Dict) -> List[Dict]:
        """
        Merge overlapping n-grams
        Example: "James", "Dillingham", "Young" + "James Dillingham Young"
        """
        # Get all bigrams and trigrams
        bigrams = set(ngrams_data.get('bigrams', []))
        trigrams = set(ngrams_data.get('trigrams', []))
        
        # Build name mapping
        name_to_fullname = {}
        
        # Check trigrams first
        for trigram in trigrams:
            parts = trigram.split()
            for part in parts:
                if part not in name_to_fullname:
                    name_to_fullname[part] = trigram
        
        # Then bigrams
        for bigram in bigrams:
            parts = bigram.split()
            for part in parts:
                if part not in name_to_fullname:
                    name_to_fullname[part] = bigram
        
        # Merge candidates
        merged = {}
        for candidate_dict in candidates:
            name = candidate_dict['name']
            
            # Check if this is part of a larger name
            if name in name_to_fullname:
                full_name = name_to_fullname[name]
                
                # Merge into full name
                if full_name not in merged:
                    merged[full_name] = candidate_dict.copy()
                    merged[full_name]['name'] = full_name
                    merged[full_name]['metadata']['merged_from'] = [name]
                else:
                    # Accumulate mentions
                    merged[full_name]['mentions'] += candidate_dict['mentions']
                    merged[full_name]['metadata']['merged_from'].append(name)
            else:
                # Keep as is
                if name not in merged:
                    merged[name] = candidate_dict
        
        return list(merged.values())
    
    def _filter_blacklist(self, candidates: List[Dict]) -> List[Dict]:
        """Filter out blacklisted entities"""
        blacklist = self.config['blacklist']
        
        filtered = []
        for candidate_dict in candidates:
            name_lower = candidate_dict['name'].lower()
            
            # Check if any part of the name is blacklisted
            is_blacklisted = False
            for blacklisted in blacklist:
                if blacklisted in name_lower or name_lower in blacklisted:
                    is_blacklisted = True
                    break
            
            if not is_blacklisted:
                filtered.append(candidate_dict)
        
        return filtered
    
    def get_confidence_score(self, candidate: str, metadata: Dict) -> float:
        """
        Calculate confidence for a candidate
        (Already calculated in _calculate_weighted_score)
        """
        return metadata.get('score', 0.0)