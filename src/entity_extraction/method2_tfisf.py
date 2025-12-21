"""
Method 2: TF-ISF Based Entity Extraction (Updated from TF-IDF)
File: src/entity_extraction/method2_tfisf.py

TF-ISF (Term Frequency - Inverse Sentence Frequency):
- More suitable for single-document analysis
- Sentence-level granularity matches literary analysis
- Better discrimination for character importance
"""

from typing import Dict, List, Any, Tuple
import numpy as np
from collections import defaultdict, Counter
import re
from .base_extractor import BaseEntityExtractor

class TFISFExtractor(BaseEntityExtractor):
    """
    Extract entities using TF-ISF (sentence-level) ranking
    
    Advantages over TF-IDF:
    - Sentence-level granularity (not document-level)
    - Meaningful for single document analysis
    - Better character importance discrimination
    - Natural alignment with context extraction
    
    Features:
    - TF-ISF calculation for PROPN candidates
    - Sentence-level importance ranking
    - Variant detection via string similarity
    - Character prominence scoring
    """
    
    def get_default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'min_tfisf_score': 0.015,   # Minimum TF-ISF threshold (adjusted for ISF)
            'max_sf': 0.8,              # Max sentence frequency (filter too common)
            'min_sf': 2,                # Min sentence frequency (filter rare)
            'top_k': 15,                # Top K entities
            'similarity_threshold': 0.6, # For variant detection
            'min_mentions': 2,          # Minimum mentions
            'prominence_boost': 0.1,     # Boost for prominent characters
            'vocabulary_type': 'propn_only'
        }
    
    def get_method_name(self) -> str:
        return "Method2_TFISF"
    
    def extract(self, preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract entities using TF-ISF ranking
        
        Pipeline:
        1. Build candidate list from PROPN + n-grams
        2. Calculate sentence frequency (SF) for each candidate
        3. Calculate TF-ISF scores
        4. Rank candidates by prominence
        5. Detect variants via similarity
        6. Apply filtering and return results
        """
        self.validate_input(preprocessed_data)
        self.logger.info("Starting TF-ISF based extraction...")
        
        # Step 1: Get candidates and sentences
        candidates = self._collect_candidates(preprocessed_data)
        sentences = preprocessed_data['sentences']
        self.logger.info(f"  Collected {len(candidates)} candidates from {len(sentences)} sentences")
        
        # Step 2: Calculate sentence frequency
        sentence_freq = self._calculate_sentence_frequency(candidates, sentences)
        self.logger.info(f"  Calculated sentence frequencies")
        
        # Step 3: Calculate TF-ISF scores
        tfisf_scores = self._calculate_tfisf(
            candidates,
            sentences,
            sentence_freq
        )
        self.logger.info(f"  Calculated TF-ISF scores")
        
        # Step 4: Calculate prominence and rank
        ranked_candidates = self._rank_by_prominence(
            tfisf_scores,
            sentence_freq,
            sentences,
            preprocessed_data.get('capitalization_patterns', {})
        )
        self.logger.info(f"  Ranked {len(ranked_candidates)} candidates by prominence")
        
        # Step 5: Detect variants
        variants_map = self._detect_variants_simple(ranked_candidates)
        self.logger.info(f"  Detected {len(variants_map)} variant groups")
        
        # Step 6: Filter by thresholds
        filtered_candidates = self._apply_filters(
            ranked_candidates,
            sentence_freq,
            len(sentences)
        )
        self.logger.info(f"  After filtering: {len(filtered_candidates)} candidates")
        
        # Sort and take top K
        final_candidates = sorted(
            filtered_candidates,
            key=lambda x: x['score'],
            reverse=True
        )[:self.config['top_k']]
        
        return {
            'candidates': final_candidates,
            'method_name': self.get_method_name(),
            'statistics': self.get_statistics({'candidates': final_candidates}),
            'variants_map': variants_map,
            'sentence_frequencies': sentence_freq
        }
    
    def _collect_candidates(self, preprocessed_data: Dict[str, Any]) -> List[str]:
        """
        Collect all candidates from PROPN and n-grams
        """
        candidates = set(preprocessed_data['propn_candidates'])
        ngrams_data = preprocessed_data.get('ngrams', {})
        
        # Add n-grams
        candidates.update(ngrams_data.get('unigrams', []))
        candidates.update(ngrams_data.get('bigrams', []))
        candidates.update(ngrams_data.get('trigrams', []))
        
        return sorted(list(candidates))
    
    def _calculate_sentence_frequency(self,
                                     candidates: List[str],
                                     sentences: List[str]) -> Dict[str, int]:
        """
        Calculate how many sentences contain each candidate
        
        SF(term) = number of sentences containing term
        
        Returns:
            Dict mapping candidate -> sentence count
        """
        sentence_freq = defaultdict(int)
        
        for sentence in sentences:
            sent_lower = sentence.lower()
            
            # Track which candidates appear in this sentence
            found_in_sentence = set()
            
            for candidate in candidates:
                candidate_lower = candidate.lower()
                
                # Use word boundary regex to avoid partial matches
                pattern = r'\b' + re.escape(candidate_lower) + r'\b'
                if re.search(pattern, sent_lower):
                    found_in_sentence.add(candidate)
            
            # Increment SF for all found candidates
            for candidate in found_in_sentence:
                sentence_freq[candidate] += 1
        
        return dict(sentence_freq)
    
    def _calculate_tfisf(self,
                        candidates: List[str],
                        sentences: List[str],
                        sentence_freq: Dict[str, int]) -> Dict[str, float]:
        """
        Calculate TF-ISF scores
        
        Formula:
        TF-ISF(term) = TF(term) × ISF(term)
        
        where:
        TF(term) = average term frequency per sentence where term appears
        ISF(term) = log(total_sentences / sentences_containing_term)
        
        Returns:
            Dict mapping candidate -> TF-ISF score
        """
        total_sentences = len(sentences)
        tfisf_scores = {}
        
        for candidate in candidates:
            sf = sentence_freq.get(candidate, 0)
            
            if sf == 0:
                tfisf_scores[candidate] = 0.0
                continue
            
            # Calculate ISF
            isf = np.log(total_sentences / sf)
            
            # Calculate average TF in sentences where candidate appears
            candidate_lower = candidate.lower()
            pattern = r'\b' + re.escape(candidate_lower) + r'\b'
            
            tf_sum = 0.0
            sentence_count = 0
            
            for sentence in sentences:
                sent_lower = sentence.lower()
                
                # Check if candidate in sentence
                if re.search(pattern, sent_lower):
                    # Count occurrences in this sentence
                    term_count = len(re.findall(pattern, sent_lower))
                    
                    # Calculate TF for this sentence
                    tokens = sentence.split()
                    if len(tokens) > 0:
                        tf = term_count / len(tokens)
                        tf_sum += tf
                        sentence_count += 1
            
            # Average TF
            avg_tf = tf_sum / sentence_count if sentence_count > 0 else 0.0
            
            # TF-ISF
            tfisf_scores[candidate] = avg_tf * isf
        
        return tfisf_scores
    
    def _rank_by_prominence(self,
                           tfisf_scores: Dict[str, float],
                           sentence_freq: Dict[str, int],
                           sentences: List[str],
                           cap_patterns: Dict[str, Any]) -> List[Dict]:
        """
        Rank candidates by prominence (TF-ISF + additional features)
        
        Scoring components:
        1. Base TF-ISF score (70%)
        2. Sentence frequency ratio (20%)
        3. Mention density (10%)
        """
        ranked = []
        total_sentences = len(sentences)
        
        for candidate, tfisf_score in tfisf_scores.items():
            sf = sentence_freq.get(candidate, 0)
            
            if sf == 0:
                continue
            
            # Component 1: Base TF-ISF (normalized)
            base_score = tfisf_score
            
            # Component 2: Sentence frequency ratio
            # Characters appearing in moderate % of sentences (10-30%) get boost
            sf_ratio = sf / total_sentences
            if 0.10 <= sf_ratio <= 0.30:
                sf_component = 0.1  # Boost for main characters
            elif sf_ratio > 0.30:
                sf_component = 0.05  # Slight penalty for ubiquitous terms
            else:
                sf_component = 0.0
            
            # Component 3: Mention density (from capitalization patterns)
            mention_count = 0
            if candidate in cap_patterns:
                mention_count = cap_patterns[candidate]['total_mentions']
            
            mention_density = min(mention_count / 100.0, 0.1)
            
            # Final prominence score
            prominence_score = base_score + sf_component + mention_density
            
            ranked.append({
                'name': candidate,
                'score': prominence_score,
                'mentions': mention_count,
                'metadata': {
                    'tfisf_score': tfisf_score,
                    'base_score': base_score,
                    'sentence_frequency': sf,
                    'sf_ratio': sf_ratio,
                    'mention_density': mention_density,
                    'sentences_with_mention': sf
                }
            })
        
        return ranked
    
    def _detect_variants_simple(self, candidates: List[Dict]) -> Dict[str, List[str]]:
        """
        Detect name variants using string similarity
        
        Simpler than TF-IDF version (no vector similarity needed)
        
        Returns:
            Dict mapping canonical name -> list of variants
        """
        variants_map = defaultdict(list)
        processed = set()
        
        candidate_names = [c['name'] for c in candidates]
        
        for i, name1 in enumerate(candidate_names):
            if name1 in processed:
                continue
            
            canonical = name1
            variants = [name1]
            
            for j, name2 in enumerate(candidate_names[i+1:], start=i+1):
                if name2 in processed:
                    continue
                
                # String similarity
                similarity = self._string_similarity(name1, name2)
                
                if similarity >= self.config['similarity_threshold']:
                    variants.append(name2)
                    processed.add(name2)
            
            if len(variants) > 1:
                # Choose canonical: prefer longer names
                canonical = max(variants, key=len)
                variants_map[canonical] = variants
            
            processed.add(name1)
        
        return dict(variants_map)
    
    def _string_similarity(self, name1: str, name2: str) -> float:
        """
        Calculate string-based similarity
        
        Rules:
        - Exact match: 1.0
        - Substring match: ratio based on length
        - First name match: 0.7
        - Character overlap (Jaccard): calculated
        """
        n1_lower = name1.lower().strip()
        n2_lower = name2.lower().strip()
        
        # Exact match
        if n1_lower == n2_lower:
            return 1.0
        
        # Substring match
        if n1_lower in n2_lower or n2_lower in n1_lower:
            shorter = min(len(n1_lower), len(n2_lower))
            longer = max(len(n1_lower), len(n2_lower))
            return shorter / longer
        
        # First name match (for multi-word names)
        parts1 = n1_lower.split()
        parts2 = n2_lower.split()
        
        if len(parts1) > 0 and len(parts2) > 0:
            if parts1[0] == parts2[0] and len(parts1[0]) >= 3:
                return 0.7
        
        # Possessive forms: "Jim" vs "Jims" or "Jim's"
        n1_clean = re.sub(r"'?s$", "", n1_lower)
        n2_clean = re.sub(r"'?s$", "", n2_lower)
        
        if n1_clean == n2_clean:
            return 0.9
        
        # Character overlap (Jaccard similarity)
        set1 = set(n1_lower)
        set2 = set(n2_lower)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        if union > 0:
            return intersection / union
        
        return 0.0
    
    def _apply_filters(self,
                      candidates: List[Dict],
                      sentence_freq: Dict[str, int],
                      total_sentences: int) -> List[Dict]:
        """
        Apply filtering based on thresholds
        
        Filters:
        1. Minimum TF-ISF score
        2. Minimum sentence frequency
        3. Maximum sentence frequency (too common)
        4. Minimum mentions
        """
        filtered = []
        
        max_sf_threshold = int(total_sentences * self.config['max_sf'])
        
        for candidate in candidates:
            name = candidate['name']
            score = candidate['score']
            mentions = candidate['mentions']
            sf = sentence_freq.get(name, 0)
            
            # Filter 1: Minimum TF-ISF score
            tfisf_score = candidate['metadata']['tfisf_score']
            if tfisf_score < self.config['min_tfisf_score']:
                continue
            
            # Filter 2: Minimum sentence frequency
            if sf < self.config['min_sf']:
                continue
            
            # Filter 3: Maximum sentence frequency (too common)
            if sf > max_sf_threshold:
                continue
            
            # Filter 4: Minimum mentions
            if mentions < self.config['min_mentions']:
                continue
            
            # Passed all filters
            filtered.append(candidate)
        
        return filtered
    
    def get_confidence_score(self, candidate: str, metadata: Dict) -> float:
        """Calculate confidence from TF-ISF score"""
        return metadata.get('score', 0.0)