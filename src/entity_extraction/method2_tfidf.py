"""
Method 2: TF-IDF Based Entity Extraction
File: src/entity_extraction/method2_tfidf.py
"""

from typing import Dict, List, Any, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
import re
from .base_extractor import BaseEntityExtractor

class TFIDFExtractor(BaseEntityExtractor):
    """
    Extract entities using TF-IDF ranking on proper nouns
    
    Features:
    - TF-IDF calculation for PROPN candidates only
    - Document-level importance ranking
    - Cross-document validation
    - Variant detection via cosine similarity
    - Statistical significance filtering
    """
    
    def get_default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'min_tfidf_score': 0.10,
            'max_df': 0.8,  # Max document frequency (filter too common)
            'min_df': 2,     # Min document frequency (filter rare)
            'top_k': 15,     # Top K entities per document
            'similarity_threshold': 0.6,  # For variant detection
            'min_mentions': 2,
            'cross_doc_penalty': 0.15,  # Penalty if appears in many docs (generic name)
            'vocabulary_type': 'propn_only'  # Use only proper nouns
        }
    
    def get_method_name(self) -> str:
        return "Method2_TFIDF"
    
    def extract(self, preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract entities using TF-IDF ranking
        
        Pipeline:
        1. Build vocabulary from PROPN candidates only
        2. Create document-term matrix
        3. Calculate TF-IDF scores
        4. Rank candidates per document
        5. Detect variants via similarity
        6. Apply cross-document validation
        7. Return scored candidates
        """
        self.validate_input(preprocessed_data)
        self.logger.info("Starting TF-IDF based extraction...")
        
        # Step 1: Extract all candidates and build corpus
        candidates, corpus = self._build_corpus(preprocessed_data)
        self.logger.info(f"  Built corpus with {len(candidates)} unique candidates")
        
        # Step 2: Calculate TF-IDF
        tfidf_matrix, feature_names, vectorizer = self._calculate_tfidf(
            corpus, 
            candidates
        )
        self.logger.info(f"  TF-IDF matrix shape: {tfidf_matrix.shape}")
        
        # Step 3: Extract and rank candidates
        ranked_candidates = self._rank_candidates(
            tfidf_matrix,
            feature_names,
            candidates,
            preprocessed_data
        )
        self.logger.info(f"  Ranked {len(ranked_candidates)} candidates")
        
        # Step 4: Detect variants
        variants_map = self._detect_variants(
            ranked_candidates,
            tfidf_matrix,
            feature_names
        )
        self.logger.info(f"  Detected {len(variants_map)} variant groups")
        
        # Step 5: Apply cross-document validation
        validated_candidates = self._cross_document_validation(
            ranked_candidates,
            corpus_size=len(corpus)
        )
        
        # Step 6: Filter by threshold
        filtered_candidates = self.filter_by_threshold(
            validated_candidates,
            threshold_key='score',
            threshold_value=self.config['min_tfidf_score']
        )
        self.logger.info(f"  After threshold filter: {len(filtered_candidates)} candidates")
        
        # Sort by score
        final_candidates = self.sort_by_score(filtered_candidates)
        
        return {
            'candidates': final_candidates,
            'method_name': self.get_method_name(),
            'statistics': self.get_statistics({'candidates': final_candidates}),
            'variants_map': variants_map
        }
    
    def _build_corpus(self, preprocessed_data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """
        Build corpus from sentences with PROPN filtering
        
        Returns:
            - candidates: List of all PROPN candidates
            - corpus: List of sentences (documents) containing only PROPNs
        """
        propn_candidates = preprocessed_data['propn_candidates']
        sentences = preprocessed_data['sentences']
        ngrams_data = preprocessed_data['ngrams']
        
        # Combine all n-grams as candidates
        all_candidates = set(propn_candidates)
        all_candidates.update(ngrams_data.get('unigrams', []))
        all_candidates.update(ngrams_data.get('bigrams', []))
        all_candidates.update(ngrams_data.get('trigrams', []))
        
        # Build corpus: each sentence as document, filtered to contain only candidates
        corpus = []
        for sentence in sentences:
            # Extract only candidate words from sentence
            sent_lower = sentence.lower()
            propn_words = []
            
            # Check each candidate
            for candidate in all_candidates:
                # Use word boundary regex to avoid partial matches
                pattern = r'\b' + re.escape(candidate.lower()) + r'\b'
                if re.search(pattern, sent_lower):
                    propn_words.append(candidate)
            
            # Create document with only PROPN candidates
            if propn_words:
                corpus.append(' '.join(propn_words))
            else:
                # Empty document (no proper nouns)
                corpus.append('')
        
        return sorted(list(all_candidates)), corpus
    
    def _calculate_tfidf(self, 
                        corpus: List[str], 
                        candidates: List[str]) -> Tuple[np.ndarray, List[str], TfidfVectorizer]:
        """
        Calculate TF-IDF matrix
        
        Returns:
            - tfidf_matrix: Document-term matrix (docs x terms)
            - feature_names: List of terms (in matrix column order)
            - vectorizer: Fitted TfidfVectorizer
        """
        # Create vectorizer with custom vocabulary
        vectorizer = TfidfVectorizer(
            vocabulary=candidates,
            max_df=self.config['max_df'],
            min_df=self.config['min_df'],
            lowercase=True,
            token_pattern=r'\b[\w\s]+\b',  # Allow multi-word terms
            ngram_range=(1, 3)  # Support up to trigrams
        )
        
        # Fit and transform
        try:
            tfidf_matrix = vectorizer.fit_transform(corpus)
            feature_names = vectorizer.get_feature_names_out().tolist()
        except ValueError as e:
            # Handle case where no terms pass the filters
            self.logger.warning(f"TF-IDF vectorization failed: {e}")
            self.logger.warning("Falling back to simpler vectorization...")
            
            # Fallback: no min_df/max_df filtering
            vectorizer = TfidfVectorizer(
                vocabulary=candidates,
                lowercase=True,
                token_pattern=r'\b[\w\s]+\b',
                ngram_range=(1, 3)
            )
            tfidf_matrix = vectorizer.fit_transform(corpus)
            feature_names = vectorizer.get_feature_names_out().tolist()
        
        return tfidf_matrix, feature_names, vectorizer
    
    def _rank_candidates(self,
                        tfidf_matrix: np.ndarray,
                        feature_names: List[str],
                        candidates: List[str],
                        preprocessed_data: Dict[str, Any]) -> List[Dict]:
        """
        Rank candidates by TF-IDF score
        
        Strategy:
        - Sum TF-IDF across all documents for each term
        - Normalize by document count
        - Add frequency-based bonus
        """
        # Convert to dense for easier manipulation
        tfidf_dense = tfidf_matrix.toarray()
        
        # Get capitalization patterns for frequency data
        cap_patterns = preprocessed_data['capitalization_patterns']
        
        ranked = []
        
        for idx, term in enumerate(feature_names):
            # Get TF-IDF scores across all documents
            scores = tfidf_dense[:, idx]
            
            # Calculate aggregate TF-IDF
            total_tfidf = np.sum(scores)
            max_tfidf = np.max(scores)
            doc_count = np.count_nonzero(scores)  # Number of docs containing term
            
            # Get actual mention count from capitalization patterns
            mention_count = 0
            if term in cap_patterns:
                mention_count = cap_patterns[term]['total_mentions']
            
            # Calculate final score
            # Combine: max TF-IDF + frequency bonus + document spread penalty
            base_score = max_tfidf
            frequency_bonus = min(mention_count / 50.0, 0.2)
            
            final_score = base_score + frequency_bonus
            
            ranked.append({
                'name': term,
                'score': float(final_score),
                'mentions': mention_count,
                'metadata': {
                    'tfidf_sum': float(total_tfidf),
                    'tfidf_max': float(max_tfidf),
                    'doc_count': int(doc_count),
                    'avg_tfidf': float(total_tfidf / max(doc_count, 1))
                }
            })
        
        # Filter by minimum mentions
        ranked = [
            r for r in ranked 
            if r['mentions'] >= self.config['min_mentions']
        ]
        
        # Sort and take top K
        ranked = sorted(ranked, key=lambda x: x['score'], reverse=True)
        
        return ranked[:self.config['top_k']]
    
    def _detect_variants(self,
                        candidates: List[Dict],
                        tfidf_matrix: np.ndarray,
                        feature_names: List[str]) -> Dict[str, List[str]]:
        """
        Detect name variants using cosine similarity
        
        Example: "Jim" and "James Dillingham Young" have high similarity
        
        Returns:
            Dict mapping canonical name -> list of variants
        """
        if len(candidates) < 2:
            return {}
        
        # Build mapping: name -> tfidf vector
        name_to_vector = {}
        tfidf_dense = tfidf_matrix.toarray()
        
        for candidate in candidates:
            name = candidate['name']
            if name in feature_names:
                idx = feature_names.index(name)
                # Get vector: sum across all documents
                vector = tfidf_dense[:, idx].reshape(1, -1)
                name_to_vector[name] = vector
        
        # Calculate pairwise similarities
        variants_map = defaultdict(list)
        processed = set()
        
        candidate_names = [c['name'] for c in candidates]
        
        for i, name1 in enumerate(candidate_names):
            if name1 in processed or name1 not in name_to_vector:
                continue
            
            canonical = name1
            variants = [name1]
            
            for j, name2 in enumerate(candidate_names[i+1:], start=i+1):
                if name2 in processed or name2 not in name_to_vector:
                    continue
                
                # Calculate cosine similarity
                vec1 = name_to_vector[name1]
                vec2 = name_to_vector[name2]
                
                # Reshape for sklearn
                similarity = cosine_similarity(vec1, vec2)[0][0]
                
                # Also check string similarity (substring matching)
                string_sim = self._string_similarity(name1, name2)
                
                # Combine similarities
                combined_sim = max(similarity, string_sim)
                
                if combined_sim >= self.config['similarity_threshold']:
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
        - Substring match
        - First name match
        - Character overlap
        """
        n1_lower = name1.lower()
        n2_lower = name2.lower()
        
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
        
        # Character overlap (Jaccard similarity)
        set1 = set(n1_lower)
        set2 = set(n2_lower)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        if union > 0:
            return intersection / union
        
        return 0.0
    
    def _cross_document_validation(self,
                                   candidates: List[Dict],
                                   corpus_size: int) -> List[Dict]:
        """
        Validate candidates across documents
        
        Penalty for appearing in too many documents (generic names)
        Boost for document-specific characters
        """
        validated = []
        
        for candidate in candidates:
            doc_count = candidate['metadata']['doc_count']
            doc_ratio = doc_count / corpus_size if corpus_size > 0 else 0
            
            # Calculate penalty/boost
            if doc_ratio > 0.5:
                # Appears in >50% of docs -> likely generic or common word
                penalty = self.config['cross_doc_penalty']
                new_score = candidate['score'] * (1 - penalty)
                candidate['metadata']['cross_doc_penalty'] = penalty
            elif doc_ratio < 0.1:
                # Appears in <10% of docs -> very specific, boost
                boost = 0.05
                new_score = min(candidate['score'] + boost, 1.0)
                candidate['metadata']['cross_doc_boost'] = boost
            else:
                # Normal range
                new_score = candidate['score']
            
            candidate['score'] = new_score
            candidate['metadata']['doc_ratio'] = doc_ratio
            
            validated.append(candidate)
        
        return validated
    
    def get_confidence_score(self, candidate: str, metadata: Dict) -> float:
        """
        Calculate confidence from TF-IDF score
        """
        return metadata.get('score', 0.0)