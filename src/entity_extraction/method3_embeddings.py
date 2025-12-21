"""
Ensemble Voting & Fusion Mechanism
File: src/entity_extraction/ensemble_voter.py
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from collections import defaultdict, Counter
import re
from sklearn.metrics.pairwise import cosine_similarity
import logging

class EnsembleVoter:
    """
    Ensemble voting mechanism untuk menggabungkan hasil dari 3 methods
    
    Features:
    - Multi-method alignment & matching
    - Weighted voting with confidence calibration
    - Conflict resolution strategy
    - Variant merging across methods
    - Quality control & filtering
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize ensemble voter
        
        Args:
            config: Configuration dict with:
                - method_weights: Dict of weights per method
                - min_confidence: Minimum confidence threshold
                - alignment_strategy: 'fuzzy' or 'strict'
                - conflict_resolution: 'majority' or 'weighted'
        """
        self.config = config or self.get_default_config()
        self.logger = self._setup_logger()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'method_weights': {
                'Method1_Capitalization': 0.30,
                'Method2_TFISF': 0.30,  # Updated from TFIDF
                'Method3_Embeddings': 0.40  # Higher weight for semantic
            },
            'min_confidence': 0.50,
            'min_mentions': 2,
            'alignment_strategy': 'fuzzy',  # or 'strict'
            'conflict_resolution': 'weighted',  # or 'majority'
            'similarity_threshold': 0.7,
            'single_word_confidence_boost': 0.75,  # Require higher for single words
            'multi_word_confidence_boost': 0.60,
            'role_based_confidence_boost': 0.50
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger"""
        logger = logging.getLogger('EnsembleVoter')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(levelname)s] Ensemble: %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def vote(self, 
             method_results: Dict[str, Dict[str, Any]],
             preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main voting function - combines results from all methods
        
        Args:
            method_results: Dict mapping method_name -> extraction results
            preprocessed_data: Original preprocessed data
        
        Returns:
            Dict with ensemble results
        """
        self.logger.info("="*70)
        self.logger.info("Starting Ensemble Voting")
        self.logger.info("="*70)
        
        # Step 1: Validate inputs
        self._validate_inputs(method_results)
        
        # Step 2: Align entities across methods
        alignment = self._align_entities(method_results)
        self.logger.info(f"  ✓ Aligned {len(alignment)} unique entities")
        
        # Step 3: Calculate ensemble scores
        scored_entities = self._calculate_ensemble_scores(
            alignment,
            method_results
        )
        self.logger.info(f"  ✓ Calculated scores for {len(scored_entities)} entities")
        
        # Step 4: Resolve conflicts
        resolved_entities = self._resolve_conflicts(scored_entities)
        self.logger.info(f"  ✓ Resolved conflicts: {len(resolved_entities)} entities")
        
        # Step 5: Merge variants
        merged_entities = self._merge_variants(resolved_entities)
        self.logger.info(f"  ✓ Merged variants: {len(merged_entities)} final entities")
        
        # Step 6: Quality control & filtering
        filtered_entities = self._quality_control(
            merged_entities,
            preprocessed_data
        )
        self.logger.info(f"  ✓ Quality control: {len(filtered_entities)} entities passed")
        
        # Step 7: Sort by confidence
        final_entities = sorted(
            filtered_entities,
            key=lambda x: x['confidence'],
            reverse=True
        )
        
        # Generate statistics
        stats = self._generate_statistics(
            final_entities,
            method_results,
            alignment
        )
        
        self.logger.info("="*70)
        self.logger.info(f"Ensemble Complete: {len(final_entities)} final entities")
        self.logger.info("="*70)
        
        return {
            'entities': final_entities,
            'statistics': stats,
            'alignment': alignment,
            'method_contributions': self._analyze_method_contributions(
                final_entities
            )
        }
    
    def _validate_inputs(self, method_results: Dict[str, Dict[str, Any]]):
        """Validate that all expected methods are present"""
        expected_methods = list(self.config['method_weights'].keys())
        
        for method in expected_methods:
            if method not in method_results:
                self.logger.warning(f"Missing results from {method}")
            elif 'candidates' not in method_results[method]:
                raise ValueError(f"{method} results missing 'candidates' field")
    
    def _align_entities(self, 
                       method_results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict]:
        """
        Align entities across methods using fuzzy/strict matching
        
        Returns:
            Dict mapping canonical_name -> {
                'method1_name': str,
                'method2_name': str,
                'method3_name': str,
                'detected_by': List[str],
                'all_variants': List[str]
            }
        """
        # Collect all candidates from all methods
        all_candidates = {}
        
        for method_name, results in method_results.items():
            candidates = results.get('candidates', [])
            all_candidates[method_name] = {
                c['name']: c for c in candidates
            }
        
        # Build alignment matrix
        alignment = {}
        processed = set()
        
        # Get list of all unique names across methods
        all_names = set()
        for method_candidates in all_candidates.values():
            all_names.update(method_candidates.keys())
        
        for name in all_names:
            if name in processed:
                continue
            
            # Find matches across methods
            matches = self._find_cross_method_matches(
                name,
                all_candidates
            )
            
            if not matches:
                continue
            
            # Determine canonical name
            canonical = self._select_canonical_name(matches)
            
            # Build alignment entry
            alignment[canonical] = {
                'matches': matches,
                'detected_by': list(matches.keys()),
                'all_variants': self._collect_variants(matches),
                'detection_count': len(matches)
            }
            
            # Mark as processed
            processed.add(canonical)
            for method_matches in matches.values():
                processed.update(method_matches)
        
        return alignment
    
    def _find_cross_method_matches(self,
                                   name: str,
                                   all_candidates: Dict[str, Dict]) -> Dict[str, List[str]]:
        """
        Find matching names across methods
        
        Returns:
            Dict mapping method_name -> list of matching names in that method
        """
        matches = {}
        
        for method_name, candidates_dict in all_candidates.items():
            method_matches = []
            
            for candidate_name in candidates_dict.keys():
                if self._are_same_entity(name, candidate_name):
                    method_matches.append(candidate_name)
            
            if method_matches:
                matches[method_name] = method_matches
        
        return matches
    
    def _are_same_entity(self, name1: str, name2: str) -> bool:
        """
        Determine if two names refer to the same entity
        
        Uses multiple strategies:
        - Exact match
        - Case-insensitive match
        - Substring match
        - First name match
        - Variant patterns (Jim vs James)
        """
        n1_lower = name1.lower().strip()
        n2_lower = name2.lower().strip()
        
        # Exact match
        if n1_lower == n2_lower:
            return True
        
        # Substring match
        if n1_lower in n2_lower or n2_lower in n1_lower:
            # But check it's meaningful (not just 1-2 chars)
            shorter = min(len(n1_lower), len(n2_lower))
            if shorter >= 3:
                return True
        
        # First name match (for multi-word names)
        parts1 = n1_lower.split()
        parts2 = n2_lower.split()
        
        if len(parts1) > 0 and len(parts2) > 0:
            if parts1[0] == parts2[0] and len(parts1[0]) >= 3:
                return True
        
        # Special case: possessive forms
        # "Jim" vs "Jims" or "Jim's"
        n1_clean = re.sub(r"'?s$", "", n1_lower)
        n2_clean = re.sub(r"'?s$", "", n2_lower)
        
        if n1_clean == n2_clean:
            return True
        
        return False
    
    def _select_canonical_name(self, matches: Dict[str, List[str]]) -> str:
        """
        Select canonical name from matches
        
        Priority:
        1. Full names (multi-word) over single names
        2. Names from Method3 (semantic clustering result)
        3. Most frequent variant
        4. Longest name
        """
        all_names = []
        for method_names in matches.values():
            all_names.extend(method_names)
        
        # Priority 1: Multi-word names
        multi_word = [n for n in all_names if len(n.split()) > 1]
        if multi_word:
            # Priority 2: From Method3
            method3_matches = matches.get('Method3_Embeddings', [])
            for name in method3_matches:
                if name in multi_word:
                    return name
            
            # Priority 4: Longest
            return max(multi_word, key=len)
        
        # Single word names
        # Priority 2: From Method3
        method3_matches = matches.get('Method3_Embeddings', [])
        if method3_matches:
            return method3_matches[0]
        
        # Priority 4: Longest
        return max(all_names, key=len)
    
    def _collect_variants(self, matches: Dict[str, List[str]]) -> List[str]:
        """Collect all unique variants from matches"""
        variants = set()
        for method_matches in matches.values():
            variants.update(method_matches)
        return sorted(list(variants))
    
    def _calculate_ensemble_scores(self,
                                   alignment: Dict[str, Dict],
                                   method_results: Dict[str, Dict[str, Any]]) -> List[Dict]:
        """
        Calculate ensemble confidence scores using weighted voting
        
        Formula:
        confidence = Σ(method_score × method_weight) / Σ(weights)
        
        Only sum over methods that detected the entity
        """
        scored_entities = []
        
        for canonical_name, alignment_data in alignment.items():
            matches = alignment_data['matches']
            detected_by = alignment_data['detected_by']
            
            # Collect scores from each method
            method_scores = {}
            total_mentions = 0
            
            for method_name in detected_by:
                # Get candidate data from method results
                method_candidates = method_results[method_name]['candidates']
                
                # Find matching candidate
                matched_names = matches[method_name]
                for candidate in method_candidates:
                    if candidate['name'] in matched_names:
                        method_scores[method_name] = candidate['score']
                        total_mentions += candidate['mentions']
                        break
            
            # Calculate weighted score
            if method_scores:
                weighted_sum = 0.0
                weight_sum = 0.0
                
                for method_name, score in method_scores.items():
                    weight = self.config['method_weights'].get(method_name, 0.3)
                    weighted_sum += score * weight
                    weight_sum += weight
                
                ensemble_confidence = weighted_sum / weight_sum if weight_sum > 0 else 0.0
            else:
                ensemble_confidence = 0.0
            
            # Build entity dict
            scored_entities.append({
                'name': canonical_name,
                'confidence': ensemble_confidence,
                'mentions': total_mentions,
                'detected_by': detected_by,
                'detection_count': len(detected_by),
                'variants': alignment_data['all_variants'],
                'method_scores': method_scores
            })
        
        return scored_entities
    
    def _resolve_conflicts(self, scored_entities: List[Dict]) -> List[Dict]:
        """
        Resolve conflicts between methods
        
        Conflicts:
        1. Different canonical names for same entity
        2. One method has false positive
        3. Borderline entities (low agreement)
        """
        resolved = []
        
        for entity in scored_entities:
            detection_count = entity['detection_count']
            confidence = entity['confidence']
            
            # Rule 1: Majority vote (2 out of 3)
            if detection_count >= 2:
                # Accept entity
                resolved.append(entity)
            
            # Rule 2: Single method detection
            elif detection_count == 1:
                detected_method = entity['detected_by'][0]
                
                # Special cases: Trust Method3 for special characters
                if detected_method == 'Method3_Embeddings':
                    # Check if it's a special character (narrator, role-based)
                    name = entity['name']
                    if 'Narrator' in name or 'The ' in name:
                        # Accept with lower confidence
                        entity['confidence'] = confidence * 0.75
                        resolved.append(entity)
                    elif confidence >= 0.7:
                        # High confidence from Method3 alone
                        entity['confidence'] = confidence * 0.80
                        resolved.append(entity)
                
                # Method1 or Method2 alone: require very high confidence
                elif confidence >= 0.85:
                    entity['confidence'] = confidence * 0.70
                    resolved.append(entity)
        
        return resolved
    
    def _merge_variants(self, entities: List[Dict]) -> List[Dict]:
        """
        Final variant merging pass
        
        Check if any entities should be merged based on:
        - Similar names
        - Overlapping variants
        """
        merged = []
        processed = set()
        
        for i, entity in enumerate(entities):
            if i in processed:
                continue
            
            canonical = entity['name']
            merged_entity = entity.copy()
            processed.add(i)
            
            # Check for mergeable entities
            for j, other_entity in enumerate(entities[i+1:], start=i+1):
                if j in processed:
                    continue
                
                other_name = other_entity['name']
                
                # Check if should merge
                if self._should_merge(entity, other_entity):
                    # Merge
                    merged_entity['mentions'] += other_entity['mentions']
                    merged_entity['variants'].extend(other_entity['variants'])
                    merged_entity['variants'] = list(set(merged_entity['variants']))
                    
                    # Update confidence (weighted by mentions)
                    total_mentions = merged_entity['mentions']
                    entity_weight = entity['mentions'] / total_mentions
                    other_weight = other_entity['mentions'] / total_mentions
                    
                    merged_entity['confidence'] = (
                        entity['confidence'] * entity_weight +
                        other_entity['confidence'] * other_weight
                    )
                    
                    # Merge detected_by
                    merged_entity['detected_by'] = list(set(
                        merged_entity['detected_by'] + other_entity['detected_by']
                    ))
                    merged_entity['detection_count'] = len(merged_entity['detected_by'])
                    
                    processed.add(j)
            
            merged.append(merged_entity)
        
        return merged
    
    def _should_merge(self, entity1: Dict, entity2: Dict) -> bool:
        """Determine if two entities should be merged"""
        name1 = entity1['name']
        name2 = entity2['name']
        
        # Use same logic as _are_same_entity
        return self._are_same_entity(name1, name2)
    
    def _quality_control(self,
                        entities: List[Dict],
                        preprocessed_data: Dict[str, Any]) -> List[Dict]:
        """
        Final quality control and filtering
        
        Apply:
        - Minimum confidence threshold (adaptive)
        - Minimum mentions threshold
        - Blacklist check
        - Special rules for different entity types
        """
        filtered = []
        
        for entity in entities:
            name = entity['name']
            confidence = entity['confidence']
            mentions = entity['mentions']
            detection_count = entity['detection_count']
            
            # Determine adaptive threshold based on entity type
            if len(name.split()) == 1:
                # Single word: require higher confidence
                required_confidence = self.config['single_word_confidence_boost']
            elif 'The ' in name or 'Narrator' in name:
                # Role-based: lower threshold
                required_confidence = self.config['role_based_confidence_boost']
            else:
                # Multi-word names: standard threshold
                required_confidence = self.config['multi_word_confidence_boost']
            
            # Apply filters
            if confidence < required_confidence:
                continue
            
            if mentions < self.config['min_mentions']:
                continue
            
            # Blacklist check (delegate to validator if needed)
            if self._is_blacklisted(name):
                continue
            
            # Passed all filters
            filtered.append(entity)
        
        return filtered
    
    def _is_blacklisted(self, name: str) -> bool:
        """Check if entity is blacklisted"""
        blacklist = {
            'christmas', 'god', 'lord', 'jesus', 'magi',
            'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
            'saturday', 'sunday', 'dear', 'young'
        }
        
        name_lower = name.lower()
        
        # Check if any blacklisted word is in name
        for blacklisted in blacklist:
            if blacklisted == name_lower or blacklisted in name_lower.split():
                return True
        
        return False
    
    def _generate_statistics(self,
                            final_entities: List[Dict],
                            method_results: Dict[str, Dict[str, Any]],
                            alignment: Dict[str, Dict]) -> Dict[str, Any]:
        """Generate ensemble statistics"""
        stats = {
            'total_entities': len(final_entities),
            'average_confidence': np.mean([e['confidence'] for e in final_entities]) if final_entities else 0.0,
            'high_confidence_entities': sum(1 for e in final_entities if e['confidence'] >= 0.8),
            'medium_confidence_entities': sum(1 for e in final_entities if 0.6 <= e['confidence'] < 0.8),
            'low_confidence_entities': sum(1 for e in final_entities if e['confidence'] < 0.6),
            'method_counts': {
                '3_methods': sum(1 for e in final_entities if e['detection_count'] == 3),
                '2_methods': sum(1 for e in final_entities if e['detection_count'] == 2),
                '1_method': sum(1 for e in final_entities if e['detection_count'] == 1)
            },
            'method_results': {
                method: len(results.get('candidates', []))
                for method, results in method_results.items()
            }
        }
        
        return stats
    
    def _analyze_method_contributions(self, final_entities: List[Dict]) -> Dict[str, Dict]:
        """Analyze how much each method contributed"""
        contributions = defaultdict(lambda: {'total': 0, 'unique': 0, 'shared': 0})
        
        for entity in final_entities:
            detected_by = entity['detected_by']
            
            for method in detected_by:
                contributions[method]['total'] += 1
                
                if len(detected_by) == 1:
                    contributions[method]['unique'] += 1
                else:
                    contributions[method]['shared'] += 1
        
        return dict(contributions)