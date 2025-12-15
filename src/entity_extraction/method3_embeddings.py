"""
Method 3: Embeddings-Based Entity Extraction with Semantic Clustering
File: src/entity_extraction/method3_embeddings.py
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from collections import defaultdict, Counter
import re
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

from .base_extractor import BaseEntityExtractor

class EmbeddingsExtractor(BaseEntityExtractor):
    """
    Extract entities using semantic embeddings and clustering
    
    Features:
    - Context-aware embeddings (Sentence-BERT)
    - Semantic clustering (HDBSCAN)
    - Variant merging based on contextual similarity
    - Role-based character detection (narrator, "The Old Man")
    - First-person narrator detection
    - Pronoun association analysis
    """
    
    def get_default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'embedding_model': 'all-MiniLM-L6-v2',  # Fast & efficient
            'context_window': 5,  # Tokens before/after mention
            'min_cluster_size': 2,
            'min_samples': 1,
            'cluster_selection_epsilon': 0.3,
            'similarity_threshold': 0.7,
            'min_mentions': 2,
            'detect_narrator': True,
            'detect_roles': True,
            'pronoun_weight': 0.2,  # Weight for pronoun association
            'role_patterns': self._get_role_patterns()
        }
    
    def _get_role_patterns(self) -> Dict[str, List[str]]:
        """Role-based character patterns"""
        return {
            'old_man': [r'\bold man\b', r'\bthe old man\b'],
            'old_woman': [r'\bold woman\b', r'\bthe old woman\b'],
            'officers': [r'\bofficers?\b', r'\bpolice\b', r'\bpolicemen\b'],
            'soldiers': [r'\bsoldiers?\b', r'\bprivates?\b'],
            'neighbor': [r'\bneighbou?rs?\b'],
            'servant': [r'\bservants?\b', r'\bmaids?\b'],
            'doctor': [r'\bdoctors?\b', r'\bphysicians?\b'],
            'narrator': [r'\bnarrator\b', r'\bi\s+(?:am|was|feel|thought|saw)\b']
        }
    
    def get_method_name(self) -> str:
        return "Method3_Embeddings"
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize with embedding model"""
        super().__init__(config)
        
        # Load embedding model
        self.logger.info(f"Loading embedding model: {self.config['embedding_model']}")
        try:
            self.model = SentenceTransformer(self.config['embedding_model'])
            self.logger.info("  ✓ Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def extract(self, preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract entities using semantic embeddings
        
        Pipeline:
        1. Extract contexts for each candidate mention
        2. Generate embeddings for contexts
        3. Cluster embeddings to merge variants
        4. Detect special cases (narrator, roles)
        5. Analyze pronoun associations
        6. Score candidates by semantic coherence
        7. Return final entities
        """
        self.validate_input(preprocessed_data)
        self.logger.info("Starting embeddings-based extraction...")
        
        sentences = preprocessed_data['sentences']
        propn_candidates = preprocessed_data['propn_candidates']
        ngrams_data = preprocessed_data['ngrams']
        
        # Step 1: Build candidate contexts
        candidate_contexts = self._extract_contexts(
            sentences,
            propn_candidates,
            ngrams_data
        )
        self.logger.info(f"  Extracted contexts for {len(candidate_contexts)} candidates")
        
        # Step 2: Generate embeddings
        candidate_embeddings = self._generate_embeddings(candidate_contexts)
        self.logger.info(f"  Generated embeddings")
        
        # Step 3: Cluster candidates
        clusters = self._cluster_candidates(
            candidate_embeddings,
            candidate_contexts
        )
        self.logger.info(f"  Formed {len(clusters)} clusters")
        
        # Step 4: Detect special characters
        if self.config['detect_narrator']:
            narrator = self._detect_narrator(sentences)
            if narrator:
                clusters.append(narrator)
                self.logger.info(f"  Detected narrator: {narrator['canonical_name']}")
        
        if self.config['detect_roles']:
            role_characters = self._detect_role_characters(sentences)
            clusters.extend(role_characters)
            if role_characters:
                self.logger.info(f"  Detected {len(role_characters)} role-based characters")
        
        # Step 5: Analyze pronoun associations
        clusters_with_pronouns = self._analyze_pronouns(clusters, sentences)
        
        # Step 6: Score clusters
        scored_candidates = self._score_clusters(
            clusters_with_pronouns,
            candidate_contexts
        )
        self.logger.info(f"  Scored {len(scored_candidates)} final candidates")
        
        # Step 7: Filter and sort
        filtered = self.filter_by_threshold(
            scored_candidates,
            threshold_key='score',
            threshold_value=0.4  # Lower threshold for semantic method
        )
        
        final_candidates = self.sort_by_score(filtered)
        
        return {
            'candidates': final_candidates,
            'method_name': self.get_method_name(),
            'statistics': self.get_statistics({'candidates': final_candidates}),
            'clusters': clusters_with_pronouns
        }
    
    def _extract_contexts(self,
                         sentences: List[str],
                         propn_candidates: set,
                         ngrams_data: Dict) -> Dict[str, List[Dict]]:
        """
        Extract context windows for each candidate mention
        
        Returns:
            Dict mapping candidate -> list of context dicts
        """
        # Combine all candidates
        all_candidates = set(propn_candidates)
        all_candidates.update(ngrams_data.get('unigrams', []))
        all_candidates.update(ngrams_data.get('bigrams', []))
        all_candidates.update(ngrams_data.get('trigrams', []))
        
        contexts = defaultdict(list)
        window = self.config['context_window']
        
        for sent_id, sentence in enumerate(sentences):
            tokens = sentence.split()
            sent_lower = sentence.lower()
            
            # Check each candidate
            for candidate in all_candidates:
                candidate_lower = candidate.lower()
                
                # Find mentions in sentence
                if candidate_lower in sent_lower:
                    # Extract context window
                    # Simple approach: take full sentence as context
                    # (More sophisticated: extract ±N tokens around mention)
                    
                    contexts[candidate].append({
                        'sentence_id': sent_id,
                        'sentence': sentence,
                        'context': sentence,  # Use full sentence as context
                        'candidate': candidate
                    })
        
        return dict(contexts)
    
    def _generate_embeddings(self, 
                            candidate_contexts: Dict[str, List[Dict]]) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for each candidate
        
        Strategy: Average embeddings of all contexts
        
        Returns:
            Dict mapping candidate -> embedding vector
        """
        candidate_embeddings = {}
        
        for candidate, contexts in candidate_contexts.items():
            if not contexts:
                continue
            
            # Extract context texts
            context_texts = [ctx['context'] for ctx in contexts]
            
            # Generate embeddings
            embeddings = self.model.encode(context_texts, show_progress_bar=False)
            
            # Average embeddings
            avg_embedding = np.mean(embeddings, axis=0)
            
            candidate_embeddings[candidate] = avg_embedding
        
        return candidate_embeddings
    
    def _cluster_candidates(self,
                           candidate_embeddings: Dict[str, np.ndarray],
                           candidate_contexts: Dict[str, List[Dict]]) -> List[Dict]:
        """
        Cluster candidates using HDBSCAN
        
        Returns:
            List of cluster dicts with canonical names and variants
        """
        if len(candidate_embeddings) < 2:
            # Not enough candidates to cluster
            return [
                {
                    'canonical_name': name,
                    'variants': [name],
                    'mentions': len(candidate_contexts[name]),
                    'confidence': 0.8,
                    'role': 'named_character'
                }
                for name in candidate_embeddings.keys()
            ]
        
        # Prepare data for clustering
        candidates = list(candidate_embeddings.keys())
        embeddings_matrix = np.array([candidate_embeddings[c] for c in candidates])
        
        # HDBSCAN clustering
        try:
            clusterer = HDBSCAN(
                min_cluster_size=self.config['min_cluster_size'],
                min_samples=self.config['min_samples'],
                cluster_selection_epsilon=self.config['cluster_selection_epsilon'],
                metric='euclidean'
            )
            
            cluster_labels = clusterer.fit_predict(embeddings_matrix)
            
        except Exception as e:
            self.logger.warning(f"HDBSCAN failed: {e}, using fallback clustering")
            # Fallback: pairwise similarity clustering
            cluster_labels = self._fallback_clustering(embeddings_matrix, candidates)
        
        # Group by cluster
        clusters_dict = defaultdict(list)
        for idx, label in enumerate(cluster_labels):
            candidate = candidates[idx]
            clusters_dict[label].append(candidate)
        
        # Build cluster objects
        clusters = []
        for cluster_id, cluster_members in clusters_dict.items():
            if cluster_id == -1:
                # Noise points - treat as individual entities
                for member in cluster_members:
                    clusters.append({
                        'canonical_name': member,
                        'variants': [member],
                        'mentions': len(candidate_contexts[member]),
                        'confidence': 0.6,  # Lower for unclustered
                        'role': 'named_character'
                    })
            else:
                # Valid cluster - merge variants
                # Choose canonical: prefer full names (longer)
                canonical = max(cluster_members, key=len)
                
                total_mentions = sum(len(candidate_contexts[m]) for m in cluster_members)
                
                clusters.append({
                    'canonical_name': canonical,
                    'variants': cluster_members,
                    'mentions': total_mentions,
                    'confidence': 0.85,
                    'role': 'named_character'
                })
        
        return clusters
    
    def _fallback_clustering(self, 
                            embeddings_matrix: np.ndarray,
                            candidates: List[str]) -> np.ndarray:
        """
        Fallback clustering using pairwise similarity
        """
        n = len(candidates)
        labels = np.arange(n)  # Initially each is its own cluster
        
        # Calculate pairwise similarities
        similarities = cosine_similarity(embeddings_matrix)
        
        # Merge similar candidates
        for i in range(n):
            for j in range(i+1, n):
                if similarities[i][j] >= self.config['similarity_threshold']:
                    # Merge j into i's cluster
                    labels[j] = labels[i]
        
        return labels
    
    def _detect_narrator(self, sentences: List[str]) -> Optional[Dict]:
        """
        Detect first-person narrator
        
        Returns:
            Cluster dict for narrator if detected, None otherwise
        """
        # Count first-person pronouns
        first_person_count = 0
        narrator_contexts = []
        
        for sent_id, sentence in enumerate(sentences):
            sent_lower = sentence.lower()
            
            # Count "I" as standalone word
            i_matches = len(re.findall(r'\bi\b', sent_lower))
            first_person_count += i_matches
            
            if i_matches > 0:
                narrator_contexts.append({
                    'sentence_id': sent_id,
                    'sentence': sentence,
                    'context': sentence,
                    'candidate': 'Narrator (I)'
                })
        
        # Threshold: at least 20 mentions of "I"
        if first_person_count >= 20:
            return {
                'canonical_name': 'Narrator (I)',
                'variants': ['I', 'narrator'],
                'mentions': first_person_count,
                'confidence': 0.75,
                'role': 'first_person_narrator',
                'contexts': narrator_contexts
            }
        
        return None
    
    def _detect_role_characters(self, sentences: List[str]) -> List[Dict]:
        """
        Detect role-based characters (e.g., "The Old Man", "The Officers")
        
        Returns:
            List of cluster dicts for role characters
        """
        role_characters = []
        role_patterns = self.config['role_patterns']
        
        for role_name, patterns in role_patterns.items():
            mentions = 0
            contexts = []
            
            for sent_id, sentence in enumerate(sentences):
                sent_lower = sentence.lower()
                
                # Check patterns
                for pattern in patterns:
                    matches = re.findall(pattern, sent_lower)
                    if matches:
                        mentions += len(matches)
                        contexts.append({
                            'sentence_id': sent_id,
                            'sentence': sentence,
                            'context': sentence,
                            'candidate': role_name
                        })
                        break  # Don't double-count same sentence
            
            # Threshold: at least 3 mentions
            if mentions >= 3:
                # Format role name nicely
                display_name = ' '.join(word.capitalize() for word in role_name.split('_'))
                if not display_name.startswith('The'):
                    display_name = f"The {display_name}"
                
                role_characters.append({
                    'canonical_name': display_name,
                    'variants': [role_name, display_name.lower()],
                    'mentions': mentions,
                    'confidence': 0.70,
                    'role': 'role_based_character',
                    'contexts': contexts
                })
        
        return role_characters
    
    def _analyze_pronouns(self, 
                         clusters: List[Dict],
                         sentences: List[str]) -> List[Dict]:
        """
        Analyze pronoun associations for each cluster
        
        Adds metadata about gender/plurality based on pronouns
        """
        pronoun_patterns = {
            'male': [r'\bhe\b', r'\bhis\b', r'\bhim\b', r'\bhimself\b'],
            'female': [r'\bshe\b', r'\bher\b', r'\bhers\b', r'\bherself\b'],
            'plural': [r'\bthey\b', r'\btheir\b', r'\bthem\b', r'\bthemselves\b']
        }
        
        for cluster in clusters:
            canonical = cluster['canonical_name']
            variants = cluster['variants']
            
            # Get contexts (if available)
            contexts = cluster.get('contexts', [])
            if not contexts:
                # Build contexts from sentences
                contexts = []
                for sent_id, sentence in enumerate(sentences):
                    sent_lower = sentence.lower()
                    if any(v.lower() in sent_lower for v in variants):
                        contexts.append({
                            'sentence': sentence,
                            'sentence_id': sent_id
                        })
            
            # Count pronouns in contexts
            pronoun_counts = defaultdict(int)
            
            for ctx in contexts:
                sentence = ctx['sentence'].lower()
                
                for gender, patterns in pronoun_patterns.items():
                    for pattern in patterns:
                        pronoun_counts[gender] += len(re.findall(pattern, sentence))
            
            # Determine dominant pronoun
            if pronoun_counts:
                dominant = max(pronoun_counts, key=pronoun_counts.get)
                cluster['pronoun_association'] = dominant
                cluster['pronoun_confidence'] = pronoun_counts[dominant] / sum(pronoun_counts.values())
            else:
                cluster['pronoun_association'] = 'unknown'
                cluster['pronoun_confidence'] = 0.0
        
        return clusters
    
    def _score_clusters(self,
                       clusters: List[Dict],
                       candidate_contexts: Dict[str, List[Dict]]) -> List[Dict]:
        """
        Calculate final scores for clusters
        
        Scoring factors:
        - Base confidence from clustering
        - Mention frequency
        - Pronoun association strength
        - Role clarity
        """
        scored = []
        
        for cluster in clusters:
            canonical = cluster['canonical_name']
            mentions = cluster['mentions']
            base_confidence = cluster['confidence']
            role = cluster['role']
            
            # Frequency score
            frequency_score = min(mentions / 50.0, 0.2)
            
            # Pronoun score
            pronoun_conf = cluster.get('pronoun_confidence', 0.0)
            pronoun_score = pronoun_conf * self.config['pronoun_weight']
            
            # Role bonus
            role_bonus = 0.0
            if role in ['first_person_narrator', 'role_based_character']:
                role_bonus = 0.1
            
            # Final score
            final_score = min(
                base_confidence + frequency_score + pronoun_score + role_bonus,
                1.0
            )
            
            scored.append({
                'name': canonical,
                'score': final_score,
                'mentions': mentions,
                'metadata': {
                    'variants': cluster['variants'],
                    'role': role,
                    'pronoun_association': cluster.get('pronoun_association', 'unknown'),
                    'pronoun_confidence': cluster.get('pronoun_confidence', 0.0),
                    'base_confidence': base_confidence,
                    'clustering_method': 'hdbscan'
                }
            })
        
        return scored
    
    def get_confidence_score(self, candidate: str, metadata: Dict) -> float:
        """Calculate confidence from metadata"""
        return metadata.get('score', 0.0)