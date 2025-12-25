"""
Method 3: Semantic Embeddings & Clustering for Entity Extraction
File: src/entity_extraction/method3_embeddings.py

Uses:
- Sentence-BERT embeddings for semantic representation
- HDBSCAN clustering for grouping similar mentions
- Context-aware character detection (narrator, roles)
"""

from typing import Dict, List, Any, Tuple
import numpy as np
from collections import defaultdict, Counter
import re
from .base_extractor import BaseEntityExtractor

# Sentence transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️  sentence-transformers not installed. Method 3 will use fallback mode.")

# Clustering
try:
    from sklearn.cluster import HDBSCAN
    HDBSCAN_AVAILABLE = True
except ImportError:
    try:
        import hdbscan
        HDBSCAN_AVAILABLE = True
    except ImportError:
        HDBSCAN_AVAILABLE = False
        print("⚠️  HDBSCAN not installed. Method 3 will use fallback clustering.")

from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingsExtractor(BaseEntityExtractor):
    """
    Extract entities using semantic embeddings and clustering
    
    Features:
    - Context-aware embeddings for each mention
    - Semantic clustering to group variants
    - Special character detection (narrator, roles)
    - Pronoun association for unnamed characters
    """
    
    def get_default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'embedding_model': 'all-MiniLM-L6-v2',
            'context_window': 5,
            'min_cluster_size': 2,
            'min_samples': 1,
            'cluster_selection_epsilon': 0.3,
            'similarity_threshold': 0.7,
            'min_mentions': 2,
            'detect_narrator': True,
            'detect_roles': True,
            'pronoun_weight': 0.2,
            'role_patterns': {
                'old_man': [r'\bold man\b', r'\bthe old man\b'],
                'old_woman': [r'\bold woman\b', r'\bthe old woman\b'],
                'officers': [r'\bofficers?\b', r'\bpolice\b', r'\bpolicemen\b'],
                'soldiers': [r'\bsoldiers?\b', r'\bprivates?\b'],
                'neighbor': [r'\bneighbou?rs?\b'],
                'servant': [r'\bservants?\b', r'\bmaids?\b'],
                'doctor': [r'\bdoctors?\b', r'\bphysicians?\b'],
                'narrator': [r'\bnarrator\b', r'\bi\s+(?:am|was|feel|thought|saw)\b']
            }
        }
    
    def get_method_name(self) -> str:
        return "Method3_Embeddings"
    
    def extract(self, preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract entities using embeddings and clustering
        
        Pipeline:
        1. Load embedding model
        2. Build context embeddings for candidates
        3. Cluster embeddings to group variants
        4. Detect special characters (narrator, roles)
        5. Score and rank candidates
        """
        self.validate_input(preprocessed_data)
        self.logger.info("Starting embeddings-based extraction...")
        
        sentences = preprocessed_data['sentences']
        propn_candidates = preprocessed_data['propn_candidates']
        
        # Check if libraries available
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            self.logger.warning("  Sentence transformers not available, using fallback")
            return self._fallback_extraction(preprocessed_data)
        
        # Step 1: Load model
        self.logger.info(f"  Loading embedding model: {self.config['embedding_model']}")
        try:
            self.model = SentenceTransformer(self.config['embedding_model'])
        except Exception as e:
            self.logger.error(f"  Failed to load model: {e}")
            return self._fallback_extraction(preprocessed_data)
        
        # Step 2: Build context embeddings
        self.logger.info("  Building context embeddings...")
        candidate_embeddings = self._build_context_embeddings(
            propn_candidates,
            sentences
        )
        self.logger.info(f"    → {len(candidate_embeddings)} candidates with embeddings")
        
        if not candidate_embeddings:
            self.logger.warning("  No candidates with embeddings found")
            return self._fallback_extraction(preprocessed_data)
        
        # Step 3: Cluster embeddings
        self.logger.info("  Clustering embeddings...")
        clusters = self._cluster_embeddings(candidate_embeddings)
        self.logger.info(f"    → {len(set(clusters.values()))} clusters formed")
        
        # Step 4: Build candidates from clusters
        cluster_candidates = self._build_candidates_from_clusters(
            clusters,
            candidate_embeddings
        )
        self.logger.info(f"    → {len(cluster_candidates)} cluster candidates")
        
        # Step 5: Detect special characters
        special_candidates = []
        
        if self.config['detect_narrator']:
            narrator = self._detect_narrator(sentences)
            if narrator:
                special_candidates.append(narrator)
                self.logger.info(f"    → Narrator detected: {narrator['name']}")
        
        if self.config['detect_roles']:
            roles = self._detect_role_based_characters(sentences)
            special_candidates.extend(roles)
            if roles:
                self.logger.info(f"    → {len(roles)} role-based characters detected")
        
        # Step 6: Combine and score
        all_candidates = cluster_candidates + special_candidates
        scored_candidates = self._score_candidates(all_candidates)
        
        # Step 7: Filter and sort
        filtered = [c for c in scored_candidates if c['mentions'] >= self.config['min_mentions']]
        final_candidates = sorted(filtered, key=lambda x: x['score'], reverse=True)
        
        self.logger.info(f"  Final: {len(final_candidates)} candidates")
        
        return {
            'candidates': final_candidates,
            'method_name': self.get_method_name(),
            'statistics': self.get_statistics({'candidates': final_candidates}),
            'clusters': clusters,
            'special_characters': [c['name'] for c in special_candidates]
        }
    
    def _build_context_embeddings(self, 
                                  candidates: set, 
                                  sentences: List[str]) -> Dict[str, List[Dict]]:
        """
        Build context embeddings for each candidate mention
        
        Returns:
            Dict mapping candidate -> list of mention embeddings with context
        """
        candidate_embeddings = defaultdict(list)
        context_window = self.config['context_window']
        
        for sent_id, sentence in enumerate(sentences):
            tokens = sentence.split()
            
            for token_id, token in enumerate(tokens):
                # Check if token is a candidate
                candidate = None
                if token in candidates:
                    candidate = token
                elif token.lower().capitalize() in candidates:
                    candidate = token.lower().capitalize()
                
                if candidate:
                    # Extract context
                    start = max(0, token_id - context_window)
                    end = min(len(tokens), token_id + context_window + 1)
                    context = ' '.join(tokens[start:end])
                    
                    # Get embedding
                    try:
                        embedding = self.model.encode(context)
                        
                        candidate_embeddings[candidate].append({
                            'embedding': embedding,
                            'context': context,
                            'sentence_id': sent_id,
                            'token_id': token_id
                        })
                    except Exception as e:
                        self.logger.debug(f"    Failed to encode context for {candidate}: {e}")
        
        return dict(candidate_embeddings)
    
    def _cluster_embeddings(self, candidate_embeddings: Dict[str, List[Dict]]) -> Dict[str, int]:
        """
        Cluster candidate embeddings to group variants
        
        Returns:
            Dict mapping candidate -> cluster_id
        """
        # Collect all embeddings with labels
        all_embeddings = []
        labels = []
        
        for candidate, mentions in candidate_embeddings.items():
            for mention in mentions:
                all_embeddings.append(mention['embedding'])
                labels.append(candidate)
        
        if len(all_embeddings) < 2:
            # Not enough data to cluster
            return {labels[0]: 0} if labels else {}
        
        all_embeddings = np.array(all_embeddings)
        
        # Try HDBSCAN first
        if HDBSCAN_AVAILABLE:
            try:
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=self.config['min_cluster_size'],
                    min_samples=self.config['min_samples'],
                    cluster_selection_epsilon=self.config['cluster_selection_epsilon'],
                    metric='euclidean'
                )
                cluster_labels = clusterer.fit_predict(all_embeddings)
                
                # Map candidates to clusters
                clusters = {}
                for label, cluster_id in zip(labels, cluster_labels):
                    if label not in clusters:
                        clusters[label] = cluster_id
                
                return clusters
            except Exception as e:
                self.logger.warning(f"  HDBSCAN failed: {e}, using fallback")
        
        # Fallback: similarity-based clustering
        return self._fallback_clustering(labels, all_embeddings)
    
    def _fallback_clustering(self, labels: List[str], embeddings: np.ndarray) -> Dict[str, int]:
        """
        Fallback clustering using cosine similarity
        """
        # Average embeddings per candidate
        candidate_avg_embeddings = {}
        
        for label, emb in zip(labels, embeddings):
            if label not in candidate_avg_embeddings:
                candidate_avg_embeddings[label] = []
            candidate_avg_embeddings[label].append(emb)
        
        # Average
        for label in candidate_avg_embeddings:
            candidate_avg_embeddings[label] = np.mean(
                candidate_avg_embeddings[label], 
                axis=0
            )
        
        # Compute similarity matrix
        candidates = list(candidate_avg_embeddings.keys())
        embeddings_matrix = np.array([candidate_avg_embeddings[c] for c in candidates])
        
        similarity_matrix = cosine_similarity(embeddings_matrix)
        
        # Simple clustering: group if similarity > threshold
        clusters = {}
        cluster_id = 0
        processed = set()
        
        for i, candidate in enumerate(candidates):
            if candidate in processed:
                continue
            
            # Start new cluster
            clusters[candidate] = cluster_id
            processed.add(candidate)
            
            # Find similar candidates
            for j, other_candidate in enumerate(candidates):
                if other_candidate in processed:
                    continue
                
                if similarity_matrix[i][j] > self.config['similarity_threshold']:
                    clusters[other_candidate] = cluster_id
                    processed.add(other_candidate)
            
            cluster_id += 1
        
        return clusters
    
    def _build_candidates_from_clusters(self,
                                       clusters: Dict[str, int],
                                       candidate_embeddings: Dict[str, List[Dict]]) -> List[Dict]:
        """
        Build final candidates from clusters
        """
        cluster_groups = defaultdict(list)
        
        for candidate, cluster_id in clusters.items():
            if cluster_id != -1:  # -1 is noise in HDBSCAN
                cluster_groups[cluster_id].append(candidate)
        
        candidates = []
        
        for cluster_id, members in cluster_groups.items():
            # Choose canonical name (longest or most frequent)
            mention_counts = {m: len(candidate_embeddings.get(m, [])) for m in members}
            canonical = max(members, key=lambda x: (mention_counts[x], len(x)))
            
            # Count total mentions
            total_mentions = sum(mention_counts.values())
            
            # Collect all variants
            variants = members
            
            candidates.append({
                'name': canonical,
                'mentions': total_mentions,
                'variants': variants,
                'cluster_id': cluster_id,
                'metadata': {
                    'cluster_size': len(members),
                    'mention_counts': mention_counts
                }
            })
        
        return candidates
    
    def _detect_narrator(self, sentences: List[str]) -> Dict:
        """
        Detect first-person narrator
        """
        text = ' '.join(sentences).lower()
        
        # Count first-person pronouns
        i_count = len(re.findall(r'\bi\b', text))
        my_count = len(re.findall(r'\bmy\b', text))
        me_count = len(re.findall(r'\bme\b', text))
        
        total_first_person = i_count + my_count + me_count
        
        # Threshold: if >20 first-person pronouns, likely has narrator
        if total_first_person >= 20:
            return {
                'name': 'Narrator (I)',
                'mentions': i_count,
                'score': min(i_count / 50.0, 1.0),
                'metadata': {
                    'type': 'narrator',
                    'i_count': i_count,
                    'my_count': my_count,
                    'me_count': me_count
                }
            }
        
        return None
    
    def _detect_role_based_characters(self, sentences: List[str]) -> List[Dict]:
        """
        Detect role-based characters (The Old Man, The Officers, etc.)
        """
        text = ' '.join(sentences).lower()
        role_candidates = []
        
        for role_name, patterns in self.config['role_patterns'].items():
            count = 0
            for pattern in patterns:
                matches = re.findall(pattern, text)
                count += len(matches)
            
            if count >= 3:  # Minimum mentions
                # Capitalize properly
                display_name = 'The ' + role_name.replace('_', ' ').title()
                
                role_candidates.append({
                    'name': display_name,
                    'mentions': count,
                    'score': min(count / 20.0, 0.9),
                    'metadata': {
                        'type': 'role_based',
                        'role': role_name
                    }
                })
        
        return role_candidates
    
    def _score_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """
        Calculate final scores for candidates
        """
        scored = []
        
        for candidate in candidates:
            mentions = candidate['mentions']
            
            # Base score from mentions
            base_score = min(mentions / 30.0, 0.7)
            
            # Type-based adjustments
            ctype = candidate.get('metadata', {}).get('type')
            
            if ctype == 'narrator':
                # Boost narrator if many mentions
                if mentions >= 30:
                    base_score = min(base_score + 0.2, 0.95)
            elif ctype == 'role_based':
                # Role-based: moderate score
                base_score = min(base_score + 0.1, 0.85)
            else:
                # Cluster-based: check cluster size
                cluster_size = candidate.get('metadata', {}).get('cluster_size', 1)
                if cluster_size > 1:
                    # Boost if multiple variants detected
                    base_score = min(base_score + 0.1, 0.90)
            
            candidate['score'] = base_score
            scored.append(candidate)
        
        return scored
    
    def _fallback_extraction(self, preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback extraction when embeddings not available
        (Just use PROPN candidates with frequency)
        """
        self.logger.warning("  Using fallback extraction (no embeddings)")
        
        propn_candidates = preprocessed_data['propn_candidates']
        sentences = preprocessed_data['sentences']
        
        # Count mentions
        mention_counts = Counter()
        
        for sentence in sentences:
            for candidate in propn_candidates:
                if candidate.lower() in sentence.lower():
                    mention_counts[candidate] += 1
        
        # Build candidates
        candidates = []
        for name, count in mention_counts.items():
            if count >= self.config['min_mentions']:
                candidates.append({
                    'name': name,
                    'mentions': count,
                    'score': min(count / 20.0, 0.75),
                    'metadata': {'fallback': True}
                })
        
        # Detect narrator
        if self.config['detect_narrator']:
            narrator = self._detect_narrator(sentences)
            if narrator:
                candidates.append(narrator)
        
        candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
        
        return {
            'candidates': candidates,
            'method_name': self.get_method_name(),
            'statistics': self.get_statistics({'candidates': candidates}),
            'fallback_mode': True
        }
    
    def get_confidence_score(self, candidate: str, metadata: Dict) -> float:
        """Calculate confidence score"""
        return metadata.get('score', 0.0)