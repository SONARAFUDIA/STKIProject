"""
Unsupervised Entity Extraction for Literary Texts
NO NER - Pure pattern mining and statistical methods

Three Methods:
1. Capitalization Pattern Mining - Detects consistently capitalized words
2. TF-ISF Statistical Ranking - Term Frequency - Inverse Sentence Frequency
3. N-gram Co-occurrence Analysis - Multi-word name detection

Author: STKI Project
Date: 2024
"""

import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter, defaultdict
import numpy as np
from typing import List, Dict, Tuple, Set
import json

# Download NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK data...")
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)


class UnsupervisedEntityExtractor:
    """
    Unsupervised entity extraction using pattern mining and statistics.
    No pre-trained models, no labeled data needed.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize extractor with optional configuration
        
        Args:
            config: Optional configuration dict with thresholds
        """
        # Default configuration
        self.config = config or self._get_default_config()
        
        # Blacklist: words that look like names but aren't characters
        self.blacklist = self._build_blacklist()
        
        print("✓ UnsupervisedEntityExtractor initialized")
    
    def _get_default_config(self) -> Dict:
        """Default configuration for extraction"""
        return {
            'min_mentions': 3,                    # Minimum mentions to be considered
            'capitalization_threshold': 0.5,      # Min consistency for capitalization
            'tfisf_min_score': 0.01,             # Min TF-ISF score
            'ngram_min_count': 2,                # Min n-gram occurrences
            'max_sentence_frequency': 0.5,       # Max % of sentences (filters common words)
            'min_sentence_frequency': 2,         # Min sentence appearances
            'ensemble_min_methods': 2,           # Min methods that must agree
            'ensemble_weights': {                # Weight for each method
                'capitalization': 0.4,
                'tfisf': 0.3,
                'ngrams': 0.3
            }
        }
    
    def _build_blacklist(self) -> Set[str]:
        """Build blacklist of non-character words"""
        return {
            # Days
            'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 
            'saturday', 'sunday',
            
            # Months
            'january', 'february', 'march', 'april', 'may', 'june', 
            'july', 'august', 'september', 'october', 'november', 'december',
            
            # Holidays
            'christmas', 'easter', 'thanksgiving',
            
            # Common words
            'the', 'and', 'but', 'for', 'with', 'dear', 'young', 'old',
            'sir', 'madam', 'miss', 'mrs', 'mr', 'dr',
            
            # Religious/Mythological (not story characters)
            'god', 'lord', 'jesus', 'christ', 'magi', 'wise', 'men',
            
            # Places (common false positives)
            'america', 'american', 'alabama', 'england', 'english', 
            'federal', 'southern', 'northern',
            
            # Time references
            'today', 'tomorrow', 'yesterday', 'now', 'then',
            
            # Generic references
            'i', 'you', 'he', 'she', 'they', 'we', 'it'
        }
    
    def extract_entities(self, filepath: str) -> Dict:
        """
        Main extraction pipeline
        
        Args:
            filepath: Path to text file
        
        Returns:
            Dict with:
                - entities: List of detected entities
                - method_results: Raw results from each method
                - statistics: Extraction statistics
        """
        print("\n" + "="*70)
        print("UNSUPERVISED ENTITY EXTRACTION")
        print("="*70)
        print(f"File: {filepath}")
        print(f"Config: min_mentions={self.config['min_mentions']}, "
              f"threshold={self.config['capitalization_threshold']}")
        
        # Step 1: Load and segment
        print("\n[1/4] Loading and segmenting text...")
        sentences = self._load_and_segment(filepath)
        print(f"  ✓ Loaded {len(sentences)} sentences")
        
        # Step 2: Method 1 - Capitalization mining
        print("\n[2/4] Method 1: Capitalization Pattern Mining...")
        cap_candidates = self._method1_capitalization(sentences)
        print(f"  ✓ Found {len(cap_candidates)} candidates")
        if cap_candidates:
            top3 = list(cap_candidates.items())[:3]
            print(f"  Top 3: {', '.join([f'{k} ({v})' for k, v in top3])}")
        
        # Step 3: Method 2 - TF-ISF ranking
        print("\n[3/4] Method 2: TF-ISF Statistical Ranking...")
        tfisf_candidates = self._method2_tfisf(sentences)
        print(f"  ✓ Found {len(tfisf_candidates)} candidates")
        if tfisf_candidates:
            top3 = list(tfisf_candidates.items())[:3]
            print(f"  Top 3: {', '.join([f'{k} ({v})' for k, v in top3])}")
        
        # Step 4: Method 3 - N-gram mining
        print("\n[4/4] Method 3: N-gram Co-occurrence Mining...")
        ngram_candidates = self._method3_ngrams(sentences)
        print(f"  ✓ Found {len(ngram_candidates)} candidates")
        if ngram_candidates:
            top3 = list(ngram_candidates.items())[:3]
            print(f"  Top 3: {', '.join([f'{k} ({v})' for k, v in top3])}")
        
        # Step 5: Ensemble voting
        print("\n[Ensemble] Combining methods with voting...")
        final_entities = self._ensemble_voting(
            cap_candidates, 
            tfisf_candidates, 
            ngram_candidates
        )
        print(f"  ✓ Final: {len(final_entities)} entities detected")
        
        # Build result
        result = {
            'entities': final_entities,
            'method_results': {
                'method1_capitalization': cap_candidates,
                'method2_tfisf': tfisf_candidates,
                'method3_ngrams': ngram_candidates
            },
            'statistics': {
                'total_sentences': len(sentences),
                'total_entities': len(final_entities),
                'method1_candidates': len(cap_candidates),
                'method2_candidates': len(tfisf_candidates),
                'method3_candidates': len(ngram_candidates),
                'config': self.config
            }
        }
        
        return result
    
    def _load_and_segment(self, filepath: str) -> List[str]:
        """Load text file and segment into sentences"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {filepath}")
        except Exception as e:
            raise Exception(f"Error reading file: {e}")
        
        # Clean text
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()
        
        # Sentence segmentation
        sentences = sent_tokenize(text)
        
        # Filter out very short sentences
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        return sentences
    
    def _method1_capitalization(self, sentences: List[str]) -> Dict[str, int]:
        """
        Method 1: Capitalization Pattern Mining
        
        Logic:
        - Track words that are consistently capitalized
        - Favor mid-sentence occurrences (more reliable than sentence-start)
        - Calculate consistency score
        - Filter by threshold
        
        Returns:
            Dict mapping candidate -> mention count
        """
        # Track capitalization patterns
        patterns = defaultdict(lambda: {
            'total': 0,           # Total occurrences
            'capitalized': 0,     # Times it appeared capitalized
            'mid_sentence': 0,    # Mid-sentence occurrences
            'sentence_start': 0   # Sentence-start occurrences
        })
        
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            
            for i, token in enumerate(tokens):
                # Skip if not alphabetic or too short
                if not token.isalpha() or len(token) < 3:
                    continue
                
                # Skip all-caps (likely acronyms)
                if token.isupper() and len(token) > 1:
                    continue
                
                # Normalize to title case
                normalized = token.capitalize()
                
                # Track position
                is_start = (i == 0)
                
                # Update stats
                patterns[normalized]['total'] += 1
                
                if token[0].isupper():
                    patterns[normalized]['capitalized'] += 1
                
                if is_start:
                    patterns[normalized]['sentence_start'] += 1
                else:
                    patterns[normalized]['mid_sentence'] += 1
        
        # Score and filter candidates
        candidates = {}
        
        for word, stats in patterns.items():
            # Must appear in mid-sentence at least twice
            if stats['mid_sentence'] < 2:
                continue
            
            # Skip blacklisted words
            if word.lower() in self.blacklist:
                continue
            
            # Calculate consistency score
            # Formula: (capitalized_ratio) × (mid_sentence_ratio)
            cap_ratio = stats['capitalized'] / stats['total']
            mid_ratio = stats['mid_sentence'] / stats['total']
            consistency = cap_ratio * mid_ratio
            
            # Filter by threshold
            if consistency >= self.config['capitalization_threshold']:
                candidates[word] = stats['total']
        
        # Sort by mention count
        candidates = dict(sorted(candidates.items(), 
                                key=lambda x: x[1], 
                                reverse=True))
        
        return candidates
    
    def _method2_tfisf(self, sentences: List[str]) -> Dict[str, int]:
        """
        Method 2: TF-ISF (Term Frequency - Inverse Sentence Frequency)
        
        Formula:
        TF = term_count / total_terms
        ISF = log(total_sentences / sentences_containing_term)
        TF-ISF = TF × ISF
        
        Higher score = more discriminative (likely a name)
        
        Returns:
            Dict mapping candidate -> mention count
        """
        # Build vocabulary (only capitalized words)
        vocab = set()
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            for token in tokens:
                if (token.isalpha() and 
                    len(token) >= 3 and 
                    token[0].isupper() and
                    not token.isupper()):
                    vocab.add(token.capitalize())
        
        # Calculate frequencies
        term_freq = Counter()
        sentence_freq = defaultdict(int)
        total_terms = 0
        
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            total_terms += len(tokens)
            
            # Track terms in this sentence (for ISF)
            terms_in_sentence = set()
            
            for token in tokens:
                normalized = token.capitalize()
                if normalized in vocab:
                    term_freq[normalized] += 1
                    terms_in_sentence.add(normalized)
            
            # Update sentence frequency
            for term in terms_in_sentence:
                sentence_freq[term] += 1
        
        # Calculate TF-ISF scores
        tfisf_scores = {}
        total_sentences = len(sentences)
        max_sf = int(total_sentences * self.config['max_sentence_frequency'])
        
        for term in vocab:
            tf = term_freq[term] / total_terms
            sf = sentence_freq[term]
            
            # Filter: too common
            if sf > max_sf:
                continue
            
            # Filter: too rare
            if sf < self.config['min_sentence_frequency']:
                continue
            
            # Calculate ISF
            isf = np.log(total_sentences / sf)
            tfisf = tf * isf
            
            # Filter by threshold
            if tfisf >= self.config['tfisf_min_score']:
                # Skip blacklisted
                if term.lower() not in self.blacklist:
                    tfisf_scores[term] = term_freq[term]
        
        # Sort by score
        tfisf_scores = dict(sorted(tfisf_scores.items(), 
                                   key=lambda x: x[1], 
                                   reverse=True)[:20])
        
        return tfisf_scores
    
    def _method3_ngrams(self, sentences: List[str]) -> Dict[str, int]:
        """
        Method 3: N-gram Co-occurrence Mining
        
        Logic:
        - Extract bigrams and trigrams with consistent capitalization
        - These are likely multi-word names (e.g., "James Dillingham Young")
        - Filter by frequency threshold
        
        Returns:
            Dict mapping n-gram -> occurrence count
        """
        bigrams = Counter()
        trigrams = Counter()
        
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            
            # Extract bigrams
            for i in range(len(tokens) - 1):
                w1, w2 = tokens[i], tokens[i+1]
                
                # Both must be alphabetic, capitalized, and min length
                if (w1.isalpha() and w2.isalpha() and 
                    len(w1) >= 3 and len(w2) >= 3 and
                    w1[0].isupper() and w2[0].isupper() and
                    not w1.isupper() and not w2.isupper()):
                    
                    bigram = f"{w1.capitalize()} {w2.capitalize()}"
                    bigrams[bigram] += 1
            
            # Extract trigrams
            for i in range(len(tokens) - 2):
                w1, w2, w3 = tokens[i], tokens[i+1], tokens[i+2]
                
                if (w1.isalpha() and w2.isalpha() and w3.isalpha() and
                    len(w1) >= 3 and len(w2) >= 3 and len(w3) >= 3 and
                    w1[0].isupper() and w2[0].isupper() and w3[0].isupper() and
                    not w1.isupper() and not w2.isupper() and not w3.isupper()):
                    
                    trigram = f"{w1.capitalize()} {w2.capitalize()} {w3.capitalize()}"
                    trigrams[trigram] += 1
        
        # Combine and filter
        candidates = {}
        min_count = self.config['ngram_min_count']
        
        # Add bigrams
        for ngram, count in bigrams.items():
            if count >= min_count:
                words = ngram.lower().split()
                # Check not blacklisted
                if not any(w in self.blacklist for w in words):
                    candidates[ngram] = count
        
        # Add trigrams
        for ngram, count in trigrams.items():
            if count >= min_count:
                words = ngram.lower().split()
                if not any(w in self.blacklist for w in words):
                    candidates[ngram] = count
        
        # Sort by frequency
        candidates = dict(sorted(candidates.items(), 
                                key=lambda x: x[1], 
                                reverse=True))
        
        return candidates
    
    def _ensemble_voting(self, 
                        cap_candidates: Dict,
                        tfisf_candidates: Dict,
                        ngram_candidates: Dict) -> List[Dict]:
        """
        Ensemble voting: combine results from all methods
        
        Voting strategy:
        - Require agreement from at least 2 methods (majority vote)
        - Calculate confidence based on number of methods agreeing
        - Calculate weighted score using method weights
        
        Returns:
            List of entity dicts sorted by confidence
        """
        # Collect all unique candidates
        all_candidates = set()
        all_candidates.update(cap_candidates.keys())
        all_candidates.update(tfisf_candidates.keys())
        all_candidates.update(ngram_candidates.keys())
        
        entities = []
        weights = self.config['ensemble_weights']
        min_methods = self.config['ensemble_min_methods']
        
        for candidate in all_candidates:
            # Check which methods detected this candidate
            detections = []
            scores = {}
            
            if candidate in cap_candidates:
                detections.append('capitalization')
                scores['capitalization'] = cap_candidates[candidate]
            
            if candidate in tfisf_candidates:
                detections.append('tfisf')
                scores['tfisf'] = tfisf_candidates[candidate]
            
            if candidate in ngram_candidates:
                detections.append('ngrams')
                scores['ngrams'] = ngram_candidates[candidate]
            
            # Filter: require minimum methods agreement
            if len(detections) < min_methods:
                continue
            
            # Calculate average mention count
            avg_mentions = int(np.mean(list(scores.values())))
            
            # Filter by minimum mentions
            if avg_mentions < self.config['min_mentions']:
                continue
            
            # Calculate confidence (0.0 to 1.0)
            # Based on: number of methods agreeing / total methods
            confidence = len(detections) / 3.0
            
            # Calculate weighted score
            weighted_score = 0.0
            for method in detections:
                method_score = scores[method]
                method_weight = weights.get(method, 0.33)
                weighted_score += method_score * method_weight
            
            # Build entity dict
            entity = {
                'name': candidate,
                'mentions': avg_mentions,
                'confidence': round(confidence, 2),
                'score': round(weighted_score, 2),
                'detected_by': detections,
                'detection_count': len(detections),
                'method_scores': scores
            }
            
            entities.append(entity)
        
        # Sort by confidence (primary) and mentions (secondary)
        entities.sort(key=lambda x: (x['confidence'], x['mentions']), reverse=True)
        
        return entities
    
    def save_results(self, results: Dict, output_path: str):
        """
        Save extraction results to JSON file
        
        Args:
            results: Results dict from extract_entities()
            output_path: Path to output JSON file
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Results saved to: {output_path}")
        except Exception as e:
            print(f"\n✗ Error saving results: {e}")
    
    def format_results(self, results: Dict) -> str:
        """
        Format results for console display
        
        Args:
            results: Results dict from extract_entities()
        
        Returns:
            Formatted string for display
        """
        lines = []
        
        lines.append("\n" + "="*70)
        lines.append("EXTRACTION RESULTS")
        lines.append("="*70)
        
        # Statistics
        stats = results['statistics']
        lines.append(f"\n📊 Document Statistics:")
        lines.append(f"  Total Sentences: {stats['total_sentences']}")
        lines.append(f"  Total Entities: {stats['total_entities']}")
        
        lines.append(f"\n🔍 Method Candidates:")
        lines.append(f"  Method 1 (Capitalization): {stats['method1_candidates']}")
        lines.append(f"  Method 2 (TF-ISF): {stats['method2_candidates']}")
        lines.append(f"  Method 3 (N-grams): {stats['method3_candidates']}")
        
        # Entities
        lines.append(f"\n" + "="*70)
        lines.append("🎭 DETECTED ENTITIES")
        lines.append("="*70)
        
        entities = results['entities']
        
        if not entities:
            lines.append("\n  ⚠️  No entities detected.")
            lines.append("  Try lowering min_mentions threshold or check input text.")
        else:
            lines.append(f"\nTotal: {len(entities)} entities\n")
            
            for i, entity in enumerate(entities, 1):
                lines.append(f"{i}. {entity['name']}")
                lines.append(f"   Mentions: {entity['mentions']}")
                lines.append(f"   Confidence: {entity['confidence']:.0%}")
                lines.append(f"   Score: {entity['score']:.2f}")
                lines.append(f"   Detected by: {', '.join(entity['detected_by'])}")
                lines.append(f"   Agreement: {entity['detection_count']}/3 methods")
                lines.append("")
        
        lines.append("="*70)
        
        return "\n".join(lines)



class UnsupervisedEntityExtractor:
    """
    Unsupervised entity extraction using:
    - Capitalization patterns
    - Statistical frequency analysis
    - N-gram mining
    """
    
    def __init__(self):
        # Blacklist: non-character words
        self.blacklist = {
            # Temporal
            'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 
            'saturday', 'sunday', 'january', 'february', 'march', 'april', 
            'may', 'june', 'july', 'august', 'september', 'october', 
            'november', 'december', 'christmas', 'today', 'tomorrow',
            
            # Common words
            'the', 'and', 'but', 'for', 'with', 'dear', 'young', 'old',
            
            # Religious/Mythological
            'god', 'lord', 'jesus', 'magi',
            
            # Places (common false positives)
            'america', 'american', 'alabama', 'england', 'federal'
        }
    
    def extract_entities(self, filepath: str, min_mentions: int = 3) -> Dict:
        """
        Main extraction pipeline
        
        Returns:
            {
                'entities': [{'name': str, 'mentions': int, 'confidence': float}],
                'method_results': {...},
                'statistics': {...}
            }
        """
        print("\n" + "="*70)
        print("UNSUPERVISED ENTITY EXTRACTION")
        print("="*70)
        
        # Read and preprocess
        print("\n[1/4] Loading document...")
        sentences = self._load_and_segment(filepath)
        print(f"  ✓ {len(sentences)} sentences")
        
        # Method 1: Capitalization mining
        print("\n[2/4] Method 1: Capitalization Pattern Mining...")
        cap_candidates = self._method1_capitalization(sentences)
        print(f"  ✓ {len(cap_candidates)} candidates")
        
        # Method 2: TF-ISF ranking
        print("\n[3/4] Method 2: TF-ISF Statistical Ranking...")
        tfisf_candidates = self._method2_tfisf(sentences)
        print(f"  ✓ {len(tfisf_candidates)} candidates")
        
        # Method 3: N-gram mining
        print("\n[4/4] Method 3: N-gram Co-occurrence...")
        ngram_candidates = self._method3_ngrams(sentences)
        print(f"  ✓ {len(ngram_candidates)} candidates")
        
        # Ensemble voting
        print("\n[Ensemble] Combining methods...")
        final_entities = self._ensemble_voting(
            cap_candidates, 
            tfisf_candidates, 
            ngram_candidates,
            min_mentions
        )
        print(f"  ✓ {len(final_entities)} final entities")
        
        return {
            'entities': final_entities,
            'method_results': {
                'method1_capitalization': cap_candidates,
                'method2_tfisf': tfisf_candidates,
                'method3_ngrams': ngram_candidates
            },
            'statistics': {
                'total_sentences': len(sentences),
                'total_entities': len(final_entities),
                'method1_count': len(cap_candidates),
                'method2_count': len(tfisf_candidates),
                'method3_count': len(ngram_candidates)
            }
        }
    
    def _load_and_segment(self, filepath: str) -> List[str]:
        """Load text and segment into sentences"""
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Clean text
        text = re.sub(r'\s+', ' ', text)
        
        # Sentence segmentation
        sentences = sent_tokenize(text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def _method1_capitalization(self, sentences: List[str]) -> Dict[str, int]:
        """
        Method 1: Mine capitalization patterns
        
        Logic:
        - Find words that are CONSISTENTLY capitalized in mid-sentence
        - Filter sentence-start positions (less reliable)
        - Score by consistency ratio
        """
        cap_patterns = defaultdict(lambda: {
            'total': 0,
            'capitalized': 0,
            'mid_sentence': 0,
            'sentence_start': 0
        })
        
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            
            for i, token in enumerate(tokens):
                # Skip if not alphabetic or too short
                if not token.isalpha() or len(token) < 3:
                    continue
                
                # Skip if all caps (likely acronym)
                if token.isupper():
                    continue
                
                # Normalize
                normalized = token.capitalize()
                
                # Track position
                is_start = (i == 0)
                
                cap_patterns[normalized]['total'] += 1
                
                if token[0].isupper():
                    cap_patterns[normalized]['capitalized'] += 1
                
                if is_start:
                    cap_patterns[normalized]['sentence_start'] += 1
                else:
                    cap_patterns[normalized]['mid_sentence'] += 1
        
        # Score candidates
        candidates = {}
        
        for word, stats in cap_patterns.items():
            # Must have mid-sentence occurrences
            if stats['mid_sentence'] < 2:
                continue
            
            # Skip blacklist
            if word.lower() in self.blacklist:
                continue
            
            # Calculate consistency score
            # Prefer words that are always capitalized when mid-sentence
            if stats['mid_sentence'] > 0:
                mid_cap_ratio = stats['capitalized'] / stats['total']
                mid_sentence_ratio = stats['mid_sentence'] / stats['total']
                
                # High mid-cap ratio + high mid-sentence ratio = likely name
                score = mid_cap_ratio * mid_sentence_ratio
                
                if score > 0.5:  # Threshold
                    candidates[word] = stats['total']
        
        return candidates
    
    def _method2_tfisf(self, sentences: List[str]) -> Dict[str, int]:
        """
        Method 2: TF-ISF (Term Frequency - Inverse Sentence Frequency)
        
        Formula:
        TF-ISF(term) = (term_freq / total_terms) * log(total_sentences / sentences_with_term)
        
        Higher score = more discriminative (likely a name)
        """
        # Build vocabulary (only capitalized words)
        vocab = set()
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            for token in tokens:
                if token.isalpha() and len(token) >= 3 and token[0].isupper():
                    vocab.add(token.capitalize())
        
        # Calculate sentence frequency
        sentence_freq = defaultdict(int)
        total_terms = 0
        term_freq = Counter()
        
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            total_terms += len(tokens)
            
            # Track which terms appear in this sentence
            terms_in_sentence = set()
            
            for token in tokens:
                if token.capitalize() in vocab:
                    normalized = token.capitalize()
                    term_freq[normalized] += 1
                    terms_in_sentence.add(normalized)
            
            # Increment sentence frequency
            for term in terms_in_sentence:
                sentence_freq[term] += 1
        
        # Calculate TF-ISF
        tfisf_scores = {}
        total_sentences = len(sentences)
        
        for term in vocab:
            tf = term_freq[term] / total_terms
            sf = sentence_freq[term]
            
            # Skip too common (appears in >50% sentences)
            if sf > total_sentences * 0.5:
                continue
            
            # Skip too rare (appears in <2 sentences)
            if sf < 2:
                continue
            
            isf = np.log(total_sentences / sf)
            tfisf = tf * isf
            
            # Filter by blacklist
            if term.lower() not in self.blacklist:
                tfisf_scores[term] = term_freq[term]
        
        # Return top candidates
        sorted_candidates = dict(
            sorted(tfisf_scores.items(), key=lambda x: x[1], reverse=True)[:20]
        )
        
        return sorted_candidates
    
    def _method3_ngrams(self, sentences: List[str]) -> Dict[str, int]:
        """
        Method 3: N-gram mining
        
        Logic:
        - Extract bigrams and trigrams
        - Filter those with consistent capitalization
        - Likely multi-word names (James Dillingham Young)
        """
        bigrams = Counter()
        trigrams = Counter()
        
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            
            # Extract bigrams
            for i in range(len(tokens) - 1):
                w1, w2 = tokens[i], tokens[i+1]
                
                # Both must be capitalized and alphabetic
                if (w1.isalpha() and w2.isalpha() and 
                    len(w1) >= 3 and len(w2) >= 3 and
                    w1[0].isupper() and w2[0].isupper()):
                    
                    bigram = f"{w1.capitalize()} {w2.capitalize()}"
                    bigrams[bigram] += 1
            
            # Extract trigrams
            for i in range(len(tokens) - 2):
                w1, w2, w3 = tokens[i], tokens[i+1], tokens[i+2]
                
                if (w1.isalpha() and w2.isalpha() and w3.isalpha() and
                    len(w1) >= 3 and len(w2) >= 3 and len(w3) >= 3 and
                    w1[0].isupper() and w2[0].isupper() and w3[0].isupper()):
                    
                    trigram = f"{w1.capitalize()} {w2.capitalize()} {w3.capitalize()}"
                    trigrams[trigram] += 1
        
        # Combine and filter
        candidates = {}
        
        for ngram, count in bigrams.items():
            if count >= 2:  # Min occurrences
                # Check not blacklisted
                words = ngram.lower().split()
                if not any(w in self.blacklist for w in words):
                    candidates[ngram] = count
        
        for ngram, count in trigrams.items():
            if count >= 2:
                words = ngram.lower().split()
                if not any(w in self.blacklist for w in words):
                    candidates[ngram] = count
        
        return candidates
    
    def _ensemble_voting(self, 
                        cap_candidates: Dict,
                        tfisf_candidates: Dict,
                        ngram_candidates: Dict,
                        min_mentions: int) -> List[Dict]:
        """
        Ensemble voting: combine all methods
        
        Scoring:
        - Detected by 3 methods = high confidence
        - Detected by 2 methods = medium confidence
        - Detected by 1 method = low confidence (filter out)
        """
        # Collect all unique candidates
        all_candidates = set()
        all_candidates.update(cap_candidates.keys())
        all_candidates.update(tfisf_candidates.keys())
        all_candidates.update(ngram_candidates.keys())
        
        entities = []
        
        for candidate in all_candidates:
            # Count detections
            detections = 0
            methods = []
            
            if candidate in cap_candidates:
                detections += 1
                methods.append('capitalization')
            
            if candidate in tfisf_candidates:
                detections += 1
                methods.append('tfisf')
            
            if candidate in ngram_candidates:
                detections += 1
                methods.append('ngrams')
            
            # Require at least 2 methods (majority vote)
            if detections < 2:
                continue
            
            # Get total mentions (average from methods)
            mentions = 0
            count = 0
            
            if candidate in cap_candidates:
                mentions += cap_candidates[candidate]
                count += 1
            if candidate in tfisf_candidates:
                mentions += tfisf_candidates[candidate]
                count += 1
            if candidate in ngram_candidates:
                mentions += ngram_candidates[candidate]
                count += 1
            
            avg_mentions = mentions // count if count > 0 else 0
            
            # Filter by min mentions
            if avg_mentions < min_mentions:
                continue
            
            # Calculate confidence
            confidence = detections / 3.0
            
            entities.append({
                'name': candidate,
                'mentions': avg_mentions,
                'confidence': confidence,
                'detected_by': methods,
                'detection_count': detections
            })
        
        # Sort by confidence, then mentions
        entities.sort(key=lambda x: (x['confidence'], x['mentions']), reverse=True)
        
        return entities


def format_results(results: Dict) -> str:
    """Format results for display"""
    output = []
    
    output.append("\n" + "="*70)
    output.append("EXTRACTION RESULTS")
    output.append("="*70)
    
    # Statistics
    stats = results['statistics']
    output.append(f"\nDocument Statistics:")
    output.append(f"  Total Sentences: {stats['total_sentences']}")
    output.append(f"  Total Entities Found: {stats['total_entities']}")
    
    output.append(f"\nMethod Results:")
    output.append(f"  Method 1 (Capitalization): {stats['method1_count']} candidates")
    output.append(f"  Method 2 (TF-ISF): {stats['method2_count']} candidates")
    output.append(f"  Method 3 (N-grams): {stats['method3_count']} candidates")
    
    # Entities
    output.append(f"\n" + "="*70)
    output.append("DETECTED ENTITIES")
    output.append("="*70)
    
    entities = results['entities']
    
    if not entities:
        output.append("\n  No entities detected.")
    else:
        output.append(f"\nTotal: {len(entities)} entities\n")
        
        for i, entity in enumerate(entities, 1):
            output.append(f"{i}. {entity['name']}")
            output.append(f"   Mentions: {entity['mentions']}")
            output.append(f"   Confidence: {entity['confidence']:.2%}")
            output.append(f"   Detected by: {', '.join(entity['detected_by'])}")
            output.append(f"   Methods: {entity['detection_count']}/3")
            output.append("")
    
    output.append("="*70 + "\n")
    
    return "\n".join(output)