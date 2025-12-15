"""
Enhanced Text Preprocessor dengan POS tagging dan N-gram extraction
File: src/preprocessing.py
"""

import re
import nltk
import spacy
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
import logging

class EnhancedTextPreprocessor:
    """
    Preprocessor dengan tambahan:
    - POS tagging
    - Capitalization pattern tracking
    - N-gram extraction
    - Proper noun filtering
    """
    
    def __init__(self):
        # Load models
        try:
            self.nlp = spacy.load('en_core_web_lg')
        except:
            print("⚠️  Downloading spaCy model...")
            import os
            os.system('python -m spacy download en_core_web_lg')
            self.nlp = spacy.load('en_core_web_lg')
        
        # NLTK resources
        self._ensure_nltk_data()
        self.stop_words = set(stopwords.words('english'))
        
        # Logging
        self.logger = self._setup_logger()
    
    def _ensure_nltk_data(self):
        """Ensure NLTK data is downloaded"""
        resources = ['punkt', 'averaged_perceptron_tagger', 'stopwords']
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                nltk.download(resource, quiet=True)
    
    def _setup_logger(self):
        """Setup logger"""
        logger = logging.getLogger('Preprocessor')
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger
    
    def preprocess_document(self, filepath, config=None):
        """
        Enhanced preprocessing pipeline
        
        Args:
            filepath: Path to text file
            config: Optional config dict with:
                - ngram_range: tuple (min, max)
                - min_propn_length: int
                - track_positions: bool
        
        Returns:
            dict with enhanced features
        """
        self.logger.info(f"Preprocessing: {filepath}")
        
        # Default config
        if config is None:
            config = {
                'ngram_range': (1, 3),
                'min_propn_length': 2,
                'track_positions': True
            }
        
        # Read file
        with open(filepath, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        
        # Step 1: Basic cleaning
        cleaned_text = self._clean_text(raw_text)
        
        # Step 2: Sentence segmentation
        sentences = self._segment_sentences(cleaned_text)
        self.logger.info(f"  ✓ {len(sentences)} sentences extracted")
        
        # Step 3: POS tagging & proper noun extraction
        pos_tagged, propn_candidates = self._extract_proper_nouns(
            sentences, 
            min_length=config['min_propn_length']
        )
        self.logger.info(f"  ✓ {len(propn_candidates)} PROPN candidates found")
        
        # Step 4: Capitalization pattern analysis
        cap_patterns = self._analyze_capitalization(
            sentences, 
            propn_candidates,
            track_positions=config['track_positions']
        )
        self.logger.info(f"  ✓ Capitalization patterns analyzed")
        
        # Step 5: N-gram extraction
        ngrams = self._extract_ngrams(
            sentences,
            propn_candidates,
            ngram_range=config['ngram_range']
        )
        self.logger.info(f"  ✓ {sum(len(v) for v in ngrams.values())} n-grams extracted")
        
        # Compile results
        result = {
            'filepath': filepath,
            'raw_text': raw_text,
            'cleaned_text': cleaned_text,
            'sentences': sentences,
            'sentence_count': len(sentences),
            'pos_tagged': pos_tagged,
            'propn_candidates': propn_candidates,
            'capitalization_patterns': cap_patterns,
            'ngrams': ngrams,
            'config': config
        }
        
        return result
    
    def _clean_text(self, text):
        """Basic text cleaning"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\'\"\-]', '', text)
        text = re.sub(r'([.!?]){2,}', r'\1', text)
        return text.strip()
    
    def _segment_sentences(self, text):
        """Sentence segmentation"""
        sentences = sent_tokenize(text)
        return [sent.strip() for sent in sentences if len(sent.strip()) > 10]
    
    def _extract_proper_nouns(self, sentences, min_length=2):
        """
        Extract proper noun candidates using spaCy POS tagging
        
        Returns:
            - pos_tagged: List of (sentence_id, tokens with POS)
            - propn_candidates: Set of proper noun strings
        """
        pos_tagged = []
        propn_candidates = set()
        
        for sent_id, sentence in enumerate(sentences):
            doc = self.nlp(sentence)
            
            tokens_with_pos = []
            for token in doc:
                tokens_with_pos.append({
                    'text': token.text,
                    'pos': token.pos_,
                    'tag': token.tag_,
                    'is_alpha': token.is_alpha
                })
                
                # Collect PROPN (proper nouns)
                if token.pos_ == 'PROPN' and len(token.text) >= min_length:
                    propn_candidates.add(token.text)
            
            pos_tagged.append({
                'sentence_id': sent_id,
                'tokens': tokens_with_pos
            })
        
        return pos_tagged, propn_candidates
    
    def _analyze_capitalization(self, sentences, propn_candidates, track_positions=True):
        """
        Analyze capitalization patterns for each candidate
        
        Returns:
            dict with capitalization statistics per candidate
        """
        patterns = defaultdict(lambda: {
            'total_mentions': 0,
            'capitalized_mentions': 0,
            'sentence_start_count': 0,
            'mid_sentence_count': 0,
            'positions': []
        })
        
        for sent_id, sentence in enumerate(sentences):
            tokens = word_tokenize(sentence)
            
            for token_id, token in enumerate(tokens):
                # Check if token or its lowercase is a candidate
                candidate = None
                if token in propn_candidates:
                    candidate = token
                elif token.lower().capitalize() in propn_candidates:
                    candidate = token.lower().capitalize()
                
                if candidate:
                    patterns[candidate]['total_mentions'] += 1
                    
                    # Check capitalization
                    if token[0].isupper():
                        patterns[candidate]['capitalized_mentions'] += 1
                    
                    # Check position
                    is_sentence_start = (token_id == 0)
                    if is_sentence_start:
                        patterns[candidate]['sentence_start_count'] += 1
                    else:
                        patterns[candidate]['mid_sentence_count'] += 1
                    
                    # Track position if needed
                    if track_positions:
                        patterns[candidate]['positions'].append((sent_id, token_id))
        
        # Calculate consistency scores
        for candidate, data in patterns.items():
            total = data['total_mentions']
            if total > 0:
                mid_sentence = data['mid_sentence_count']
                if mid_sentence > 0:
                    cap_ratio = data['capitalized_mentions'] / total
                    mid_ratio = mid_sentence / total
                    data['consistency_score'] = cap_ratio * mid_ratio
                else:
                    data['consistency_score'] = 0.5
            else:
                data['consistency_score'] = 0.0
        
        return dict(patterns)
    
    def _extract_ngrams(self, sentences, propn_candidates, ngram_range=(1, 3)):
        """
        Extract n-grams containing at least one PROPN
        
        Returns:
            dict with unigrams, bigrams, trigrams
        """
        ngrams = {
            'unigrams': set(),
            'bigrams': set(),
            'trigrams': set()
        }
        
        min_n, max_n = ngram_range
        
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            
            # Generate n-grams
            for n in range(min_n, max_n + 1):
                for i in range(len(tokens) - n + 1):
                    ngram_tokens = tokens[i:i+n]
                    ngram_text = ' '.join(ngram_tokens)
                    
                    # Check if contains at least one PROPN candidate
                    has_propn = any(token in propn_candidates for token in ngram_tokens)
                    
                    # Additional filter: must have consistent capitalization
                    all_capitalized = all(
                        token[0].isupper() if token[0].isalpha() else True 
                        for token in ngram_tokens
                    )
                    
                    if has_propn and all_capitalized:
                        if n == 1:
                            ngrams['unigrams'].add(ngram_text)
                        elif n == 2:
                            ngrams['bigrams'].add(ngram_text)
                        elif n == 3:
                            ngrams['trigrams'].add(ngram_text)
        
        # Convert to sorted lists
        return {
            'unigrams': sorted(list(ngrams['unigrams'])),
            'bigrams': sorted(list(ngrams['bigrams'])),
            'trigrams': sorted(list(ngrams['trigrams']))
        }
    
    def get_statistics(self, preprocessed_data):
        """Get preprocessing statistics"""
        return {
            'total_sentences': preprocessed_data['sentence_count'],
            'propn_candidates': len(preprocessed_data['propn_candidates']),
            'unigrams': len(preprocessed_data['ngrams']['unigrams']),
            'bigrams': len(preprocessed_data['ngrams']['bigrams']),
            'trigrams': len(preprocessed_data['ngrams']['trigrams']),
            'high_confidence_caps': sum(
                1 for data in preprocessed_data['capitalization_patterns'].values()
                if data['consistency_score'] > 0.75
            )
        }