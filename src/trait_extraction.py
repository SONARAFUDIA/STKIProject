import spacy
from textblob import TextBlob
import re
from collections import Counter

class TraitExtractor:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_lg')
        
        # Expanded trait keywords (lebih banyak kata sifat)
        self.trait_keywords = {
            'positive': [
                'kind', 'brave', 'honest', 'loyal', 'generous', 'wise', 'good',
                'gentle', 'patient', 'loving', 'caring', 'compassionate', 'beautiful',
                'noble', 'heroic', 'virtuous', 'faithful', 'trustworthy', 'sweet',
                'precious', 'dear', 'wonderful', 'fine', 'excellent', 'proud',
                'happy', 'cheerful', 'bright', 'clever', 'smart', 'quick'
            ],
            'negative': [
                'cruel', 'evil', 'dishonest', 'selfish', 'greedy', 'foolish', 'bad',
                'harsh', 'impatient', 'hateful', 'wicked', 'mean', 'brutal', 'poor',
                'villainous', 'treacherous', 'malicious', 'suspicious', 'ugly',
                'terrible', 'horrible', 'awful', 'dreadful', 'miserable', 'sad'
            ],
            'emotional': [
                'sad', 'happy', 'angry', 'fearful', 'anxious', 'nervous', 'hysterical',
                'excited', 'depressed', 'joyful', 'melancholy', 'passionate', 'emotional',
                'sentimental', 'tender', 'sobbing', 'crying', 'laughing', 'smiling'
            ],
            'physical': [
                'tall', 'short', 'thin', 'fat', 'slender', 'beautiful', 'handsome',
                'ugly', 'young', 'old', 'strong', 'weak', 'pale', 'dark', 'fair',
                'graceful', 'clumsy', 'quick', 'slow', 'delicate', 'robust'
            ],
            'behavioral': [
                'aggressive', 'passive', 'cautious', 'reckless', 'calm', 'quiet',
                'violent', 'peaceful', 'active', 'lazy', 'diligent', 'careful',
                'careless', 'thoughtful', 'impulsive', 'deliberate', 'shy', 'bold'
            ]
        }
        
        # Action verbs yang mengindikasikan trait
        self.trait_indicating_actions = {
            'positive': ['helped', 'saved', 'protected', 'loved', 'cared', 'gave', 'sacrificed'],
            'negative': ['hurt', 'killed', 'destroyed', 'hated', 'stole', 'betrayed'],
            'emotional': ['cried', 'laughed', 'sobbed', 'smiled', 'wept', 'rejoiced']
        }
    
    def extract_traits(self, character_name, character_contexts):
        """
        Ekstraksi watak karakter dari konteks kemunculannya
        """
        all_traits = []
        trait_sentences = []
        
        for context in character_contexts:
            sentence = context['sentence']
            doc = self.nlp(sentence)
            
            # Method 1: Adjacent adjectives
            traits_found = self._find_adjacent_adjectives(doc, character_name)
            
            # Method 2: Pattern matching (CHARACTER is/was/seems ADJECTIVE)
            pattern_traits = self._pattern_matching(sentence, character_name)
            
            # Method 3: Possessive descriptions (CHARACTER's ADJECTIVE NOUN)
            possessive_traits = self._possessive_descriptions(sentence, character_name)
            
            # Method 4: Action-based trait inference
            action_traits = self._action_based_traits(doc, character_name)
            
            # Method 5: Descriptive phrases
            descriptive_traits = self._descriptive_phrases(sentence, character_name)
            
            # Method 6: Sentiment analysis
            sentiment_trait = self._analyze_sentiment(sentence)
            
            # Combine all
            combined_traits = (traits_found + pattern_traits + possessive_traits + 
                             action_traits + descriptive_traits)
            
            if sentiment_trait:
                combined_traits.append(sentiment_trait)
            
            if combined_traits:
                all_traits.extend(combined_traits)
                trait_sentences.append({
                    'sentence': sentence,
                    'traits': combined_traits
                })
        
        # Aggregate and classify
        trait_summary = self._classify_traits(all_traits)
        
        return {
            'character': character_name,
            'raw_traits': all_traits,
            'trait_frequency': dict(Counter(all_traits)),
            'classified_traits': trait_summary,
            'evidence_sentences': trait_sentences
        }
    
    def _find_adjacent_adjectives(self, doc, character_name):
        """
        Mencari adjektiva di sekitar nama karakter (expanded window)
        """
        traits = []
        char_tokens = character_name.lower().split()
        
        for i, token in enumerate(doc):
            if token.text.lower() in char_tokens:
                # Expand window to 5 tokens before and after
                window_start = max(0, i - 5)
                window_end = min(len(doc), i + 6)
                
                for j in range(window_start, window_end):
                    if doc[j].pos_ == 'ADJ':
                        traits.append(doc[j].text.lower())
        
        return traits
    
    def _pattern_matching(self, sentence, character_name):
        """
        Pattern matching untuk struktur kalimat umum
        """
        traits = []
        sent_lower = sentence.lower()
        char_lower = character_name.lower()
        
        # Escape special regex characters in character name
        char_escaped = re.escape(char_lower)
        
        # Expanded patterns
        patterns = [
            # "CHARACTER is/was/seemed ADJECTIVE"
            rf'\b{char_escaped}\s+(?:is|was|seemed?|appeared?|looked?|became?)\s+(\w+)',
            
            # "ADJECTIVE CHARACTER"
            rf'(\w+)\s+{char_escaped}',
            
            # "CHARACTER, who is/was ADJECTIVE"
            rf'{char_escaped},?\s+who\s+(?:is|was)\s+(\w+)',
            
            # "the ADJECTIVE CHARACTER"
            rf'the\s+(\w+)\s+{char_escaped}',
            
            # "CHARACTER looked/seemed/appeared ADJECTIVE"
            rf'{char_escaped}\s+(?:looked|seemed|appeared)\s+(\w+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, sent_lower)
            for match in matches:
                potential_trait = match if isinstance(match, str) else match[-1]
                
                # Verify it's an adjective
                doc = self.nlp(potential_trait)
                if len(doc) > 0 and doc[0].pos_ == 'ADJ':
                    traits.append(potential_trait.lower())
        
        return traits
    
    def _possessive_descriptions(self, sentence, character_name):
        """
        Extract traits from possessive descriptions
        e.g., "Della's beautiful hair", "Jim's kind eyes"
        """
        traits = []
        char_lower = character_name.lower()
        char_escaped = re.escape(char_lower)
        
        # Pattern: "CHARACTER's ADJECTIVE NOUN"
        pattern = rf"{char_escaped}'?s?\s+(\w+)\s+\w+"
        matches = re.findall(pattern, sentence.lower())
        
        for match in matches:
            doc = self.nlp(match)
            if len(doc) > 0 and doc[0].pos_ == 'ADJ':
                traits.append(match.lower())
        
        return traits
    
    def _action_based_traits(self, doc, character_name):
        """
        Infer traits from character actions
        """
        traits = []
        char_tokens = set(character_name.lower().split())
        
        # Find if character is subject of action verbs
        for token in doc:
            # Check if this token is the character
            if token.text.lower() in char_tokens:
                # Look at the verb this character is doing
                if token.head.pos_ == 'VERB':
                    verb = token.head.text.lower()
                    
                    # Check if verb indicates a trait
                    for category, verbs in self.trait_indicating_actions.items():
                        if verb in verbs:
                            traits.append(f"{category}_action")
        
        return traits
    
    def _descriptive_phrases(self, sentence, character_name):
        """
        Extract traits from descriptive phrases
        """
        traits = []
        char_lower = character_name.lower()
        
        # Common descriptive patterns
        descriptive_patterns = [
            r'poor\s+' + re.escape(char_lower),
            r'dear\s+' + re.escape(char_lower),
            r'young\s+' + re.escape(char_lower),
            r'old\s+' + re.escape(char_lower),
            r'beautiful\s+' + re.escape(char_lower),
            r'handsome\s+' + re.escape(char_lower),
        ]
        
        for pattern in descriptive_patterns:
            if re.search(pattern, sentence.lower()):
                # Extract the adjective
                adj = pattern.split(r'\\s\+')[0].replace(r'\\b', '')
                traits.append(adj)
        
        return traits
    
    def _analyze_sentiment(self, sentence):
        """
        Analisis sentimen kalimat
        """
        blob = TextBlob(sentence)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.3:
            return 'positive_context'
        elif polarity < -0.3:
            return 'negative_context'
        return None
    
    def _classify_traits(self, traits):
        """
        Klasifikasi traits ke kategori
        """
        classified = {
            'positive': [],
            'negative': [],
            'emotional': [],
            'physical': [],
            'behavioral': [],
            'other': []
        }
        
        for trait in traits:
            categorized = False
            
            # Check each category
            for category, keywords in self.trait_keywords.items():
                if trait in keywords:
                    classified[category].append(trait)
                    categorized = True
                    break
            
            # Check action-based traits
            if '_action' in trait or '_context' in trait:
                category = trait.split('_')[0]
                if category in classified:
                    classified[category].append(trait)
                    categorized = True
            
            if not categorized:
                classified['other'].append(trait)
        
        return classified