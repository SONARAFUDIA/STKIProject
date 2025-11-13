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
        Ekstraksi watak karakter dari konteks - ENHANCED for narrator
        """
        all_traits = []
        trait_sentences = []
        
        # SPECIAL: For "Narrator (I)", extract from first-person perspective
        is_narrator = character_name.lower() == "narrator (i)"
        
        for context in character_contexts:
            sentence = context['sentence']
            doc = self.nlp(sentence)
            
            if is_narrator:
                # For narrator, extract traits from "I am/was/feel ADJECTIVE"
                traits_found = self._extract_narrator_traits(doc, sentence)
            else:
                # Regular character trait extraction
                # Method 1: Adjacent adjectives
                traits_found = self._find_adjacent_adjectives(doc, character_name)
                
                # Method 2: Pattern matching
                pattern_traits = self._pattern_matching(sentence, character_name)
                traits_found.extend(pattern_traits)
                
                # Method 3: Possessive descriptions
                possessive_traits = self._possessive_descriptions(sentence, character_name)
                traits_found.extend(possessive_traits)
                
                # Method 4: Action-based trait inference
                action_traits = self._action_based_traits(doc, character_name)
                traits_found.extend(action_traits)
                
                # Method 5: Descriptive phrases
                descriptive_traits = self._descriptive_phrases(sentence, character_name)
                traits_found.extend(descriptive_traits)
            
            # Method 6: Sentiment analysis (untuk semua)
            sentiment_trait = self._analyze_sentiment(sentence)
            
            combined_traits = traits_found
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

    def _extract_narrator_traits(self, doc, sentence):
        """
        Extract traits specifically for first-person narrator
        """
        traits = []
        sent_lower = sentence.lower()
        
        # Pattern 1: "I am/was/feel ADJECTIVE"
        patterns = [
            r'\bi\s+(?:am|was|feel|felt|became)\s+(\w+)',
            r'\bi\s+(?:seem|seemed|appear|appeared|look|looked)\s+(\w+)',
            r'\bi\s+(?:get|got|grow|grew)\s+(\w+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, sent_lower)
            for match in matches:
                # Verify it's an adjective
                test_doc = self.nlp(match)
                if len(test_doc) > 0 and test_doc[0].pos_ == 'ADJ':
                    traits.append(match.lower())
        
        # Pattern 2: "I'm so/very/quite ADJECTIVE"
        intensity_pattern = r'\bi\'?m\s+(?:so|very|quite|too|really)\s+(\w+)'
        matches = re.findall(intensity_pattern, sent_lower)
        for match in matches:
            test_doc = self.nlp(match)
            if len(test_doc) > 0 and test_doc[0].pos_ == 'ADJ':
                traits.append(match.lower())
        
        # Pattern 3: Find adjectives near "I" in the parse tree
        for token in doc:
            if token.text.lower() == 'i':
                # Look at nearby adjectives (within 5 tokens)
                start = max(0, token.i - 5)
                end = min(len(doc), token.i + 6)
                
                for j in range(start, end):
                    if doc[j].pos_ == 'ADJ':
                        traits.append(doc[j].text.lower())
        
        return traits
    
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