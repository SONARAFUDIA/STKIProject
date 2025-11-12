import spacy
from textblob import TextBlob
from collections import Counter
import re

class TraitExtractor:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_lg')
        
        # Daftar kata sifat watak (bisa diperluas)
        self.trait_keywords = {
            'positive': [
                'kind', 'brave', 'honest', 'loyal', 'generous', 'wise', 
                'gentle', 'patient', 'loving', 'caring', 'compassionate',
                'noble', 'heroic', 'virtuous', 'faithful', 'trustworthy'
            ],
            'negative': [
                'cruel', 'evil', 'dishonest', 'selfish', 'greedy', 'foolish',
                'harsh', 'impatient', 'hateful', 'wicked', 'mean', 'brutal',
                'villainous', 'treacherous', 'malicious', 'suspicious'
            ],
            'emotional': [
                'sad', 'happy', 'angry', 'fearful', 'anxious', 'nervous',
                'excited', 'depressed', 'joyful', 'melancholy', 'passionate'
            ],
            'behavioral': [
                'aggressive', 'passive', 'cautious', 'reckless', 'calm',
                'violent', 'peaceful', 'active', 'lazy', 'diligent'
            ]
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
            
            # Rule 1: Cari adjektiva yang dekat dengan nama karakter
            traits_found = self._find_adjacent_adjectives(doc, character_name)
            
            # Rule 2: Pattern "CHARACTER is/was/seems ADJECTIVE"
            pattern_traits = self._pattern_matching(sentence, character_name)
            
            # Rule 3: Sentiment analysis untuk konteks emosional
            sentiment_trait = self._analyze_sentiment(sentence)
            
            combined_traits = traits_found + pattern_traits
            if sentiment_trait:
                combined_traits.append(sentiment_trait)
            
            if combined_traits:
                all_traits.extend(combined_traits)
                trait_sentences.append({
                    'sentence': sentence,
                    'traits': combined_traits
                })
        
        # Agregasi dan klasifikasi
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
        Mencari adjektiva di sekitar nama karakter
        """
        traits = []
        char_tokens = character_name.lower().split()
        
        for i, token in enumerate(doc):
            # Cek apakah token adalah bagian dari nama karakter
            if token.text.lower() in char_tokens:
                # Cek 3 token sebelum dan sesudah
                window_start = max(0, i - 3)
                window_end = min(len(doc), i + 4)
                
                for j in range(window_start, window_end):
                    if doc[j].pos_ == 'ADJ':
                        traits.append(doc[j].text.lower())
        
        return traits
    
    def _pattern_matching(self, sentence, character_name):
        """
        Pattern matching untuk struktur kalimat tertentu
        """
        traits = []
        
        # Pattern: "CHARACTER is/was/seems ADJECTIVE"
        patterns = [
            rf'{character_name}\s+(is|was|seems|appeared|looked)\s+(\w+)',
            rf'(\w+)\s+{character_name}',  # ADJECTIVE CHARACTER
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, sentence, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    # Cek apakah kata adalah adjektiva
                    potential_trait = match[-1] if len(match) > 1 else match[0]
                    doc = self.nlp(potential_trait)
                    if doc[0].pos_ == 'ADJ':
                        traits.append(potential_trait.lower())
        
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
            'behavioral': [],
            'other': []
        }
        
        for trait in traits:
            categorized = False
            for category, keywords in self.trait_keywords.items():
                if trait in keywords:
                    classified[category].append(trait)
                    categorized = True
                    break
            
            if not categorized:
                classified['other'].append(trait)
        
        return classified