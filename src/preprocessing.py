import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

class TextPreprocessor:
    def __init__(self):
        # Download NLTK resources jika belum ada
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """
        Membersihkan teks dari karakter yang tidak diperlukan
        """
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters tapi pertahankan punctuation untuk analisis
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\'\"\-]', '', text)
        
        # Remove multiple punctuation
        text = re.sub(r'([.!?]){2,}', r'\1', text)
        
        return text.strip()
    
    def segment_sentences(self, text):
        """
        Memecah teks menjadi kalimat-kalimat
        """
        sentences = sent_tokenize(text)
        return [sent.strip() for sent in sentences if len(sent.strip()) > 10]
    
    def tokenize_words(self, text):
        """
        Tokenisasi kata dengan preservasi konteks
        """
        return word_tokenize(text)
    
    def preprocess_document(self, filepath):
        """
        Pipeline preprocessing lengkap untuk satu dokumen
        """
        # Baca file
        with open(filepath, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        
        # Cleaning
        cleaned_text = self.clean_text(raw_text)
        
        # Segmentasi kalimat
        sentences = self.segment_sentences(cleaned_text)
        
        return {
            'raw_text': raw_text,
            'cleaned_text': cleaned_text,
            'sentences': sentences,
            'sentence_count': len(sentences)
        }