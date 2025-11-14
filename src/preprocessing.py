import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

class TextPreprocessor:
    def __init__(self):
        # Download NLTK resources jika belum ada
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab')
        
        # Indonesian stopwords - Manual definition (lebih lengkap dari NLTK)
        self.stop_words = set([
            # Kata sambung
            'dan', 'atau', 'tetapi', 'tapi', 'namun', 'sedangkan', 'padahal',
            'bahwa', 'karena', 'sebab', 'jika', 'kalau', 'bila', 'saat', 'ketika',
            'untuk', 'bagi', 'kepada', 'pada', 'di', 'ke', 'dari', 'oleh', 'dengan',
            
            # Kata ganti
            'saya', 'aku', 'kamu', 'anda', 'dia', 'ia', 'mereka', 'kami', 'kita',
            'ini', 'itu', 'tersebut',
            
            # Kata kerja bantu
            'adalah', 'ialah', 'merupakan', 'yaitu', 'yakni',
            'ada', 'tidak', 'bukan', 'belum', 'sudah', 'telah', 'akan', 'sedang',
            
            # Kata keterangan
            'sangat', 'amat', 'sekali', 'lebih', 'paling', 'terlalu',
            'masih', 'lagi', 'juga', 'pun', 'saja', 'hanya', 'cuma',
            'selalu', 'sering', 'kadang', 'jarang',
            
            # Kata tanya
            'apa', 'siapa', 'kapan', 'dimana', 'kemana', 'mengapa', 'kenapa', 'bagaimana',
            
            # Preposisi & konjungsi
            'dalam', 'luar', 'atas', 'bawah', 'depan', 'belakang', 'samping',
            'sebelum', 'sesudah', 'setelah', 'sejak', 'hingga', 'sampai',
            'antara', 'antar', 'diantara',
            
            # Kata umum lainnya
            'yang', 'sebuah', 'suatu', 'para', 'semua', 'setiap', 'masing', 'tiap',
            'pula', 'lah', 'kah', 'nya',
            
            # Angka & waktu
            'satu', 'dua', 'tiga', 'empat', 'lima', 'enam', 'tujuh', 'delapan', 
            'sembilan', 'sepuluh', 'ratus', 'ribu', 'juta',
            'hari', 'minggu', 'bulan', 'tahun', 'jam', 'menit', 'detik',
            
            # Kata sifat umum (bukan watak)
            'besar', 'kecil', 'tinggi', 'rendah', 'panjang', 'pendek',
            'banyak', 'sedikit', 'baru', 'lama',
        ])
        
        # Try to use Sastrawi if available
        try:
            from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
            factory = StopWordRemoverFactory()
            self.stop_words.update(factory.get_stop_words())
        except ImportError:
            print("⚠️  Sastrawi tidak terinstall. Menggunakan stopwords manual.")
    
    def clean_text(self, text):
        """
        Membersihkan teks dari karakter yang tidak diperlukan
        """
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters tapi pertahankan punctuation untuk analisis
        # Juga pertahankan tanda hubung dalam kata (mis: "putus-putus")
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\'\"\-]', '', text)
        
        # Remove multiple punctuation
        text = re.sub(r'([.!?]){2,}', r'\1', text)
        
        # Remove character encoding artifacts (em dash, etc)
        text = text.replace('â€"', '-')
        text = text.replace('â€™', "'")
        
        return text.strip()
    
    def segment_sentences(self, text):
        """
        Memecah teks menjadi kalimat-kalimat
        Menggunakan sent_tokenize yang support bahasa Indonesia
        """
        # NLTK sent_tokenize works reasonably well for Indonesian
        sentences = sent_tokenize(text, language='english')  # No Indonesian model, but works
        
        # Filter kalimat yang terlalu pendek (< 10 karakter)
        sentences = [sent.strip() for sent in sentences if len(sent.strip()) > 10]
        
        # Additional cleanup: split on dialog patterns if needed
        cleaned_sentences = []
        for sent in sentences:
            # Split kalimat yang mengandung dialog dengan tanda kutip
            if '"' in sent and sent.count('"') >= 2:
                # Keep as is, dialog is important for character analysis
                cleaned_sentences.append(sent)
            else:
                cleaned_sentences.append(sent)
        
        return cleaned_sentences
    
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