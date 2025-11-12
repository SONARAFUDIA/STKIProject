import sys
import os

# Tambahkan root directory ke Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import TextPreprocessor
from src.ner_extraction import CharacterExtractor
import json

def test_ner_extraction():
    """
    Eksperimen untuk testing ekstraksi tokoh dengan NER
    """
    print("="*60)
    print("EKSPERIMEN 1: TESTING NER EXTRACTION")
    print("="*60)
    
    # Inisialisasi
    preprocessor = TextPreprocessor()
    extractor = CharacterExtractor()
    
    # File yang akan ditest
    test_files = [
        # 'data/raw/the_tell_tale_heart.txt',
        'data/raw/the_gift_of_magi.txt'
    ]
    
    results = {}
    
    for filepath in test_files:
        print(f"\n📖 Processing: {filepath}")
        print("-"*60)
        
        # Preprocessing
        preprocessed = preprocessor.preprocess_document(filepath)
        print(f"✓ Sentences extracted: {preprocessed['sentence_count']}")
        
        # NER Extraction
        extraction = extractor.extract_characters(
            preprocessed['cleaned_text'],
            preprocessed['sentences']
        )
        
        # Statistics
        stats = extractor.get_character_statistics(extraction)
        
        print(f"✓ Characters found: {stats['total_characters']}")
        print(f"✓ Most mentioned: {stats['most_mentioned']}")
        
        # Tampilkan detail
        print("\n📋 Main Characters:")
        for char, count in extraction['main_characters'].items():
            print(f"  - {char}: {count} mentions")
        
        # Simpan hasil
        results[filepath] = {
            'extraction': extraction,
            'statistics': stats
        }
    
    # Save ke file
    with open('outputs/exp_01_ner_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n✅ Results saved to: outputs/exp_01_ner_results.json")
    return results

if __name__ == "__main__":
    test_ner_extraction()