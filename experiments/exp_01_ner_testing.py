import sys
import os

# Tambahkan root directory ke Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import TextPreprocessor
from src.ner_extraction import CharacterExtractor
import json

def test_ner_extraction():
    """
    Eksperimen untuk testing ekstraksi tokoh dengan NER - Versi Indonesia
    """
    print("="*60)
    print("EKSPERIMEN 1: TESTING NER EXTRACTION (BAHASA INDONESIA)")
    print("="*60)
    
    # Inisialisasi
    preprocessor = TextPreprocessor()
    extractor = CharacterExtractor()
    
    # File yang akan ditest - cerita bahasa Indonesia
    test_files = [
        'data/raw/senja_di_ujung_kios.txt',
        'data/raw/rapat_warung_yopi_yang_batal.txt',
        'data/raw/aroma_kayu_cendana.txt',
        'data/raw/asing_di_cermin_itu.txt',
        'data/raw/garis_putus-putus.txt',
    ]
    
    results = {}
    
    for filepath in test_files:
        if not os.path.exists(filepath):
            print(f"\n⚠️  File tidak ditemukan: {filepath}")
            continue
            
        if os.path.getsize(filepath) == 0:
            print(f"\n⚠️  File kosong, dilewati: {filepath}")
            continue
        
        print(f"\n📖 Memproses: {filepath}")
        print("-"*60)
        
        # Preprocessing
        preprocessed = preprocessor.preprocess_document(filepath)
        print(f"✓ Kalimat diekstrak: {preprocessed['sentence_count']}")
        
        # NER Extraction
        extraction = extractor.extract_characters(
            preprocessed['cleaned_text'],
            preprocessed['sentences']
        )
        
        # Statistics
        stats = extractor.get_character_statistics(extraction)
        
        print(f"✓ Tokoh ditemukan: {stats['total_characters']}")
        if stats['most_mentioned']:
            print(f"✓ Paling sering disebut: {stats['most_mentioned'][0]} ({stats['most_mentioned'][1]}x)")
        
        # Tampilkan detail
        print("\n📋 Tokoh Utama:")
        for char, count in sorted(extraction['main_characters'].items(), 
                                   key=lambda x: x[1], reverse=True):
            print(f"  - {char}: {count} sebutan")
        
        # Simpan hasil
        results[filepath] = {
            'extraction': extraction,
            'statistics': stats
        }
    
    # Save ke file
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'exp_01_ner_results.json')
    
    # Convert for JSON serialization
    json_results = {}
    for filepath, data in results.items():
        json_results[filepath] = {
            'main_characters': data['extraction']['main_characters'],
            'statistics': data['statistics'],
            'total_entities_found': len(data['extraction']['all_entities'])
        }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print(f"✅ Hasil tersimpan di: {output_file}")
    print("="*60)
    
    # Summary
    print("\n📊 RINGKASAN:")
    total_chars = sum(len(r['extraction']['main_characters']) for r in results.values())
    print(f"  Total file diproses: {len(results)}")
    print(f"  Total tokoh terdeteksi: {total_chars}")
    print(f"  Rata-rata tokoh per cerita: {total_chars/len(results):.1f}")
    
    return results

if __name__ == "__main__":
    test_ner_extraction()