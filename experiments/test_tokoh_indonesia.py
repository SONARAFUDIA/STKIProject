import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import TextPreprocessor
from src.ner_extraction import CharacterExtractor

def test_tokoh_detection(filename='senja_di_ujung_kios.txt'):
    """
    Test khusus untuk masalah deteksi tokoh Indonesia
    """
    print("="*70)
    print(f"TESTING: Deteksi Tokoh - {filename}")
    print("="*70)
    
    filepath = os.path.join(PROJECT_ROOT, f'data/raw/{filename}')
    
    if not os.path.exists(filepath):
        print(f"❌ File tidak ditemukan: {filepath}")
        return
    
    if os.path.getsize(filepath) == 0:
        print(f"❌ File kosong: {filepath}")
        return
    
    # Read and check raw text
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    
    # Manual count untuk tokoh yang diharapkan
    # Untuk "Senja di Ujung Kios" - Pak Suroto sebagai tokoh utama
    print("\n📊 Manual Count dalam Raw Text:")
    
    # Define expected characters based on the story
    if 'senja_di_ujung_kios' in filename:
        expected_chars = {
            'Pak Suroto': ['Pak Suroto', 'Suroto'],
            'wanita': ['wanita muda', 'wanita itu', 'wanita'],
        }
    elif 'rapat_warung' in filename:
        expected_chars = {
            'Leo': ['Leo'],
            'Ardi': ['Ardi'],
            'Rina': ['Rina'],
            'Guntur': ['Guntur'],
            'Pak Hasan': ['Pak Hasan', 'Hasan'],
        }
    elif 'aroma_kayu' in filename:
        expected_chars = {
            'Elara': ['Elara'],
            'Sandi': ['Sandi'],
            'Ibu': ['Ibu', 'ibunya'],
        }
    elif 'asing_di_cermin' in filename:
        expected_chars = {
            'Narator (Aku)': [' aku ', ' Aku '],  # First person
            'Bima': ['Bima'],
        }
    elif 'garis_putus' in filename:
        expected_chars = {
            'Aria': ['Aria', 'ARIA'],
        }
    else:
        expected_chars = {}
    
    for char_name, variants in expected_chars.items():
        total_count = 0
        for variant in variants:
            count = raw_text.count(variant)
            if count > 0:
                print(f"  '{variant}': {count}x")
            total_count += count
        print(f"  → Total '{char_name}': {total_count}x")
    
    # Process with system
    preprocessor = TextPreprocessor()
    extractor = CharacterExtractor()
    
    print("\n" + "="*70)
    preprocessed = preprocessor.preprocess_document(filepath)
    extraction = extractor.extract_characters(
        preprocessed['cleaned_text'],
        preprocessed['sentences'],
        min_mentions=2
    )
    print("="*70)
    
    print("\n🤖 Hasil Deteksi NER:")
    print(f"  Total entitas ditemukan: {len(extraction['all_entities'])}")
    print(f"  Entitas unik: {len(set(extraction['all_entities']))}")
    
    print("\n📋 Raw Frequency (setelah normalisasi):")
    for name, count in sorted(extraction['raw_frequency'].items(), 
                              key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {name}: {count}")
    
    print("\n✨ Filtered Frequency (setelah blacklist):")
    for name, count in sorted(extraction['filtered_frequency'].items(), 
                              key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {name}: {count}")
    
    print("\n🎭 Tokoh Utama (final, min_mentions=2):")
    if extraction['main_characters']:
        for name, count in sorted(extraction['main_characters'].items(), 
                                  key=lambda x: x[1], reverse=True):
            context_count = len(extraction['characters_with_context'].get(name, []))
            print(f"  {name}: {count} sebutan, {context_count} konteks")
    else:
        print("  ⚠️  Tidak ada tokoh utama terdeteksi!")
    
    # Validation
    print("\n✅ Validasi:")
    if expected_chars and extraction['main_characters']:
        detected_names = set(extraction['main_characters'].keys())
        expected_names = set(expected_chars.keys())
        
        # Check overlap (case insensitive)
        detected_lower = {n.lower() for n in detected_names}
        
        correctly_detected = []
        for expected in expected_names:
            # Check if any variant is detected
            for detected in detected_names:
                if expected.lower() in detected.lower() or detected.lower() in expected.lower():
                    correctly_detected.append(expected)
                    break
        
        print(f"  Tokoh yang diharapkan: {len(expected_names)}")
        print(f"  Tokoh terdeteksi dengan benar: {len(correctly_detected)}")
        print(f"  Akurasi: {len(correctly_detected)/len(expected_names)*100:.1f}%")
        
        if correctly_detected:
            print(f"\n  ✓ Terdeteksi: {', '.join(correctly_detected)}")
        
        missing = set(expected_names) - set(correctly_detected)
        if missing:
            print(f"  ✗ Tidak terdeteksi: {', '.join(missing)}")

def test_all_stories():
    """
    Test semua cerita yang tersedia
    """
    print("\n" + "="*70)
    print("TESTING SEMUA CERITA")
    print("="*70)
    
    test_files = [
        'senja_di_ujung_kios.txt',
        'rapat_warung_yopi_yang_batal.txt',
        'aroma_kayu_cendana.txt',
        'asing_di_cermin_itu.txt',
        'garis_putus-putus.txt',
    ]
    
    for filename in test_files:
        filepath = os.path.join(PROJECT_ROOT, f'data/raw/{filename}')
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            test_tokoh_detection(filename)
            print("\n")

if __name__ == "__main__":
    # Test satu file atau semua
    import sys
    
    if len(sys.argv) > 1:
        test_tokoh_detection(sys.argv[1])
    else:
        test_all_stories()