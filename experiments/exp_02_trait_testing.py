import sys
import os

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import TextPreprocessor
from src.ner_extraction import CharacterExtractor
from src.trait_extraction import TraitExtractor
from collections import Counter
import json

def test_trait_extraction():
    """
    Eksperimen untuk testing ekstraksi watak - Versi Indonesia
    """
    print("="*60)
    print("EKSPERIMEN 2: TESTING TRAIT EXTRACTION (BAHASA INDONESIA)")
    print("="*60)
    
    # Inisialisasi
    preprocessor = TextPreprocessor()
    char_extractor = CharacterExtractor()
    trait_extractor = TraitExtractor()
    
    # Test dengan cerita Indonesia
    test_files = [
        'senja_di_ujung_kios.txt',
        'rapat_warung_yopi_yang_batal.txt',
        'aroma_kayu_cendana.txt',
        'asing_di_cermin_itu.txt',
        'garis_putus-putus.txt',
    ]
    
    all_results = {}
    
    for filename in test_files:
        filepath = os.path.join(PROJECT_ROOT, 'data/raw', filename)
        
        if not os.path.exists(filepath):
            print(f"\n⚠️  File tidak ditemukan: {filepath}")
            continue
        
        if os.path.getsize(filepath) == 0:
            print(f"\n⚠️  File kosong, dilewati: {filename}")
            continue
        
        print(f"\n{'='*60}")
        print(f"📖 Memproses: {filename}")
        print(f"{'='*60}")
        
        try:
            # Step 1: Preprocessing
            print("\n[1/3] Preprocessing...")
            preprocessed = preprocessor.preprocess_document(filepath)
            print(f"  ✓ {preprocessed['sentence_count']} kalimat diekstrak")
            
            # Step 2: Extract characters
            print("\n[2/3] Ekstraksi tokoh...")
            char_extraction = char_extractor.extract_characters(
                preprocessed['cleaned_text'],
                preprocessed['sentences'],
                min_mentions=2
            )
            
            if len(char_extraction['main_characters']) == 0:
                print("  ⚠️  Tidak ada tokoh ditemukan dalam cerita ini. Melewati ekstraksi watak.")
                continue
            
            print(f"  ✓ {len(char_extraction['main_characters'])} tokoh utama ditemukan")
            print(f"  Tokoh: {', '.join(char_extraction['main_characters'].keys())}")
            
            # Step 3: Extract traits untuk setiap karakter
            print("\n[3/3] Ekstraksi watak tokoh...")
            trait_results = {}
            
            for character, contexts in char_extraction['characters_with_context'].items():
                print(f"\n🎭 Menganalisis tokoh: {character}")
                print("-"*60)
                
                if not contexts:
                    print(f"  ⚠️  Tidak ada konteks ditemukan untuk {character}")
                    continue
                
                traits = trait_extractor.extract_traits(character, contexts)
                
                print(f"  ✓ Total sebutan watak: {len(traits['raw_traits'])}")
                print(f"  ✓ Watak unik: {len(traits['trait_frequency'])}")
                
                if traits['trait_frequency']:
                    print(f"\n  📊 Top 5 Watak:")
                    top_traits = sorted(traits['trait_frequency'].items(), 
                                       key=lambda x: x[1], reverse=True)[:5]
                    for trait, count in top_traits:
                        print(f"    - {trait}: {count}x")
                
                print(f"\n  🏷️  Watak Terklasifikasi:")
                category_map = {
                    'positive': 'Positif',
                    'negative': 'Negatif',
                    'emotional': 'Emosional',
                    'physical': 'Fisik',
                    'behavioral': 'Perilaku',
                    'other': 'Lainnya'
                }
                
                for category, trait_list in traits['classified_traits'].items():
                    if trait_list:
                        unique_traits = list(set(trait_list))
                        if unique_traits:
                            category_indo = category_map.get(category, category.title())
                            print(f"    {category_indo}: {', '.join(unique_traits[:5])}")
                
                # Simplify for JSON serialization
                trait_results[character] = {
                    'raw_traits': traits['raw_traits'],
                    'trait_frequency': traits['trait_frequency'],
                    'classified_traits': traits['classified_traits'],
                    'total_traits': len(traits['raw_traits']),
                    'unique_traits': len(traits['trait_frequency'])
                }
            
            all_results[filename] = trait_results
            
        except Exception as e:
            print(f"\n❌ Kesalahan memproses {filename}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Save results
    if all_results:
        output_dir = os.path.join(PROJECT_ROOT, 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, 'exp_02_trait_results.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print("\n" + "="*60)
        print(f"✅ Hasil tersimpan di: {output_file}")
        print("="*60)
        
        # Summary
        print("\n📊 RINGKASAN:")
        for filename, results in all_results.items():
            print(f"\n  📖 {filename}:")
            for char, data in results.items():
                print(f"    - {char}: {data['unique_traits']} watak unik")
    else:
        print("\n⚠️  Tidak ada hasil untuk disimpan")
    
    return all_results

if __name__ == "__main__":
    test_trait_extraction()