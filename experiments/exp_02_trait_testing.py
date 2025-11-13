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
    Eksperimen untuk testing ekstraksi watak
    """
    print("="*60)
    print("EKSPERIMEN 2: TESTING TRAIT EXTRACTION")
    print("="*60)
    
    # Inisialisasi
    preprocessor = TextPreprocessor()
    char_extractor = CharacterExtractor()
    trait_extractor = TraitExtractor()
    
    # Test dengan multiple files untuk variasi
    test_files = [
        'owl_creek_bridge.txt',
        'the_gift_of_magi.txt',
        'the_tell_tale_heart.txt',
        'the_yellow_wallpaper.txt',
    ]
    
    all_results = {}
    
    for filename in test_files:
        filepath = os.path.join(PROJECT_ROOT, 'data/raw', filename)
        
        if not os.path.exists(filepath):
            print(f"\n⚠️  File not found: {filepath}")
            continue
        
        print(f"\n{'='*60}")
        print(f"📖 Processing: {filename}")
        print(f"{'='*60}")
        
        try:
            # Step 1: Preprocessing
            print("\n[1/3] Preprocessing...")
            preprocessed = preprocessor.preprocess_document(filepath)
            print(f"  ✓ {preprocessed['sentence_count']} sentences extracted")
            
            # Step 2: Extract characters
            print("\n[2/3] Extracting characters...")
            char_extraction = char_extractor.extract_characters(
                preprocessed['cleaned_text'],
                preprocessed['sentences'],
                min_mentions=2
            )
            
            if len(char_extraction['main_characters']) == 0:
                print("  ⚠️  No characters found in this story. Skipping trait extraction.")
                continue
            
            print(f"  ✓ {len(char_extraction['main_characters'])} main characters found")
            print(f"  Characters: {', '.join(char_extraction['main_characters'].keys())}")
            
            # Step 3: Extract traits untuk setiap karakter
            print("\n[3/3] Extracting character traits...")
            trait_results = {}
            
            for character, contexts in char_extraction['characters_with_context'].items():
                print(f"\n🎭 Analyzing character: {character}")
                print("-"*60)
                
                if not contexts:
                    print(f"  ⚠️  No contexts found for {character}")
                    continue
                
                traits = trait_extractor.extract_traits(character, contexts)
                
                print(f"  ✓ Total trait mentions: {len(traits['raw_traits'])}")
                print(f"  ✓ Unique traits: {len(traits['trait_frequency'])}")
                
                if traits['trait_frequency']:
                    print(f"\n  📊 Top 5 Traits:")
                    top_traits = sorted(traits['trait_frequency'].items(), 
                                       key=lambda x: x[1], reverse=True)[:5]
                    for trait, count in top_traits:
                        print(f"    - {trait}: {count}x")
                
                print(f"\n  🏷️  Classified Traits:")
                for category, trait_list in traits['classified_traits'].items():
                    if trait_list:
                        unique_traits = list(set(trait_list))
                        if unique_traits:
                            print(f"    {category.capitalize()}: {', '.join(unique_traits[:5])}")
                
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
            print(f"\n❌ Error processing {filename}: {str(e)}")
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
        print(f"✅ Results saved to: {output_file}")
        print("="*60)
        
        # Summary
        print("\n📊 SUMMARY:")
        for filename, results in all_results.items():
            print(f"\n  {filename}:")
            for char, data in results.items():
                print(f"    - {char}: {data['unique_traits']} unique traits")
    else:
        print("\n⚠️  No results to save")
    
    return all_results

if __name__ == "__main__":
    test_trait_extraction()