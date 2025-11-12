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
    
    # Test file - ganti ke cerita yang lebih cocok untuk trait extraction
    filepath = os.path.join(PROJECT_ROOT, 'data/raw/the_gift_of_magi.txt')
    
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        print(f"   Looking for: {filepath}")
        return None
    
    print(f"\n📖 Processing: {os.path.basename(filepath)}")
    
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
        
        print(f"  ✓ {len(char_extraction['main_characters'])} main characters found")
        print(f"  Characters: {', '.join(char_extraction['main_characters'].keys())}")
        
        # Step 3: Extract traits untuk setiap karakter
        print("\n[3/3] Extracting character traits...")
        trait_results = {}
        
        for character, contexts in char_extraction['characters_with_context'].items():
            print(f"\n🎭 Analyzing character: {character}")
            print("-"*60)
            
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
                    print(f"    {category.capitalize()}: {', '.join(unique_traits[:5])}")
            
            # Simplify for JSON serialization
            trait_results[character] = {
                'raw_traits': traits['raw_traits'],
                'trait_frequency': traits['trait_frequency'],
                'classified_traits': traits['classified_traits'],
                'total_traits': len(traits['raw_traits']),
                'unique_traits': len(traits['trait_frequency'])
            }
        
        # Save results
        output_dir = os.path.join(PROJECT_ROOT, 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, 'exp_02_trait_results.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(trait_results, f, indent=2, ensure_ascii=False)
        
        print("\n" + "="*60)
        print(f"✅ Results saved to: {output_file}")
        print("="*60)
        
        return trait_results
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_trait_extraction()