import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import TextPreprocessor
from src.ner_extraction import CharacterExtractor

def test_della_detection():
    """
    Test khusus untuk masalah deteksi Della
    """
    print("="*70)
    print("TESTING: Della Detection Problem")
    print("="*70)
    
    filepath = os.path.join(PROJECT_ROOT, 'data/raw/the_gift_of_magi.txt')
    
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return
    
    # Read and check raw text
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    
    # Count manual
    print("\n📊 Manual Count in Raw Text:")
    della_count = raw_text.count('Della')
    della_possessive = raw_text.count("Della's")
    jim_count = raw_text.count('Jim')
    jim_possessive = raw_text.count("Jim's")
    jims_count = raw_text.count("Jims")  # plural form
    
    print(f"  'Della' appears: {della_count} times")
    print(f"  'Della's' appears: {della_possessive} times")
    print(f"  'Jim' appears: {jim_count} times")
    print(f"  'Jim's' appears: {jim_possessive} times")
    print(f"  'Jims' appears: {jims_count} times")
    print(f"  Total 'Della' variants: {della_count + della_possessive}")
    print(f"  Total 'Jim' variants: {jim_count + jim_possessive + jims_count}")
    
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
    
    print("\n🤖 NER Detection Results:")
    print(f"  Total entities found: {len(extraction['all_entities'])}")
    print(f"  Unique entities: {len(set(extraction['all_entities']))}")
    
    print("\n📋 Raw Frequency (after normalization):")
    for name, count in sorted(extraction['raw_frequency'].items(), 
                              key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {name}: {count}")
    
    print("\n✨ Filtered Frequency (after blacklist):")
    for name, count in sorted(extraction['filtered_frequency'].items(), 
                              key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {name}: {count}")
    
    print("\n🎭 Main Characters (final, min_mentions=2):")
    for name, count in sorted(extraction['main_characters'].items(), 
                              key=lambda x: x[1], reverse=True):
        context_count = len(extraction['characters_with_context'].get(name, []))
        print(f"  {name}: {count} mentions, {context_count} contexts")
    
    # Validation
    print("\n✅ Validation:")
    della_detected = extraction['main_characters'].get('Della', 0)
    jim_detected = extraction['main_characters'].get('Jim', 0)
    
    della_accuracy = (della_detected / (della_count + della_possessive)) * 100 if della_count > 0 else 0
    jim_accuracy = (jim_detected / (jim_count + jim_possessive + jims_count)) * 100 if jim_count > 0 else 0
    
    print(f"  Della detection: {della_detected}/{della_count + della_possessive} = {della_accuracy:.1f}% accuracy")
    print(f"  Jim detection: {jim_detected}/{jim_count + jim_possessive + jims_count} = {jim_accuracy:.1f}% accuracy")
    
    if della_accuracy >= 90 and jim_accuracy >= 90:
        print("\n🎉 SUCCESS: Character detection is accurate!")
    else:
        print("\n⚠️  WARNING: Character detection needs improvement")

if __name__ == "__main__":
    test_della_detection()