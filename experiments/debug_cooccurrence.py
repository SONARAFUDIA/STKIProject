import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import TextPreprocessor
from src.ner_extraction import CharacterExtractor
import re

def debug_character_cooccurrence():
    """
    Debug kenapa co-occurrence count rendah
    """
    print("="*70)
    print("DEBUG: Character Co-occurrence")
    print("="*70)
    
    filepath = os.path.join(PROJECT_ROOT, 'data/raw/the_gift_of_magi.txt')
    
    preprocessor = TextPreprocessor()
    char_extractor = CharacterExtractor()
    
    preprocessed = preprocessor.preprocess_document(filepath)
    char_extraction = char_extractor.extract_characters(
        preprocessed['cleaned_text'],
        preprocessed['sentences'],
        min_mentions=2
    )
    
    # Manual check
    della_jim_together = 0
    della_sentences = []
    jim_sentences = []
    both_sentences = []
    
    for i, sentence in enumerate(preprocessed['sentences']):
        sent_lower = sentence.lower()
        
        has_della = bool(re.search(r'\bdella\b', sent_lower))
        has_jim = bool(re.search(r'\bjim\b', sent_lower))
        
        if has_della:
            della_sentences.append(i)
        if has_jim:
            jim_sentences.append(i)
        if has_della and has_jim:
            della_jim_together += 1
            both_sentences.append((i, sentence))
    
    print(f"\n📊 Manual Count:")
    print(f"  Sentences with 'Della': {len(della_sentences)}")
    print(f"  Sentences with 'Jim': {len(jim_sentences)}")
    print(f"  Sentences with BOTH: {della_jim_together}")
    
    print(f"\n📝 Sample sentences with both characters:")
    for i, (sent_id, sentence) in enumerate(both_sentences[:5], 1):
        print(f"\n  {i}. [Sentence {sent_id}]")
        print(f"     {sentence[:100]}...")
    
    # Check context detection
    print(f"\n🔍 Context Detection from CharacterExtractor:")
    della_contexts = char_extraction['characters_with_context'].get('Della', [])
    jim_contexts = char_extraction['characters_with_context'].get('Jim', [])
    
    print(f"  Della contexts: {len(della_contexts)}")
    print(f"  Jim contexts: {len(jim_contexts)}")
    
    # Find overlap
    della_sent_ids = set(ctx['sentence_id'] for ctx in della_contexts)
    jim_sent_ids = set(ctx['sentence_id'] for ctx in jim_contexts)
    overlap = della_sent_ids & jim_sent_ids
    
    print(f"  Overlapping contexts: {len(overlap)}")
    
    if len(overlap) < della_jim_together:
        print(f"\n⚠️  PROBLEM: Manual count ({della_jim_together}) > Context overlap ({len(overlap)})")
        print(f"  → Context detection is missing some occurrences!")

if __name__ == "__main__":
    debug_character_cooccurrence()