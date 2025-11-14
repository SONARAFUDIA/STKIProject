"""
Debug script untuk cek kenapa Gift of Magi tidak detect relasi
Save sebagai: experiments/debug_gift_of_magi.py
Run: python experiments/debug_gift_of_magi.py
"""

import sys
import os
import re

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import TextPreprocessor
from src.ner_extraction import CharacterExtractor
from src.relation_extraction import RelationExtractor

def debug_gift_of_magi():
    """
    Debug kenapa Gift of Magi tidak detect relasi Della-Jim
    """
    print("="*70)
    print("DEBUG: Gift of Magi Relation Detection")
    print("="*70)
    
    filepath = os.path.join(PROJECT_ROOT, 'data/raw/the_gift_of_magi.txt')
    
    # Initialize
    preprocessor = TextPreprocessor()
    char_extractor = CharacterExtractor()
    rel_extractor = RelationExtractor()
    
    # Process
    print("\n[1] Preprocessing...")
    preprocessed = preprocessor.preprocess_document(filepath)
    print(f"  ✓ {preprocessed['sentence_count']} sentences")
    
    print("\n[2] Extract characters...")
    char_results = char_extractor.extract_characters(
        preprocessed['cleaned_text'],
        preprocessed['sentences'],
        min_mentions=2
    )
    print(f"  ✓ Characters: {list(char_results['main_characters'].keys())}")
    
    # Manual check for relationship keywords
    print("\n[3] Manual check for relationship keywords...")
    
    # Read raw text
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read().lower()
    
    relationship_keywords = {
        'husband': text.count('husband'),
        'wife': text.count('wife'),
        'married': text.count('married'),
        'his wife': text.count('his wife'),
        'her husband': text.count('her husband'),
        'love': text.count('love'),
        'darling': text.count('darling'),
        'treasure': text.count('treasure'),
        'precious': text.count('precious')
    }
    
    print("  Relationship keyword counts:")
    for keyword, count in relationship_keywords.items():
        if count > 0:
            print(f"    '{keyword}': {count}x")
    
    # Check co-occurrence in sentences
    print("\n[4] Check Della-Jim co-occurrence...")
    
    della_jim_sentences = []
    for i, sentence in enumerate(preprocessed['sentences']):
        sent_lower = sentence.lower()
        has_della = 'della' in sent_lower or 'dell' in sent_lower
        has_jim = 'jim' in sent_lower
        
        if has_della and has_jim:
            della_jim_sentences.append((i, sentence))
    
    print(f"  ✓ Sentences with BOTH Della and Jim: {len(della_jim_sentences)}")
    
    if della_jim_sentences:
        print("\n  Sample sentences:")
        for i, (sent_id, sentence) in enumerate(della_jim_sentences[:5], 1):
            print(f"\n  {i}. [Sentence {sent_id}]")
            print(f"     {sentence[:150]}...")
            
            # Check for relationship keywords in this sentence
            sent_lower = sentence.lower()
            found_keywords = []
            for keyword in relationship_keywords.keys():
                if keyword in sent_lower and relationship_keywords[keyword] > 0:
                    found_keywords.append(keyword)
            
            if found_keywords:
                print(f"     Keywords found: {', '.join(found_keywords)}")
    
    # Try extraction
    print("\n[5] Run relation extraction...")
    relations = rel_extractor.extract_relations(
        char_results['main_characters'],
        preprocessed['sentences']
    )
    
    print(f"  ✓ Co-occurrence pairs: {len(relations['cooccurrence'])}")
    print(f"  ✓ Proximity pairs: {len(relations.get('proximity_pairs', {}))}")
    print(f"  ✓ Specific relations: {len(relations['specific_relations'])}")
    print(f"  ✓ Possessive relations: {len(relations.get('possessive_relations', []))}")
    print(f"  ✓ Merged relations: {len(relations['merged_relations'])}")
    
    # Check cooccurrence details
    if relations['cooccurrence']:
        print("\n[6] Co-occurrence details:")
        for pair, data in relations['cooccurrence'].items():
            print(f"\n  Pair: {pair[0]} ↔ {pair[1]}")
            print(f"    Count: {data['count']}")
            print(f"    Sample contexts (first 3):")
            for ctx in data['contexts'][:3]:
                print(f"      - {ctx[:120]}...")
    
    # If still no relations detected, check patterns
    if len(relations['merged_relations']) == 0:
        print("\n[7] PROBLEM DETECTED: No relations found!")
        print("\n  Checking why patterns didn't match...")
        
        # Check if patterns exist
        print("\n  Available patterns for 'married-couple':")
        for pattern in rel_extractor.specific_relation_patterns.get('married-couple', []):
            print(f"    - {pattern}")
            
            # Try manual match on sample sentence
            if della_jim_sentences:
                sample_sent = della_jim_sentences[0][1].lower()
                if re.search(pattern, sample_sent):
                    print(f"      ✓ MATCHES: '{sample_sent[:80]}...'")
                else:
                    print(f"      ✗ No match")
    
    else:
        print("\n[7] SUCCESS: Relations detected!")
        for rel in relations['merged_relations']:
            print(f"\n  {rel['character1']} ↔ {rel['character2']}")
            print(f"    Primary: {rel['primary_relation']}")
            print(f"    All: {rel['all_relations']}")
            print(f"    Confidence: {rel['confidence']:.2f}")
            print(f"    Strength: {rel['strength']:.2f}")
    
    print("\n" + "="*70)
    print("DEBUG COMPLETE")
    print("="*70)

if __name__ == "__main__":
    debug_gift_of_magi()