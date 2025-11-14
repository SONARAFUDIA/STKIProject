import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import TextPreprocessor
from src.ner_extraction import CharacterExtractor
import re

def debug_character_cooccurrence(filename='rapat_warung_yopi_yang_batal.txt'):
    """
    Debug kenapa co-occurrence count rendah - Versi Indonesia
    """
    print("="*70)
    print(f"DEBUG: Character Co-occurrence - {filename}")
    print("="*70)
    
    filepath = os.path.join(PROJECT_ROOT, f'data/raw/{filename}')
    
    if not os.path.exists(filepath):
        print(f"❌ File tidak ditemukan: {filepath}")
        return
    
    if os.path.getsize(filepath) == 0:
        print(f"❌ File kosong: {filepath}")
        return
    
    preprocessor = TextPreprocessor()
    char_extractor = CharacterExtractor()
    
    preprocessed = preprocessor.preprocess_document(filepath)
    char_extraction = char_extractor.extract_characters(
        preprocessed['cleaned_text'],
        preprocessed['sentences'],
        min_mentions=2
    )
    
    characters = list(char_extraction['main_characters'].keys())
    
    if len(characters) < 2:
        print(f"⚠️  Hanya {len(characters)} tokoh ditemukan. Butuh minimal 2 untuk co-occurrence.")
        return
    
    print(f"\n📊 Tokoh yang Dianalisis: {', '.join(characters)}")
    
    # Manual check untuk setiap pasangan
    print(f"\n🔍 Manual Co-occurrence Check:")
    
    for i in range(len(characters)):
        for j in range(i + 1, len(characters)):
            char1 = characters[i]
            char2 = characters[j]
            
            print(f"\n  Pasangan: {char1} ↔ {char2}")
            print(f"  {'-'*60}")
            
            # Count manual
            together_count = 0
            char1_sentences = []
            char2_sentences = []
            both_sentences = []
            
            for idx, sentence in enumerate(preprocessed['sentences']):
                sent_lower = sentence.lower()
                
                # Prepare search patterns
                char1_lower = char1.lower()
                char2_lower = char2.lower()
                
                # Remove honorifics untuk search
                char1_base = char1_lower.replace('pak ', '').replace('bu ', '').replace('mas ', '').replace('mbak ', '')
                char2_base = char2_lower.replace('pak ', '').replace('bu ', '').replace('mas ', '').replace('mbak ', '')
                
                # Check presence
                has_char1 = bool(re.search(r'\b' + re.escape(char1_base) + r'\b', sent_lower))
                has_char2 = bool(re.search(r'\b' + re.escape(char2_base) + r'\b', sent_lower))
                
                if has_char1:
                    char1_sentences.append(idx)
                if has_char2:
                    char2_sentences.append(idx)
                if has_char1 and has_char2:
                    together_count += 1
                    both_sentences.append((idx, sentence))
            
            print(f"    Kalimat dengan '{char1}': {len(char1_sentences)}")
            print(f"    Kalimat dengan '{char2}': {len(char2_sentences)}")
            print(f"    Kalimat dengan KEDUANYA: {together_count}")
            
            if both_sentences:
                print(f"\n    Sample kalimat dengan kedua tokoh (max 3):")
                for k, (sent_id, sentence) in enumerate(both_sentences[:3], 1):
                    print(f"\n      {k}. [Kalimat {sent_id}]")
                    print(f"         {sentence[:120]}...")
    
    # Check context detection
    print(f"\n🔍 Context Detection dari CharacterExtractor:")
    for char, contexts in char_extraction['characters_with_context'].items():
        print(f"\n  {char}: {len(contexts)} konteks")
        if contexts:
            print(f"    Sample konteks pertama:")
            print(f"      Sentence ID: {contexts[0]['sentence_id']}")
            print(f"      Text: {contexts[0]['sentence'][:100]}...")
    
    # Find overlap
    print(f"\n📊 Overlap Analysis:")
    if len(characters) >= 2:
        char1 = characters[0]
        char2 = characters[1]
        
        char1_contexts = char_extraction['characters_with_context'].get(char1, [])
        char2_contexts = char_extraction['characters_with_context'].get(char2, [])
        
        char1_sent_ids = set(ctx['sentence_id'] for ctx in char1_contexts)
        char2_sent_ids = set(ctx['sentence_id'] for ctx in char2_contexts)
        overlap = char1_sent_ids & char2_sent_ids
        
        print(f"  {char1} sentence IDs: {len(char1_sent_ids)}")
        print(f"  {char2} sentence IDs: {len(char2_sent_ids)}")
        print(f"  Overlapping contexts: {len(overlap)}")
        
        if overlap:
            print(f"  Overlapping sentence IDs: {sorted(list(overlap))[:10]}")

def test_multiple_stories():
    """
    Test beberapa cerita
    """
    test_files = [
        'rapat_warung_yopi_yang_batal.txt',  # Multiple characters
        'aroma_kayu_cendana.txt',             # Elara & Sandi
        'senja_di_ujung_kios.txt',            # Pak Suroto & wanita
    ]
    
    for filename in test_files:
        filepath = os.path.join(PROJECT_ROOT, f'data/raw/{filename}')
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            debug_character_cooccurrence(filename)
            print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        debug_character_cooccurrence(sys.argv[1])
    else:
        test_multiple_stories()