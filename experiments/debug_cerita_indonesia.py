"""
Debug script untuk cek kenapa cerita Indonesia tidak detect relasi
Run: python experiments/debug_cerita_indonesia.py
"""

import sys
import os
import re

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import TextPreprocessor
from src.ner_extraction import CharacterExtractor
from src.relation_extraction import RelationExtractor

def debug_cerita_indonesia(filename='senja_di_ujung_kios.txt'):
    """
    Debug kenapa cerita Indonesia tidak detect relasi dengan baik
    """
    print("="*70)
    print(f"DEBUG: {filename}")
    print("="*70)
    
    filepath = os.path.join(PROJECT_ROOT, f'data/raw/{filename}')
    
    if not os.path.exists(filepath):
        print(f"❌ File tidak ditemukan: {filepath}")
        return
    
    if os.path.getsize(filepath) == 0:
        print(f"❌ File kosong: {filepath}")
        return
    
    # Initialize
    preprocessor = TextPreprocessor()
    char_extractor = CharacterExtractor()
    rel_extractor = RelationExtractor()
    
    # Process
    print("\n[1] Preprocessing...")
    preprocessed = preprocessor.preprocess_document(filepath)
    print(f"  ✓ {preprocessed['sentence_count']} kalimat")
    
    print("\n[2] Ekstraksi tokoh...")
    char_results = char_extractor.extract_characters(
        preprocessed['cleaned_text'],
        preprocessed['sentences'],
        min_mentions=2
    )
    print(f"  ✓ Tokoh: {list(char_results['main_characters'].keys())}")
    
    # Manual check untuk kata kunci hubungan
    print("\n[3] Manual check untuk kata kunci hubungan...")
    
    # Read raw text
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read().lower()
    
    relationship_keywords = {
        # Keluarga
        'istri': text.count('istri'),
        'suami': text.count('suami'),
        'ibu': text.count('ibu'),
        'bapak': text.count('bapak'),
        'ayah': text.count('ayah'),
        'anak': text.count('anak'),
        'kakak': text.count('kakak'),
        'adik': text.count('adik'),
        
        # Profesional
        'pelanggan': text.count('pelanggan'),
        'atasan': text.count('atasan'),
        'rekan kerja': text.count('rekan kerja'),
        'bos': text.count('bos'),
        'karyawan': text.count('karyawan'),
        
        # Sosial
        'teman': text.count('teman'),
        'sahabat': text.count('sahabat'),
        'kenalan': text.count('kenalan'),
        
        # Possessive
        'nya': text.count('nya'),
        'mu': text.count('mu'),
        'ku': text.count('ku'),
    }
    
    print("  Kata kunci hubungan yang ditemukan:")
    for keyword, count in relationship_keywords.items():
        if count > 0:
            print(f"    '{keyword}': {count}x")
    
    # Check co-occurrence
    print("\n[4] Check kemunculan tokoh bersama...")
    
    if len(char_results['main_characters']) >= 2:
        chars = list(char_results['main_characters'].keys())
        together_sentences = []
        
        for i, sentence in enumerate(preprocessed['sentences']):
            sent_lower = sentence.lower()
            present_chars = []
            
            for char in chars:
                char_lower = char.lower()
                # Remove honorifics for checking
                char_base = char_lower.replace('pak ', '').replace('bu ', '').replace('mas ', '')
                
                if char_base in sent_lower or char_lower in sent_lower:
                    present_chars.append(char)
            
            if len(present_chars) >= 2:
                together_sentences.append((i, sentence, present_chars))
        
        print(f"  ✓ Kalimat dengan 2+ tokoh: {len(together_sentences)}")
        
        if together_sentences:
            print("\n  Sample kalimat (5 pertama):")
            for i, (sent_id, sentence, present) in enumerate(together_sentences[:5], 1):
                print(f"\n  {i}. [Kalimat {sent_id}] - Tokoh: {', '.join(present)}")
                print(f"     {sentence[:150]}...")
    else:
        print(f"  ⚠️  Hanya {len(char_results['main_characters'])} tokoh ditemukan, butuh min 2")
    
    # Try extraction
    print("\n[5] Jalankan ekstraksi relasi...")
    relations = rel_extractor.extract_relations(
        char_results['main_characters'],
        preprocessed['sentences']
    )
    
    print(f"  ✓ Pasangan co-occurrence: {len(relations['cooccurrence'])}")
    print(f"  ✓ Pasangan proximity: {len(relations.get('proximity_pairs', {}))}")
    print(f"  ✓ Relasi spesifik: {len(relations['specific_relations'])}")
    print(f"  ✓ Relasi possessive: {len(relations.get('possessive_relations', []))}")
    print(f"  ✓ Relasi merged: {len(relations['merged_relations'])}")
    
    # Check cooccurrence details
    if relations['cooccurrence']:
        print("\n[6] Detail co-occurrence:")
        for pair, data in relations['cooccurrence'].items():
            print(f"\n  Pasangan: {pair[0]} ↔ {pair[1]}")
            print(f"    Jumlah: {data['count']}")
            print(f"    Contoh konteks (3 pertama):")
            for ctx in data['contexts'][:3]:
                print(f"      - {ctx[:120]}...")
    
    # If no relations
    if len(relations['merged_relations']) == 0:
        print("\n[7] MASALAH: Tidak ada relasi ditemukan!")
        print("\n  Mengecek pattern yang tersedia...")
        
        print("\n  Pattern untuk 'keluarga':")
        for pattern in rel_extractor.specific_relation_patterns.get('parent-child', [])[:3]:
            print(f"    - {pattern}")
    else:
        print("\n[7] SUKSES: Relasi terdeteksi!")
        for rel in relations['merged_relations']:
            print(f"\n  {rel['character1']} ↔ {rel['character2']}")
            print(f"    Primary: {rel['primary_relation']}")
            print(f"    Semua: {rel['all_relations']}")
            print(f"    Confidence: {rel['confidence']:.2f}")
            print(f"    Strength: {rel['strength']:.2f}")
    
    print("\n" + "="*70)
    print("DEBUG SELESAI")
    print("="*70)

if __name__ == "__main__":
    # Test dengan beberapa file
    test_files = [
        'senja_di_ujung_kios.txt',
        'rapat_warung_yopi_yang_batal.txt',
        'aroma_kayu_cendana.txt',
    ]
    
    for filename in test_files:
        filepath = os.path.join(PROJECT_ROOT, f'data/raw/{filename}')
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            debug_cerita_indonesia(filename)
            print("\n" + "="*70 + "\n")
            break  # Test satu file dulu