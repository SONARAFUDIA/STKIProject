import sys
import os

# Tambahkan root directory ke Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import TextPreprocessor
from src.ner_extraction import CharacterExtractor
from src.trait_extraction import TraitExtractor
from src.relation_extraction import RelationExtractor
import json
import os
from datetime import datetime

def full_pipeline_analysis(filepath):
    """
    Pipeline lengkap untuk satu dokumen - Versi Indonesia
    """
    print(f"\n{'='*60}")
    print(f"MENGANALISIS: {os.path.basename(filepath)}")
    print(f"{'='*60}")
    
    # Inisialisasi semua modul
    preprocessor = TextPreprocessor()
    char_extractor = CharacterExtractor()
    trait_extractor = TraitExtractor()
    rel_extractor = RelationExtractor()
    
    # STEP 1: Preprocessing
    print("\n[1/4] Preprocessing...")
    preprocessed = preprocessor.preprocess_document(filepath)
    print(f"  ✓ {preprocessed['sentence_count']} kalimat diekstrak")
    
    # STEP 2: Character Extraction
    print("\n[2/4] Ekstraksi tokoh...")
    char_results = char_extractor.extract_characters(
        preprocessed['cleaned_text'],
        preprocessed['sentences']
    )
    char_stats = char_extractor.get_character_statistics(char_results)
    print(f"  ✓ {char_stats['total_characters']} tokoh utama ditemukan")
    
    # STEP 3: Trait Extraction
    print("\n[3/4] Ekstraksi watak tokoh...")
    trait_results = {}
    for character, contexts in char_results['characters_with_context'].items():
        traits = trait_extractor.extract_traits(character, contexts)
        trait_results[character] = traits
        print(f"  ✓ {character}: {len(traits['raw_traits'])} watak diekstrak")
    
    # STEP 4: Relation Extraction
    print("\n[4/4] Ekstraksi hubungan...")
    relation_results = rel_extractor.extract_relations(
        char_results['main_characters'],
        preprocessed['sentences']
    )
    print(f"  ✓ {len(relation_results['merged_relations'])} hubungan ditemukan")
    
    # Compile full report
    full_report = {
        'metadata': {
            'filename': os.path.basename(filepath),
            'processed_at': datetime.now().isoformat(),
            'sentence_count': preprocessed['sentence_count']
        },
        'characters': {
            'statistics': char_stats,
            'details': char_results['main_characters']
        },
        'traits': trait_results,
        'relations': {
            'summary': {
                'total_relations': len(relation_results['merged_relations']),
                'cooccurrence_pairs': len(relation_results['cooccurrence'])
            },
            'details': relation_results['merged_relations']
        },
        'graph': relation_results['relation_graph']
    }
    
    return full_report

def analyze_all_documents():
    """
    Analisis semua dokumen dalam folder data/raw - Versi Indonesia
    """
    print("\n" + "="*60)
    print("ANALISIS PIPELINE LENGKAP - SEMUA DOKUMEN")
    print("="*60)
    
    documents = [
        'data/raw/senja_di_ujung_kios.txt',
        'data/raw/rapat_warung_yopi_yang_batal.txt',
        'data/raw/aroma_kayu_cendana.txt',
        'data/raw/asing_di_cermin_itu.txt',
        'data/raw/garis_putus-putus.txt',
    ]
    
    all_results = {}
    
    for doc in documents:
        if not os.path.exists(doc):
            print(f"\n⚠️  File tidak ditemukan: {doc}")
            continue
        
        if os.path.getsize(doc) == 0:
            print(f"\n⚠️  File kosong, dilewati: {doc}")
            continue
            
        result = full_pipeline_analysis(doc)
        doc_name = os.path.basename(doc).replace('.txt', '')
        all_results[doc_name] = result
        
        # Save individual report
        output_dir = 'outputs/reports'
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = f'{output_dir}/{doc_name}_analisis.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Tersimpan: {output_file}")
    
    # Save combined report
    if all_results:
        combined_output = 'outputs/reports/semua_dokumen_analisis.json'
        with open(combined_output, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print("\n" + "="*60)
        print(f"✅ ANALISIS SELESAI")
        print(f"✅ Laporan gabungan tersimpan: {combined_output}")
        print("="*60)
        
        # Generate summary statistics
        generate_summary_statistics(all_results)
    else:
        print("\n⚠️  Tidak ada file yang berhasil diproses")
    
    return all_results

def generate_summary_statistics(all_results):
    """
    Generate statistik ringkasan dari semua dokumen
    """
    print("\n📊 STATISTIK RINGKASAN")
    print("-"*60)
    
    summary = {
        'total_documents': len(all_results),
        'documents': {}
    }
    
    for doc_name, result in all_results.items():
        summary['documents'][doc_name] = {
            'sentences': result['metadata']['sentence_count'],
            'characters': result['characters']['statistics']['total_characters'],
            'relations': result['relations']['summary']['total_relations']
        }
        
        print(f"\n📖 {doc_name}:")
        print(f"   Kalimat: {result['metadata']['sentence_count']}")
        print(f"   Tokoh: {result['characters']['statistics']['total_characters']}")
        print(f"   Hubungan: {result['relations']['summary']['total_relations']}")
    
    # Overall statistics
    if all_results:
        total_chars = sum(d['characters'] for d in summary['documents'].values())
        total_relations = sum(d['relations'] for d in summary['documents'].values())
        avg_chars = total_chars / len(all_results)
        
        print(f"\n📈 KESELURUHAN:")
        print(f"   Total Tokoh: {total_chars}")
        print(f"   Total Hubungan: {total_relations}")
        print(f"   Rata-rata Tokoh per Dokumen: {avg_chars:.1f}")
    
    # Save summary
    output_dir = 'outputs/reports'
    os.makedirs(output_dir, exist_ok=True)
    
    summary_file = f'{output_dir}/ringkasan_statistik.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Ringkasan tersimpan: {summary_file}")

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('outputs/reports', exist_ok=True)
    os.makedirs('outputs/visualizations', exist_ok=True)
    
    # Run full analysis
    analyze_all_documents()