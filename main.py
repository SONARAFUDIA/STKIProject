"""
Script utama untuk menjalankan sistem ekstraksi informasi karakter
pada karya sastra berbahasa Indonesia
"""

import os
import argparse
from src.preprocessing import TextPreprocessor
from src.ner_extraction import CharacterExtractor
from src.trait_extraction import TraitExtractor
from src.relation_extraction import RelationExtractor
from src.utils import ReportGenerator, save_processed_data
import json

def setup_directories():
    """
    Setup direktori yang diperlukan
    """
    directories = [
        'data/raw',
        'data/processed',
        'data/results',
        'outputs/characters',
        'outputs/visualizations',
        'outputs/reports'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Setup direktori selesai")

def process_single_document(filepath, save_reports=True):
    """
    Proses satu dokumen lengkap
    """
    print(f"\n{'='*70}")
    print(f"MEMPROSES: {os.path.basename(filepath)}")
    print(f"{'='*70}")
    
    # Inisialisasi
    preprocessor = TextPreprocessor()
    char_extractor = CharacterExtractor()
    trait_extractor = TraitExtractor()
    rel_extractor = RelationExtractor()
    report_gen = ReportGenerator()
    
    try:
        # Langkah 1: Preprocessing
        print("\n[LANGKAH 1/4] Preprocessing Teks...")
        preprocessed = preprocessor.preprocess_document(filepath)
        print(f"  ✓ Teks dibersihkan: {len(preprocessed['cleaned_text'])} karakter")
        print(f"  ✓ Kalimat diekstrak: {preprocessed['sentence_count']}")
        
        # Langkah 2: Ekstraksi Tokoh
        print("\n[LANGKAH 2/4] Ekstraksi Tokoh (NER)...")
        char_results = char_extractor.extract_characters(
            preprocessed['cleaned_text'],
            preprocessed['sentences']
        )
        char_stats = char_extractor.get_character_statistics(char_results)
        print(f"  ✓ Total tokoh ditemukan: {char_stats['total_characters']}")
        
        if char_stats['most_mentioned']:
            print(f"  ✓ Paling sering disebut: {char_stats['most_mentioned'][0]} ({char_stats['most_mentioned'][1]}x)")
        
        # Langkah 3: Ekstraksi Watak
        print("\n[LANGKAH 3/4] Ekstraksi Watak Tokoh...")
        trait_results = {}
        for character, contexts in char_results['characters_with_context'].items():
            print(f"  → Menganalisis {character}...")
            traits = trait_extractor.extract_traits(character, contexts)
            trait_results[character] = traits
            print(f"    ✓ {len(set(traits['raw_traits']))} watak unik ditemukan")
        
        # Langkah 4: Ekstraksi Hubungan
        print("\n[LANGKAH 4/4] Ekstraksi Hubungan Antar Tokoh...")
        relation_results = rel_extractor.extract_relations(
            char_results['main_characters'],
            preprocessed['sentences']
        )
        print(f"  ✓ Hubungan terdeteksi: {len(relation_results['merged_relations'])}")
        
        # Kompilasi hasil
        from datetime import datetime
        full_results = {
            'metadata': {
                'filename': os.path.basename(filepath),
                'processed_at': datetime.now().isoformat(),
                'sentence_count': preprocessed['sentence_count']
            },
            'characters': {
                'statistics': char_stats,
                'details': char_results['main_characters'],
                'contexts': char_results['characters_with_context']
            },
            'traits': trait_results,
            'relations': {
                'summary': {
                    'total_relations': len(relation_results['merged_relations']),
                    'cooccurrence_pairs': len(relation_results['cooccurrence'])
                },
                'details': relation_results['merged_relations'],
                'graph': relation_results['relation_graph']
            }
        }
        
        # Simpan laporan jika diminta
        if save_reports:
            doc_name = os.path.basename(filepath).replace('.txt', '')
            
            # Laporan JSON
            json_path = f'outputs/reports/{doc_name}_laporan_lengkap.json'
            save_processed_data(full_results, json_path)
            
            # Laporan Markdown
            md_path = f'outputs/reports/{doc_name}_laporan.md'
            report_gen.generate_markdown_report(full_results, md_path)
            print(f"\n✅ Laporan Markdown: {md_path}")
            
            # Laporan HTML
            html_path = f'outputs/reports/{doc_name}_laporan.html'
            report_gen.generate_html_report(full_results, html_path)
            print(f"✅ Laporan HTML: {html_path}")
        
        print(f"\n{'='*70}")
        print("✅ PEMROSESAN SELESAI")
        print(f"{'='*70}\n")
        
        return full_results
        
    except Exception as e:
        print(f"\n❌ KESALAHAN: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def process_all_documents():
    """
    Proses semua dokumen dalam folder data/raw
    """
    print("\n" + "="*70)
    print("PEMROSESAN BATCH: SEMUA DOKUMEN")
    print("="*70)
    
    documents = [
        'data/raw/senja_di_ujung_kios.txt',
        'data/raw/rapat_warung_yopi_yang_batal.txt',
        'data/raw/aroma_kayu_cendana.txt',
        'data/raw/asing_di_cermin_itu.txt',
        'data/raw/garis_putus-putus.txt'
    ]
    
    all_results = {}
    successful = 0
    failed = 0
    skipped = 0
    
    for doc_path in documents:
        if not os.path.exists(doc_path):
            print(f"\n⚠️  File tidak ditemukan: {doc_path}")
            failed += 1
            continue
        
        # Check if file is empty
        if os.path.getsize(doc_path) == 0:
            print(f"\n⚠️  File kosong, dilewati: {doc_path}")
            skipped += 1
            continue
            
        result = process_single_document(doc_path, save_reports=True)
        if result:
            doc_name = os.path.basename(doc_path).replace('.txt', '')
            all_results[doc_name] = result
            successful += 1
        else:
            failed += 1
    
    # Simpan hasil gabungan
    if all_results:
        combined_path = 'outputs/reports/semua_dokumen_gabungan.json'
        save_processed_data(all_results, combined_path)
        print(f"\n✅ Hasil gabungan: {combined_path}")
    
    # Ringkasan
    print("\n" + "="*70)
    print("RINGKASAN PEMROSESAN BATCH")
    print("="*70)
    print(f"✅ Berhasil: {successful}")
    print(f"⚠️  Dilewati (kosong): {skipped}")
    print(f"❌ Gagal: {failed}")
    print(f"📊 Total: {successful + failed + skipped}")
    print("="*70 + "\n")
    
    return all_results

def main():
    """
    Fungsi utama dengan command line interface
    """
    parser = argparse.ArgumentParser(
        description='Sistem Ekstraksi Informasi Karakter dalam Karya Sastra Indonesia'
    )
    
    parser.add_argument(
        '--mode',
        choices=['single', 'batch', 'experiment'],
        default='batch',
        help='Mode pemrosesan (default: batch)'
    )
    
    parser.add_argument(
        '--file',
        type=str,
        help='Path ke file untuk mode single'
    )
    
    parser.add_argument(
        '--experiment',
        choices=['ner', 'trait', 'relation', 'full'],
        help='Tipe eksperimen yang akan dijalankan'
    )
    
    args = parser.parse_args()
    
    # Setup direktori
    setup_directories()
    
    if args.mode == 'single':
        if not args.file:
            print("❌ Kesalahan: --file diperlukan untuk mode single")
            return
        
        if not os.path.exists(args.file):
            print(f"❌ Kesalahan: File tidak ditemukan: {args.file}")
            return
        
        if os.path.getsize(args.file) == 0:
            print(f"❌ Kesalahan: File kosong: {args.file}")
            return
            
        process_single_document(args.file, save_reports=True)
    
    elif args.mode == 'batch':
        process_all_documents()
    
    elif args.mode == 'experiment':
        if not args.experiment:
            print("❌ Kesalahan: --experiment diperlukan untuk mode experiment")
            return
        
        if args.experiment == 'ner':
            from experiments.exp_01_ner_testing import test_ner_extraction
            test_ner_extraction()
        
        elif args.experiment == 'trait':
            from experiments.exp_02_trait_testing import test_trait_extraction
            test_trait_extraction()
        
        elif args.experiment == 'relation':
            from experiments.exp_03_relation_testing import test_relation_extraction
            test_relation_extraction()
        
        elif args.experiment == 'full':
            from experiments.exp_04_full_pipeline import analyze_all_documents
            analyze_all_documents()

if __name__ == "__main__":
    main()