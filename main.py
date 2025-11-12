"""
Main script untuk menjalankan sistem ekstraksi informasi karakter
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
    
    print("✅ Directories setup complete")

def process_single_document(filepath, save_reports=True):
    """
    Proses satu dokumen lengkap
    """
    print(f"\n{'='*70}")
    print(f"PROCESSING: {os.path.basename(filepath)}")
    print(f"{'='*70}")
    
    # Inisialisasi
    preprocessor = TextPreprocessor()
    char_extractor = CharacterExtractor()
    trait_extractor = TraitExtractor()
    rel_extractor = RelationExtractor()
    report_gen = ReportGenerator()
    
    try:
        # Step 1: Preprocessing
        print("\n[STEP 1/4] Text Preprocessing...")
        preprocessed = preprocessor.preprocess_document(filepath)
        print(f"  ✓ Cleaned text: {len(preprocessed['cleaned_text'])} characters")
        print(f"  ✓ Sentences extracted: {preprocessed['sentence_count']}")
        
        # Step 2: Character Extraction
        print("\n[STEP 2/4] Character Extraction (NER)...")
        char_results = char_extractor.extract_characters(
            preprocessed['cleaned_text'],
            preprocessed['sentences']
        )
        char_stats = char_extractor.get_character_statistics(char_results)
        print(f"  ✓ Total characters found: {char_stats['total_characters']}")
        
        if char_stats['most_mentioned']:
            print(f"  ✓ Most mentioned: {char_stats['most_mentioned'][0]} ({char_stats['most_mentioned'][1]}x)")
        
        # Step 3: Trait Extraction
        print("\n[STEP 3/4] Character Trait Extraction...")
        trait_results = {}
        for character, contexts in char_results['characters_with_context'].items():
            print(f"  → Analyzing {character}...")
            traits = trait_extractor.extract_traits(character, contexts)
            trait_results[character] = traits
            print(f"    ✓ {len(set(traits['raw_traits']))} unique traits found")
        
        # Step 4: Relation Extraction
        print("\n[STEP 4/4] Relation Extraction...")
        relation_results = rel_extractor.extract_relations(
            char_results['main_characters'],
            preprocessed['sentences']
        )
        print(f"  ✓ Relations detected: {len(relation_results['merged_relations'])}")
        
        # Compile results
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
        
        # Save reports if requested
        if save_reports:
            doc_name = os.path.basename(filepath).replace('.txt', '')
            
            # JSON report
            json_path = f'outputs/reports/{doc_name}_full_report.json'
            save_processed_data(full_results, json_path)
            
            # Markdown report
            md_path = f'outputs/reports/{doc_name}_report.md'
            report_gen.generate_markdown_report(full_results, md_path)
            print(f"\n✅ Markdown report: {md_path}")
            
            # HTML report
            html_path = f'outputs/reports/{doc_name}_report.html'
            report_gen.generate_html_report(full_results, html_path)
            print(f"✅ HTML report: {html_path}")
        
        print(f"\n{'='*70}")
        print("✅ PROCESSING COMPLETE")
        print(f"{'='*70}\n")
        
        return full_results
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def process_all_documents():
    """
    Proses semua dokumen dalam folder data/raw
    """
    print("\n" + "="*70)
    print("BATCH PROCESSING: ALL DOCUMENTS")
    print("="*70)
    
    documents = [
        'data/raw/the_tell_tale_heart.txt',
        'data/raw/the_gift_of_magi.txt',
        'data/raw/the_yellow_wallpaper.txt',
        'data/raw/the_lottery.txt',
        'data/raw/owl_creek_bridge.txt'
    ]
    
    all_results = {}
    successful = 0
    failed = 0
    
    for doc_path in documents:
        if os.path.exists(doc_path):
            result = process_single_document(doc_path, save_reports=True)
            if result:
                doc_name = os.path.basename(doc_path).replace('.txt', '')
                all_results[doc_name] = result
                successful += 1
            else:
                failed += 1
        else:
            print(f"\n⚠️  File not found: {doc_path}")
            failed += 1
    
    # Save combined results
    if all_results:
        combined_path = 'outputs/reports/all_documents_combined.json'
        save_processed_data(all_results, combined_path)
        print(f"\n✅ Combined results: {combined_path}")
    
    # Summary
    print("\n" + "="*70)
    print("BATCH PROCESSING SUMMARY")
    print("="*70)
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {successful + failed}")
    print("="*70 + "\n")
    
    return all_results

def main():
    """
    Main function dengan command line interface
    """
    parser = argparse.ArgumentParser(
        description='Sistem Ekstraksi Informasi Karakter dalam Karya Sastra'
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
    
    # Setup directories
    setup_directories()
    
    if args.mode == 'single':
        if not args.file:
            print("❌ Error: --file required for single mode")
            return
        
        if not os.path.exists(args.file):
            print(f"❌ Error: File not found: {args.file}")
            return
        
        process_single_document(args.file, save_reports=True)
    
    elif args.mode == 'batch':
        process_all_documents()
    
    elif args.mode == 'experiment':
        if not args.experiment:
            print("❌ Error: --experiment required for experiment mode")
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