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
    Pipeline lengkap untuk satu dokumen
    """
    print(f"\n{'='*60}")
    print(f"ANALYZING: {os.path.basename(filepath)}")
    print(f"{'='*60}")
    
    # Inisialisasi semua modul
    preprocessor = TextPreprocessor()
    char_extractor = CharacterExtractor()
    trait_extractor = TraitExtractor()
    rel_extractor = RelationExtractor()
    
    # STEP 1: Preprocessing
    print("\n[1/4] Preprocessing...")
    preprocessed = preprocessor.preprocess_document(filepath)
    print(f"  ✓ {preprocessed['sentence_count']} sentences extracted")
    
    # STEP 2: Character Extraction
    print("\n[2/4] Extracting characters...")
    char_results = char_extractor.extract_characters(
        preprocessed['cleaned_text'],
        preprocessed['sentences']
    )
    char_stats = char_extractor.get_character_statistics(char_results)
    print(f"  ✓ {char_stats['total_characters']} main characters found")
    
    # STEP 3: Trait Extraction
    print("\n[3/4] Extracting character traits...")
    trait_results = {}
    for character, contexts in char_results['characters_with_context'].items():
        traits = trait_extractor.extract_traits(character, contexts)
        trait_results[character] = traits
        print(f"  ✓ {character}: {len(traits['raw_traits'])} traits extracted")
    
    # STEP 4: Relation Extraction
    print("\n[4/4] Extracting relations...")
    relation_results = rel_extractor.extract_relations(
        char_results['main_characters'],
        preprocessed['sentences']
    )
    print(f"  ✓ {len(relation_results['merged_relations'])} relations found")
    
    # Compile full report
    full_report = {
        'metadata': {
            'filename': os.path.basename(filepath),
            'processed_at': datetime.now().isoformat,
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
    Analisis semua dokumen dalam folder data/raw
    """
    print("\n" + "="*60)
    print("FULL PIPELINE ANALYSIS - ALL DOCUMENTS")
    print("="*60)
    
    documents = [
        'data/raw/owl_creek_bridge.txt'
        'data/raw/the_tell_tale_heart.txt',
        'data/raw/the_gift_of_magi.txt',
        'data/raw/the_yellow_wallpaper.txt',
    ]
    
    all_results = {}
    
    for doc in documents:
        if os.path.exists(doc):
            result = full_pipeline_analysis(doc)
            doc_name = os.path.basename(doc).replace('.txt', '')
            all_results[doc_name] = result
            
            # Save individual report
            output_file = f'outputs/reports/{doc_name}_analysis.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n✅ Saved: {output_file}")
        else:
            print(f"\n⚠️  File not found: {doc}")
    
    # Save combined report
    combined_output = 'outputs/reports/all_documents_analysis.json'
    with open(combined_output, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print(f"✅ ALL ANALYSIS COMPLETE")
    print(f"✅ Combined report saved: {combined_output}")
    print("="*60)
    
    # Generate summary statistics
    generate_summary_statistics(all_results)
    
    return all_results

def generate_summary_statistics(all_results):
    """
    Generate statistik ringkasan dari semua dokumen
    """
    print("\n📊 SUMMARY STATISTICS")
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
        print(f"   Sentences: {result['metadata']['sentence_count']}")
        print(f"   Characters: {result['characters']['statistics']['total_characters']}")
        print(f"   Relations: {result['relations']['summary']['total_relations']}")
    
    # Overall statistics
    total_chars = sum(d['characters'] for d in summary['documents'].values())
    total_relations = sum(d['relations'] for d in summary['documents'].values())
    avg_chars = total_chars / len(all_results)
    
    print(f"\n📈 OVERALL:")
    print(f"   Total Characters: {total_chars}")
    print(f"   Total Relations: {total_relations}")
    print(f"   Avg Characters per Document: {avg_chars:.1f}")
    
    # Save summary
    with open('outputs/reports/summary_statistics.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n✅ Summary saved: outputs/reports/summary_statistics.json")

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('outputs/reports', exist_ok=True)
    os.makedirs('outputs/visualizations', exist_ok=True)
    
    # Run full analysis
    analyze_all_documents()