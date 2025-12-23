"""
Updated main.py - CLI Interface for Hybrid Ensemble Entity Extraction
"""

import os
import sys
import argparse
from pathlib import Path
import yaml
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.entity_extraction.pipeline_orchestrator import EntityExtractionPipeline
from src.trait_extraction import TraitExtractor
from src.relation_extraction import RelationExtractor
from src.utils import ReportGenerator

def setup_directories():
    """Setup required directories"""
    directories = [
        'data/raw',
        'data/processed',
        'outputs/intermediate',
        'outputs/final',
        'outputs/reports',
        'outputs/visualizations',
        'outputs/metrics',
        'configs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✓ Directories setup complete")

def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file"""
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    # Return default config if file not found
    return None

def run_entity_extraction(args):
    """Run entity extraction pipeline"""
    print("\n" + "="*80)
    print("ENTITY EXTRACTION PIPELINE (Hybrid Ensemble)")
    print("="*80)
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize pipeline
    pipeline = EntityExtractionPipeline(
        config=config,
        debug_mode=args.debug
    )
    
    # Single or batch mode
    if args.mode == 'single':
        if not args.file:
            print("Error: --file required for single mode")
            sys.exit(1)
        
        if not Path(args.file).exists():
            print(f"Error: File not found: {args.file}")
            sys.exit(1)
        
        # Process single document
        results = pipeline.process_document(
            args.file,
            output_dir='outputs'
        )
        
        return results
    
    elif args.mode == 'batch':
        # Get all files from data/raw
        raw_dir = Path('data/raw')
        files = list(raw_dir.glob('*.txt'))
        
        if not files:
            print(f"Error: No .txt files found in {raw_dir}")
            sys.exit(1)
        
        print(f"\nFound {len(files)} documents to process")
        
        # Process batch
        results = pipeline.process_batch(
            [str(f) for f in files],
            output_dir='outputs'
        )
        
        return results

def run_full_pipeline(args):
    """Run complete pipeline: entity + trait + relation extraction"""
    print("\n" + "="*80)
    print("FULL ANALYSIS PIPELINE")
    print("="*80)
    
    # Step 1: Entity extraction
    print("\n[PHASE 1] Entity Extraction")
    entity_results = run_entity_extraction(args)
    
    if args.mode == 'single':
        # Process single document
        results = process_full_analysis(
            entity_results,
            args.file
        )
        
        # Save report
        save_full_report(results, args.file)
        
    elif args.mode == 'batch':
        # Process all documents
        all_results = {}
        for filepath, entity_result in entity_results.items():
            if 'error' in entity_result:
                continue
            
            results = process_full_analysis(entity_result, filepath)
            all_results[filepath] = results
            save_full_report(results, filepath)
        
        # Generate combined report
        generate_combined_report(all_results)

def process_full_analysis(entity_results: dict, filepath: str) -> dict:
    """Process full analysis for one document"""
    print(f"\n[PHASE 2] Trait & Relation Extraction: {Path(filepath).name}")
    
    entities = entity_results.get('entities', [])
    
    if not entities:
        print("  Warning: No entities found, skipping trait/relation extraction")
        return entity_results
    
    # Initialize extractors
    trait_extractor = TraitExtractor()
    relation_extractor = RelationExtractor()
    
    # Extract traits
    print("  → Extracting character traits...")
    traits = {}
    for entity in entities:
        name = entity['name']
        contexts = entity.get('contexts', [])
        
        if contexts:
            entity_traits = trait_extractor.extract_traits(name, contexts)
            traits[name] = entity_traits
    
    print(f"  ✓ Extracted traits for {len(traits)} characters")
    
    # Extract relations
    print("  → Extracting character relations...")
    
    # Build character dict for relation extractor (expects {name: count})
    characters = {e['name']: e['mentions'] for e in entities}
    
    # Get sentences from entity_results metadata or reload
    sentences = []
    if 'metadata' in entity_results:
        # Would need to reload from preprocessed data
        # For now, simple approach
        pass
    
    # Simplified: use contexts to build sentence list
    all_sentences = set()
    for entity in entities:
        for ctx in entity.get('contexts', []):
            all_sentences.add(ctx['sentence'])
    sentences = sorted(list(all_sentences))
    
    relations = relation_extractor.extract_relations(characters, sentences)
    
    print(f"  ✓ Found {len(relations.get('merged_relations', []))} relations")
    
    # Combine results
    full_results = {
        **entity_results,
        'traits': traits,
        'relations': relations
    }
    
    return full_results

def save_full_report(results: dict, filepath: str):
    """Save full analysis report"""
    output_dir = Path('outputs/reports')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = Path(filepath).stem
    
    # JSON report
    json_path = output_dir / f'{filename}_full_report.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n  ✓ Full report saved: {json_path}")
    
    # Generate markdown report
    report_gen = ReportGenerator()
    md_path = output_dir / f'{filename}_report.md'
    report_gen.generate_markdown_report(results, str(md_path))
    
    print(f"  ✓ Markdown report: {md_path}")

def generate_combined_report(all_results: dict):
    """Generate combined report for all documents"""
    output_dir = Path('outputs/reports')
    
    combined = {
        'total_documents': len(all_results),
        'documents': {}
    }
    
    for filepath, results in all_results.items():
        doc_name = Path(filepath).stem
        combined['documents'][doc_name] = {
            'entities': len(results.get('entities', [])),
            'traits_analyzed': len(results.get('traits', {})),
            'relations': len(results.get('relations', {}).get('merged_relations', []))
        }
    
    # Save combined
    combined_path = output_dir / 'combined_analysis.json'
    with open(combined_path, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    
    print(f"\n  ✓ Combined report: {combined_path}")

def run_evaluation(args):
    """Run evaluation against ground truth"""
    print("\n" + "="*80)
    print("EVALUATION MODE")
    print("="*80)
    
    if not args.ground_truth:
        print("Error: --ground-truth required for evaluation")
        sys.exit(1)
    
    # Load ground truth
    with open(args.ground_truth, 'r') as f:
        ground_truth = json.load(f)
    
    # Run extraction
    entity_results = run_entity_extraction(args)
    
    # Calculate metrics
    from evaluation import evaluate_extraction
    metrics = evaluate_extraction(entity_results, ground_truth)
    
    # Display results
    print("\n📊 EVALUATION RESULTS")
    print("-"*80)
    print(f"  Precision: {metrics['precision']:.2%}")
    print(f"  Recall:    {metrics['recall']:.2%}")
    print(f"  F1-Score:  {metrics['f1']:.2%}")
    
    # Save metrics
    metrics_path = Path('outputs/metrics') / 'evaluation_results.json'
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n  ✓ Metrics saved: {metrics_path}")

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Hybrid Ensemble Entity Extraction for Literary Texts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single document
  python main.py --mode single --file data/raw/the_gift_of_magi.txt
  
  # Process all documents
  python main.py --mode batch
  
  # Full pipeline (entity + trait + relation)
  python main.py --mode batch --pipeline full
  
  # Evaluation mode
  python main.py --mode single --file data/raw/the_gift_of_magi.txt \\
                 --evaluate --ground-truth experiments/ground_truth/gift_of_magi_gt.json
  
  # Debug mode
  python main.py --mode single --file data/raw/the_gift_of_magi.txt --debug
        """
    )
    
    # Main arguments
    parser.add_argument(
        '--mode',
        choices=['single', 'batch'],
        default='batch',
        help='Processing mode (default: batch)'
    )
    
    parser.add_argument(
        '--file',
        type=str,
        help='Input file for single mode'
    )
    
    parser.add_argument(
        '--pipeline',
        choices=['entity', 'full'],
        default='entity',
        help='Pipeline type: entity-only or full (entity+trait+relation)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Configuration file (YAML)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with verbose logging'
    )
    
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Run evaluation against ground truth'
    )
    
    parser.add_argument(
        '--ground-truth',
        type=str,
        help='Ground truth file for evaluation (JSON)'
    )
    
    args = parser.parse_args()
    
    # Setup directories
    setup_directories()
    
    # Run appropriate pipeline
    try:
        if args.evaluate:
            run_evaluation(args)
        elif args.pipeline == 'full':
            run_full_pipeline(args)
        else:
            run_entity_extraction(args)
        
        print("\n✓ All tasks completed successfully\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()