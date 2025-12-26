"""
Main CLI for Unsupervised Entity Extraction

Usage:
    python main.py <path_to_text_file> [options]

Examples:
    python main.py data/raw/the_gift_of_magi.txt
    python main.py data/raw/story.txt --min-mentions 2
    python main.py data/raw/story.txt --threshold 0.6
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.entity_extractor import UnsupervisedEntityExtractor


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Unsupervised Entity Extraction for Literary Texts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py data/raw/the_gift_of_magi.txt
  python main.py data/raw/story.txt --min-mentions 2
  python main.py data/raw/story.txt --threshold 0.6 --output custom_output.json
        """
    )
    
    parser.add_argument(
        'filepath',
        type=str,
        help='Path to text file (.txt)'
    )
    
    parser.add_argument(
        '--min-mentions',
        type=int,
        default=3,
        help='Minimum mentions required (default: 3)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Capitalization consistency threshold (default: 0.5)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Custom output file path (default: outputs/entities/<filename>_entities.json)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results to file (display only)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed extraction process'
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    
    # Parse arguments
    args = parse_arguments()
    filepath = args.filepath
    
    # Check file exists
    if not os.path.exists(filepath):
        print(f"✗ Error: File not found: {filepath}")
        sys.exit(1)
    
    # Check file extension
    if not filepath.endswith('.txt'):
        print(f"⚠️  Warning: Expected .txt file, got: {filepath}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    print(f"\n{'='*70}")
    print(f"Processing: {os.path.basename(filepath)}")
    print(f"{'='*70}")
    
    # Build config
    config = {
        'min_mentions': args.min_mentions,
        'capitalization_threshold': args.threshold,
    }
    
    if args.verbose:
        print(f"\nConfiguration:")
        print(f"  min_mentions: {config['min_mentions']}")
        print(f"  threshold: {config['capitalization_threshold']}")
    
    try:
        # Initialize extractor
        extractor = UnsupervisedEntityExtractor(config=config)
        
        # Extract entities
        results = extractor.extract_entities(filepath)
        
        # Display results
        print(extractor.format_results(results))
        
        # Save results
        if not args.no_save:
            # Determine output path
            if args.output:
                output_path = args.output
            else:
                output_dir = Path('outputs/entities')
                output_dir.mkdir(parents=True, exist_ok=True)
                filename = Path(filepath).stem
                output_path = output_dir / f"{filename}_entities.json"
            
            # Save
            extractor.save_results(results, str(output_path))
        
        # Exit successfully
        sys.exit(0)
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()