"""
Main Pipeline Orchestrator - Integrates all components
File: src/entity_extraction/pipeline_orchestrator.py
"""

from typing import Dict, List, Any, Optional
import time
import logging
from pathlib import Path
import json

# Import preprocessing
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing import EnhancedTextPreprocessor

# Import extraction methods
from entity_extraction.method1_capitalization import CapitalizationExtractor
from entity_extraction.method2_tfisf import TFISFExtractor
from entity_extraction.method3_embeddings import EmbeddingsExtractor
from entity_extraction.ensemble_voter import EnsembleVoter

class EntityExtractionPipeline:
    """
    Main orchestrator for the entire entity extraction pipeline
    
    Pipeline stages:
    1. Preprocessing (enhanced with POS, n-grams)
    2. Parallel extraction (3 methods)
    3. Ensemble voting & fusion
    4. Quality control & output
    
    Features:
    - Configurable method weights
    - Parallel or sequential execution
    - Detailed timing & statistics
    - Error handling & fallbacks
    - Debug mode for troubleshooting
    """
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 debug_mode: bool = False):
        """
        Initialize pipeline
        
        Args:
            config: Pipeline configuration
            debug_mode: Enable detailed logging
        """
        self.config = config or self.get_default_config()
        self.debug_mode = debug_mode
        self.logger = self._setup_logger()
        
        # Initialize components
        self.logger.info("Initializing Entity Extraction Pipeline...")
        self._initialize_components()
        
        # Timing statistics
        self.timing_stats = {}
    
    def get_default_config(self) -> Dict[str, Any]:
        """Default pipeline configuration"""
        return {
            'preprocessing': {
                'ngram_range': (1, 3),
                'min_propn_length': 2,
                'track_positions': True
            },
            'method1': {
                'min_mentions': 3,
                'consistency_threshold': 0.65,
            },
            'method2': {
                'min_tfidf_score': 0.10,
                'top_k': 15,
            },
            'method3': {
                'embedding_model': 'all-MiniLM-L6-v2',
                'min_cluster_size': 2,
                'detect_narrator': True,
                'detect_roles': True
            },
            'ensemble': {
                'method_weights': {
                    'Method1_Capitalization': 0.30,
                    'Method2_TFIDF': 0.30,
                    'Method3_Embeddings': 0.40
                },
                'min_confidence': 0.50,
                'alignment_strategy': 'fuzzy'
            },
            'execution': {
                'parallel': False,  # Future: parallel execution
                'save_intermediate': True,
                'save_debug_info': True
            }
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Setup pipeline logger"""
        logger = logging.getLogger('EntityExtractionPipeline')
        logger.handlers = []  # Clear existing handlers
        
        handler = logging.StreamHandler()
        if self.debug_mode:
            formatter = logging.Formatter(
                '[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
                datefmt='%H:%M:%S'
            )
            logger.setLevel(logging.DEBUG)
        else:
            formatter = logging.Formatter('[%(levelname)s] %(message)s')
            logger.setLevel(logging.INFO)
        
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _initialize_components(self):
        """Initialize all pipeline components"""
        try:
            # Preprocessor
            self.logger.info("  → Loading preprocessor...")
            self.preprocessor = EnhancedTextPreprocessor()
            
            # Method 1: Capitalization
            self.logger.info("  → Initializing Method 1 (Capitalization)...")
            self.method1 = CapitalizationExtractor(
                config=self.config.get('method1')
            )
            
            # Method 2: TF-ISF
            self.logger.info("  → Initializing Method 2 (TF-ISF)...")
            self.method2 = TFISFExtractor(
                config=self.config.get('method2')
            )
            
            # Method 3: Embeddings
            self.logger.info("  → Initializing Method 3 (Embeddings)...")
            self.method3 = EmbeddingsExtractor(
                config=self.config.get('method3')
            )
            
            # Ensemble Voter
            self.logger.info("  → Initializing Ensemble Voter...")
            self.ensemble = EnsembleVoter(
                config=self.config.get('ensemble')
            )
            
            self.logger.info("✓ All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def process_document(self, 
                        filepath: str,
                        output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a single document through the entire pipeline
        
        Args:
            filepath: Path to document
            output_dir: Optional output directory for results
        
        Returns:
            Dict with complete extraction results
        """
        self.logger.info("\n" + "="*80)
        self.logger.info(f"PROCESSING DOCUMENT: {Path(filepath).name}")
        self.logger.info("="*80)
        
        pipeline_start = time.time()
        
        try:
            # Stage 1: Preprocessing
            self.logger.info("\n[STAGE 1/4] Preprocessing")
            self.logger.info("-"*80)
            stage_start = time.time()
            
            preprocessed = self.preprocessor.preprocess_document(
                filepath,
                config=self.config.get('preprocessing')
            )
            
            self.timing_stats['preprocessing'] = time.time() - stage_start
            self.logger.info(f"✓ Preprocessing complete ({self.timing_stats['preprocessing']:.2f}s)")
            
            # Save intermediate if configured
            if self.config['execution']['save_intermediate'] and output_dir:
                self._save_intermediate(
                    preprocessed,
                    output_dir,
                    'preprocessed.json',
                    filepath
                )
            
            # Stage 2: Parallel Extraction (3 methods)
            self.logger.info("\n[STAGE 2/4] Entity Extraction (3 Methods)")
            self.logger.info("-"*80)
            
            method_results = self._run_extraction_methods(preprocessed)
            
            # Save method results
            if self.config['execution']['save_intermediate'] and output_dir:
                for method_name, results in method_results.items():
                    self._save_intermediate(
                        results,
                        output_dir,
                        f'{method_name.lower()}_results.json',
                        filepath
                    )
            
            # Stage 3: Ensemble Voting
            self.logger.info("\n[STAGE 3/4] Ensemble Voting & Fusion")
            self.logger.info("-"*80)
            stage_start = time.time()
            
            ensemble_results = self.ensemble.vote(
                method_results,
                preprocessed
            )
            
            self.timing_stats['ensemble'] = time.time() - stage_start
            self.logger.info(f"✓ Ensemble complete ({self.timing_stats['ensemble']:.2f}s)")
            
            # Stage 4: Finalization
            self.logger.info("\n[STAGE 4/4] Finalization")
            self.logger.info("-"*80)
            
            final_results = self._finalize_results(
                ensemble_results,
                preprocessed,
                filepath,
                method_results
            )
            
            self.timing_stats['total'] = time.time() - pipeline_start
            final_results['timing'] = self.timing_stats.copy()
            
            # Display summary
            self._display_summary(final_results)
            
            # Save final results
            if output_dir:
                self._save_final_results(final_results, output_dir, filepath)
            
            self.logger.info("\n" + "="*80)
            self.logger.info(f"✓ PIPELINE COMPLETE ({self.timing_stats['total']:.2f}s)")
            self.logger.info("="*80 + "\n")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"\n✗ PIPELINE FAILED: {e}")
            if self.debug_mode:
                import traceback
                traceback.print_exc()
            raise
    
    def _run_extraction_methods(self, 
                                preprocessed: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Run all three extraction methods
        
        Returns:
            Dict mapping method_name -> results
        """
        method_results = {}
        
        # Method 1: Capitalization
        self.logger.info("\n  [Method 1] Capitalization-Based Extraction")
        stage_start = time.time()
        try:
            method1_results = self.method1.extract(preprocessed)
            method_results['Method1_Capitalization'] = method1_results
            elapsed = time.time() - stage_start
            self.timing_stats['method1'] = elapsed
            self.logger.info(f"    ✓ Found {len(method1_results['candidates'])} candidates ({elapsed:.2f}s)")
        except Exception as e:
            self.logger.error(f"    ✗ Method 1 failed: {e}")
            method_results['Method1_Capitalization'] = {'candidates': []}
        
        # Method 2: TF-ISF
        self.logger.info("\n  [Method 2] TF-ISF Statistical Ranking")
        stage_start = time.time()
        try:
            method2_results = self.method2.extract(preprocessed)
            method_results['Method2_TFISF'] = method2_results
            elapsed = time.time() - stage_start
            self.timing_stats['method2'] = elapsed
            self.logger.info(f"    ✓ Found {len(method2_results['candidates'])} candidates ({elapsed:.2f}s)")
        except Exception as e:
            self.logger.error(f"    ✗ Method 2 failed: {e}")
            method_results['Method2_TFISF'] = {'candidates': []}
        
        # Method 3: Embeddings
        self.logger.info("\n  [Method 3] Semantic Embeddings & Clustering")
        stage_start = time.time()
        try:
            method3_results = self.method3.extract(preprocessed)
            method_results['Method3_Embeddings'] = method3_results
            elapsed = time.time() - stage_start
            self.timing_stats['method3'] = elapsed
            self.logger.info(f"    ✓ Found {len(method3_results['candidates'])} candidates ({elapsed:.2f}s)")
        except Exception as e:
            self.logger.error(f"    ✗ Method 3 failed: {e}")
            method_results['Method3_Embeddings'] = {'candidates': []}
        
        return method_results
    
    def _finalize_results(self,
                         ensemble_results: Dict[str, Any],
                         preprocessed: Dict[str, Any],
                         filepath: str,
                         method_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Finalize and structure results
        """
        final_entities = ensemble_results['entities']
        
        # Add contexts for each entity (for trait/relation extraction)
        entities_with_contexts = self._add_entity_contexts(
            final_entities,
            preprocessed['sentences']
        )
        
        # Build final result structure
        final_results = {
            'metadata': {
                'filename': Path(filepath).name,
                'filepath': filepath,
                'sentence_count': preprocessed['sentence_count'],
                'propn_candidates': len(preprocessed['propn_candidates']),
                'processing_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'entities': entities_with_contexts,
            'statistics': {
                'total_entities': len(entities_with_contexts),
                'ensemble_stats': ensemble_results['statistics'],
                'method_contributions': ensemble_results['method_contributions'],
                'confidence_distribution': self._calculate_confidence_distribution(
                    entities_with_contexts
                )
            },
            'method_results_summary': {
                method: {
                    'candidates_found': len(results.get('candidates', [])),
                    'method_stats': results.get('statistics', {})
                }
                for method, results in method_results.items()
            },
            'config': self.config
        }
        
        return final_results
    
    def _add_entity_contexts(self,
                            entities: List[Dict],
                            sentences: List[str]) -> List[Dict]:
        """
        Add context sentences for each entity
        (Needed for downstream trait/relation extraction)
        """
        entities_with_contexts = []
        
        for entity in entities:
            name = entity['name']
            variants = entity.get('variants', [name])
            
            # Find all sentences containing this entity
            contexts = []
            for sent_id, sentence in enumerate(sentences):
                sent_lower = sentence.lower()
                
                # Check if any variant appears in sentence
                for variant in variants:
                    variant_lower = variant.lower()
                    if variant_lower in sent_lower:
                        contexts.append({
                            'sentence_id': sent_id,
                            'sentence': sentence
                        })
                        break  # Don't double-count same sentence
            
            # Add contexts to entity
            entity_with_context = entity.copy()
            entity_with_context['contexts'] = contexts
            entity_with_context['context_count'] = len(contexts)
            
            entities_with_contexts.append(entity_with_context)
        
        return entities_with_contexts
    
    def _calculate_confidence_distribution(self, entities: List[Dict]) -> Dict[str, int]:
        """Calculate confidence distribution"""
        distribution = {
            'very_high': 0,  # >= 0.9
            'high': 0,       # 0.8 - 0.9
            'medium': 0,     # 0.6 - 0.8
            'low': 0         # < 0.6
        }
        
        for entity in entities:
            conf = entity['confidence']
            if conf >= 0.9:
                distribution['very_high'] += 1
            elif conf >= 0.8:
                distribution['high'] += 1
            elif conf >= 0.6:
                distribution['medium'] += 1
            else:
                distribution['low'] += 1
        
        return distribution
    
    def _display_summary(self, results: Dict[str, Any]):
        """Display pipeline summary"""
        self.logger.info("\n📊 PIPELINE SUMMARY")
        self.logger.info("-"*80)
        
        # Entities found
        entities = results['entities']
        self.logger.info(f"  Total Entities: {len(entities)}")
        
        # Top entities
        if entities:
            self.logger.info(f"\n  Top 5 Entities:")
            for i, entity in enumerate(entities[:5], 1):
                conf = entity['confidence']
                mentions = entity['mentions']
                detected = len(entity['detected_by'])
                self.logger.info(
                    f"    {i}. {entity['name']:<30} "
                    f"Conf: {conf:.2%}  Mentions: {mentions:3d}  Methods: {detected}/3"
                )
        
        # Method contributions
        self.logger.info(f"\n  Method Contributions:")
        contrib = results['statistics']['method_contributions']
        for method, stats in contrib.items():
            method_short = method.replace('Method', 'M').replace('_', ' ')
            self.logger.info(
                f"    {method_short:<25} "
                f"Total: {stats['total']:2d}  "
                f"Unique: {stats['unique']:2d}  "
                f"Shared: {stats['shared']:2d}"
            )
        
        # Timing
        self.logger.info(f"\n  Timing Breakdown:")
        timing = results['timing']
        total = timing['total']
        for stage, duration in timing.items():
            if stage != 'total':
                pct = (duration / total * 100) if total > 0 else 0
                self.logger.info(f"    {stage:<15} {duration:6.2f}s  ({pct:5.1f}%)")
        self.logger.info(f"    {'total':<15} {total:6.2f}s  (100.0%)")
    
    def _save_intermediate(self,
                          data: Dict[str, Any],
                          output_dir: str,
                          filename: str,
                          source_filepath: str):
        """Save intermediate results"""
        output_path = Path(output_dir) / 'intermediate' / Path(source_filepath).stem
        output_path.mkdir(parents=True, exist_ok=True)
        
        filepath = output_path / filename
        
        # Handle numpy types for JSON serialization
        def convert_types(obj):
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=convert_types)
        
        if self.debug_mode:
            self.logger.debug(f"    Saved intermediate: {filepath}")
    
    def _save_final_results(self,
                           results: Dict[str, Any],
                           output_dir: str,
                           source_filepath: str):
        """Save final results"""
        output_path = Path(output_dir) / 'final'
        output_path.mkdir(parents=True, exist_ok=True)
        
        filename = Path(source_filepath).stem + '_entities.json'
        filepath = output_path / filename
        
        # Handle numpy types
        def convert_types(obj):
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=convert_types)
        
        self.logger.info(f"  ✓ Results saved: {filepath}")
    
    def process_batch(self,
                     filepaths: List[str],
                     output_dir: str) -> Dict[str, Dict[str, Any]]:
        """
        Process multiple documents
        
        Args:
            filepaths: List of document paths
            output_dir: Output directory
        
        Returns:
            Dict mapping filepath -> results
        """
        self.logger.info("\n" + "="*80)
        self.logger.info(f"BATCH PROCESSING: {len(filepaths)} documents")
        self.logger.info("="*80 + "\n")
        
        all_results = {}
        
        for i, filepath in enumerate(filepaths, 1):
            self.logger.info(f"\n[Document {i}/{len(filepaths)}]")
            
            try:
                results = self.process_document(filepath, output_dir)
                all_results[filepath] = results
            except Exception as e:
                self.logger.error(f"Failed to process {filepath}: {e}")
                all_results[filepath] = {'error': str(e)}
        
        # Generate batch summary
        self._generate_batch_summary(all_results, output_dir)
        
        return all_results
    
    def _generate_batch_summary(self,
                               all_results: Dict[str, Dict[str, Any]],
                               output_dir: str):
        """Generate summary across all documents"""
        self.logger.info("\n" + "="*80)
        self.logger.info("BATCH SUMMARY")
        self.logger.info("="*80)
        
        successful = sum(1 for r in all_results.values() if 'error' not in r)
        failed = len(all_results) - successful
        
        self.logger.info(f"\n  Documents Processed: {len(all_results)}")
        self.logger.info(f"    ✓ Successful: {successful}")
        self.logger.info(f"    ✗ Failed: {failed}")
        
        if successful > 0:
            total_entities = sum(
                len(r.get('entities', [])) 
                for r in all_results.values() 
                if 'error' not in r
            )
            avg_entities = total_entities / successful
            
            self.logger.info(f"\n  Total Entities: {total_entities}")
            self.logger.info(f"  Average per Document: {avg_entities:.1f}")
        
        # Save batch summary
        summary_path = Path(output_dir) / 'batch_summary.json'
        summary = {
            'total_documents': len(all_results),
            'successful': successful,
            'failed': failed,
            'documents': {
                Path(fp).name: {
                    'status': 'success' if 'error' not in r else 'failed',
                    'entities_found': len(r.get('entities', [])) if 'error' not in r else 0,
                    'error': r.get('error')
                }
                for fp, r in all_results.items()
            }
        }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"\n  ✓ Batch summary saved: {summary_path}")