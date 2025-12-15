"""
Abstract Base Class untuk Entity Extraction Methods
File: src/entity_extraction/base_extractor.py
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple
import logging

class BaseEntityExtractor(ABC):
    """
    Abstract base class untuk semua extraction methods
    
    Setiap method harus implement:
    1. extract() - main extraction logic
    2. get_confidence_score() - calculate confidence per entity
    3. get_method_name() - return method identifier
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize extractor
        
        Args:
            config: Method-specific configuration
        """
        self.config = config or self.get_default_config()
        self.logger = self._setup_logger()
        
    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        """
        Return default configuration for this method
        
        Returns:
            dict with default hyperparameters
        """
        pass
    
    @abstractmethod
    def extract(self, preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main extraction method
        
        Args:
            preprocessed_data: Output from EnhancedTextPreprocessor
        
        Returns:
            dict with structure:
            {
                'candidates': [
                    {
                        'name': str,
                        'score': float,
                        'mentions': int,
                        'metadata': dict
                    },
                    ...
                ],
                'method_name': str,
                'statistics': dict
            }
        """
        pass
    
    @abstractmethod
    def get_confidence_score(self, candidate: str, metadata: Dict) -> float:
        """
        Calculate confidence score for a candidate entity
        
        Args:
            candidate: Entity name
            metadata: Metadata about the candidate
        
        Returns:
            float: confidence score [0.0, 1.0]
        """
        pass
    
    @abstractmethod
    def get_method_name(self) -> str:
        """Return method identifier"""
        pass
    
    def _setup_logger(self) -> logging.Logger:
        """Setup method-specific logger"""
        logger = logging.getLogger(f'EntityExtraction.{self.get_method_name()}')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'[%(levelname)s] {self.get_method_name()}: %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def validate_input(self, preprocessed_data: Dict[str, Any]) -> bool:
        """
        Validate preprocessed data structure
        
        Args:
            preprocessed_data: Input data
        
        Returns:
            bool: True if valid
        
        Raises:
            ValueError: If required fields missing
        """
        required_fields = [
            'sentences', 
            'propn_candidates', 
            'capitalization_patterns',
            'ngrams'
        ]
        
        for field in required_fields:
            if field not in preprocessed_data:
                raise ValueError(f"Missing required field: {field}")
        
        return True
    
    def filter_by_threshold(self, 
                           candidates: List[Dict], 
                           threshold_key: str = 'score',
                           threshold_value: float = 0.5) -> List[Dict]:
        """
        Filter candidates by threshold
        
        Args:
            candidates: List of candidate dicts
            threshold_key: Key to check threshold on
            threshold_value: Minimum threshold value
        
        Returns:
            Filtered list of candidates
        """
        return [
            c for c in candidates 
            if c.get(threshold_key, 0) >= threshold_value
        ]
    
    def sort_by_score(self, candidates: List[Dict], descending: bool = True) -> List[Dict]:
        """Sort candidates by score"""
        return sorted(
            candidates, 
            key=lambda x: x.get('score', 0), 
            reverse=descending
        )
    
    def get_statistics(self, extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate statistics from extraction result
        
        Args:
            extraction_result: Result from extract()
        
        Returns:
            dict with statistics
        """
        candidates = extraction_result.get('candidates', [])
        
        if not candidates:
            return {
                'total_candidates': 0,
                'average_score': 0.0,
                'max_score': 0.0,
                'min_score': 0.0
            }
        
        scores = [c['score'] for c in candidates]
        
        return {
            'total_candidates': len(candidates),
            'average_score': sum(scores) / len(scores),
            'max_score': max(scores),
            'min_score': min(scores),
            'method_name': self.get_method_name()
        }