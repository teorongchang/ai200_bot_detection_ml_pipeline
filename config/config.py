# config/config.py
"""
Configuration module for bot detection pipeline.
Matches the exact setup from my winning Kaggle notebook.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Any

@dataclass
class DataConfig:
    """Configuration for data handling - matching my notebook file structure"""
    # File paths matching my notebook
    BIDS_FILE: str = "data/bids.csv"
    TRAIN_FILE: str = "data/train.csv"
    TEST_FILE: str = "data/test.csv"
    
    # Processed data paths
    PROCESSED_TRAIN: str = "data/processed/train_processed.csv"
    PROCESSED_TEST: str = "data/processed/test_processed.csv"
    FEATURES_TRAIN: str = "data/features/train_features.csv"
    FEATURES_TEST: str = "data/features/test_features.csv"

@dataclass
class ModelConfig:
    """Configuration for model training - matching my notebook approach"""
    # Target variable
    TARGET_COLUMN: str = "outcome"
    
    # Columns to exclude from modeling (as I did in my notebook)
    EXCLUDE_FEATURES: List[str] = None
    
    def __post_init__(self):
        if self.EXCLUDE_FEATURES is None:
            self.EXCLUDE_FEATURES = ['bidder_id', 'payment_account', 'address']
    
    # SMOTE parameters (as I used in my notebook)
    SMOTE_RANDOM_STATE: int = 42
    
    # ExtraTreesClassifier parameters (my winning model)
    EXTRA_TREES_PARAMS: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.EXTRA_TREES_PARAMS is None:
            self.EXTRA_TREES_PARAMS = {
                'n_estimators': 300,
                'max_depth': 25,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'sqrt',
                'random_state': 42,
                'n_jobs': -1
            }
    
    # Hyperparameter tuning grid (as I used in RandomizedSearchCV)
    HYPERPARAMETER_GRID: Dict[str, List] = None
    
    def __post_init__(self):
        if self.HYPERPARAMETER_GRID is None:
            self.HYPERPARAMETER_GRID = {
                'n_estimators': [120, 300, 500, 800, 1200],
                'max_depth': [5, 8, 15, 25, 30, None],
                'min_samples_split': [1, 2, 5, 10, 15, 100],
                'min_samples_leaf': [1, 2, 5, 10],
                'max_features': ['log2', 'sqrt', None]
            }

@dataclass
class TrainingConfig:
    """Configuration for training process"""
    RANDOM_SEED: int = 42
    MODEL_SAVE_PATH: str = "models/trained_models/"
    PERFORM_HYPERPARAMETER_TUNING: bool = True
    N_ITER_SEARCH: int = 50
    CV_FOLDS: int = 5

# Main configuration class
class Config:
    """Main configuration class"""
    
    def __init__(self):
        self.data = DataConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()

# Global config instance
config = Config()