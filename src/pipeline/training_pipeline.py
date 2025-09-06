# src/pipeline/training_pipeline.py
"""
End-to-end training pipeline matching my exact notebook workflow.
Recreates the complete process from data loading to model training.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
import os
from datetime import datetime
import logging

# Import modules matching my notebook approach
from src.data.data_loader import load_and_prepare_data
from src.features.feature_engineer import engineer_features
from src.models.preprocessor import BotDetectionPreprocessor
from src.models.model_trainer import BotDetectionModelTrainer
from config.config import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BotDetectionTrainingPipeline:
    """
    Complete training pipeline matching my exact notebook workflow.
    Recreates cells 2-21 and the modeling process that won 1st place.
    """
    
    def __init__(self):
        """Initialize pipeline"""
        self.bids_df = None
        self.train_df = None
        self.test_df = None
        self.train_processed = None
        self.test_processed = None
        self.preprocessor = None
        self.model_trainer = None
        self.pipeline_results = {}
        
    def load_data(self) -> 'BotDetectionTrainingPipeline':
        """
        Load data exactly as in my notebook cells 2-3.
        """
        logger.info("Loading competition data...")
        
        try:
            # Load data exactly as in notebook cells 2-3
            self.bids_df = pd.read_csv('data/bids.csv')
            self.train_df = pd.read_csv('data/train.csv')
            self.test_df = pd.read_csv('data/test.csv')
            
            logger.info(f"Loaded bids_df: {self.bids_df.shape}")
            logger.info(f"Loaded train_df: {self.train_df.shape}")
            logger.info(f"Loaded test_df: {self.test_df.shape}")
            
            # Store data loading results
            self.pipeline_results['data_loading'] = {
                'bids_shape': self.bids_df.shape,
                'train_shape': self.train_df.shape,
                'test_shape': self.test_df.shape
            }
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
        
        return self
    
    def engineer_features(self) -> 'BotDetectionTrainingPipeline':
        """
        Engineer features exactly as in my notebook cells 16-21.
        """
        if self.bids_df is None:
            raise ValueError("Data must be loaded first. Call load_data().")
        
        logger.info("Starting feature engineering...")
        
        try:
            # Apply feature engineering exactly as in my notebook
            self.train_processed, self.test_processed = engineer_features(
                self.bids_df, self.train_df, self.test_df
            )
            
            logger.info(f"Feature engineering completed!")
            logger.info(f"Train shape: {self.train_processed.shape}")
            logger.info(f"Test shape: {self.test_processed.shape}")
            
            # Store feature engineering results
            self.pipeline_results['feature_engineering'] = {
                'train_shape': self.train_processed.shape,
                'test_shape': self.test_processed.shape,
                'num_features': self.train_processed.shape[1] - 1,  # Excluding target
                'feature_names': self.train_processed.columns.tolist()
            }
            
            # Save engineered features
            os.makedirs('data/processed', exist_ok=True)
            self.train_processed.to_csv('data/processed/train_with_features.csv', index=False)
            self.test_processed.to_csv('data/processed/test_with_features.csv', index=False)
            logger.info("Engineered features saved to data/processed/")
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            raise
        
        return self
    
    def preprocess_data(self) -> 'BotDetectionTrainingPipeline':
        """
        Preprocess data exactly as I did in my notebook.
        Apply SMOTE + StandardScaler pipeline.
        """
        if self.train_processed is None:
            raise ValueError("Features must be engineered first. Call engineer_features().")
        
        logger.info("Starting data preprocessing...")
        
        try:
            # Separate features and target exactly as in my notebook
            X = self.train_processed.drop('outcome', axis=1)
            y = self.train_processed['outcome']
            
            # Create and apply preprocessor (SMOTE + StandardScaler)
            self.preprocessor = BotDetectionPreprocessor()
            self.X_processed, self.y_processed = self.preprocessor.fit_transform(X, y)
            
            # Log preprocessing results
            original_class_dist = np.bincount(y.astype(int))
            processed_class_dist = np.bincount(self.y_processed.astype(int))
            
            logger.info(f"Preprocessing completed!")
            logger.info(f"Original samples: {len(X)} -> Processed samples: {len(self.X_processed)}")
            logger.info(f"Original class distribution: {original_class_dist}")
            logger.info(f"After SMOTE class distribution: {processed_class_dist}")
            
            # Store preprocessing results
            self.pipeline_results['preprocessing'] = {
                'original_samples': len(X),
                'processed_samples': len(self.X_processed),
                'original_class_distribution': original_class_dist.tolist(),
                'processed_class_distribution': processed_class_dist.tolist(),
                'features_used': len(self.preprocessor.get_feature_names())
            }
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise
        
        return self
    
    def train_model(self, use_hyperparameter_tuning: bool = True) -> 'BotDetectionTrainingPipeline':
        """
        Train model exactly as I did in my notebook.
        Uses ExtraTreesClassifier with optional hyperparameter tuning.
        """
        if not hasattr(self, 'X_processed'):
            raise ValueError("Data must be preprocessed first. Call preprocess_data().")
        
        logger.info("Starting model training...")
        
        try:
            # Create and train model
            self.model_trainer = BotDetectionModelTrainer()
            
            if use_hyperparameter_tuning:
                self.model_trainer.train_with_hyperparameter_tuning(self.X_processed, self.y_processed)
            else:
                self.model_trainer.train_with_default_params(self.X_processed, self.y_processed)
            
            # Get training results
            training_results = self.model_trainer.get_training_summary()
            
            logger.info("Model training completed!")
            if 'best_score' in training_results:
                logger.info(f"Best CV ROC-AUC: {training_results['best_score']:.6f}")
            elif 'cv_mean' in training_results:
                logger.info(f"CV ROC-AUC: {training_results['cv_mean']:.6f} (+/- {training_results['cv_std']:.6f})")
            
            # Store training results
            self.pipeline_results['model_training'] = training_results
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise
        
        return self
    
    def evaluate_model(self) -> Dict[str, Any]:
        """
        Evaluate the trained model and get feature importance.
        """
        if self.model_trainer is None:
            raise ValueError("Model must be trained first. Call train_model().")
        
        logger.info("Evaluating model...")
        
        # Get feature importance with feature names
        feature_names = self.preprocessor.get_feature_names()
        feature_importance = self.model_trainer.get_feature_importance(feature_names, top_n=20)
        
        logger.info("Top 20 Most Important Features:")
        for idx, row in feature_importance.iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.6f}")
        
        # Store evaluation results
        evaluation_results = {
            'feature_importance': feature_importance.to_dict('records'),
            'top_features': feature_importance['feature'].tolist()
        }
        
        self.pipeline_results['evaluation'] = evaluation_results
        
        return evaluation_results
    
    def compare_models(self) -> Dict[str, float]:
        """
        Compare multiple models as I did in my notebook.
        Shows why ExtraTreesClassifier won.
        """
        if not hasattr(self, 'X_processed'):
            raise ValueError("Data must be preprocessed first.")
        
        logger.info("Comparing multiple models...")
        
        # Use the model trainer's comparison method
        model_scores = self.model_trainer.evaluate_models(self.X_processed, self.y_processed)
        
        # Store comparison results
        self.pipeline_results['model_comparison'] = model_scores
        
        return model_scores
    
    def save_model(self, model_name: Optional[str] = None) -> str:
        """
        Save the trained model and pipeline results.
        """
        if self.model_trainer is None:
            raise ValueError("Model must be trained first.")
        
        # Generate model filename
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"bot_detection_model_{timestamp}.pkl"
        
        # Create model path
        model_path = os.path.join(config.training.MODEL_SAVE_PATH, model_name)
        
        # Save model
        saved_path = self.model_trainer.save_model(model_path)
        
        # Save pipeline results
        results_path = model_path.replace('.pkl', '_pipeline_results.json')
        import json
        with open(results_path, 'w') as f:
            json.dump(self.pipeline_results, f, indent=2, default=str)
        
        logger.info(f"Pipeline results saved to {results_path}")
        
        return saved_path
    
    def run_full_pipeline(self, use_hyperparameter_tuning: bool = True, 
                         save_model: bool = True, compare_models: bool = False) -> Dict[str, Any]:
        """
        Run the complete pipeline exactly as I did in my notebook.
        
        Parameters:
        use_hyperparameter_tuning: Whether to use RandomizedSearchCV
        save_model: Whether to save the trained model
        compare_models: Whether to compare with other models
        
        Returns:
        Dictionary with complete pipeline results
        """
        pipeline_start_time = datetime.now()
        
        logger.info("Starting complete bot detection training pipeline...")
        logger.info("Recreating the exact process that won 1st place in Kaggle competition!")
        
        try:
            # Execute pipeline steps
            self.load_data()
            self.engineer_features()
            self.preprocess_data()
            self.train_model(use_hyperparameter_tuning)
            self.evaluate_model()
            
            # Optional model comparison
            if compare_models:
                self.compare_models()
            
            # Save model if requested
            if save_model:
                model_path = self.save_model()
                self.pipeline_results['saved_model_path'] = model_path
            
            # Calculate total time
            pipeline_end_time = datetime.now()
            total_time = (pipeline_end_time - pipeline_start_time).total_seconds()
            
            logger.info(f"Complete pipeline finished in {total_time:.2f} seconds")
            
            # Add pipeline summary
            self.pipeline_results['pipeline_summary'] = {
                'total_execution_time_seconds': total_time,
                'pipeline_start_time': pipeline_start_time.isoformat(),
                'pipeline_end_time': pipeline_end_time.isoformat(),
                'pipeline_success': True,
                'notebook_recreated': True
            }
            
            # Print final summary
            self._print_final_summary()
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            self.pipeline_results['pipeline_summary'] = {
                'pipeline_success': False,
                'error_message': str(e)
            }
            raise
        
        return self.pipeline_results
    
    def _print_final_summary(self):
        """Print final summary of pipeline results"""
        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETED - NOTEBOOK RECREATED SUCCESSFULLY!")
        logger.info("="*80)
        
        # Data summary
        if 'data_loading' in self.pipeline_results:
            data_info = self.pipeline_results['data_loading']
            logger.info(f"Data loaded: {data_info['bids_shape'][0]} bids, {data_info['train_shape'][0]} training bidders")
        
        # Feature engineering summary
        if 'feature_engineering' in self.pipeline_results:
            feat_info = self.pipeline_results['feature_engineering']
            logger.info(f"Features engineered: {feat_info['num_features']} features created")
        
        # Preprocessing summary
        if 'preprocessing' in self.pipeline_results:
            prep_info = self.pipeline_results['preprocessing']
            logger.info(f"SMOTE applied: {prep_info['original_samples']} -> {prep_info['processed_samples']} samples")
        
        # Model performance
        if 'model_training' in self.pipeline_results:
            model_info = self.pipeline_results['model_training']
            if 'best_score' in model_info:
                logger.info(f"Best model CV ROC-AUC: {model_info['best_score']:.6f}")
            elif 'cv_mean' in model_info:
                logger.info(f"Model CV ROC-AUC: {model_info['cv_mean']:.6f}")
        
        logger.info("="*80)
        logger.info("This recreates the exact process that achieved 1st place!")
        logger.info("="*80)
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get complete pipeline summary"""
        return self.pipeline_results.copy()


def run_training_pipeline(use_hyperparameter_tuning: bool = True, 
                         save_model: bool = True, 
                         compare_models: bool = False) -> BotDetectionTrainingPipeline:
    """
    Convenience function to run the complete training pipeline.
    Recreates my exact notebook process that won 1st place.
    
    Parameters:
    use_hyperparameter_tuning: Whether to use RandomizedSearchCV
    save_model: Whether to save the trained model
    compare_models: Whether to compare with other models
    
    Returns:
    BotDetectionTrainingPipeline: Completed pipeline instance
    """
    pipeline = BotDetectionTrainingPipeline()
    pipeline.run_full_pipeline(
        use_hyperparameter_tuning=use_hyperparameter_tuning,
        save_model=save_model,
        compare_models=compare_models
    )
    return pipeline