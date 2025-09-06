# scripts/train_model.py
"""
Main training script that recreates my exact notebook workflow.
Runs the complete pipeline that achieved 1st place in Kaggle competition.
"""

import sys
import os
import argparse
import logging
from datetime import datetime

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline.training_pipeline import BotDetectionTrainingPipeline
from config.config import config

def setup_logging(log_level: str = 'INFO'):
    """Set up logging for the training script"""
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/training_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Training log: {log_filename}")
    
    return logger

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train bot detection model - recreate 1st place Kaggle solution')
    
    # Model training options
    parser.add_argument('--no-hyperparameter-tuning', action='store_true',
                       help='Skip hyperparameter tuning (faster training)')
    parser.add_argument('--compare-models', action='store_true',
                       help='Compare ExtraTreesClassifier with other models')
    parser.add_argument('--model-name', type=str,
                       help='Custom name for saved model file')
    
    # Data options
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory containing bids.csv, train.csv, test.csv')
    
    # Pipeline control
    parser.add_argument('--no-save-model', action='store_true',
                       help='Do not save the trained model')
    
    # Logging
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    return parser.parse_args()

def validate_data_files(data_dir: str, logger):
    """Validate that required data files exist"""
    required_files = ['bids.csv', 'train.csv', 'test.csv']
    
    for filename in required_files:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            logger.error(f"Required data file not found: {filepath}")
            raise FileNotFoundError(f"Required data file not found: {filepath}")
    
    logger.info("All required data files found")

def main():
    """
    Main function that recreates my exact notebook workflow.
    This runs the complete process that won 1st place in the competition.
    """
    # Parse arguments
    args = parse_arguments()
    
    # Set up logging
    logger = setup_logging(args.log_level)
    
    try:
        logger.info("="*80)
        logger.info("BOT DETECTION MODEL TRAINING")
        logger.info("Recreating 1st place Kaggle competition solution")
        logger.info("="*80)
        
        # Validate data files
        validate_data_files(args.data_dir, logger)
        
        # Update working directory if needed
        if args.data_dir != 'data':
            os.chdir(os.path.dirname(args.data_dir))
        
        # Create training pipeline
        pipeline = BotDetectionTrainingPipeline()
        
        # Set training options
        use_hyperparameter_tuning = not args.no_hyperparameter_tuning
        save_model = not args.no_save_model
        compare_models = args.compare_models
        
        logger.info(f"Training configuration:")
        logger.info(f"  Hyperparameter tuning: {use_hyperparameter_tuning}")
        logger.info(f"  Model comparison: {compare_models}")
        logger.info(f"  Save model: {save_model}")
        
        # Run complete pipeline (recreating my notebook process)
        logger.info("Starting pipeline - recreating notebook cells 2-21 and modeling...")
        
        results = pipeline.run_full_pipeline(
            use_hyperparameter_tuning=use_hyperparameter_tuning,
            save_model=save_model,
            compare_models=compare_models
        )
        
        # Print detailed results
        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        
        # Data loading results
        if 'data_loading' in results:
            data_info = results['data_loading']
            logger.info(f"Data loaded:")
            logger.info(f"   Bids: {data_info['bids_shape']}")
            logger.info(f"   Train: {data_info['train_shape']}")
            logger.info(f"   Test: {data_info['test_shape']}")
        
        # Feature engineering results
        if 'feature_engineering' in results:
            feat_info = results['feature_engineering']
            logger.info(f"Feature engineering:")
            logger.info(f"   Features created: {feat_info['num_features']}")
            logger.info(f"   Final train shape: {feat_info['train_shape']}")
        
        # Preprocessing results
        if 'preprocessing' in results:
            prep_info = results['preprocessing']
            logger.info(f"Preprocessing (SMOTE + StandardScaler):")
            logger.info(f"   Original samples: {prep_info['original_samples']}")
            logger.info(f"   After SMOTE: {prep_info['processed_samples']}")
            logger.info(f"   Original class dist: {prep_info['original_class_distribution']}")
            logger.info(f"   After SMOTE class dist: {prep_info['processed_class_distribution']}")
        
        # Model training results
        if 'model_training' in results:
            model_info = results['model_training']
            logger.info(f"Model training (ExtraTreesClassifier):")
            
            if 'best_score' in model_info:
                logger.info(f"   Best CV ROC-AUC: {model_info['best_score']:.6f}")
                logger.info(f"   Best parameters: {model_info['best_params']}")
            elif 'cv_mean' in model_info:
                logger.info(f"   CV ROC-AUC: {model_info['cv_mean']:.6f} (+/- {model_info['cv_std']:.6f})")
        
        # Model comparison results
        if 'model_comparison' in results:
            comp_info = results['model_comparison']
            logger.info(f"Model comparison results:")
            for i, (model_name, score) in enumerate(comp_info.items(), 1):
                logger.info(f"   {i}. {model_name}: {score:.6f}")
        
        # Feature importance
        if 'evaluation' in results:
            eval_info = results['evaluation']
            logger.info(f"Top 10 most important features:")
            for i, feature_info in enumerate(eval_info['feature_importance'][:10], 1):
                logger.info(f"   {i}. {feature_info['feature']}: {feature_info['importance']:.6f}")
        
        # Saved model info
        if 'saved_model_path' in results:
            logger.info(f"Model saved: {results['saved_model_path']}")
        
        # Timing info
        if 'pipeline_summary' in results:
            timing_info = results['pipeline_summary']
            total_time = timing_info.get('total_execution_time_seconds', 0)
            logger.info(f"Total execution time: {total_time:.2f} seconds")
        
        logger.info("="*80)
        logger.info("SUCCESS: Recreated the exact process that won 1st place!")
        logger.info("Your notebook workflow has been productionized!")
        logger.info("="*80)
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        logger.exception("Full error traceback:")
        sys.exit(1)

if __name__ == "__main__":
    main()