# scripts/predict.py
"""
Prediction script for bot detection model.
Uses the trained model to make predictions on new data.
"""

import sys
import os
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.features.feature_engineer import engineer_features
from src.models.model_trainer import BotDetectionModelTrainer
from src.models.preprocessor import BotDetectionPreprocessor

def setup_logging(log_level: str = 'INFO'):
    """Set up logging for prediction script"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Make predictions with trained bot detection model')
    
    # Required arguments
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model file')
    
    # Data arguments
    parser.add_argument('--bids-file', type=str, default='data/bids.csv',
                       help='Path to bids CSV file')
    parser.add_argument('--test-file', type=str, default='data/test.csv',
                       help='Path to test CSV file (bidders to predict)')
    
    # Output arguments
    parser.add_argument('--output-file', type=str,
                       help='Path to save predictions CSV file')
    parser.add_argument('--include-probabilities', action='store_true',
                       help='Include prediction probabilities in output')
    
    # Logging
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    return parser.parse_args()

def load_test_data(bids_file: str, test_file: str, logger):
    """Load test data for prediction"""
    logger.info("Loading test data...")
    
    # Load files
    bids_df = pd.read_csv(bids_file)
    test_df = pd.read_csv(test_file)
    
    logger.info(f"Loaded bids_df: {bids_df.shape}")
    logger.info(f"Loaded test_df: {test_df.shape}")
    
    return bids_df, test_df

def prepare_test_features(bids_df: pd.DataFrame, test_df: pd.DataFrame, logger):
    """Prepare test features using the same process as training"""
    logger.info("Engineering features for test data...")
    
    # Create dummy train_df for feature engineering (we only need test features)
    dummy_train = pd.DataFrame({'bidder_id': [], 'payment_account': [], 'address': [], 'outcome': []})
    
    # Engineer features (this will create features for test_df)
    _, test_with_features = engineer_features(bids_df, dummy_train, test_df)
    
    logger.info(f"Test features prepared: {test_with_features.shape}")
    
    return test_with_features

def make_predictions(model_path: str, test_features: pd.DataFrame, 
                    include_probabilities: bool, logger):
    """Make predictions using trained model"""
    logger.info("Loading trained model...")
    
    # Load model
    model_trainer = BotDetectionModelTrainer()
    model_trainer.load_model(model_path)
    
    # Prepare features (remove bidder_id and other non-feature columns)
    exclude_cols = ['bidder_id', 'payment_account', 'address', 'outcome']
    feature_cols = [col for col in test_features.columns if col not in exclude_cols]
    X_test = test_features[feature_cols]
    
    # Handle missing values and scaling (simplified for prediction)
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    
    # Fill missing values
    imputer = SimpleImputer(strategy='median')
    X_test_imputed = imputer.fit_transform(X_test)
    
    # Scale features
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test_imputed)
    
    logger.info("Making predictions...")
    
    # Make predictions
    predictions = model_trainer.predict(X_test_scaled)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'bidder_id': test_features['bidder_id'],
        'prediction': predictions
    })
    
    # Add probabilities if requested
    if include_probabilities:
        probabilities = model_trainer.predict_proba(X_test_scaled)
        results_df['probability_human'] = probabilities[:, 0]
        results_df['probability_bot'] = probabilities[:, 1]
    
    logger.info(f"Predictions completed for {len(results_df)} bidders")
    
    # Log prediction summary
    bot_count = (predictions == 1.0).sum()
    human_count = (predictions == 0.0).sum()
    logger.info(f"Predicted bots: {bot_count} ({bot_count/len(predictions):.1%})")
    logger.info(f"Predicted humans: {human_count} ({human_count/len(predictions):.1%})")
    
    return results_df

def main():
    """Main prediction function"""
    args = parse_arguments()
    logger = setup_logging(args.log_level)
    
    try:
        logger.info("="*60)
        logger.info("BOT DETECTION PREDICTIONS")
        logger.info("="*60)
        
        # Validate model file exists
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Model file not found: {args.model_path}")
        
        # Load test data
        bids_df, test_df = load_test_data(args.bids_file, args.test_file, logger)
        
        # Prepare test features
        test_features = prepare_test_features(bids_df, test_df, logger)
        
        # Make predictions
        predictions_df = make_predictions(
            args.model_path, 
            test_features, 
            args.include_probabilities, 
            logger
        )
        
        # Save predictions
        if args.output_file:
            output_path = args.output_file
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"predictions_{timestamp}.csv"
        
        predictions_df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to: {output_path}")
        
        # Show sample predictions
        logger.info("\nSample predictions:")
        sample_size = min(10, len(predictions_df))
        for idx, row in predictions_df.head(sample_size).iterrows():
            bidder_id = row['bidder_id']
            prediction = "BOT" if row['prediction'] == 1.0 else "HUMAN"
            
            if args.include_probabilities:
                bot_prob = row['probability_bot']
                logger.info(f"  {bidder_id}: {prediction} (bot prob: {bot_prob:.4f})")
            else:
                logger.info(f"  {bidder_id}: {prediction}")
        
        logger.info("="*60)
        logger.info("PREDICTIONS COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        logger.exception("Full error traceback:")
        sys.exit(1)

if __name__ == "__main__":
    main()