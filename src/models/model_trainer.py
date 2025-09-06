# src/models/model_trainer.py
"""
Model training module matching my exact notebook implementation.
Uses ExtraTreesClassifier with hyperparameter tuning as in my winning solution.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report
from typing import Dict, Any, Tuple, Optional
import joblib
import os
from datetime import datetime
from config.config import config

class BotDetectionModelTrainer:
    """
    Model trainer matching my exact notebook approach.
    Uses ExtraTreesClassifier as my winning model with hyperparameter tuning.
    """
    
    def __init__(self):
        """Initialize with the same parameters I used in my notebook"""
        # ExtraTreesClassifier parameters from my winning solution
        self.base_params = {
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Hyperparameter grid exactly as I used in my notebook
        self.param_grid = {
            'n_estimators': [120, 300, 500, 800, 1200],
            'max_depth': [5, 8, 15, 25, 30, None],
            'min_samples_split': [1, 2, 5, 10, 15, 100],
            'min_samples_leaf': [1, 2, 5, 10],
            'max_features': ['log2', 'sqrt', None]
        }
        
        self.model = None
        self.best_model = None
        self.training_results = {}
        
    def train_with_hyperparameter_tuning(self, X: np.ndarray, y: np.ndarray) -> 'BotDetectionModelTrainer':
        """
        Train model with hyperparameter tuning exactly as I did in my notebook.
        Uses RandomizedSearchCV as mentioned in my README.
        """
        print("Starting hyperparameter tuning for ExtraTreesClassifier...")
        
        # Create base model
        base_model = ExtraTreesClassifier(**self.base_params)
        
        # Set up RandomizedSearchCV exactly as I used it
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=self.param_grid,
            n_iter=50,  # Number of parameter combinations to try
            cv=5,       # 5-fold cross-validation
            scoring='roc_auc',  # ROC-AUC as main metric (as mentioned in my README)
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        # Fit the randomized search
        random_search.fit(X, y)
        
        # Store the best model
        self.best_model = random_search.best_estimator_
        self.model = self.best_model
        
        # Store training results
        self.training_results = {
            'best_score': random_search.best_score_,
            'best_params': random_search.best_params_,
            'cv_results': random_search.cv_results_
        }
        
        print(f"Hyperparameter tuning completed!")
        print(f"Best CV ROC-AUC score: {random_search.best_score_:.6f}")
        print(f"Best parameters: {random_search.best_params_}")
        
        return self
    
    def train_with_default_params(self, X: np.ndarray, y: np.ndarray) -> 'BotDetectionModelTrainer':
        """
        Train model with default parameters (faster option).
        """
        print("Training ExtraTreesClassifier with default parameters...")
        
        # Default parameters that worked well in my notebook
        default_params = {
            'n_estimators': 300,
            'max_depth': 25,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Create and train model
        self.model = ExtraTreesClassifier(**default_params)
        self.model.fit(X, y)
        self.best_model = self.model
        
        # Evaluate with cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='roc_auc')
        
        self.training_results = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist(),
            'model_params': default_params
        }
        
        print(f"Training completed!")
        print(f"CV ROC-AUC: {cv_scores.mean():.6f} (+/- {cv_scores.std():.6f})")
        
        return self
    
    def evaluate_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate multiple models as I did in my notebook.
        Compare ExtraTreesClassifier with other models to show why it won.
        """
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
        from lightgbm import LGBMClassifier
        from xgboost import XGBClassifier
        from catboost import CatBoostClassifier
        
        print("Evaluating multiple models as in my notebook...")
        
        # Models I tested in my notebook (as mentioned in README)
        models = {
            'ExtraTreesClassifier': ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1),
            'RandomForestClassifier': RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
            'LGBMClassifier': LGBMClassifier(n_estimators=300, random_state=42, n_jobs=-1, verbose=-1),
            'XGBClassifier': XGBClassifier(n_estimators=300, random_state=42, n_jobs=-1, eval_metric='logloss'),
            'HistGradientBoostingClassifier': HistGradientBoostingClassifier(random_state=42),
            'CatBoostClassifier': CatBoostClassifier(n_estimators=300, random_state=42, verbose=False),
            'GradientBoostingClassifier': GradientBoostingClassifier(n_estimators=300, random_state=42)
        }
        
        model_scores = {}
        
        for name, model in models.items():
            # Cross-validation with ROC-AUC
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
            model_scores[name] = cv_scores.mean()
            print(f"{name}: {cv_scores.mean():.6f}")
        
        # Sort by performance (as shown in my README)
        sorted_scores = dict(sorted(model_scores.items(), key=lambda x: x[1], reverse=True))
        
        print("\nModel rankings (as achieved in my notebook):")
        for i, (name, score) in enumerate(sorted_scores.items(), 1):
            print(f"{i}. {name}: {score:.6f}")
        
        return sorted_scores
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using trained model"""
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using trained model"""
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, feature_names: Optional[List[str]] = None, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from trained model.
        Returns top features as shown in my notebook results.
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        # Get feature importance
        importance_values = self.model.feature_importances_
        
        # Create feature names if not provided
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importance_values))]
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_values
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def save_model(self, filepath: str) -> str:
        """Save trained model to disk"""
        if self.model is None:
            raise ValueError("No trained model to save")
        
        # Create directory if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model and results
        model_data = {
            'model': self.model,
            'training_results': self.training_results,
            'feature_importance': self.model.feature_importances_.tolist() if hasattr(self.model, 'feature_importances_') else None
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
        
        return filepath
    
    def load_model(self, filepath: str) -> 'BotDetectionModelTrainer':
        """Load trained model from disk"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.best_model = self.model
        self.training_results = model_data.get('training_results', {})
        
        print(f"Model loaded from {filepath}")
        return self
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training results"""
        return self.training_results.copy()


def train_bot_detection_model(X: np.ndarray, y: np.ndarray, use_hyperparameter_tuning: bool = True) -> BotDetectionModelTrainer:
    """
    Train bot detection model exactly as I did in my notebook.
    
    Parameters:
    X: Feature matrix
    y: Target vector
    use_hyperparameter_tuning: Whether to use RandomizedSearchCV
    
    Returns:
    BotDetectionModelTrainer: Trained model
    """
    trainer = BotDetectionModelTrainer()
    
    if use_hyperparameter_tuning:
        trainer.train_with_hyperparameter_tuning(X, y)
    else:
        trainer.train_with_default_params(X, y)
    
    return trainer


def compare_models(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Compare multiple models as I did in my notebook.
    Shows why ExtraTreesClassifier performed best.
    """
    trainer = BotDetectionModelTrainer()
    return trainer.evaluate_models(X, y)