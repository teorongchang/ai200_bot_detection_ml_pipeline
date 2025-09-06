# src/models/preprocessor.py
"""
Preprocessing pipeline matching my exact notebook implementation.
Simple SMOTE + StandardScaler pipeline as I used to win the competition.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from typing import Tuple, List, Optional

class BotDetectionPreprocessor:
    """
    Preprocessor matching my exact notebook implementation.
    Simple pipeline: SMOTE -> StandardScaler (as mentioned in my README)
    """
    
    def __init__(self):
        """Initialize exactly as I did in my notebook"""
        self.pipeline = None
        self.feature_columns = None
        self.is_fitted = False
        
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features exactly as I did in my notebook.
        Remove bidder_id, payment_account, address, outcome.
        """
        # Columns to exclude (as I did in my notebook)
        exclude_cols = ['bidder_id', 'payment_account', 'address', 'outcome']
        
        # Get feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        return df[feature_cols]
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit preprocessing pipeline exactly as in my notebook.
        Uses imblearn.Pipeline with SMOTE and StandardScaler.
        """
        # Prepare features
        X_features = self._prepare_features(X)
        self.feature_columns = X_features.columns.tolist()
        
        # Create pipeline exactly as I used in my notebook
        # SMOTE for class imbalance + StandardScaler for feature standardization
        self.pipeline = Pipeline([
            ('smote', SMOTE(random_state=42)),
            ('scaler', StandardScaler())
        ])
        
        # Fit the pipeline
        self.pipeline.fit(X_features, y)
        self.is_fitted = True
        
        print(f"Preprocessing pipeline fitted on {len(X_features)} samples")
        print(f"Features used: {len(self.feature_columns)}")
        
        return self
    
    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Transform data using fitted pipeline.
        """
        if not self.is_fitted:
            raise ValueError("Must fit preprocessor before transform")
        
        # Prepare features
        X_features = self._prepare_features(X)[self.feature_columns]
        
        if y is not None:
            # Training mode: apply SMOTE + scaling
            X_transformed, y_transformed = self.pipeline.transform(X_features, y)
            print(f"SMOTE applied: {len(X_features)} -> {len(X_transformed)} samples")
            return X_transformed, y_transformed
        else:
            # Prediction mode: only scaling (no SMOTE)
            X_scaled = self.pipeline.named_steps['scaler'].transform(X_features)
            return X_scaled, None
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit and transform in one step as I did in my notebook.
        """
        return self.fit(X, y).transform(X, y)
    
    def get_feature_names(self) -> List[str]:
        """Get feature names used in preprocessing"""
        return self.feature_columns if self.feature_columns else []

def preprocess_data(train_df: pd.DataFrame):
    """
    Preprocess data exactly as I did in my notebook.
    Separates features and target, applies SMOTE + StandardScaler.
    """
    # Separate features and target (as I did in my notebook)
    X = train_df.drop('outcome', axis=1)
    y = train_df['outcome']
    
    # Create and apply preprocessor
    preprocessor = BotDetectionPreprocessor()
    X_processed, y_processed = preprocessor.fit_transform(X, y)
    
    print("Preprocessing completed:")
    print(f"Original class distribution: {np.bincount(y.astype(int))}")
    print(f"After SMOTE class distribution: {np.bincount(y_processed.astype(int))}")
    
    return X_processed, y_processed, preprocessor