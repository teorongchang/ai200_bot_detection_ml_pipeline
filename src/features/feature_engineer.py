# src/features/feature_engineer.py
"""
Feature engineering module matching my exact notebook implementation.
Creates the exact features that won 1st place in the Kaggle competition.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')

class BotDetectionFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature engineer that recreates my exact notebook feature engineering process.
    Implements cells 16-21 from my winning notebook.
    """
    
    def __init__(self):
        """Initialize feature engineer"""
        self.feature_names_ = []
    
    def fit(self, X=None, y=None):
        """Fit method for sklearn compatibility"""
        return self
    
    def transform(self, data_tuple: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Transform data exactly as I did in my notebook.
        
        Parameters:
        data_tuple: Tuple containing (bids_df, train_df, test_df)
        
        Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_processed, test_processed)
        """
        bids_df, train_df, test_df = data_tuple
        
        # Fill missing country values with mode (cell 5)
        bids_df = self._clean_bids_data(bids_df)
        
        # Create bidder features (cells 16-21)
        bidder_features = self._create_bidder_features(bids_df)
        
        # Merge and prepare final datasets (cell 17)
        train_processed, test_processed = self._merge_and_prepare(train_df, test_df, bidder_features)
        
        return train_processed, test_processed
    
    def _clean_bids_data(self, bids_df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean bids data exactly as in notebook cell 5.
        """
        # Fill country missing values with mode
        bids_df['country'] = bids_df['country'].fillna(bids_df['country'].mode()[0])
        return bids_df
    
    def _create_bidder_features(self, bids_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create bidder features exactly as in notebook cells 16-21.
        This is the core feature engineering that won the competition.
        """
        # Cell 16: Basic nunique features
        bids_df_cols = ['bid_id', 'bidder_id', 'auction', 'merchandise', 'device', 'time', 'country', 'ip', 'url']
        
        # groupby bidder to get nunique of each feature
        bidder_unique = bids_df[bids_df_cols].groupby("bidder_id").nunique()
        
        # drop bidder_id nunique
        bidder_unique = bidder_unique.drop(columns='bidder_id', errors='ignore')
        
        # reset index
        bidder_unique = bidder_unique.reset_index()
        
        # rename columns (exactly as in my notebook)
        bidder_unique.columns = ['bidder_id', 'number_of_bids', 'number_of_auctions', 'number_of_merchandise',
                                 'number_of_devices', 'time_nunique', 'country_nunique', 'ip_nunique', 'url_nunique']
        
        # Cell 19: Mean/Max devices per auction analysis
        describe_devices_per_auction = bids_df.groupby(["bidder_id", 'auction'])['device'].nunique().groupby('bidder_id').describe()
        describe_devices_per_auction = describe_devices_per_auction.reset_index()
        describe_devices_per_auction = describe_devices_per_auction[['bidder_id', 'mean', 'max']]
        describe_devices_per_auction.columns = ['bidder_id', 'mean_devices_per_auction', 'max_devices_per_auction']
        
        # Merge devices analysis
        bidder_unique = bidder_unique.merge(describe_devices_per_auction, on='bidder_id', how='left')
        
        # Cell 21: Mean/Max IP per auction analysis
        describe_ip_per_auction = bids_df.groupby(["bidder_id", 'auction'])['ip'].nunique().groupby('bidder_id').describe()
        describe_ip_per_auction = describe_ip_per_auction.reset_index()
        describe_ip_per_auction = describe_ip_per_auction[['bidder_id', 'mean', 'max']]
        describe_ip_per_auction.columns = ['bidder_id', 'mean_ip_per_auction', 'max_ip_per_auction']
        
        # Merge IP analysis
        bidder_unique = bidder_unique.merge(describe_ip_per_auction, on='bidder_id', how='left')
        
        # URL analysis (following same pattern)
        describe_url_per_auction = bids_df.groupby(["bidder_id", 'auction'])['url'].nunique().groupby('bidder_id').describe()
        describe_url_per_auction = describe_url_per_auction.reset_index()
        describe_url_per_auction = describe_url_per_auction[['bidder_id', 'mean', 'max']]
        describe_url_per_auction.columns = ['bidder_id', 'mean_url_per_auction', 'max_url_per_auction']
        
        # Merge URL analysis
        bidder_unique = bidder_unique.merge(describe_url_per_auction, on='bidder_id', how='left')
        
        # Time analysis
        describe_time_per_auction = bids_df.groupby(["bidder_id", 'auction'])['time'].nunique().groupby('bidder_id').describe()
        describe_time_per_auction = describe_time_per_auction.reset_index()
        describe_time_per_auction = describe_time_per_auction[['bidder_id', 'mean', 'max']]
        describe_time_per_auction.columns = ['bidder_id', 'mean_time_per_auction', 'max_time_per_auction']
        
        # Merge time analysis
        bidder_unique = bidder_unique.merge(describe_time_per_auction, on='bidder_id', how='left')
        
        return bidder_unique
    
    def _merge_and_prepare(self, train_df: pd.DataFrame, test_df: pd.DataFrame, bidder_features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Merge and prepare final datasets exactly as in notebook cell 17.
        """
        # Merge train data with features and drop unnecessary columns
        train = train_df.merge(bidder_features, on='bidder_id', how='left')
        train = train.drop(columns=['payment_account', 'address'], errors='ignore')
        
        # Merge test data with features and drop unnecessary columns
        test = test_df.merge(bidder_features, on='bidder_id', how='left')
        test = test.drop(columns=['payment_account', 'address'], errors='ignore')
        
        # Store feature names
        self.feature_names_ = train.columns.tolist()
        
        return train, test
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names created"""
        return self.feature_names_


def engineer_features(bids_df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main function to engineer features exactly as in my notebook.
    Recreates the feature engineering process from cells 16-21.
    
    Parameters:
    bids_df: Bids dataframe
    train_df: Train dataframe
    test_df: Test dataframe
    
    Returns:
    Tuple[pd.DataFrame, pd.DataFrame]: (train_with_features, test_with_features)
    """
    # Initialize feature engineer
    feature_engineer = BotDetectionFeatureEngineer()
    
    # Apply feature engineering
    train_processed, test_processed = feature_engineer.fit_transform((bids_df, train_df, test_df))
    
    print(f"Feature engineering completed!")
    print(f"Train shape: {train_processed.shape}")
    print(f"Test shape: {test_processed.shape}")
    print(f"Features created: {len(feature_engineer.get_feature_names())}")
    
    return train_processed, test_processed


# Additional helper functions matching my notebook approach
def load_and_engineer_features():
    """
    Complete workflow matching my notebook cells 2-21.
    Loads data and creates all features in one go.
    """
    # Load data (cells 2-3)
    bids_df = pd.read_csv('data/bids.csv')
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    
    print(f"Loaded bids_df: {bids_df.shape}")
    print(f"Loaded train_df: {train_df.shape}")
    print(f"Loaded test_df: {test_df.shape}")
    
    # Engineer features
    train_processed, test_processed = engineer_features(bids_df, train_df, test_df)
    
    return train_processed, test_processed