# src/data/data_loader.py
"""
Data loading utilities matching my exact notebook implementation.
Loads bids.csv, train.csv, test.csv and handles country missing values.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import os

def load_competition_data():
    """
    Load data exactly as I did in my notebook.
    Returns the three main dataframes: bids_df, train_df, test_df
    """
    # Load data exactly as in notebook cell 2
    bids_df = pd.read_csv('data/bids.csv')
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    
    print(f"Loaded bids_df: {bids_df.shape}")
    print(f"Loaded train_df: {train_df.shape}")
    print(f"Loaded test_df: {test_df.shape}")
    
    return bids_df, train_df, test_df

def clean_bids_data(bids_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean bids data exactly as I did in my notebook.
    Fill missing country values with mode (cell 5).
    """
    # Check missing values (as in cell 4)
    print("Missing values in bids_df:")
    print(bids_df.isnull().sum())
    
    # Fill country missing values with mode (cell 5)
    bids_df['country'] = bids_df['country'].fillna(bids_df['country'].mode()[0])
    
    print(f"Filled missing country values with mode: {bids_df['country'].mode()[0]}")
    
    return bids_df

def create_bidder_features(bids_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create bidder features exactly as I did in notebook cells 16-21.
    This recreates my exact feature engineering process.
    """
    print("Creating bidder features...")
    
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
    
    print(f"Created basic features: {bidder_unique.shape}")
    
    # Cell 19: Mean/Max devices per auction analysis
    describe_devices_per_auction = bids_df.groupby(["bidder_id", 'auction'])['device'].nunique().groupby('bidder_id').describe()
    describe_devices_per_auction = describe_devices_per_auction.reset_index()
    describe_devices_per_auction = describe_devices_per_auction[['bidder_id', 'mean', 'max']]
    describe_devices_per_auction.columns = ['bidder_id', 'mean_devices_per_auction', 'max_devices_per_auction']
    
    # Merge devices analysis
    bidder_unique = bidder_unique.merge(describe_devices_per_auction, on='bidder_id', how='left')
    
    # Continue with IP analysis (cell 21 pattern)
    describe_ip_per_auction = bids_df.groupby(["bidder_id", 'auction'])['ip'].nunique().groupby('bidder_id').describe()
    describe_ip_per_auction = describe_ip_per_auction.reset_index()
    describe_ip_per_auction = describe_ip_per_auction[['bidder_id', 'mean', 'max']]
    describe_ip_per_auction.columns = ['bidder_id', 'mean_ip_per_auction', 'max_ip_per_auction']
    
    # Merge IP analysis
    bidder_unique = bidder_unique.merge(describe_ip_per_auction, on='bidder_id', how='left')
    
    # URL analysis
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
    
    print(f"Final features created: {bidder_unique.shape}")
    
    return bidder_unique

def prepare_train_test_data(train_df: pd.DataFrame, test_df: pd.DataFrame, bidder_features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare train and test data exactly as I did in my notebook (cell 17).
    """
    # Merge with bidder features and drop payment_account, address (as in cell 17)
    train = train_df.merge(bidder_features, on='bidder_id', how='left')
    train = train.drop(columns=['payment_account', 'address'], errors='ignore')
    
    test = test_df.merge(bidder_features, on='bidder_id', how='left')
    test = test.drop(columns=['payment_account', 'address'], errors='ignore')
    
    print(f"Train data prepared: {train.shape}")
    print(f"Test data prepared: {test.shape}")
    
    return train, test

def load_and_prepare_data():
    """
    Complete data loading and preparation pipeline matching my notebook.
    This recreates cells 2-21 from my notebook.
    """
    # Load raw data (cells 2-3)
    bids_df, train_df, test_df = load_competition_data()
    
    # Clean bids data (cells 4-5)
    bids_df = clean_bids_data(bids_df)
    
    # Create features (cells 16-21)
    bidder_features = create_bidder_features(bids_df)
    
    # Prepare final datasets (cell 17)
    train, test = prepare_train_test_data(train_df, test_df, bidder_features)
    
    print("\nData preparation completed!")
    print(f"Final train shape: {train.shape}")
    print(f"Final test shape: {test.shape}")
    
    return train, test, bids_df

# Quick loading function
def load_data_for_training():
    """Quick function to load training data"""
    train, test, bids_df = load_and_prepare_data()
    return train, test