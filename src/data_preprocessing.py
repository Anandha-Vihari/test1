"""
Data preprocessing utilities for house price prediction.

This module provides functions to load, clean, and transform housing data
for model training and inference.
"""

import argparse
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def load_data(filepath):
    """
    Load housing data from CSV file.
    
    Args:
        filepath (str): Path to CSV file
        
    Returns:
        pd.DataFrame: Loaded data
    """
    return pd.read_csv(filepath)


def preprocess_data(df, target_col='SalePrice', is_training=True, encoder=None, training_columns=None):
    """
    Preprocess housing data with cleaning, imputation, feature engineering, and encoding.
    
    Steps:
    1. Drop Id columns
    2. Median imputation for numeric features
    3. Fill categorical nulls with "Missing"
    4. Feature engineering (price_per_sqft if applicable)
    5. One-hot encoding for categorical features
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Name of target column
        is_training (bool): Whether this is training data (has target)
        encoder (dict): Pre-fitted encoders for categorical columns (for inference)
        training_columns (list): Column order from training (for inference)
        
    Returns:
        tuple: (X, y, encoder, columns) if training, else (X, encoder, columns)
    """
    df = df.copy()
    
    # 1. Drop Id columns
    id_cols = [col for col in df.columns if 'id' in col.lower()]
    df = df.drop(columns=id_cols, errors='ignore')
    
    # 2. Separate features and target
    if is_training and target_col in df.columns:
        y = df[target_col].copy()
        X = df.drop(columns=[target_col])
    else:
        y = None
        X = df.copy()
    
    # 3. Identify numeric and categorical columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    # 4. Median imputation for numeric features
    for col in numeric_cols:
        if X[col].isnull().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
    
    # 5. Fill categorical nulls with "Missing"
    for col in categorical_cols:
        X[col] = X[col].fillna('Missing')
    
    # 6. Feature engineering: price_per_sqft (only if we have both columns and target)
    if is_training and y is not None and 'GrLivArea' in X.columns:
        # Create feature for training
        X['price_per_sqft'] = y / (X['GrLivArea'] + 1)  # +1 to avoid division by zero
    elif 'GrLivArea' in X.columns and not is_training:
        # For inference, we can't compute price_per_sqft (no target), so we'll add a placeholder
        # This feature won't be very useful for inference, but keeps column consistency
        X['price_per_sqft'] = 0  # Placeholder
    
    # 7. One-hot encoding for categorical features
    if is_training:
        # Fit new encoders
        X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        encoder_info = {
            'categorical_cols': categorical_cols,
            'numeric_cols': numeric_cols
        }
        columns = X_encoded.columns.tolist()
    else:
        # Use existing encoder (from training)
        X_encoded = pd.get_dummies(X, columns=encoder['categorical_cols'], drop_first=True)
        
        # Align columns with training data
        # Add missing columns with zeros
        for col in training_columns:
            if col not in X_encoded.columns:
                X_encoded[col] = 0
        
        # Remove extra columns and reorder to match training
        X_encoded = X_encoded[training_columns]
        encoder_info = encoder
        columns = training_columns
    
    if is_training:
        return X_encoded, y, encoder_info, columns
    else:
        return X_encoded, encoder_info, columns


def save_processed_data(X, y, encoder, columns, output_path):
    """
    Save processed data and metadata to pickle file.
    
    Args:
        X (pd.DataFrame): Processed features
        y (pd.Series): Target variable
        encoder (dict): Encoder information
        columns (list): Column names
        output_path (str): Path to save pickle file
    """
    data = {
        'X': X,
        'y': y,
        'encoder': encoder,
        'columns': columns
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Processed data saved to {output_path}")


def load_processed_data(filepath):
    """
    Load processed data from pickle file.
    
    Args:
        filepath (str): Path to pickle file
        
    Returns:
        dict: Dictionary containing X, y, encoder, and columns
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


def main():
    """Command-line interface for preprocessing."""
    parser = argparse.ArgumentParser(description='Preprocess housing data')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output', type=str, required=True, help='Output pickle file path')
    parser.add_argument('--target', type=str, default='SalePrice', help='Target column name')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input}...")
    df = load_data(args.input)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Preprocess
    print("Preprocessing data...")
    X, y, encoder, columns = preprocess_data(df, target_col=args.target, is_training=True)
    print(f"Processed data shape: {X.shape}")
    
    # Save
    save_processed_data(X, y, encoder, columns, args.output)
    print("Done!")


if __name__ == '__main__':
    main()
