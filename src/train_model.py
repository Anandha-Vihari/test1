"""
Model training and evaluation for house price prediction.

This module trains multiple regression models, evaluates them, and saves the best model.
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

try:
    from .data_preprocessing import load_data, preprocess_data
except ImportError:
    from data_preprocessing import load_data, preprocess_data


def get_models():
    """
    Get dictionary of regression models to train.
    
    Returns:
        dict: Dictionary of model name to model instance
    """
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    }
    return models


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluate a model on train and test sets.
    
    Args:
        model: Trained model
        X_train, X_test: Feature matrices
        y_train, y_test: Target vectors
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred)
    }
    
    return metrics


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Train multiple models and evaluate them.
    
    Args:
        X_train, X_test: Feature matrices
        y_train, y_test: Target vectors
        
    Returns:
        tuple: (results_df, trained_models)
    """
    models = get_models()
    results = []
    trained_models = {}
    
    print("\nTraining and evaluating models...")
    print("-" * 80)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Evaluate
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
        metrics['model'] = name
        results.append(metrics)
        
        # Print results
        print(f"  Train RMSE: ₹{metrics['train_rmse']:,.0f} ({metrics['train_rmse']/100000:.2f}L)")
        print(f"  Test RMSE:  ₹{metrics['test_rmse']:,.0f} ({metrics['test_rmse']/100000:.2f}L)")
        print(f"  Test R²:    {metrics['test_r2']:.4f}")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('test_rmse')
    
    return results_df, trained_models


def save_model(model, encoder, columns, output_path, model_name=None, metrics=None):
    """
    Save trained model and associated metadata.
    
    Args:
        model: Trained model
        encoder (dict): Encoder information from preprocessing
        columns (list): Feature column names
        output_path (str): Path to save model
        model_name (str): Name of the model
        metrics (dict): Model performance metrics
    """
    model_artifact = {
        'model': model,
        'encoder': encoder,
        'columns': columns,
        'model_name': model_name,
        'metrics': metrics
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(model_artifact, f)
    
    print(f"\nModel saved to {output_path}")


def load_model(filepath):
    """
    Load trained model and metadata.
    
    Args:
        filepath (str): Path to saved model
        
    Returns:
        dict: Dictionary containing model, encoder, columns, etc.
    """
    with open(filepath, 'rb') as f:
        model_artifact = pickle.load(f)
    return model_artifact


def main():
    """Command-line interface for model training."""
    parser = argparse.ArgumentParser(description='Train house price prediction models')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output', type=str, default='models/model.pkl', help='Output model path')
    parser.add_argument('--target', type=str, default='SalePrice', help='Target column name')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size (default: 0.2)')
    parser.add_argument('--random-state', type=int, default=42, help='Random state for reproducibility')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input}...")
    df = load_data(args.input)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Preprocess
    print("\nPreprocessing data...")
    X, y, encoder, columns = preprocess_data(df, target_col=args.target, is_training=True)
    print(f"Processed data shape: {X.shape}")
    print(f"Target statistics - Mean: ₹{y.mean():,.0f} ({y.mean()/100000:.1f}L), Median: ₹{y.median():,.0f} ({y.median()/100000:.1f}L), Std: ₹{y.std():,.0f}")
    
    # Train/test split
    print(f"\nSplitting data (test size: {args.test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set:  {len(X_test)} samples")
    
    # Train and evaluate models
    results_df, trained_models = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Display results
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print(results_df.to_string(index=False))
    
    # Select best model (lowest test RMSE)
    best_model_name = results_df.iloc[0]['model']
    best_model = trained_models[best_model_name]
    best_metrics = results_df.iloc[0].to_dict()
    
    print("\n" + "=" * 80)
    print(f"BEST MODEL: {best_model_name}")
    print(f"Test RMSE: ₹{best_metrics['test_rmse']:,.0f} (₹{best_metrics['test_rmse']/100000:.2f} Lakhs)")
    print(f"Test R²:   {best_metrics['test_r2']:.4f}")
    print("=" * 80)
    
    # Save best model
    save_model(best_model, encoder, columns, args.output, 
               model_name=best_model_name, metrics=best_metrics)
    
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
