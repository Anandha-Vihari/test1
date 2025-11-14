"""
Prediction utilities for house price prediction.

This module provides functions to load a trained model and make predictions
on new data samples.
"""

import argparse
import json
import pickle
import pandas as pd
from pathlib import Path

try:
    from .data_preprocessing import preprocess_data
    from .train_model import load_model
except ImportError:
    from data_preprocessing import preprocess_data
    from train_model import load_model


def predict_price(model_artifact, input_data):
    """
    Predict house price for new data.
    
    Args:
        model_artifact (dict): Dictionary containing model, encoder, and columns
        input_data (dict or pd.DataFrame): Input features
        
    Returns:
        float or np.ndarray: Predicted price(s)
    """
    # Convert dict to DataFrame if needed
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data
    
    # Preprocess input data using the same pipeline as training
    X_processed, _, _ = preprocess_data(
        df, 
        is_training=False,
        encoder=model_artifact['encoder'],
        training_columns=model_artifact['columns']
    )
    
    # Make prediction
    prediction = model_artifact['model'].predict(X_processed)
    
    return prediction[0] if len(prediction) == 1 else prediction


def predict_from_csv(model_path, input_csv, output_csv=None):
    """
    Make predictions for a CSV file of houses.
    
    Args:
        model_path (str): Path to saved model
        input_csv (str): Path to input CSV file
        output_csv (str, optional): Path to save predictions
        
    Returns:
        pd.DataFrame: DataFrame with predictions
    """
    # Load model
    print(f"Loading model from {model_path}...")
    model_artifact = load_model(model_path)
    print(f"Model: {model_artifact.get('model_name', 'Unknown')}")
    
    # Load input data
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} samples")
    
    # Make predictions
    print("Making predictions...")
    predictions = predict_price(model_artifact, df)
    
    # Add predictions to dataframe
    df['PredictedPrice'] = predictions
    
    # Save if output path specified
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Predictions saved to {output_csv}")
    
    return df


def main():
    """Command-line interface for predictions."""
    parser = argparse.ArgumentParser(description='Predict house prices using trained model')
    parser.add_argument('--model', type=str, required=True, help='Path to saved model (.pkl)')
    parser.add_argument('--input', type=str, required=True, 
                       help='Input data (JSON string or path to CSV file)')
    parser.add_argument('--output', type=str, help='Output CSV file path (for batch predictions)')
    
    args = parser.parse_args()
    
    # Check if input is a file or JSON string
    input_path = Path(args.input)
    
    if input_path.exists() and input_path.suffix == '.csv':
        # Batch prediction from CSV
        df = predict_from_csv(args.model, args.input, args.output)
        
        print("\nPrediction Summary:")
        print(f"  Mean Price: ₹{df['PredictedPrice'].mean():,.0f} (₹{df['PredictedPrice'].mean()/100000:.2f} Lakhs)")
        print(f"  Min Price:  ₹{df['PredictedPrice'].min():,.0f} (₹{df['PredictedPrice'].min()/100000:.2f} Lakhs)")
        print(f"  Max Price:  ₹{df['PredictedPrice'].max():,.0f} (₹{df['PredictedPrice'].max()/100000:.2f} Lakhs)")
        
        print("\nFirst 5 predictions:")
        print(df[['PredictedPrice']].head().to_string())
        
    else:
        # Single prediction from JSON
        try:
            input_data = json.loads(args.input)
        except json.JSONDecodeError:
            print("Error: Input must be valid JSON string or path to CSV file")
            return
        
        # Load model
        print(f"Loading model from {args.model}...")
        model_artifact = load_model(args.model)
        print(f"Model: {model_artifact.get('model_name', 'Unknown')}")
        
        # Make prediction
        print("\nMaking prediction...")
        print(f"Input features: {input_data}")
        
        prediction = predict_price(model_artifact, input_data)
        
        print(f"\nPredicted Price: ₹{prediction:,.0f}")
        print(f"In Lakhs: ₹{prediction/100000:.2f} Lakhs")
        
        # Show model metrics if available
        if 'metrics' in model_artifact and model_artifact['metrics']:
            metrics = model_artifact['metrics']
            print(f"\nModel Performance (Test Set):")
            if 'test_rmse' in metrics:
                rmse = metrics.get('test_rmse')
                print(f"  RMSE: ₹{rmse:,.0f} (₹{rmse/100000:.2f} Lakhs)")
            if 'test_r2' in metrics:
                print(f"  R²:   {metrics.get('test_r2'):.4f}")


if __name__ == '__main__':
    main()
