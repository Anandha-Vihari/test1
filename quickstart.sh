#!/bin/bash
# Quick start script for house price prediction project

set -e  # Exit on error

echo "=========================================="
echo "House Price Prediction - Quick Start"
echo "=========================================="
echo

# Step 1: Generate sample data
echo "Step 1: Generating sample data..."
python generate_sample_data.py
echo

# Step 2: Train model
echo "Step 2: Training models (this may take a minute)..."
python -m src.train_model --input data/house_prices.csv --output models/model.pkl
echo

# Step 3: Test prediction
echo "Step 3: Testing single prediction..."
python -m src.predict --model models/model.pkl --input '{"LotArea":8400,"OverallQual":7,"GrLivArea":1710,"FullBath":2,"BedroomAbvGr":3,"YearBuilt":2003,"Neighborhood":"CollgCr"}'
echo

echo "=========================================="
echo "✅ Setup complete!"
echo "=========================================="
echo
echo "Next steps:"
echo "  1. Launch Streamlit app: streamlit run app/streamlit_app.py"
echo "  2. Or make more predictions: python -m src.predict --model models/model.pkl --input <data>"
echo "  3. Check the Readme.md for detailed usage instructions"
echo
