# House Price Prediction (mini-project)

## Overview
This project builds a regression pipeline to predict residential house prices from structured housing data (area, bedrooms, bathrooms, location, age, etc.). It demonstrates the full ML workflow: data loading, cleaning, feature engineering, model training & evaluation, and exporting a production-ready model.

## Repository Structure
```
.
├── src/
│   ├── data_preprocessing.py   # Data loading and preprocessing utilities
│   ├── train_model.py          # Model training, evaluation, and model export
│   └── predict.py              # Helper to load model and predict on new samples
├── app/
│   └── streamlit_app.py        # Simple Streamlit front-end for quick predictions
├── data/                       # Place your dataset here (not checked in)
├── models/                     # Saved models (model.pkl) after training
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
└── Readme.md                   # This file
```

## Features

### Data Preprocessing (`data_preprocessing.py`)
- Automatic handling of missing values (median imputation for numeric, "Missing" for categorical)
- Feature engineering (price per square foot)
- One-hot encoding for categorical features
- Consistent preprocessing pipeline for training and inference

### Model Training (`train_model.py`)
- Trains multiple regression models:
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - Random Forest
  - Gradient Boosting
  - XGBoost
- Comprehensive evaluation metrics (RMSE, MAE, R²)
- Automatic selection and export of best model
- Saves model with metadata (encoder, feature columns, metrics)

### Prediction (`predict.py`)
- Single prediction from JSON input
- Batch prediction from CSV files
- Automatic preprocessing using saved encoder
- CLI and programmatic interfaces

### Web Application (`app/streamlit_app.py`)
- User-friendly interface for predictions
- Single prediction with interactive inputs
- Batch prediction from CSV upload
- Real-time model performance metrics
- Download predictions as CSV

## Quickstart

### 1. Create a Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows PowerShell
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Your Dataset
Place your CSV dataset in `data/` (e.g., `data/house_prices.csv`). 

**Expected columns:**
- Target: `SalePrice` (or update code accordingly)
- Features: `LotArea`, `OverallQual`, `GrLivArea`, `FullBath`, `BedroomAbvGr`, `YearBuilt`, `Neighborhood`, etc.

### 4. Train Models
```bash
python -m src.train_model --input data/house_prices.csv --output models/model.pkl
```

This will:
- Preprocess the data
- Train 6 different regression models
- Evaluate and compare them
- Save the best model to `models/model.pkl`

**Optional flags:**
- `--target`: Specify target column name (default: `SalePrice`)
- `--test-size`: Test set fraction (default: 0.2)
- `--random-state`: Random seed for reproducibility (default: 42)

### 5. Make Predictions

#### Single Prediction (CLI)
```bash
python -m src.predict --model models/model.pkl --input '{"LotArea":8400,"OverallQual":7,"GrLivArea":1710,"FullBath":2,"BedroomAbvGr":3,"YearBuilt":2003,"Neighborhood":"CollgCr"}'
```

#### Batch Prediction (CSV)
```bash
python -m src.predict --model models/model.pkl --input data/new_houses.csv --output data/predictions.csv
```

### 6. Launch Web Application
```bash
streamlit run app/streamlit_app.py --server.port 8501
```

Then open your browser to `http://localhost:8501`

## Advanced Usage

### Preprocessing Only
```bash
python -m src.data_preprocessing --input data/house_prices.csv --output data/processed.pkl
```

### Programmatic Use

```python
from src.train_model import load_model
from src.predict import predict_price

# Load model
model_artifact = load_model('models/model.pkl')

# Make prediction
input_data = {
    'LotArea': 8400,
    'OverallQual': 7,
    'GrLivArea': 1710,
    'FullBath': 2,
    'BedroomAbvGr': 3,
    'YearBuilt': 2003,
    'Neighborhood': 'CollgCr'
}

price = predict_price(model_artifact, input_data)
print(f"Predicted price: ${price:,.2f}")
```

## Technical Details

### Preprocessing Pipeline
1. **Drop ID columns**: Removes any column with 'id' in the name
2. **Missing value imputation**:
   - Numeric: Filled with median
   - Categorical: Filled with "Missing"
3. **Feature engineering**: Creates `price_per_sqft` feature
4. **One-hot encoding**: Converts categorical variables to numeric

### Model Evaluation
Models are evaluated on:
- **RMSE** (Root Mean Squared Error): Primary metric for model selection
- **MAE** (Mean Absolute Error): Average prediction error
- **R²** (Coefficient of Determination): Explained variance

### Model Artifacts
The saved model file (`model.pkl`) contains:
- Trained model object
- Preprocessing encoder information
- Feature column names (for consistency)
- Model name and performance metrics

## Extensions & Ideas

- [ ] Convert preprocessing to sklearn `Pipeline` with `ColumnTransformer`
- [ ] Add hyperparameter tuning (`GridSearchCV` or `RandomizedSearchCV`)
- [ ] Implement SHAP for model explainability
- [ ] Add data validation with Great Expectations
- [ ] Create REST API with FastAPI
- [ ] Dockerize application for deployment
- [ ] Add logging and monitoring
- [ ] Implement A/B testing framework
- [ ] Add feature importance visualization

## Dataset Recommendations

This project is designed to work with datasets similar to the [Kaggle House Prices dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques), but can be adapted to any structured housing dataset.

**Minimum required columns:**
- Numeric: `LotArea`, `GrLivArea`, `YearBuilt`
- Categorical: `Neighborhood`
- Target: `SalePrice`

## Troubleshooting

### Model not found error
Make sure you've trained a model first:
```bash
python -m src.train_model --input data/house_prices.csv --output models/model.pkl
```

### Import errors in Streamlit
The app automatically adds the `src/` directory to the Python path. If you still have issues, run from the project root.

### Column mismatch during prediction
Ensure your prediction data has the same feature columns as the training data. The preprocessing pipeline will handle missing or extra columns automatically.

## License
MIT License - Feel free to use this project for learning and development.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
