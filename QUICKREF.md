# House Price Prediction - Quick Reference

## Setup (One-time)
```bash
# Install dependencies
pip install -r requirements.txt

# Generate sample data
python generate_sample_data.py

# Train models
python -m src.train_model --input data/house_prices.csv --output models/model.pkl
```

## Make Predictions

### Single Prediction (CLI)
```bash
python -m src.predict \
  --model models/model.pkl \
  --input '{"LotArea":8400,"OverallQual":7,"GrLivArea":1710,"FullBath":2,"BedroomAbvGr":3,"YearBuilt":2003,"Neighborhood":"CollgCr"}'
```

### Batch Prediction (CSV)
```bash
python -m src.predict \
  --model models/model.pkl \
  --input data/new_houses.csv \
  --output data/predictions.csv
```

## Launch Web Application
```bash
streamlit run app/streamlit_app.py --server.port 8501
```

Then open: http://localhost:8501

## Advanced Usage

### Custom Training Parameters
```bash
python -m src.train_model \
  --input data/house_prices.csv \
  --output models/model.pkl \
  --test-size 0.2 \
  --random-state 42 \
  --target SalePrice
```

### Preprocessing Only
```bash
python -m src.data_preprocessing \
  --input data/house_prices.csv \
  --output data/processed.pkl
```

## Model Information
- **Best Model**: Gradient Boosting
- **Test RMSE**: $7,663.71
- **Test R²**: 0.9944
- **Trained on**: 800 samples
- **Models compared**: Linear, Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost

## File Structure
```
src/
  data_preprocessing.py  # Data pipeline
  train_model.py         # Model training
  predict.py             # Predictions
app/
  streamlit_app.py       # Web UI
data/
  house_prices.csv       # Dataset
models/
  model.pkl             # Trained model
```

## Troubleshooting

### Import errors
Make sure you run commands from the project root directory using `-m` flag:
```bash
python -m src.train_model  # ✓ Correct
python src/train_model.py  # ✗ Wrong (import issues)
```

### Missing packages
```bash
pip install -r requirements.txt
```

### Model not found
Train the model first:
```bash
python -m src.train_model --input data/house_prices.csv --output models/model.pkl
```
