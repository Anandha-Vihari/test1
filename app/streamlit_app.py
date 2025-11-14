"""
Streamlit web application for house price prediction.

This app provides a user-friendly interface to predict house prices
using a trained machine learning model.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src directory to path to import modules
src_path = str(Path(__file__).parent.parent / 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import after adding to path
import train_model
import predict


# Page configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Title and description
st.title("🏠 House Price Prediction - India")
st.markdown("""
This app predicts house prices in Indian cities based on various features like area, number of bedrooms, 
location, and other property characteristics. Prices are displayed in Indian Rupees (₹).
""")

# Sidebar for model loading
st.sidebar.header("Model Configuration")

# Get absolute path to models directory
default_model_path = str(Path(__file__).parent.parent / "models" / "model.pkl")

model_path = st.sidebar.text_input(
    "Model Path", 
    value=default_model_path,
    help="Path to the trained model file"
)

# Load model
@st.cache_resource
def load_model_cached(path):
    """Load and cache the model."""
    try:
        model_artifact = train_model.load_model(path)
        return model_artifact, None
    except Exception as e:
        return None, str(e)


model_artifact, error = load_model_cached(model_path)

if error:
    st.sidebar.error(f"Error loading model: {error}")
    st.warning("⚠️ Please train a model first using the training script.")
    st.code("python -m src.train_model --input data/house_prices.csv --output models/model.pkl")
    st.stop()
else:
    model_name = model_artifact.get('model_name', 'Unknown')
    st.sidebar.success(f"✅ Model loaded: {model_name}")
    
    # Display model metrics if available
    if 'metrics' in model_artifact and model_artifact['metrics']:
        metrics = model_artifact['metrics']
        st.sidebar.subheader("Model Performance")
        rmse = metrics.get('test_rmse', 0)
        st.sidebar.metric("Test RMSE", f"₹{rmse:,.0f} ({rmse/100000:.2f}L)")
        st.sidebar.metric("Test R²", f"{metrics.get('test_r2', 0):.4f}")

# Main content - two tabs
tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

# Tab 1: Single Prediction
with tab1:
    st.header("Enter House Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Property Size")
        lot_area = st.number_input("Plot Area (sq ft)", min_value=100, max_value=10000, value=1200, step=50, 
                                   help="Typical: 800-2000 sq ft for apartments, 1500-3000 for independent houses")
        gr_liv_area = st.number_input("Built-up Area (sq ft)", min_value=100, max_value=10000, value=1500, step=50,
                                      help="Typical: 1000-1500 (2BHK), 1500-2000 (3BHK), 2000+ (4BHK)")
        
        # Show warnings for unusual sizes
        if gr_liv_area < 500:
            st.warning("⚠️ Very small property (<500 sq ft) - predictions may be less accurate")
        elif gr_liv_area > 5000:
            st.info("ℹ️ Large luxury property (>5000 sq ft) - prediction based on available data")
        
    with col2:
        st.subheader("Property Details")
        overall_qual = st.slider("Overall Quality (1-10)", min_value=1, max_value=10, value=7)
        year_built = st.number_input("Year Built", min_value=1980, max_value=2030, value=2010)
        full_bath = st.number_input("Bathrooms", min_value=0, max_value=5, value=2)
        bedroom_abv_gr = st.number_input("Bedrooms (BHK)", min_value=0, max_value=10, value=3)
        
    with col3:
        st.subheader("Location & More")
        
        # City and neighborhood mapping
        city_neighborhoods = {
            'Bangalore': ['Koramangala', 'Indiranagar', 'Whitefield', 'HSR Layout', 'Jayanagar', 'Marathahalli', 'Electronic City', 'Yelahanka'],
            'Mumbai': ['Bandra', 'Andheri', 'Powai', 'Worli', 'Juhu', 'Goregaon', 'Thane', 'Navi Mumbai'],
            'Delhi': ['Dwarka', 'Rohini', 'Vasant Kunj', 'Greater Kailash', 'Saket', 'Noida', 'Gurgaon', 'Hauz Khas'],
            'Hyderabad': ['Gachibowli', 'Banjara Hills', 'Jubilee Hills', 'Kondapur', 'Madhapur', 'HITEC City', 'Kukatpally', 'Miyapur'],
            'Pune': ['Koregaon Park', 'Hinjewadi', 'Wakad', 'Baner', 'Kothrud', 'Viman Nagar', 'Magarpatta', 'Kalyani Nagar'],
            'Chennai': ['Adyar', 'Anna Nagar', 'T Nagar', 'Velachery', 'OMR', 'Nungambakkam', 'Porur', 'Tambaram']
        }
        
        city = st.selectbox("City", list(city_neighborhoods.keys()), index=0)
        neighborhood = st.selectbox("Neighborhood", city_neighborhoods[city], index=0)
        
        # Optional additional features
        with st.expander("Additional Features (Optional)"):
            garage_cars = st.number_input("Garage Capacity (cars)", min_value=0, max_value=5, value=2)
            total_bsmt_sf = st.number_input("Basement Area (sq ft)", min_value=0, value=900)
            fireplaces = st.number_input("Fireplaces", min_value=0, max_value=5, value=0)
    
    # Create input dictionary
    input_data = {
        'City': city,
        'Neighborhood': neighborhood,
        'LotArea': lot_area,
        'OverallQual': overall_qual,
        'GrLivArea': gr_liv_area,
        'FullBath': full_bath,
        'BedroomAbvGr': bedroom_abv_gr,
        'YearBuilt': year_built,
        'GarageCars': garage_cars,
        'TotalBsmtSF': total_bsmt_sf,
        'Fireplaces': fireplaces
    }
    
    # Predict button
    if st.button("🔮 Predict Price", type="primary"):
        try:
            with st.spinner("Making prediction..."):
                prediction = predict.predict_price(model_artifact, input_data)
            
            # Display result
            st.success("Prediction Complete!")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    label="Estimated House Price",
                    value=f"₹{prediction:,.0f}",
                    delta=None
                )
            with col2:
                st.metric(
                    label="In Lakhs/Crores",
                    value=f"₹{prediction/100000:.2f}L" if prediction < 10000000 else f"₹{prediction/10000000:.2f} Cr",
                    delta=None
                )
            
            # Additional info with price breakdown
            if gr_liv_area > 0:
                price_per_sqft = prediction / gr_liv_area
                st.info(f"💡 **Price per sq ft:** ₹{price_per_sqft:,.0f}")
                
                # Market context
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Built-up", f"{gr_liv_area:,} sq ft")
                with col2:
                    st.metric("Plot Area", f"{lot_area:,} sq ft")
                with col3:
                    bhk = bedroom_abv_gr
                    st.metric("Configuration", f"{bhk}BHK, {full_bath}Bath")
                
                # Price range context
                st.markdown("---")
                st.markdown(f"**Market Context for {city} - {neighborhood}:**")
                estimated_range_low = prediction * 0.85
                estimated_range_high = prediction * 1.15
                st.write(f"Expected price range: ₹{estimated_range_low/100000:.1f}L - ₹{estimated_range_high/100000:.1f}L")
                st.caption(f"Based on {city} market rates, {neighborhood} location premium, and {year_built} construction")
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

# Tab 2: Batch Prediction
with tab2:
    st.header("Batch Prediction from CSV")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file with house features",
        type=['csv'],
        help="Upload a CSV file with the same features as used in training"
    )
    
    if uploaded_file is not None:
        try:
            # Read uploaded file
            df = pd.read_csv(uploaded_file)
            
            st.subheader("Preview of uploaded data")
            st.dataframe(df.head(), use_container_width=True)
            
            st.info(f"📊 Loaded {len(df)} rows, {len(df.columns)} columns")
            
            # Predict button
            if st.button("🔮 Predict All Prices", type="primary", key="batch_predict"):
                with st.spinner("Making predictions..."):
                    predictions = predict.predict_price(model_artifact, df)
                    df['PredictedPrice'] = predictions
                
                st.success("Predictions complete!")
                
                # Display statistics
                col1, col2, col3 = st.columns(3)
                col1.metric("Mean Price", f"₹{predictions.mean():,.0f} ({predictions.mean()/100000:.1f}L)")
                col2.metric("Min Price", f"₹{predictions.min():,.0f} ({predictions.min()/100000:.1f}L)")
                col3.metric("Max Price", f"₹{predictions.max():,.0f} ({predictions.max()/100000:.1f}L)")
                
                # Display results
                st.subheader("Results")
                st.dataframe(df, use_container_width=True)
                
                # Download button
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv,
                    file_name="house_price_predictions.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>House Price Prediction System | Built with Streamlit and scikit-learn</p>
</div>
""", unsafe_allow_html=True)
