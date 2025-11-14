"""
Sample data generator for testing the house price prediction pipeline.

This script generates synthetic house price data for testing purposes.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 1000

# Cities and their neighborhoods
cities = ['Bangalore', 'Mumbai', 'Delhi', 'Hyderabad', 'Pune', 'Chennai']
neighborhoods_by_city = {
    'Bangalore': ['Koramangala', 'Indiranagar', 'Whitefield', 'HSR Layout', 'Jayanagar', 'Marathahalli', 'Electronic City', 'Yelahanka'],
    'Mumbai': ['Bandra', 'Andheri', 'Powai', 'Worli', 'Juhu', 'Goregaon', 'Thane', 'Navi Mumbai'],
    'Delhi': ['Dwarka', 'Rohini', 'Vasant Kunj', 'Greater Kailash', 'Saket', 'Noida', 'Gurgaon', 'Hauz Khas'],
    'Hyderabad': ['Gachibowli', 'Banjara Hills', 'Jubilee Hills', 'Kondapur', 'Madhapur', 'HITEC City', 'Kukatpally', 'Miyapur'],
    'Pune': ['Koregaon Park', 'Hinjewadi', 'Wakad', 'Baner', 'Kothrud', 'Viman Nagar', 'Magarpatta', 'Kalyani Nagar'],
    'Chennai': ['Adyar', 'Anna Nagar', 'T Nagar', 'Velachery', 'OMR', 'Nungambakkam', 'Porur', 'Tambaram']
}

# Assign random city and corresponding neighborhood
assigned_cities = np.random.choice(cities, n_samples)
neighborhoods = [np.random.choice(neighborhoods_by_city[city]) for city in assigned_cities]

# Generate sample data
data = {
    'City': assigned_cities,
    'Neighborhood': neighborhoods,
    'LotArea': np.random.randint(500, 3000, n_samples),  # sq ft (smaller plots in India)
    'OverallQual': np.random.randint(1, 11, n_samples),
    'GrLivArea': np.random.randint(600, 3000, n_samples),  # sq ft
    'FullBath': np.random.randint(1, 4, n_samples),
    'BedroomAbvGr': np.random.randint(1, 5, n_samples),
    'YearBuilt': np.random.randint(1980, 2024, n_samples),
    'GarageCars': np.random.randint(0, 3, n_samples),
    'TotalBsmtSF': np.random.randint(0, 1000, n_samples),
    'Fireplaces': np.random.randint(0, 2, n_samples)
}

df = pd.DataFrame(data)

# City-based price multipliers (based on real estate market rates)
city_multipliers = {
    'Bangalore': 1.3,
    'Mumbai': 1.8,      # Most expensive
    'Delhi': 1.5,
    'Hyderabad': 1.0,   # Base rate
    'Pune': 1.1,
    'Chennai': 1.0
}

# Neighborhood premium (prime locations cost more)
neighborhood_premium = {
    # Bangalore
    'Koramangala': 1.4, 'Indiranagar': 1.5, 'Whitefield': 1.1, 'HSR Layout': 1.3,
    'Jayanagar': 1.2, 'Marathahalli': 1.0, 'Electronic City': 0.9, 'Yelahanka': 0.85,
    # Mumbai
    'Bandra': 1.8, 'Andheri': 1.4, 'Powai': 1.5, 'Worli': 2.0, 'Juhu': 1.9,
    'Goregaon': 1.3, 'Thane': 1.0, 'Navi Mumbai': 0.9,
    # Delhi
    'Dwarka': 1.2, 'Rohini': 1.0, 'Vasant Kunj': 1.6, 'Greater Kailash': 1.8,
    'Saket': 1.7, 'Noida': 1.1, 'Gurgaon': 1.5, 'Hauz Khas': 1.6,
    # Hyderabad
    'Gachibowli': 1.3, 'Banjara Hills': 1.6, 'Jubilee Hills': 1.7, 'Kondapur': 1.2,
    'Madhapur': 1.3, 'HITEC City': 1.4, 'Kukatpally': 0.9, 'Miyapur': 0.85,
    # Pune
    'Koregaon Park': 1.5, 'Hinjewadi': 1.2, 'Wakad': 1.1, 'Baner': 1.3,
    'Kothrud': 1.2, 'Viman Nagar': 1.3, 'Magarpatta': 1.4, 'Kalyani Nagar': 1.4,
    # Chennai
    'Adyar': 1.4, 'Anna Nagar': 1.3, 'T Nagar': 1.5, 'Velachery': 1.1,
    'OMR': 1.2, 'Nungambakkam': 1.6, 'Porur': 1.0, 'Tambaram': 0.85
}

# Realistic pricing model for Indian real estate (2024-2025 rates)
# Land cost + Construction cost + Premium for quality/location
for idx, row in df.iterrows():
    city = row['City']
    neighborhood = row['Neighborhood']
    
    # Construction cost: ₹2500-5000 per sq ft based on quality (2024 rates)
    # Quality 1-3: Budget (₹2500-3000), 4-7: Standard (₹3000-4000), 8-10: Premium (₹4000-5000)
    if row['OverallQual'] <= 3:
        construction_rate = 2500 + (row['OverallQual'] * 150)
    elif row['OverallQual'] <= 7:
        construction_rate = 3000 + (row['OverallQual'] * 150)
    else:
        construction_rate = 4000 + (row['OverallQual'] * 120)
    
    construction_cost = row['GrLivArea'] * construction_rate
    
    # Land cost: ₹12000-40000 per sq ft depending on city and location (2024 realistic rates)
    base_land_rate = 12000  # Increased base rate
    land_rate = base_land_rate * city_multipliers[city] * neighborhood_premium[neighborhood]
    land_cost = row['LotArea'] * land_rate
    
    # Additional amenities (realistic values)
    bathroom_value = row['FullBath'] * 300000  # Each bathroom adds value
    bedroom_value = row['BedroomAbvGr'] * 250000  # Each bedroom adds value  
    parking_value = row['GarageCars'] * 400000  # Parking premium
    
    # Age depreciation (2% per year for buildings older than 5 years)
    age = 2024 - row['YearBuilt']
    depreciation_factor = 1.0
    if age > 5:
        depreciation_factor = max(0.75, 1.0 - (age - 5) * 0.02)
    
    # Total price
    base_price = (land_cost + construction_cost + bathroom_value + bedroom_value + parking_value) * depreciation_factor
    
    df.loc[idx, 'SalePrice'] = base_price

# Add market variation (±10% random fluctuation)
noise = np.random.normal(1.0, 0.10, n_samples)
df['SalePrice'] = df['SalePrice'] * noise

# Ensure minimum prices and round
df['SalePrice'] = df['SalePrice'].clip(lower=3000000)  # Minimum 30 Lakhs
df['SalePrice'] = (df['SalePrice'] / 100000).round() * 100000  # Round to nearest lakh

# Add some missing values to make it realistic (5% missing for some columns)
for col in ['GarageCars', 'TotalBsmtSF', 'Fireplaces']:
    mask = np.random.random(n_samples) < 0.05
    df.loc[mask, col] = np.nan

# Save to CSV
output_path = Path('data/house_prices.csv')
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print(f"✅ Generated {n_samples} sample houses")
print(f"📁 Saved to: {output_path}")
print(f"\nDataset statistics:")
print(f"  Price range: ₹{df['SalePrice'].min():,.0f} (₹{df['SalePrice'].min()/100000:.1f}L) - ₹{df['SalePrice'].max():,.0f} (₹{df['SalePrice'].max()/100000:.1f}L)")
print(f"  Mean price: ₹{df['SalePrice'].mean():,.0f} (₹{df['SalePrice'].mean()/100000:.1f} Lakhs)")
print(f"  Median price: ₹{df['SalePrice'].median():,.0f} (₹{df['SalePrice'].median()/100000:.1f} Lakhs)")
print(f"\nYou can now train the model with:")
print(f"  python -m src.train_model --input {output_path} --output models/model.pkl")
