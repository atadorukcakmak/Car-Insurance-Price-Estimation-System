import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import random
import streamlit as st

@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the insurance dataset."""
    # Load the insurance dataset
    df = pd.read_csv('data/newinsurancedata.csv')
    
    # Store original categorical values for UI
    categorical_values = {}
    for col in ['GENDER', 'RACE', 'DRIVING_EXPERIENCE', 'EDUCATION', 
                'VEHICLE_OWNERSHIP']:
        if col in df.columns:
            categorical_values[col] = sorted(df[col].unique())
    
    # Handle missing values
    # Fill numeric columns with mean
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    
    # Fill categorical columns with mode
    categorical_columns = ['GENDER', 'RACE', 'DRIVING_EXPERIENCE', 'EDUCATION', 
                         'VEHICLE_OWNERSHIP', 'VEHICLE_YEAR', 'VEHICLE_TYPE']
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # Create a copy for display
    df_display = df.copy()
    
    # Convert categorical variables
    le = LabelEncoder()
    for col in categorical_columns:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
    
    # Handle special cases for numeric columns
    if 'AGE' in df.columns:
        df['AGE'] = df['AGE'].apply(convert_age)
    
    if 'INCOME' in df.columns:
        df['INCOME'] = df['INCOME'].apply(convert_income)
    
    if 'CREDIT_SCORE' in df.columns:
        df['CREDIT_SCORE'] = pd.to_numeric(df['CREDIT_SCORE'], errors='coerce')
        df['CREDIT_SCORE'] = df['CREDIT_SCORE'].fillna(df['CREDIT_SCORE'].mean())
    
    if 'ANNUAL_MILEAGE' in df.columns:
        df['ANNUAL_MILEAGE'] = pd.to_numeric(df['ANNUAL_MILEAGE'], errors='coerce')
        df['ANNUAL_MILEAGE'] = df['ANNUAL_MILEAGE'].fillna(df['ANNUAL_MILEAGE'].mean())
    
    if 'SPEEDING_VIOLATIONS' in df.columns:
        df['SPEEDING_VIOLATIONS'] = pd.to_numeric(df['SPEEDING_VIOLATIONS'], errors='coerce')
        df['SPEEDING_VIOLATIONS'] = df['SPEEDING_VIOLATIONS'].fillna(0)
    
    if 'DUIS' in df.columns:
        df['DUIS'] = pd.to_numeric(df['DUIS'], errors='coerce')
        df['DUIS'] = df['DUIS'].fillna(0)
    
    if 'PAST_ACCIDENTS' in df.columns:
        df['PAST_ACCIDENTS'] = pd.to_numeric(df['PAST_ACCIDENTS'], errors='coerce')
        df['PAST_ACCIDENTS'] = df['PAST_ACCIDENTS'].fillna(0)
    
    return df, df_display, categorical_values

def convert_age(x):
    """Convert age ranges to numeric values."""
    if str(x) == '65+':
        return 65
    elif '-' in str(x):
        start, end = map(int, str(x).split('-'))
        return (start + end) / 2
    return float(x)

def convert_income(x):
    """Convert income categories to numeric values."""
    income_mapping = {
        'poverty': 15000,
        'working class': 35000,
        'middle class': 65000,
        'upper middle class': 95000,
        'upper class': 150000
    }
    
    if isinstance(x, (int, float)):
        return float(x)
    elif isinstance(x, str):
        x = x.replace('$', '').replace(',', '')
        try:
            return float(x)
        except ValueError:
            return income_mapping.get(x.lower(), 65000)
    return 65000

@st.cache_resource
def train_model(df):
    """Train the insurance price prediction model."""
    feature_columns = ['AGE', 'GENDER', 'RACE', 'DRIVING_EXPERIENCE', 
                      'EDUCATION', 'INCOME', 'CREDIT_SCORE', 
                      'VEHICLE_OWNERSHIP', 'VEHICLE_YEAR', 'MARRIED',
                      'CHILDREN', 'ANNUAL_MILEAGE', 'VEHICLE_TYPE', 
                      'SPEEDING_VIOLATIONS', 'DUIS', 'PAST_ACCIDENTS']
    
    available_features = [col for col in feature_columns if col in df.columns]
    X = df[available_features]
    y = df['OUTCOME']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, available_features, mse, r2

def adjust_price_for_turkish_market(base_price, user_data, df):
    """Adjust the base price for the Turkish market."""
    print(f"Base price from model: {base_price}")
    
    base_conversion = 0.25
    age = float(str(user_data['AGE']).replace('np.int64(', '').replace(')', ''))
    
    # Calculate all adjustment factors
    age_factor = calculate_age_factor(age)
    exp_factor = calculate_experience_factor(user_data)
    accident_factor = calculate_accident_factor(user_data)
    vehicle_factor = calculate_vehicle_factor(user_data)
    year_factor = calculate_year_factor(user_data)
    mileage_factor = calculate_mileage_factor(user_data)
    
    # Calculate base Turkish price
    base_turkish_price = max(5000, base_price * base_conversion)
    
    # Print factors
    print("Factors applied:")
    print(f"Age factor: {age_factor:.3f}")
    print(f"Experience factor: {exp_factor:.3f}")
    print(f"Accident factor: {accident_factor:.3f}")
    print(f"Vehicle type factor: {vehicle_factor:.3f}")
    print(f"Vehicle year factor: {year_factor:.3f}")
    print(f"Mileage factor: {mileage_factor:.3f}")
    
    # Calculate adjusted price
    adjusted_price = (base_turkish_price * age_factor * exp_factor * 
                     accident_factor * vehicle_factor * year_factor * 
                     mileage_factor)
    
    # Add randomness (±2%)
    random_factor = random.uniform(0.98, 1.02)
    adjusted_price *= random_factor
    
    print(f"Base Turkish price: {base_turkish_price:.2f}")
    print(f"Adjusted price before random: {adjusted_price:.2f}")
    print(f"Random factor: {random_factor:.3f}")
    
    # Ensure price is within market range
    final_price = max(5000, min(adjusted_price, 25000))
    print(f"Final price: {final_price:.2f}")
    
    return round(final_price, 2)

def calculate_age_factor(age):
    """Calculate age adjustment factor."""
    if age < 25:
        return 1.0 + (25 - age) * 0.02
    elif age < 30:
        return 1.0 + (30 - age) * 0.01
    elif age > 65:
        return 1.0 + (age - 65) * 0.01
    return 1.0

def calculate_experience_factor(user_data):
    """Calculate driving experience adjustment factor."""
    driving_exp = float(str(user_data['DRIVING_EXPERIENCE']).replace('np.int64(', '').replace(')', ''))
    if driving_exp < 3:
        return 1.0 + (3 - driving_exp) * 0.05
    elif driving_exp > 7:
        return 1.0 - (driving_exp - 7) * 0.02
    return 1.0

def calculate_accident_factor(user_data):
    """Calculate accident history adjustment factor."""
    past_accidents = float(str(user_data['PAST_ACCIDENTS']).replace('np.int64(', '').replace(')', ''))
    return 1.0 + (past_accidents * 0.08)

def calculate_vehicle_factor(user_data):
    """Calculate vehicle type adjustment factor."""
    vehicle_type = float(str(user_data['VEHICLE_TYPE']).replace('np.int64(', '').replace(')', ''))
    vehicle_factors = {
        0: 1.25,  # Luxury
        1: 1.30,  # Sports
        2: 1.15,  # SUV
        3: 1.05,  # Sedan
        4: 1.00,  # Hatchback
        5: 1.20   # Commercial
    }
    return vehicle_factors.get(vehicle_type, 1.0)

def calculate_year_factor(user_data):
    """Calculate vehicle year adjustment factor."""
    vehicle_year = float(str(user_data['VEHICLE_YEAR']).replace('np.int64(', '').replace(')', ''))
    vehicle_age = 2024 - vehicle_year
    
    if vehicle_age <= 1:
        return 1.15
    elif vehicle_age <= 3:
        return 1.10
    elif vehicle_age <= 5:
        return 1.05
    elif vehicle_age <= 10:
        return 1.00
    elif vehicle_age <= 15:
        return 0.95
    return 0.90

def calculate_mileage_factor(user_data):
    """Calculate annual mileage adjustment factor."""
    annual_mileage = float(str(user_data['ANNUAL_MILEAGE']).replace('np.int64(', '').replace(')', ''))
    if annual_mileage > 20000:
        return 1.0 + (annual_mileage - 20000) / 100000
    return 1.0 