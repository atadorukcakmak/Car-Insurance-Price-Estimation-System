import pandas as pd
import numpy as np
from datetime import datetime

def calculate_car_age(year):
    """Calculate the age of the car."""
    current_year = datetime.now().year
    return current_year - year

def adjust_price_for_negotiation(predicted_price, negotiation_factor):
    """Adjust price based on negotiation factor (-1 to 1)."""
    # Negotiation factor: -1 (maximum reduction) to 1 (maximum increase)
    adjustment = 1 + (negotiation_factor * 0.1)  # Max 10% adjustment
    return predicted_price * adjustment

def calculate_risk_factor(car_age, mileage, accidents=0):
    """Calculate risk factor for insurance premium."""
    age_risk = min(1 + (car_age * 0.05), 2.0)  # Max 100% increase for age
    mileage_risk = min(1 + (mileage / 100000) * 0.1, 1.5)  # Max 50% increase for mileage
    accident_risk = 1 + (accidents * 0.2)  # 20% increase per accident
    
    return age_risk * mileage_risk * accident_risk

def format_currency(amount):
    """Format amount as currency."""
    return f"₺{amount:,.2f}"

def validate_input_data(data):
    """Validate input data for prediction."""
    required_fields = ['year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner']
    
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
    
    # Validate numeric fields
    if not isinstance(data['year'], (int, float)) or data['year'] < 1900:
        raise ValueError("Invalid year")
    
    if not isinstance(data['km_driven'], (int, float)) or data['km_driven'] < 0:
        raise ValueError("Invalid kilometers driven")
    
    return True

def prepare_prediction_data(input_data, label_encoders):
    """Prepare input data for model prediction."""
    data = input_data.copy()
    
    # Encode categorical variables
    for column, encoder in label_encoders.items():
        if column in data:
            try:
                data[column] = encoder.transform([data[column]])[0]
            except ValueError:
                raise ValueError(f"Invalid value for {column}")
    
    return data 