import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer
from src.utils import (
    calculate_car_age,
    adjust_price_for_negotiation,
    calculate_risk_factor,
    format_currency,
    validate_input_data,
    prepare_prediction_data
)

# Initialize session state
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()
    st.session_state.model_trainer = ModelTrainer()

def load_and_train_models():
    """Load data and train models if not already trained."""
    try:
        data = st.session_state.data_processor.load_data('data/car_data.csv')
        processed_data = st.session_state.data_processor.preprocess_data()
        
        price_features, insurance_features = st.session_state.model_trainer.prepare_features(processed_data)
        
        with st.spinner('Training price prediction model...'):
            price_metrics = st.session_state.model_trainer.train_price_model(processed_data, price_features)
        
        with st.spinner('Training insurance prediction model...'):
            insurance_metrics = st.session_state.model_trainer.train_insurance_model(processed_data, insurance_features)
        
        return True
    except Exception as e:
        st.error(f"Error loading data and training models: {str(e)}")
        return False

def main():
    st.title('Car Insurance and Price Estimation System')
    
    # Sidebar for data analysis
    st.sidebar.title('Data Analysis')
    if st.sidebar.button('Show Data Visualizations'):
        fig = st.session_state.data_processor.generate_visualizations()
        st.pyplot(fig)
        plt.close()
    
    # Main input form
    st.header('Enter Car Details')
    
    col1, col2 = st.columns(2)
    
    with col1:
        year = st.number_input('Year of Manufacture', min_value=1900, max_value=2024, value=2020)
        km_driven = st.number_input('Kilometers Driven', min_value=0, value=50000)
        fuel = st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'CNG', 'Electric'])
    
    with col2:
        transmission = st.selectbox('Transmission', ['Manual', 'Automatic'])
        owner = st.selectbox('Number of Previous Owners', ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above'])
        seller_type = st.selectbox('Seller Type', ['Individual', 'Dealer'])
    
    # Additional inputs for insurance calculation
    st.header('Insurance Factors')
    accidents = st.number_input('Number of Previous Accidents', min_value=0, max_value=10, value=0)
    usage_purpose = st.selectbox('Usage Purpose', ['Personal', 'Commercial'])
    
    # Negotiation factor
    st.header('Price Negotiation')
    negotiation_factor = st.slider('Negotiation Factor', min_value=-1.0, max_value=1.0, value=0.0,
                                 help='-1: Maximum reduction, 1: Maximum increase')
    
    if st.button('Calculate Estimates'):
        try:
            # Prepare input data
            input_data = {
                'year': year,
                'km_driven': km_driven,
                'fuel': fuel,
                'seller_type': seller_type,
                'transmission': transmission,
                'owner': owner
            }
            
            # Validate input
            validate_input_data(input_data)
            
            # Prepare data for prediction
            prediction_data = prepare_prediction_data(
                input_data,
                st.session_state.data_processor.label_encoders
            )
            
            # Get predictions
            prediction_features = list(prediction_data.values())
            predicted_price = st.session_state.model_trainer.predict_price([prediction_features])[0]
            
            # Add the predicted price to features for insurance prediction
            insurance_features = prediction_features + [predicted_price]
            predicted_insurance = st.session_state.model_trainer.predict_insurance([insurance_features])[0]
            
            # Apply adjustments
            car_age = calculate_car_age(year)
            risk_factor = calculate_risk_factor(car_age, km_driven, accidents)
            
            # Adjust insurance based on usage
            if usage_purpose == 'Commercial':
                predicted_insurance *= 1.2  # 20% increase for commercial use
            
            # Adjust price for negotiation
            final_price = adjust_price_for_negotiation(predicted_price, negotiation_factor)
            
            # Display results
            st.success('Calculations completed successfully!')
            
            results_col1, results_col2 = st.columns(2)
            
            with results_col1:
                st.subheader('Price Estimates')
                st.write(f'Base Price: {format_currency(predicted_price)}')
                st.write(f'Negotiated Price: {format_currency(final_price)}')
            
            with results_col2:
                st.subheader('Insurance Estimates')
                st.write(f'Annual Premium: {format_currency(predicted_insurance * risk_factor)}')
                st.write(f'Monthly Premium: {format_currency((predicted_insurance * risk_factor) / 12)}')
            
            # Additional information
            st.info(f'''
                Risk Factors Considered:
                - Car Age: {car_age} years
                - Mileage: {km_driven:,} km
                - Accidents: {accidents}
                - Usage: {usage_purpose}
            ''')
            
        except Exception as e:
            st.error(f"Error calculating estimates: {str(e)}")

if __name__ == '__main__':
    if load_and_train_models():
        main()
    else:
        st.error('Failed to initialize the application. Please check the data and try again.') 