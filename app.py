import streamlit as st
import pandas as pd
from models.insurance_model import (
    load_and_preprocess_data,
    train_model,
    adjust_price_for_turkish_market
)

# Set page config
st.set_page_config(
    page_title="Car Insurance Price Estimation System",
    page_icon="🚗",
    layout="wide"
)

# Load CSS
with open('static/styles.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def create_user_summary(user_data, vehicle_types, selected_age, selected_exp):
    """Create a summary of user inputs in a readable format."""
    summary = {
        'Özellik': [
            'Yaş',
            'Sürüş Deneyimi',
            'Geçmiş Kaza Sayısı',
            'Araç Tipi',
            'Araç Yılı',
            'Yıllık Kilometre'
        ],
        'Değer': [
            selected_age,  # Use the actual selected age range
            selected_exp,  # Use the actual selected experience text
            f"{int(user_data['PAST_ACCIDENTS'])}",
            vehicle_types[user_data['VEHICLE_TYPE']],
            str(user_data['VEHICLE_YEAR']),
            f"{int(user_data['ANNUAL_MILEAGE']):,} km"
        ]
    }
    return pd.DataFrame(summary)

def main():
    st.title("🚗 Araç Sigorta Fiyat Tahmin Sistemi")
    
    # Load data and train model
    df, df_display, categorical_values = load_and_preprocess_data()
    model, available_features, mse, r2 = train_model(df)
    
    # Define vehicle types for UI
    vehicle_types = {
        0: "Lüks Araç",
        1: "Spor Araç",
        2: "SUV",
        3: "Sedan",
        4: "Hatchback",
        5: "Ticari Araç"
    }
    
    # Define vehicle years
    current_year = 2024
    vehicle_years = list(range(current_year, current_year - 30, -1))
    
    st.markdown("""
    ### Araç Sigorta Fiyat Tahmin Sistemine Hoş Geldiniz!
    Lütfen aşağıdaki bilgileri doldurarak tahmini sigorta fiyatınızı öğrenin.
    """)
    
    # Create input columns
    col1, col2 = st.columns(2)
    user_data = {}
    selected_exp = None
    selected_age = None
    
    with col1:
        st.markdown("#### Kişisel Bilgiler")
        # Age selection
        age_ranges = sorted(df_display['AGE'].unique())
        selected_age = st.selectbox("Yaş Aralığı", age_ranges, key="age_select")
        user_data['AGE'] = 65 if selected_age == '65+' else sum(map(int, str(selected_age).split('-'))) / 2
        
        # Driving experience
        if 'DRIVING_EXPERIENCE' in categorical_values:
            selected_exp = st.selectbox("Sürüş Deneyimi", 
                                      categorical_values['DRIVING_EXPERIENCE'],
                                      key="exp_select")
            user_data['DRIVING_EXPERIENCE'] = categorical_values['DRIVING_EXPERIENCE'].index(selected_exp)
        
        # Past accidents
        user_data['PAST_ACCIDENTS'] = st.number_input("Geçmiş Kaza Sayısı", 
                                                    min_value=0, 
                                                    max_value=10, 
                                                    value=0)
    
    with col2:
        st.markdown("#### Araç Bilgileri")
        # Vehicle type
        selected_vehicle_type = st.selectbox(
            "Araç Tipi",
            options=list(vehicle_types.keys()),
            format_func=lambda x: vehicle_types[x]
        )
        user_data['VEHICLE_TYPE'] = selected_vehicle_type
        
        # Vehicle year
        selected_year = st.selectbox(
            "Araç Yılı",
            options=vehicle_years
        )
        user_data['VEHICLE_YEAR'] = selected_year
        
        # Annual mileage
        user_data['ANNUAL_MILEAGE'] = st.number_input("Yıllık Kilometre", 
                                                    min_value=0, 
                                                    max_value=100000, 
                                                    value=10000)
    
    if st.button("Sigorta Fiyatını Hesapla", type="primary"):
        # Prepare user data
        user_df = pd.DataFrame([user_data])
        for feature in available_features:
            if feature not in user_df.columns:
                user_df[feature] = df[feature].mean()
        user_df = user_df[available_features]
        
        # Make prediction
        base_prediction = model.predict(user_df)[0]
        turkish_price = adjust_price_for_turkish_market(base_prediction, user_data, df)
        
        # Load and format prediction template
        with open('templates/prediction_box.html') as f:
            prediction_template = f.read()
        st.markdown(
            prediction_template.format(price=turkish_price),
            unsafe_allow_html=True
        )
        
        # Display user input summary
        st.markdown("### Girilen Bilgiler")
        user_summary = create_user_summary(user_data, vehicle_types, selected_age, selected_exp)
        st.table(user_summary.set_index('Özellik'))

if __name__ == "__main__":
    main() 