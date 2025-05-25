import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, r2_score
import random

# Set page config
st.set_page_config(
    page_title="Car Insurance Price Estimation System",
    page_icon="🚗",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Load and preprocess data
@st.cache_data
def load_data():
    # Load the insurance dataset
    df = pd.read_csv('data/newinsurancedata.csv')
    
    # Store original categorical values for UI
    categorical_values = {}
    for col in ['GENDER', 'RACE', 'DRIVING_EXPERIENCE', 'EDUCATION', 
                'VEHICLE_OWNERSHIP']:
        if col in df.columns:
            categorical_values[col] = sorted(df[col].unique())
    
    # Define vehicle types for UI
    vehicle_types = {
        0: "Lüks Araç",
        1: "Spor Araç",
        2: "SUV",
        3: "Sedan",
        4: "Hatchback",
        5: "Ticari Araç"
    }
    categorical_values['VEHICLE_TYPE'] = list(vehicle_types.values())
    
    # Define vehicle years for UI (last 30 years)
    current_year = 2024
    vehicle_years = list(range(current_year, current_year - 30, -1))
    categorical_values['VEHICLE_YEAR'] = [str(year) for year in vehicle_years]
    
    # Handle missing values
    # Fill numeric columns with mean
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    
    # Fill categorical columns with mode (most frequent value)
    categorical_columns = ['GENDER', 'RACE', 'DRIVING_EXPERIENCE', 'EDUCATION', 
                         'VEHICLE_OWNERSHIP', 'VEHICLE_YEAR', 'VEHICLE_TYPE']
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # Create a copy of the dataframe for UI display
    df_display = df.copy()
    
    # Convert categorical variables for model
    le = LabelEncoder()
    for col in categorical_columns:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
    
    # Handle special cases for numeric columns
    if 'AGE' in df.columns:
        # Convert age ranges to numeric values by taking midpoint
        def convert_age(x):
            if str(x) == '65+':
                return 65
            elif '-' in str(x):
                # Split the range and take midpoint
                start, end = map(int, str(x).split('-'))
                return (start + end) / 2
            else:
                return float(x)
        
        df['AGE'] = df['AGE'].apply(convert_age)
    
    if 'INCOME' in df.columns:
        # Convert income categories to numeric values
        income_mapping = {
            'poverty': 15000,
            'working class': 35000,
            'middle class': 65000,
            'upper middle class': 95000,
            'upper class': 150000
        }
        
        def convert_income(x):
            if isinstance(x, (int, float)):
                return float(x)
            elif isinstance(x, str):
                # Remove currency symbols and commas
                x = x.replace('$', '').replace(',', '')
                try:
                    return float(x)
                except ValueError:
                    # If not a number, try to map the category
                    return income_mapping.get(x.lower(), 65000)  # Default to middle class if unknown
            return 65000  # Default value if conversion fails
        
        df['INCOME'] = df['INCOME'].apply(convert_income)
    
    if 'CREDIT_SCORE' in df.columns:
        # Ensure credit score is numeric
        df['CREDIT_SCORE'] = pd.to_numeric(df['CREDIT_SCORE'], errors='coerce')
        df['CREDIT_SCORE'] = df['CREDIT_SCORE'].fillna(df['CREDIT_SCORE'].mean())
    
    if 'ANNUAL_MILEAGE' in df.columns:
        # Convert mileage to numeric
        df['ANNUAL_MILEAGE'] = pd.to_numeric(df['ANNUAL_MILEAGE'], errors='coerce')
        df['ANNUAL_MILEAGE'] = df['ANNUAL_MILEAGE'].fillna(df['ANNUAL_MILEAGE'].mean())
    
    if 'SPEEDING_VIOLATIONS' in df.columns:
        # Ensure speeding violations is numeric
        df['SPEEDING_VIOLATIONS'] = pd.to_numeric(df['SPEEDING_VIOLATIONS'], errors='coerce')
        df['SPEEDING_VIOLATIONS'] = df['SPEEDING_VIOLATIONS'].fillna(0)
    
    if 'DUIS' in df.columns:
        # Ensure DUIs is numeric
        df['DUIS'] = pd.to_numeric(df['DUIS'], errors='coerce')
        df['DUIS'] = df['DUIS'].fillna(0)
    
    if 'PAST_ACCIDENTS' in df.columns:
        # Ensure past accidents is numeric
        df['PAST_ACCIDENTS'] = pd.to_numeric(df['PAST_ACCIDENTS'], errors='coerce')
        df['PAST_ACCIDENTS'] = df['PAST_ACCIDENTS'].fillna(0)
    
    return df, df_display, categorical_values

# Train model
@st.cache_resource
def train_model(df):
    # Prepare features
    feature_columns = ['AGE', 'GENDER', 'RACE', 'DRIVING_EXPERIENCE', 
                      'EDUCATION', 'INCOME', 'CREDIT_SCORE', 
                      'VEHICLE_OWNERSHIP', 'VEHICLE_YEAR', 'MARRIED',
                      'CHILDREN', 'ANNUAL_MILEAGE', 'VEHICLE_TYPE', 
                      'SPEEDING_VIOLATIONS', 'DUIS', 'PAST_ACCIDENTS']
    
    # Filter available features
    available_features = [col for col in feature_columns if col in df.columns]
    
    # Ensure all features are numeric
    X = df[available_features].copy()
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')
            X[col] = X[col].fillna(X[col].mean())
    
    y = df['OUTCOME']  # Using OUTCOME as the target variable
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate metrics
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, available_features, mse, r2

def adjust_price_for_turkish_market(base_price, user_data, df):
    """
    Adjust the base price based on user's profile and current Turkish insurance market prices.
    """
    # Debug: Print base price
    print(f"Base price from model: {base_price}")
    
    # Base conversion factor for Turkish Lira (adjusted for current market)
    base_conversion = 0.25
    
    # Age factor adjustment (more granular)
    age = float(str(user_data['AGE']).replace('np.int64(', '').replace(')', ''))
    age_factor = 1.0
    if age < 25:
        age_factor = 1.0 + (25 - age) * 0.02  # 2% increase per year under 25
    elif age < 30:
        age_factor = 1.0 + (30 - age) * 0.01  # 1% increase per year under 30
    elif age > 65:
        age_factor = 1.0 + (age - 65) * 0.01  # 1% increase per year over 65
    
    # Driving experience factor (more granular)
    driving_exp = float(str(user_data['DRIVING_EXPERIENCE']).replace('np.int64(', '').replace(')', ''))
    exp_factor = 1.0
    if driving_exp < 3:  # Less than 9 years
        exp_factor = 1.0 + (3 - driving_exp) * 0.05  # 5% increase per level of inexperience
    elif driving_exp > 7:  # More than 20 years
        exp_factor = 1.0 - (driving_exp - 7) * 0.02  # 2% decrease per level of experience
    
    # Accident factor (more granular)
    accident_factor = 1.0
    past_accidents = float(str(user_data['PAST_ACCIDENTS']).replace('np.int64(', '').replace(')', ''))
    
    if past_accidents > 0:
        accident_factor += (past_accidents * 0.08)  # 8% increase per accident
    
    # Vehicle type factor (more realistic)
    vehicle_factor = 1.0
    if 'VEHICLE_TYPE' in user_data:
        vehicle_type = float(str(user_data['VEHICLE_TYPE']).replace('np.int64(', '').replace(')', ''))
        # More detailed vehicle type factors based on Turkish market
        if vehicle_type == 0:  # Luxury cars
            vehicle_factor = 1.25  # 25% increase
        elif vehicle_type == 1:  # Sports cars
            vehicle_factor = 1.30  # 30% increase
        elif vehicle_type == 2:  # SUVs
            vehicle_factor = 1.15  # 15% increase
        elif vehicle_type == 3:  # Sedan
            vehicle_factor = 1.05  # 5% increase
        elif vehicle_type == 4:  # Hatchback
            vehicle_factor = 1.00  # Base rate
        elif vehicle_type == 5:  # Commercial vehicles
            vehicle_factor = 1.20  # 20% increase
    
    # Vehicle year factor (new, more realistic)
    year_factor = 1.0
    if 'VEHICLE_YEAR' in user_data:
        vehicle_year = float(str(user_data['VEHICLE_YEAR']).replace('np.int64(', '').replace(')', ''))
        current_year = 2024  # Current year
        
        # Calculate vehicle age
        vehicle_age = current_year - vehicle_year
        
        # Adjust factor based on vehicle age
        if vehicle_age <= 1:  # Brand new
            year_factor = 1.15  # 15% increase
        elif vehicle_age <= 3:  # 1-3 years old
            year_factor = 1.10  # 10% increase
        elif vehicle_age <= 5:  # 3-5 years old
            year_factor = 1.05  # 5% increase
        elif vehicle_age <= 10:  # 5-10 years old
            year_factor = 1.00  # Base rate
        elif vehicle_age <= 15:  # 10-15 years old
            year_factor = 0.95  # 5% decrease
        else:  # Over 15 years old
            year_factor = 0.90  # 10% decrease
    
    # Annual mileage factor (new)
    mileage_factor = 1.0
    annual_mileage = float(str(user_data['ANNUAL_MILEAGE']).replace('np.int64(', '').replace(')', ''))
    if annual_mileage > 20000:
        mileage_factor = 1.0 + (annual_mileage - 20000) / 100000  # 1% increase per 1000 km over 20000
    
    # Calculate base Turkish price with all factors
    base_turkish_price = max(5000, base_price * base_conversion)  # Ensure minimum base price
    
    # Debug: Print all factors
    print(f"Factors applied:")
    print(f"Age factor: {age_factor:.3f}")
    print(f"Experience factor: {exp_factor:.3f}")
    print(f"Accident factor: {accident_factor:.3f}")
    print(f"Vehicle type factor: {vehicle_factor:.3f}")
    print(f"Vehicle year factor: {year_factor:.3f}")
    print(f"Mileage factor: {mileage_factor:.3f}")
    
    # Calculate final price with all factors
    adjusted_price = base_turkish_price * age_factor * exp_factor * accident_factor * vehicle_factor * year_factor * mileage_factor
    
    # Add some randomness to make prices more realistic (within ±2%)
    random_factor = random.uniform(0.98, 1.02)
    adjusted_price *= random_factor
    
    # Debug: Print intermediate calculations
    print(f"Base Turkish price: {base_turkish_price:.2f}")
    print(f"Adjusted price before random: {adjusted_price:.2f}")
    print(f"Random factor: {random_factor:.3f}")
    
    # Ensure price is within current Turkish market range
    min_price = 5000
    max_price = 25000
    
    final_price = max(min_price, min(adjusted_price, max_price))
    
    # Debug: Print final price
    print(f"Final price: {final_price:.2f}")
    
    return round(final_price, 2)

def find_similar_profiles(df, user_data, n=5):
    # Calculate similarity scores based on available features
    similarity_scores = []
    
    for _, row in df.iterrows():
        score = 0
        for feature in user_data:
            if feature in df.columns:
                if df[feature].dtype in ['int64', 'float64']:
                    # For numerical features, use normalized difference
                    max_val = df[feature].max()
                    min_val = df[feature].min()
                    if max_val != min_val:
                        score += 1 - abs(row[feature] - user_data[feature]) / (max_val - min_val)
                else:
                    # For categorical features, use exact match
                    score += 1 if row[feature] == user_data[feature] else 0
        
        similarity_scores.append(score)
    
    # Add similarity scores to dataframe
    df['similarity_score'] = similarity_scores
    
    # Get top n similar profiles
    similar_profiles = df.nlargest(n, 'similarity_score')
    
    return similar_profiles

def main():
    st.title("🚗 Araç Sigorta Fiyat Tahmin Sistemi")
    
    # Load data
    df, df_display, categorical_values = load_data()
    
    # Define vehicle types for UI
    vehicle_types = {
        0: "Lüks Araç",
        1: "Spor Araç",
        2: "SUV",
        3: "Sedan",
        4: "Hatchback",
        5: "Ticari Araç"
    }
    
    # Define vehicle years for UI (last 30 years)
    current_year = 2024
    vehicle_years = list(range(current_year, current_year - 30, -1))
    
    # Train model
    model, available_features, mse, r2 = train_model(df)
    
    # Main content area
    st.markdown("""
    ### Araç Sigorta Fiyat Tahmin Sistemine Hoş Geldiniz!
    Lütfen aşağıdaki bilgileri doldurarak tahmini sigorta fiyatınızı öğrenin.
    """)
    
    # Create two columns for input fields
    col1, col2 = st.columns(2)
    
    user_data = {}
    
    with col1:
        st.markdown("#### Kişisel Bilgiler")
        # Age selection
        age_ranges = sorted(df_display['AGE'].unique())
        selected_age = st.selectbox("Yaş Aralığı", age_ranges, key="age_select")
        if selected_age == '65+':
            user_data['AGE'] = 65
        else:
            start, end = map(int, str(selected_age).split('-'))
            user_data['AGE'] = (start + end) / 2
        
        # Driving experience
        if 'DRIVING_EXPERIENCE' in categorical_values:
            selected_exp = st.selectbox("Sürüş Deneyimi", 
                                      categorical_values['DRIVING_EXPERIENCE'],
                                      key="exp_select")
            le = LabelEncoder()
            le.fit(df_display['DRIVING_EXPERIENCE'])
            user_data['DRIVING_EXPERIENCE'] = le.transform([selected_exp])[0]
        
        # Past accidents
        user_data['PAST_ACCIDENTS'] = st.number_input("Geçmiş Kaza Sayısı", 
                                                    min_value=0, 
                                                    max_value=10, 
                                                    value=0,
                                                    key="accidents_input")
    
    with col2:
        st.markdown("#### Araç Bilgileri")
        # Vehicle type
        selected_vehicle_type = st.selectbox(
            "Araç Tipi",
            options=list(vehicle_types.keys()),
            format_func=lambda x: vehicle_types[x],
            key="vehicle_type_select"
        )
        user_data['VEHICLE_TYPE'] = selected_vehicle_type
        
        # Vehicle year
        selected_year = st.selectbox(
            "Araç Yılı",
            options=vehicle_years,
            key="vehicle_year_select"
        )
        user_data['VEHICLE_YEAR'] = selected_year
        
        # Annual mileage
        user_data['ANNUAL_MILEAGE'] = st.number_input("Yıllık Kilometre", 
                                                    min_value=0, 
                                                    max_value=100000, 
                                                    value=10000,
                                                    key="mileage_input")
    
    # Calculate button
    if st.button("Sigorta Fiyatını Hesapla", type="primary"):
        # Debug: Print user inputs
        st.write("Debug - User Inputs:")
        st.write(user_data)
        
        # Prepare user data for prediction
        user_df = pd.DataFrame([user_data])
        
        # Ensure all required features are present and in the correct order
        for feature in available_features:
            if feature not in user_df.columns:
                user_df[feature] = df[feature].mean()
        
        # Reorder columns to match training data
        user_df = user_df[available_features]
        
        # Debug: Print prepared data
        st.write("Debug - Prepared Data:")
        st.write(user_df)
        
        # Make prediction
        base_prediction = model.predict(user_df)[0]
        
        # Debug: Print base prediction
        st.write(f"Debug - Base Prediction: {base_prediction}")
        
        # Adjust price for Turkish market with all factors
        turkish_price = adjust_price_for_turkish_market(base_prediction, user_data, df)
        
        # Display results
        st.markdown("---")
        st.markdown("### Tahmini Sigorta Fiyatınız")
        st.markdown(f"""
            <div class="prediction-box">
                <h2>₺{turkish_price:,.2f}</h2>
            </div>
        """, unsafe_allow_html=True)
        
        # Find similar profiles
        similar_profiles = find_similar_profiles(df, user_data)
        
        # Show additional information in expanders
        with st.expander("Benzer Profilleri Görüntüle"):
            st.write("İşte sizin profilinize benzer kişilerin sigorta ücretleri:")
            display_columns = ['AGE', 'DRIVING_EXPERIENCE', 
                             'VEHICLE_TYPE', 'OUTCOME']
            st.dataframe(similar_profiles[display_columns].head())
        
        with st.expander("Veri Analizini Görüntüle"):
            tab1, tab2 = st.tabs(["Demografik", "Risk Faktörleri"])
            
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.histogram(df_display, x='AGE', title='Yaş Dağılımı')
                    st.plotly_chart(fig)
                with col2:
                    fig = px.box(df_display, x='DRIVING_EXPERIENCE', y='OUTCOME', 
                                title='Sürüş Deneyimine Göre Sigorta Ücretleri')
                    st.plotly_chart(fig)
            
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.histogram(df_display, x='VEHICLE_TYPE', 
                                     title='Araç Tipi Dağılımı')
                    st.plotly_chart(fig)
                with col2:
                    fig = px.scatter(df_display, x='ANNUAL_MILEAGE', y='OUTCOME',
                                   title='Yıllık Kilometre ve Sigorta Ücreti İlişkisi')
                    st.plotly_chart(fig)

if __name__ == "__main__":
    main() 