import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

class DataProcessor:
    def __init__(self):
        self.data = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self, file_path):
        """Load and perform initial cleaning of the dataset."""
        self.data = pd.read_csv(file_path)
        return self.data
    
    def preprocess_data(self):
        """Preprocess the data for model training."""
        # Create copy of data
        df = self.data.copy()
        
        # Handle categorical variables
        categorical_columns = ['fuel', 'seller_type', 'transmission', 'owner']
        for column in categorical_columns:
            if column in df.columns:
                self.label_encoders[column] = LabelEncoder()
                df[column] = self.label_encoders[column].fit_transform(df[column])
        
        # Handle numerical variables
        numerical_columns = ['year', 'selling_price', 'km_driven']
        df[numerical_columns] = self.scaler.fit_transform(df[numerical_columns])
        
        return df
    
    def calculate_insurance_base(self, car_value, car_age):
        """Calculate base insurance premium."""
        # Basic formula: 4% of car value + age factor
        age_factor = max(1.0, 1.5 - (car_age * 0.05))  # Older cars have lower factor
        base_premium = (car_value * 0.04) * age_factor
        return base_premium
    
    def generate_visualizations(self):
        """Generate various visualizations for data analysis."""
        plt.figure(figsize=(15, 10))
        
        # Price distribution
        plt.subplot(2, 2, 1)
        sns.histplot(self.data['selling_price'], bins=30)
        plt.title('Price Distribution')
        
        # Mileage vs Price
        plt.subplot(2, 2, 2)
        sns.scatterplot(data=self.data, x='km_driven', y='selling_price')
        plt.title('Mileage vs Price')
        
        # Average price by fuel type
        plt.subplot(2, 2, 3)
        sns.boxplot(data=self.data, x='fuel', y='selling_price')
        plt.title('Price by Fuel Type')
        
        # Average price by transmission
        plt.subplot(2, 2, 4)
        sns.boxplot(data=self.data, x='transmission', y='selling_price')
        plt.title('Price by Transmission')
        
        plt.tight_layout()
        return plt.gcf()
    
    def analyze_price_factors(self):
        """Analyze key factors affecting car prices."""
        analysis = {
            'avg_price_by_fuel': self.data.groupby('fuel')['selling_price'].mean().to_dict(),
            'avg_price_by_transmission': self.data.groupby('transmission')['selling_price'].mean().to_dict(),
            'price_km_correlation': self.data['selling_price'].corr(self.data['km_driven']),
            'price_year_correlation': self.data['selling_price'].corr(self.data['year'])
        }
        return analysis 