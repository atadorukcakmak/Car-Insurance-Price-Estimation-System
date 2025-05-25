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
        """Calculate base insurance premium for Turkey."""
        # Ensure positive car value
        car_value = abs(car_value)
        
        # Base rate varies by car age (newer cars have lower rates)
        if car_age <= 1:
            base_rate = 0.03  # 3% for new cars
        elif car_age <= 3:
            base_rate = 0.035  # 3.5% for 2-3 year old cars
        elif car_age <= 5:
            base_rate = 0.04  # 4% for 4-5 year old cars
        else:
            base_rate = 0.045  # 4.5% for older cars
            
        # Calculate base premium
        base_premium = car_value * base_rate
        
        # Apply age factor (slightly higher for older cars)
        age_factor = 1.0 + (car_age * 0.01)  # 1% increase per year
        
        # Calculate final premium
        final_premium = base_premium * age_factor
        
        # Ensure minimum premium
        min_premium = 5000  # Minimum annual premium in Turkey
        return max(final_premium, min_premium)
    
    def generate_visualizations(self):
        """Generate enhanced visualizations using seaborn."""
        # Set the style
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Price Distribution with KDE
        plt.subplot(3, 2, 1)
        sns.histplot(data=self.data, x='selling_price', bins=30, kde=True)
        plt.title('Price Distribution with KDE', fontsize=12, pad=15)
        plt.xlabel('Price (TRY)')
        
        # 2. Mileage vs Price with regression line
        plt.subplot(3, 2, 2)
        sns.regplot(data=self.data, x='km_driven', y='selling_price', 
                   scatter_kws={'alpha':0.5}, line_kws={'color': 'red'})
        plt.title('Mileage vs Price with Regression Line', fontsize=12, pad=15)
        plt.xlabel('Kilometers Driven')
        plt.ylabel('Price (TRY)')
        
        # 3. Price by Fuel Type with enhanced boxplot
        plt.subplot(3, 2, 3)
        sns.boxplot(data=self.data, x='fuel', y='selling_price', 
                   showfliers=False, width=0.7)
        sns.stripplot(data=self.data, x='fuel', y='selling_price', 
                     size=4, alpha=0.3, color='black')
        plt.title('Price Distribution by Fuel Type', fontsize=12, pad=15)
        plt.xticks(rotation=45)
        
        # 4. Price by Transmission with violin plot
        plt.subplot(3, 2, 4)
        sns.violinplot(data=self.data, x='transmission', y='selling_price')
        plt.title('Price Distribution by Transmission', fontsize=12, pad=15)
        
        # 5. Year vs Price with enhanced scatter
        plt.subplot(3, 2, 5)
        sns.scatterplot(data=self.data, x='year', y='selling_price', 
                       hue='transmission', alpha=0.6)
        plt.title('Price Trends by Year and Transmission', fontsize=12, pad=15)
        plt.xlabel('Year')
        plt.ylabel('Price (TRY)')
        
        # 6. Correlation Heatmap
        plt.subplot(3, 2, 6)
        numeric_cols = ['year', 'km_driven', 'selling_price']
        correlation = self.data[numeric_cols].corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Heatmap', fontsize=12, pad=15)
        
        plt.tight_layout()
        return fig
    
    def generate_additional_visualizations(self):
        """Generate additional detailed visualizations."""
        fig = plt.figure(figsize=(20, 10))
        
        # 1. Price by Owner Count
        plt.subplot(2, 2, 1)
        sns.boxplot(data=self.data, x='owner', y='selling_price')
        plt.title('Price Distribution by Number of Owners', fontsize=12, pad=15)
        plt.xticks(rotation=45)
        
        # 2. Price by Seller Type
        plt.subplot(2, 2, 2)
        sns.violinplot(data=self.data, x='seller_type', y='selling_price')
        plt.title('Price Distribution by Seller Type', fontsize=12, pad=15)
        
        # 3. Year-wise Price Trends
        plt.subplot(2, 2, 3)
        yearly_avg = self.data.groupby('year')['selling_price'].mean().reset_index()
        sns.lineplot(data=yearly_avg, x='year', y='selling_price', marker='o')
        plt.title('Average Price Trends Over Years', fontsize=12, pad=15)
        
        # 4. Mileage Distribution by Fuel Type
        plt.subplot(2, 2, 4)
        sns.boxplot(data=self.data, x='fuel', y='km_driven')
        plt.title('Mileage Distribution by Fuel Type', fontsize=12, pad=15)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        return fig
    
    def analyze_price_factors(self):
        """Analyze key factors affecting car prices."""
        analysis = {
            'avg_price_by_fuel': self.data.groupby('fuel')['selling_price'].mean().to_dict(),
            'avg_price_by_transmission': self.data.groupby('transmission')['selling_price'].mean().to_dict(),
            'price_km_correlation': self.data['selling_price'].corr(self.data['km_driven']),
            'price_year_correlation': self.data['selling_price'].corr(self.data['year'])
        }
        return analysis 