import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

class ModelTrainer:
    def __init__(self):
        self.price_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.insurance_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
    
    def prepare_features(self, data):
        """Prepare feature sets for model training."""
        # Features for both price and insurance prediction
        features = ['year', 'km_driven', 'fuel', 'seller_type', 
                   'transmission', 'owner', 'selling_price']
        
        return features, features
    
    def train_price_model(self, data, features):
        """Train the price prediction model."""
        # Remove selling_price from input features for price prediction
        X = data[features[:-1]]  # All features except selling_price
        y = data['selling_price']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.price_model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.price_model.predict(X_test)
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        return metrics
    
    def train_insurance_model(self, data, features):
        """Train the insurance prediction model."""
        X = data[features]
        # Create synthetic insurance values based on car attributes
        base_insurance = data['selling_price'] * 0.04  # 4% of car value
        age_factor = (2023 - data['year']) * 0.02  # Age impact
        synthetic_insurance = base_insurance * (1 + age_factor)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, synthetic_insurance, test_size=0.2, random_state=42
        )
        
        self.insurance_model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.insurance_model.predict(X_test)
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        return metrics
    
    def predict_price(self, features):
        """Predict car price based on features."""
        return self.price_model.predict(features)
    
    def predict_insurance(self, features):
        """Predict insurance premium based on features."""
        return self.insurance_model.predict(features)
    
    def save_models(self, price_model_path, insurance_model_path):
        """Save trained models to disk."""
        joblib.dump(self.price_model, price_model_path)
        joblib.dump(self.insurance_model, insurance_model_path)
    
    def load_models(self, price_model_path, insurance_model_path):
        """Load trained models from disk."""
        self.price_model = joblib.load(price_model_path)
        self.insurance_model = joblib.load(insurance_model_path) 