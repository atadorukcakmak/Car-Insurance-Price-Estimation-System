import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import pandas as pd

class ModelTrainer:
    def __init__(self):
        # Define numeric and categorical features
        self.numeric_features = ['year', 'km_driven']
        self.categorical_features = ['fuel', 'seller_type', 'transmission', 'owner']
        
        # Create preprocessing pipeline
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), self.numeric_features),
                ('cat', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ]), self.categorical_features)
            ])
        
        # Create model pipelines
        self.price_pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ))
        ])
        
        self.insurance_pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ))
        ])
    
    def prepare_features(self, data):
        """Prepare feature sets for model training."""
        # Features for both price and insurance prediction
        features = self.numeric_features + self.categorical_features + ['selling_price']
        return features, features
    
    def train_price_model(self, data, features):
        """Train the price prediction model using the pipeline."""
        X = data[self.numeric_features + self.categorical_features]
        y = data['selling_price']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.price_pipeline.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.price_pipeline.predict(X_test)
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        return metrics
    
    def train_insurance_model(self, data, features):
        """Train the insurance prediction model using the pipeline."""
        X = data[self.numeric_features + self.categorical_features]
        # Create synthetic insurance values based on car attributes
        base_insurance = data['selling_price'] * 0.04  # 4% of car value
        age_factor = (2023 - data['year']) * 0.02  # Age impact
        synthetic_insurance = base_insurance * (1 + age_factor)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, synthetic_insurance, test_size=0.2, random_state=42
        )
        
        self.insurance_pipeline.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.insurance_pipeline.predict(X_test)
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        return metrics
    
    def predict_price(self, features):
        """Predict car price based on features."""
        # Convert features to DataFrame if it's not already
        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(features, columns=self.numeric_features + self.categorical_features)
        return self.price_pipeline.predict(features)
    
    def predict_insurance(self, features):
        """Predict insurance premium based on features."""
        # Convert features to DataFrame if it's not already
        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(features, columns=self.numeric_features + self.categorical_features)
        return self.insurance_pipeline.predict(features)
    
    def save_models(self, price_model_path, insurance_model_path):
        """Save trained models to disk."""
        joblib.dump(self.price_pipeline, price_model_path)
        joblib.dump(self.insurance_pipeline, insurance_model_path)
    
    def load_models(self, price_model_path, insurance_model_path):
        """Load trained models from disk."""
        self.price_pipeline = joblib.load(price_model_path)
        self.insurance_pipeline = joblib.load(insurance_model_path) 