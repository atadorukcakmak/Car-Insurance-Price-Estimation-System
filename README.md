# Car Insurance and Pricing Estimation System

This system provides car insurance estimates and price predictions based on various vehicle attributes using machine learning algorithms.

## Features
- Insurance cost estimation
- Car price prediction (buying and selling)
- Data visualization and analytics
- User-friendly interface for input

## Setup
1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Place the vehicle dataset CSV file in the `data` directory

3. Run the application:
```bash
streamlit run app.py
```

## Project Structure
- `data/` - Contains the dataset
- `src/` - Source code
  - `data_processor.py` - Data preprocessing and analysis
  - `model_trainer.py` - Machine learning model training
  - `utils.py` - Utility functions
- `models/` - Trained model files
- `app.py` - Main Streamlit application 
