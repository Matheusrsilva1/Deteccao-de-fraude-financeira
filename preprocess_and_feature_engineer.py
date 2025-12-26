import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from datetime import datetime, timedelta
import joblib

def create_temporal_features(df):
    # Convert timestamp to datetime if it's not already
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Extract temporal features
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    return df

def create_time_based_features(df):
    # Convert timestamp to datetime if it's not already
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by user and timestamp
    df = df.sort_values(['user_id', 'timestamp'])
    
    # Time since last transaction for each user
    df['time_since_last_transaction_user'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds()
    
    # Transaction frequency in last 24h for each user
    def count_transactions_last_24h(group):
        timestamps = pd.to_datetime(group['timestamp'])
        result = []
        for i, current_time in enumerate(timestamps):
            last_24h = current_time - pd.Timedelta(days=1)
            count = sum((timestamps <= current_time) & (timestamps > last_24h)) - 1
            result.append(count)
        return result
    
    df['transaction_frequency_last_24h_user'] = df.groupby('user_id')['timestamp'].transform(
        lambda x: count_transactions_last_24h(pd.DataFrame({'timestamp': x}))
    )
    
    return df

def create_behavioral_features(df):
    # Calculate historical average amount per user (using expanding window)
    df = df.sort_values(['user_id', 'timestamp'])
    df['user_avg_amount_history'] = df.groupby('user_id')['transaction_amount'].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    
    # Calculate amount deviation from user's average
    df['amount_deviation_from_avg'] = df['transaction_amount'] / (df['user_avg_amount_history'] + 1e-6)
    
    return df

def build_preprocessor():
    # Define numeric and categorical columns
    numeric_features = [
        'transaction_amount', 'hour_of_day', 'day_of_week',
        'time_since_last_transaction_user', 'transaction_frequency_last_24h_user',
        'user_avg_amount_history', 'amount_deviation_from_avg'
    ]
    
    categorical_features = [
        'transaction_type', 'merchant_category', 'device_used', 'location_country'
    ]
    
    # Create preprocessing steps for numeric and categorical features
    numeric_transformer = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Drop any other columns not specified
    )
    
    return preprocessor

def preprocess_data(df):
    # Create all features
    df = create_temporal_features(df)
    df = create_time_based_features(df)
    df = create_behavioral_features(df)
    
    # Fill missing values
    df['time_since_last_transaction_user'] = df['time_since_last_transaction_user'].fillna(-1)
    df['transaction_frequency_last_24h_user'] = df['transaction_frequency_last_24h_user'].fillna(0)
    df['user_avg_amount_history'] = df['user_avg_amount_history'].fillna(
        df['transaction_amount'].mean()
    )
    df['amount_deviation_from_avg'] = df['amount_deviation_from_avg'].fillna(1.0)
    
    return df

if __name__ == '__main__':
    # Load data
    df = pd.read_csv('synthetic_transactions_enhanced.csv')
    
    # Preprocess data
    df_processed = preprocess_data(df)
    
    # Build and save preprocessor
    preprocessor = build_preprocessor()
    joblib.dump(preprocessor, 'preprocessor_enhanced.joblib')
    
    print("Preprocessing pipeline created and saved.")
    print("\nProcessed features:")
    print(df_processed.columns.tolist())