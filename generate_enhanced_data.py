import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_synthetic_data(n_samples=75000, fraud_ratio=0.03):
    np.random.seed(42)
    
    # Generate base data
    data = {
        'transaction_id': range(1, n_samples + 1),
        'user_id': np.random.randint(1, 10001, n_samples),
        'transaction_amount': np.random.lognormal(3, 1, n_samples),
        'transaction_type': np.random.choice(['PURCHASE', 'TRANSFER', 'WITHDRAWAL', 'PAYMENT'], n_samples),
        'merchant_category': np.random.choice(['RETAIL', 'TRAVEL', 'ENTERTAINMENT', 'GROCERY', 'ELECTRONICS'], n_samples),
        'device_used': np.random.choice(['MOBILE', 'WEB', 'ATM', 'POS'], n_samples),
        'location_country': np.random.choice(['BR', 'US', 'UK', 'FR', 'DE', 'JP', 'AU'], n_samples)
    }
    
    # Generate timestamps over last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    timestamps = [start_date + timedelta(seconds=int(x)) for x in np.random.randint(0, 30*24*60*60, n_samples)]
    data['timestamp'] = sorted(timestamps)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Assign base country to users
    user_countries = pd.Series(np.random.choice(['BR', 'US', 'UK', 'FR', 'DE'], 10000), index=range(1, 10001))
    df['user_base_country'] = df['user_id'].map(user_countries)
    
    # Initialize fraud labels
    df['is_fraud'] = 0
    
    # Scenario 1: High value transactions
    user_avg_amount = df.groupby('user_id')['transaction_amount'].transform('mean')
    high_value_mask = (df['transaction_amount'] > user_avg_amount * 5) & (np.random.random(n_samples) < 0.7)
    
    # Scenario 2: Unusual location
    unusual_location_mask = (df['location_country'] != df['user_base_country']) & (np.random.random(n_samples) < 0.4)
    
    # Scenario 3: Unusual hour (between 1 AM and 4 AM)
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    unusual_hour_mask = (df['hour'].between(1, 4)) & (np.random.random(n_samples) < 0.3)
    
    # Scenario 4: Rapid small transactions
    df['time_diff'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds()
    df['amount_rolling_mean'] = df.groupby('user_id')['transaction_amount'].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
    rapid_small_mask = (df['time_diff'] < 300) & (df['transaction_amount'] < df['amount_rolling_mean'] * 0.5) & (np.random.random(n_samples) < 0.5)
    
    # Combine fraud scenarios
    fraud_probability = (high_value_mask.astype(float) + 
                       unusual_location_mask.astype(float) + 
                       unusual_hour_mask.astype(float) + 
                       rapid_small_mask.astype(float)) / 4
    
    # Assign fraud labels ensuring desired fraud ratio
    fraud_threshold = np.percentile(fraud_probability, (1 - fraud_ratio) * 100)
    df['is_fraud'] = (fraud_probability >= fraud_threshold).astype(int)
    
    # Add some noise: flip some labels randomly
    noise_mask = np.random.random(n_samples) < 0.01
    df.loc[noise_mask, 'is_fraud'] = 1 - df.loc[noise_mask, 'is_fraud']
    
    # Clean up intermediate columns
    df = df.drop(['user_base_country', 'hour', 'time_diff', 'amount_rolling_mean'], axis=1)
    
    return df

if __name__ == '__main__':
    # Generate synthetic dataset
    df = generate_synthetic_data()
    
    # Save to CSV
    df.to_csv('synthetic_transactions_enhanced.csv', index=False)
    
    # Print summary statistics
    print("Dataset Summary:")
    print(f"Total transactions: {len(df)}")
    print(f"Fraud ratio: {df['is_fraud'].mean():.3f}")
    print("\nSample of the generated data:")
    print(df.head())
    print("\nFeature distributions:")
    print(df.describe())