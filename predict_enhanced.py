import pandas as pd
import numpy as np
import joblib
import shap
from preprocess_and_feature_engineer import preprocess_data

def predict_fraud_enhanced(transaction_data):
    # Load model, preprocessor and threshold
    model = joblib.load('fraud_detection_model_enhanced.joblib')
    preprocessor = joblib.load('preprocessor_enhanced.joblib')
    optimal_threshold = joblib.load('optimal_threshold.joblib')
    
    # Convert single transaction to DataFrame
    df = pd.DataFrame([transaction_data])
    
    # Preprocess the transaction
    df_processed = preprocess_data(df)
    
    # Apply preprocessor
    X_processed = preprocessor.transform(df_processed)
    
    # Get prediction probability
    fraud_probability = model.predict_proba(X_processed)[0, 1]
    
    # Make prediction using optimal threshold
    is_fraud = int(fraud_probability >= optimal_threshold)
    
    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_processed)
    
    # If shap_values is a list (for binary classification), take the second element
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # Get feature names after preprocessing
    feature_names = []
    for name, transformer, features in preprocessor.transformers_:
        if name == 'num':
            feature_names.extend(features)
        elif name == 'cat':
            for feature in features:
                feature_names.extend([f"{feature}_{val}" for val in 
                                   transformer.named_steps['onehot'].get_feature_names_out([feature])])
    
    # Create DataFrame with feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'shap_value': shap_values[0]
    })
    feature_importance['abs_shap_value'] = abs(feature_importance['shap_value'])
    feature_importance = feature_importance.sort_values('abs_shap_value', ascending=False)
    
    # Get top 5 features and their SHAP values
    top_features = feature_importance.head(5).to_dict('records')
    
    # Determine risk level
    if fraud_probability < 0.3:
        risk_level = 'LOW'
    elif fraud_probability < 0.7:
        risk_level = 'MEDIUM'
    else:
        risk_level = 'HIGH'
    
    # Prepare response
    result = {
        'is_fraud': is_fraud,
        'fraud_probability': float(fraud_probability),
        'threshold_used': float(optimal_threshold),
        'risk_level': risk_level,
        'explanation': {
            'top_features': top_features
        }
    }
    
    return result

if __name__ == '__main__':
    # Example transaction for testing
    example_transaction = {
        'transaction_id': 1,
        'user_id': 1001,
        'timestamp': pd.Timestamp.now().isoformat(),
        'transaction_amount': 5000.0,
        'transaction_type': 'PURCHASE',
        'merchant_category': 'ELECTRONICS',
        'device_used': 'WEB',
        'location_country': 'US'
    }
    
    # Make prediction
    result = predict_fraud_enhanced(example_transaction)
    
    # Print results
    print("\nFraud Detection Results:")
    print(f"Is Fraud: {result['is_fraud']}")
    print(f"Fraud Probability: {result['fraud_probability']:.3f}")
    print(f"Threshold Used: {result['threshold_used']:.3f}")
    print(f"Risk Level: {result['risk_level']}")
    print("\nTop Contributing Features:")
    for feature in result['explanation']['top_features']:
        print(f"- {feature['feature']}: {feature['shap_value']:.4f}")