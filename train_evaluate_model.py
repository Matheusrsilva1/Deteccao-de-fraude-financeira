import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc, f1_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess_and_feature_engineer import preprocess_data, build_preprocessor

def load_and_prepare_data():
    # Load and preprocess data
    df = pd.read_csv('synthetic_transactions_enhanced.csv')
    df_processed = preprocess_data(df)
    
    # Prepare features and target
    target = 'is_fraud'
    features = [col for col in df_processed.columns if col != target]
    
    X = df_processed[features]
    y = df_processed[target]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Build and fit preprocessor
    preprocessor = build_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Save preprocessor
    joblib.dump(preprocessor, 'preprocessor_enhanced.joblib')
    
    return X_train_transformed, X_test_transformed, y_train, y_test

def train_and_optimize_model(X_train, y_train):
    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    # Define model and parameter space
    model = XGBClassifier(random_state=42)
    param_space = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5]
    }
    
    # Perform randomized search
    search = RandomizedSearchCV(
        model,
        param_space,
        n_iter=20,
        scoring='f1',
        cv=3,
        random_state=42,
        n_jobs=-1
    )
    
    # Fit the model
    search.fit(X_resampled, y_resampled)
    
    print("Best parameters:", search.best_params_)
    print("Best F1 score:", search.best_score_)
    
    return search.best_estimator_

def optimize_threshold(model, X_test, y_test):
    # Get probabilities for positive class
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    
    # Calculate F1 score for different thresholds
    f1_scores = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        f1 = f1_score(y_test, y_pred)
        f1_scores.append(f1)
    
    # Find optimal threshold
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Plot Precision-Recall curve
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, label=f'PR curve (AUC = {auc(recall, precision):.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig('precision_recall_curve.png')
    plt.close()
    
    return optimal_threshold

def evaluate_model(model, X_test, y_test, threshold=0.5):
    # Get probabilities and make predictions
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    
    # Calculate metrics
    print(f"\nEvaluation with threshold = {threshold}:")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix (threshold = {threshold})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'confusion_matrix_threshold_{threshold}.png')
    plt.close()
    
    # Calculate ROC AUC
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"\nROC AUC Score: {roc_auc:.3f}")

def save_model_and_threshold(model, threshold):
    # Save model and threshold
    joblib.dump(model, 'fraud_detection_model_enhanced.joblib')
    joblib.dump(threshold, 'optimal_threshold.joblib')

if __name__ == '__main__':
    # Load and prepare data
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    
    # Train and optimize model
    print("Training and optimizing model...")
    model = train_and_optimize_model(X_train, y_train)
    
    # Evaluate with default threshold
    print("\nEvaluating model with default threshold (0.5)...")
    evaluate_model(model, X_test, y_test)
    
    # Find optimal threshold
    print("\nFinding optimal threshold...")
    optimal_threshold = optimize_threshold(model, X_test, y_test)
    print(f"Optimal threshold: {optimal_threshold:.3f}")
    
    # Evaluate with optimal threshold
    print("\nEvaluating model with optimal threshold...")
    evaluate_model(model, X_test, y_test, optimal_threshold)
    
    # Save model and threshold
    save_model_and_threshold(model, optimal_threshold)
    print("\nModel and optimal threshold saved.")