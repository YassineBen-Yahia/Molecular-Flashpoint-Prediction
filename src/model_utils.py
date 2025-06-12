

import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os

def load_latest_model(model_type='full'):
  
   
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(os.path.dirname(current_dir), "models")
    
    if model_type == 'full':
        model_path = os.path.join(models_dir, "rf_flashpoint_full_latest.pkl")
    elif model_type == 'reduced':
        model_path = os.path.join(models_dir, "rf_flashpoint_reduced_latest.pkl")
    else:
        raise ValueError("model_type must be either 'full' or 'reduced'")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = joblib.load(model_path)
    print(f"[SUCCESS] Loaded {model_type} model from {model_path}")
    return model

def load_feature_info():
  
    # Get the absolute path to the models directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(os.path.dirname(current_dir), "models")
    feature_info_path = os.path.join(models_dir, "feature_info_latest.pkl")
    
    if not os.path.exists(feature_info_path):
        raise FileNotFoundError(f"Feature info file not found: {feature_info_path}")
    
    feature_info = joblib.load(feature_info_path)
    print(f"[SUCCESS] Loaded feature information from {feature_info_path}")
    return feature_info

def predict_flashpoint(model, X_new, model_type='full'):
   
    
    feature_info = load_feature_info()
    top_features = feature_info['top_features']
    X_new = X_new[top_features]
    predictions = model.predict(X_new)
    print(f"[SUCCESS] Made predictions for {len(X_new)} samples")
    return predictions

def list_saved_models():
   
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(os.path.dirname(current_dir), "models")
    
    if not os.path.exists(models_dir):
        print("No models directory found")
        return
    
    print("[INFO] Saved Models:")
    print("=" * 50)
    
    for filename in os.listdir(models_dir):
        if filename.endswith('.pkl'):
            filepath = os.path.join(models_dir, filename)
            file_size = os.path.getsize(filepath) / (1024 * 1024)  # Convert to MB
            modified_time = datetime.fromtimestamp(os.path.getmtime(filepath))
            
            print(f"[FILE] {filename}")
            print(f"   Size: {file_size:.2f} MB")
            print(f"   Modified: {modified_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print()

def model_summary():
   
    try:
        feature_info = load_feature_info()
        
        print("[INFO] Model Performance Summary")
        print("=" * 40)
        
        full_perf = feature_info['model_performance']['full_model']
        reduced_perf = feature_info['model_performance']['reduced_model']
        
        print(f"Full Model:")
        print(f"  R2 Score: {full_perf['R2']:.4f}")
        print(f"  RMSE: {full_perf['RMSE']:.4f}")
        print(f"  MAE: {full_perf['MAE']:.4f}")
        
        print(f"\nReduced Model:")
        print(f"  R2 Score: {reduced_perf['R2']:.4f}")
        print(f"  RMSE: {reduced_perf['RMSE']:.4f}")
        print(f"  MAE: {reduced_perf['MAE']:.4f}")
        
        print(f"\n[STATS] Features:")
        print(f"  Total features: {len(feature_info['all_features'])}")
        print(f"  Top features used: {len(feature_info['top_features'])}")
        print(f"  Feature reduction: {(1 - len(feature_info['top_features'])/len(feature_info['all_features']))*100:.1f}%")
        
        print(f"\n[TIME] Model created: {feature_info['timestamp']}")
        
    except Exception as e:
        print(f"[ERROR] Error loading model summary: {e}")

if __name__ == "__main__":
   
    print("[INFO] Flashpoint Prediction Model Utilities")
    print("=" * 50)
    
    list_saved_models()

    model_summary()
