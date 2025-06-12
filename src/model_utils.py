

import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os

def load_latest_model():
    """
    Load the latest trained reduced model.
    
    Returns:
    --------
    model : sklearn model object
        The loaded trained reduced model
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(os.path.dirname(current_dir), "models")
    model_path = os.path.join(models_dir, "rf_flashpoint_reduced_latest.pkl")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = joblib.load(model_path)
    print(f"[SUCCESS] Loaded reduced model from {model_path}")
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

def predict_flashpoint(model, X_new):
    """
    Make flashpoint predictions using the trained reduced model.
    
    Parameters:
    -----------
    model : sklearn model object
        The trained model
    X_new : pd.DataFrame
        New data to make predictions on
        
    Returns:
    --------
    predictions : np.array
        Predicted flashpoint values
    """
    feature_info = load_feature_info()
    top_features = feature_info['top_features']
    X_new = X_new[top_features]
    predictions = model.predict(X_new)
    print(f"[SUCCESS] Made predictions for {len(X_new)} samples using top {len(top_features)} features")
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
    """
    Display a summary of the latest model performance.
    """
    try:
        feature_info = load_feature_info()
        
        print("[INFO] Model Performance Summary")
        print("=" * 40)
        
        reduced_perf = feature_info['model_performance']['reduced_model']
        
        print(f"Reduced Model Performance:")
        print(f"  R2 Score: {reduced_perf['R2']:.4f}")
        print(f"  RMSE: {reduced_perf['RMSE']:.4f}")
        print(f"  MAE: {reduced_perf['MAE']:.4f}")
        
        print(f"\n[STATS] Features:")
        print(f"  Total features available: {len(feature_info['all_features'])}")
        print(f"  Features used in model: {len(feature_info['top_features'])}")
        print(f"  Feature reduction: {(1 - len(feature_info['top_features'])/len(feature_info['all_features']))*100:.1f}%")
        
        print(f"\n[TIME] Model created: {feature_info['timestamp']}")
        
    except Exception as e:
        print(f"[ERROR] Error loading model summary: {e}")

if __name__ == "__main__":
   
    print("[INFO] Flashpoint Prediction Model Utilities")
    print("=" * 50)
    
    list_saved_models()

    model_summary()
