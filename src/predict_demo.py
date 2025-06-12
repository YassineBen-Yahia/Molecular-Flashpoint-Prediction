
import os
import pandas as pd
import numpy as np
from model_utils import load_latest_model, load_feature_info, predict_flashpoint, model_summary

def demo_prediction():
    """
    Demonstrate flashpoint prediction using the trained reduced model.
    """
    print("[INFO] Flashpoint Prediction Demo - Reduced Model")
    print("=" * 50)
    
    try:
        # Load the test data to demonstrate predictions
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed'))
        
        # Check if test data exists
        X_test_path = os.path.join(data_dir, "X_test.csv")
        y_test_path = os.path.join(data_dir, "y_test.csv")
        
        if not os.path.exists(X_test_path) or not os.path.exists(y_test_path):
            print(f"[ERROR] Test data not found in {data_dir}")
            print("Please run data preprocessing first!")
            return
        
        X_test = pd.read_csv(X_test_path)
        y_test = pd.read_csv(y_test_path).squeeze()
        
        # Load the trained reduced model
        model = load_latest_model()
        
        # Make predictions on a subset of test data
        sample_size = 10
        X_sample = X_test.head(sample_size)
        y_actual = y_test.head(sample_size)
        
        predictions = predict_flashpoint(model, X_sample)
        
        # Display results
        print(f"\n[RESULTS] Prediction Results (Sample of {sample_size} compounds):")
        print("=" * 60)
        print(f"{'Index':<6} {'Actual':<10} {'Predicted':<12} {'Error':<10} {'Error %':<10}")
        print("-" * 60)
        
        for i in range(sample_size):
            actual = y_actual.iloc[i]
            pred = predictions[i]
            error = abs(actual - pred)
            error_pct = (error / actual) * 100 if actual != 0 else 0
            
            print(f"{i:<6} {actual:<10.2f} {pred:<12.2f} {error:<10.2f} {error_pct:<10.1f}%")
        
        # Calculate summary statistics
        errors = np.abs(y_actual - predictions)
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        mean_error_pct = np.mean((errors / y_actual) * 100)
        
        print("-" * 60)
        print(f"Mean Absolute Error: {mean_error:.2f} K")
        print(f"Max Error: {max_error:.2f} K")
        print(f"Mean Error %: {mean_error_pct:.1f}%")
        
        print(f"\n[SUCCESS] Demo completed successfully using reduced model!")
        
    except Exception as e:
        print(f"[ERROR] Error during prediction demo: {e}")
        print("Make sure you have trained and saved the reduced model first!")

def custom_prediction_demo():
    """
    Allow users to input custom SMILES strings for prediction.
    Note: This would require the full preprocessing pipeline to be available.
    """
    print("\n[INFO] Custom SMILES Prediction Demo")
    print("=" * 40)
    print("Note: This demo requires the preprocessing pipeline to convert SMILES to features.")
    print("For now, use the test data demo above.")

if __name__ == "__main__":
    # Show model summary
    print("[INFO] Molecular Flashpoint Prediction System")
    print("=" * 50)
    model_summary()
    
    # Run prediction demo
    demo_prediction()
    
    # Show custom prediction option
    custom_prediction_demo()
    
