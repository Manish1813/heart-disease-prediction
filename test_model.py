import pandas as pd
import numpy as np
import joblib

def run_tests():
    print("--- Starting Backend Model Tests ---")
    
    # 1. Load Scaler and Model
    print("Loading scaler and model...")
    scaler = joblib.load('scaler.joblib')
    model = joblib.load('heart_disease_model.joblib')
    print("Loaded successfully!")
    
    # 2. Define features structure
    features = [
        'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'Diabetes', 
        'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 
        'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 
        'Age', 'Education', 'Income'
    ]
    
    # 3. Create a Low-Risk Profile
    # Normal weight, active, non-smoker, good health, college grad, high income, young-middle aged
    low_risk = {
        'HighBP': 0.0, 'HighChol': 0.0, 'CholCheck': 1.0, 'BMI': 22.0, 
        'Smoker': 0.0, 'Stroke': 0.0, 'Diabetes': 0.0, 'PhysActivity': 1.0, 
        'Fruits': 1.0, 'Veggies': 1.0, 'HvyAlcoholConsump': 0.0, 'AnyHealthcare': 1.0, 
        'NoDocbcCost': 0.0, 'GenHlth': 1.0, 'MentHlth': 0.0, 'PhysHlth': 0.0, 
        'DiffWalk': 0.0, 'Sex': 0.0, 'Age': 4.0, 'Education': 6.0, 'Income': 8.0
    }
    
    # 4. Create a High-Risk Profile
    # High BP, high chol, obese, smoker, stroke history, diabetic, poor health, sedentary
    high_risk = {
        'HighBP': 1.0, 'HighChol': 1.0, 'CholCheck': 1.0, 'BMI': 34.0, 
        'Smoker': 1.0, 'Stroke': 1.0, 'Diabetes': 2.0, 'PhysActivity': 0.0, 
        'Fruits': 0.0, 'Veggies': 0.0, 'HvyAlcoholConsump': 0.0, 'AnyHealthcare': 1.0, 
        'NoDocbcCost': 0.0, 'GenHlth': 5.0, 'MentHlth': 15.0, 'PhysHlth': 20.0, 
        'DiffWalk': 1.0, 'Sex': 1.0, 'Age': 11.0, 'Education': 4.0, 'Income': 4.0
    }
    
    df_low = pd.DataFrame([low_risk])
    df_high = pd.DataFrame([high_risk])
    
    # Scale numerical fields (BMI, MentHlth, PhysHlth)
    num_cols = ['BMI', 'MentHlth', 'PhysHlth']
    df_low[num_cols] = scaler.transform(df_low[num_cols])
    df_high[num_cols] = scaler.transform(df_high[num_cols])
    
    # Predict
    prob_low = model.predict_proba(df_low)[0][1]
    pred_low = model.predict(df_low)[0]
    
    prob_high = model.predict_proba(df_high)[0][1]
    pred_high = model.predict(df_high)[0]
    
    print(f"\n[Test 1] Low-Risk Profile Prediction:")
    print(f" - Raw predicted class: {pred_low}")
    print(f" - Heart Disease Probability: {prob_low * 100:.2f}%")
    
    print(f"\n[Test 2] High-Risk Profile Prediction:")
    print(f" - Raw predicted class: {pred_high}")
    print(f" - Heart Disease Probability: {prob_high * 100:.2f}%")
    
    # Assertions
    assert prob_low < prob_high, "High-risk profile should have a higher risk than low-risk profile."
    print("\nAssertion passed: High-risk probability is higher than low-risk probability.")
    print("Backend tests completed successfully!")

if __name__ == '__main__':
    run_tests()
