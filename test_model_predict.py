import pickle
import joblib

try:
    with open('random_forest_tpot_model.pkl', 'rb') as model_file:
        model = joblib.load(model_file)  # Adjust encoding if necessary
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
