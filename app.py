import logging  # Import the logging module

# Configure logging (adjust level as needed)
logging.basicConfig(level=logging.DEBUG)

# Rest of your Flask application code
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load your model
model_path = 'random_forest_tpot_model.pkl'
model = None

try:
    with open(model_path, 'rb') as model_file:
        model = joblib.load(model_file)
        logging.info("Model loaded successfully")  # New logging statement
except Exception as e:
    logging.error(f"Error loading model: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check the server logs for more information.'})

    try:
        data = request.get_json(force=True)
        logging.info(f"Received input data: {data}")

        # Basic data validation (assuming 'input' key exists)
        if 'input' not in data:
            return jsonify({'error': 'Missing required key "input" in JSON data.'})

        input_data = np.array(data['input']).reshape(1, -1)
        prediction = model.predict(input_data)
        logging.info(f"Prediction: {prediction}")
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        if isinstance(e, (ValueError, TypeError)):  # Specific error handling
            return jsonify({'error': 'Invalid data format. Please check the input data.'})
        else:
            return jsonify({'error': str(e)})  # Generic error for others

if __name__ == '__main__':
    app.run(debug=True)
