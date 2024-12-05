from flask import Flask, request, jsonify
import numpy as np
import joblib
import logging
from flask_cors import CORS

# Set up basic logging
logging.basicConfig(level=logging.DEBUG)

# Provide the absolute path to your model file
model_path = r"E:\Data Science Projects\Daibetes_prediction\Backend\decision_tree_model.pkl"

# Load model using joblib
logging.info('Loading the model...')
model = joblib.load(model_path)
logging.info('Model loaded successfully')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/')
def home():
    return "Hello, World!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logging.info('Received prediction request')
        # Receive JSON data from the request
        data = request.get_json()
        logging.debug(f'Request data: {data}')
        
        # Extract Data fields
        age = data.get('age')
        glucose = data.get('glucose')
        blood_pressure = data.get('bloodPressure')
        insulin = data.get('insulin')
        bmi = data.get('bmi')
        diabetes_pedigree = data.get('diabetesPedigree')
        
        # Create a feature array in the order your model expects.
        features = np.array([[age, glucose, blood_pressure, insulin, bmi, diabetes_pedigree]])
        
        # Perform your prediction using the loaded model
        prediction = model.predict(features)
        
        # Convert numeric Prediction to a meaningful label if necessary
        prediction_label = 'Diabetic' if prediction[0] == 1 else 'Not Diabetic'
        
        logging.info('Prediction successful')
        return jsonify({'prediction': prediction_label})
    except Exception as e:
        logging.error(f'Error in prediction: {e}')
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    logging.info('Starting the Flask app...')
    app.run(debug=True, host='0.0.0.0', port=8000)
