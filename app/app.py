import os
import json
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

# Initialize Flask app with the correct paths
app = Flask(__name__, static_folder='static', template_folder='templates')

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Note: You need to ensure the parent directories exist or adjust these paths
MODELS_DIR = os.path.join(BASE_DIR, '../models')
METADATA_DIR = os.path.join(BASE_DIR, '../metadata')
DATA_DIR = os.path.join(BASE_DIR, '../data/processed')

# Load the encoded data mapping
ENCODED_DATA_PATH = os.path.join(DATA_DIR, 'train_encoded_data.json')
encoded_data_mapping = {}
try:
    if not os.path.exists(ENCODED_DATA_PATH):
        # Fallback path if file is in the same directory as app.py
        ENCODED_DATA_PATH = os.path.join(BASE_DIR, 'train_encoded_data.json')
        if not os.path.exists(ENCODED_DATA_PATH):
            raise FileNotFoundError(f"Encoded data file not found at {os.path.join(DATA_DIR, 'train_encoded_data.json')} or {os.path.join(BASE_DIR, 'train_encoded_data.json')}")

    with open(ENCODED_DATA_PATH, 'r') as f:
        encoded_data_mapping = json.load(f)
    print(f"Successfully loaded encoded data mapping from: {ENCODED_DATA_PATH}")
except (FileNotFoundError, json.JSONDecodeError) as e:
    print(f"Error loading encoded data mapping: {e}")
    # Handle this error gracefully in a production app

# Load the model
MODEL_PATH = os.path.join(MODELS_DIR, 'current_best_model.pkl')
model = None
try:
    if not os.path.exists(MODEL_PATH):
        # Fallback path if file is in the same directory as app.py
        MODEL_PATH = os.path.join(BASE_DIR, 'current_best_model.pkl')
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {os.path.join(MODELS_DIR, 'current_best_model.pkl')} or {os.path.join(BASE_DIR, 'current_best_model.pkl')}")

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print(f"Successfully loaded model from: {MODEL_PATH}")

    # Add a sanity check to ensure the loaded object has the necessary methods
    if not hasattr(model, 'predict') or not hasattr(model, 'predict_proba'):
        raise TypeError("Loaded object does not have 'predict' and 'predict_proba' methods. Is it a trained model?")

except (FileNotFoundError, pickle.UnpicklingError, TypeError) as e:
    print(f"Error loading or validating model: {e}")
    model = None # Ensure model is None if loading failed

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives input data from the frontend, preprocesses it,
    and returns a prediction.
    """
    if model is None:
        return jsonify({'error': 'Model not loaded correctly.'}), 500

    try:
        # Get the JSON data from the request
        data = request.get_json(force=True)
        print(f"Received data: {data}")

        # Define the feature order as expected by the model
        feature_order = [
            "city", "city_development_index", "gender", "relevent_experience",
            "enrolled_university", "education_level", "major_discipline", "experience",
            "company_size", "company_type", "last_new_job", "training_hours"
        ]

        input_features = []
        for feature_name in feature_order:
            value = data.get(feature_name)
            if value is None:
                return jsonify({'error': f"Missing feature: '{feature_name}'"}), 400

            # Convert categorical string values to their encoded integer values
            if feature_name in encoded_data_mapping:
                encoded_value = encoded_data_mapping[feature_name].get(value)
                if encoded_value is None:
                    return jsonify({'error': f"Unknown value for '{feature_name}': '{value}'"}), 400
                input_features.append(encoded_value)
            else:
                # For numerical features, convert to float
                input_features.append(float(value))

        # --- CORRECTED PART STARTS HERE ---
        # Convert to a Pandas DataFrame with the correct feature names
        import pandas as pd
        final_features = pd.DataFrame([input_features], columns=feature_order)

        # Make prediction
        # The .predict() and .predict_proba() methods should now work without warnings
        prediction = model.predict(final_features)[0]
        probabilities = model.predict_proba(final_features)[0].tolist()

        # Return prediction and probabilities
        return jsonify({
            'prediction': int(prediction),
            'probabilities': probabilities
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Ensure static and templates folders exist for Flask to find them
    os.makedirs(os.path.join(BASE_DIR, 'static/css'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'static/js'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'templates'), exist_ok=True)

    # Create dummy style.css only if it doesn't exist
    style_path = os.path.join(BASE_DIR, 'static/css/style.css')
    if not os.path.exists(style_path):
        with open(style_path, 'w') as f:
            f.write("/* Add your custom styles here */\n")
            
    # Create dummy script.js only if it doesn't exist
    script_path = os.path.join(BASE_DIR, 'static/js/script.js')
    if not os.path.exists(script_path):
        with open(script_path, 'w') as f:
            f.write("/* Add your custom scripts here */\n")
            
    # Create dummy index.html only if it doesn't exist
    index_path = os.path.join(BASE_DIR, 'templates/index.html')
    if not os.path.exists(index_path):
        with open(index_path, 'w') as f:
            f.write("<!-- This is a dummy index.html -->\n")
            
    app.run(debug=True)
