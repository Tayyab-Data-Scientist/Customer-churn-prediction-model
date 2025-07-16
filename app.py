import os
import pickle
from flask import Flask, render_template, request, jsonify
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = None
model_path = 'churn_predictor.pkl'
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
else:
    raise FileNotFoundError(f"Model file '{model_path}' not found!")

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')


# Safe conversion functions
def safe_int_cast(value, field_name):
    if isinstance(value, str) and value.lower().strip() in ['nan', 'inf', '-inf', 'infinity', '-infinity']:
        raise ValueError(f"Invalid value for {field_name}: {value}")
    try:
        return int(value)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid integer value for {field_name}: {value}")


def safe_float_cast(value, field_name):
    if isinstance(value, str) and value.lower().strip() in ['nan', 'inf', '-inf', 'infinity', '-infinity']:
        raise ValueError(f"Invalid value for {field_name}: {value}")
    try:
        result = float(value)
        if not np.isfinite(result):
            raise ValueError(f"Invalid finite value for {field_name}: {value}")
        return result
    except (ValueError, TypeError):
        raise ValueError(f"Invalid float value for {field_name}: {value}")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from the form with safe casting
        tenure_months = safe_int_cast(request.form['tenure_months'], 'tenure_months')
        monthly_charges = safe_float_cast(request.form['monthly_charges'], 'monthly_charges')
        city_encoded = safe_int_cast(request.form['city_encoded'], 'city_encoded')
        gender = safe_int_cast(request.form['gender'], 'gender')
        senior_citizen = safe_int_cast(request.form['senior_citizen'], 'senior_citizen')
        partner = safe_int_cast(request.form['partner'], 'partner')
        dependents = safe_int_cast(request.form['dependents'], 'dependents')
        multiple_lines = safe_float_cast(request.form['multiple_lines'], 'multiple_lines')
        internet_service = safe_int_cast(request.form['internet_service'], 'internet_service')
        online_security = safe_int_cast(request.form['online_security'], 'online_security')
        online_backup = safe_int_cast(request.form['online_backup'], 'online_backup')
        tech_support = safe_int_cast(request.form['tech_support'], 'tech_support')
        streaming_tv = safe_int_cast(request.form['streaming_tv'], 'streaming_tv')
        streaming_movies = safe_int_cast(request.form['streaming_movies'], 'streaming_movies')
        contract_type = safe_int_cast(request.form['contract_type'], 'contract_type')
        payment_method = safe_int_cast(request.form['payment_method'], 'payment_method')

        # Prepare the feature vector
        feature_vector = np.array([[
            tenure_months, monthly_charges, city_encoded, gender,
            senior_citizen, partner, dependents, multiple_lines,
            internet_service, online_security, online_backup, tech_support,
            streaming_tv, streaming_movies, contract_type, payment_method
        ]])

        # Get prediction from the model
        prediction = model.predict(feature_vector)
        prediction_label = 'Churn' if prediction[0] == 1 else 'No Churn'

        return render_template('index.html', prediction=prediction_label)
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")


# Run the app on Replit
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
