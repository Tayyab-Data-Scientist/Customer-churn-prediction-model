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

# Route to predict churn
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from the form
        tenure_months = int(request.form['tenure_months'])
        monthly_charges = float(request.form['monthly_charges'])
        city_encoded = int(request.form['city_encoded'])
        gender = int(request.form['gender'])
        senior_citizen = int(request.form['senior_citizen'])
        partner = int(request.form['partner'])
        dependents = int(request.form['dependents'])
        multiple_lines = float(request.form['multiple_lines'])
        internet_service = int(request.form['internet_service'])
        online_security = int(request.form['online_security'])
        online_backup = int(request.form['online_backup'])
        tech_support = int(request.form['tech_support'])
        streaming_tv = int(request.form['streaming_tv'])
        streaming_movies = int(request.form['streaming_movies'])
        contract_type = int(request.form['contract_type'])
        payment_method = int(request.form['payment_method'])

        # Prepare the feature vector
        feature_vector = np.array([[tenure_months, monthly_charges, city_encoded, gender, senior_citizen, partner,
                                    dependents, multiple_lines, internet_service, online_security, online_backup,
                                    tech_support, streaming_tv, streaming_movies, contract_type, payment_method]])

        # Get prediction from the model
        prediction = model.predict(feature_vector)
        prediction_label = 'Churn' if prediction[0] == 1 else 'No Churn'

        return render_template('index.html', prediction=prediction_label)
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

# Run the app on Replit
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)