from flask import Flask, request, render_template
import numpy as np
import pickle

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    """Renders the home page with the input form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request."""
    # Get the feature values from the form
    # The order must match the training data columns
    features = [
        float(request.form['age']),
        int(request.form['anaemia']),
        float(request.form['creatinine_phosphokinase']),
        int(request.form['diabetes']),
        float(request.form['ejection_fraction']),
        int(request.form['high_blood_pressure']),
        float(request.form['platelets']),
        float(request.form['serum_creatinine']),
        float(request.form['serum_sodium']),
        int(request.form['sex']),
        int(request.form['smoking']),
        float(request.form['time'])
    ]

    # Convert features to a numpy array and reshape for a single prediction
    final_features = np.array(features).reshape(1, -1)

    # Scale the input features using the loaded scaler
    scaled_features = scaler.transform(final_features)

    # Make prediction
    prediction = model.predict(scaled_features)
    prediction_prob = model.predict_proba(scaled_features)

    # Determine the output message
    if prediction[0] == 1:
        # Calculate probability for the "likely to die" class
        probability = round(prediction_prob[0][1] * 100, 2)
        output_message = f"High Risk of a Heart Failure Event"
        output_prob = f"Probability: {probability}%"
        result_class = "high-risk"
    else:
        # Calculate probability for the "not likely to die" class
        probability = round(prediction_prob[0][0] * 100, 2)
        output_message = f"Low Risk of a Heart Failure Event"
        output_prob = f"Probability: {probability}%"
        result_class = "low-risk"

    # Render the result page with the prediction
    return render_template('result.html', prediction_text=output_message, prediction_prob=output_prob, result_class=result_class)

if __name__ == "__main__":
    app.run(debug=True)