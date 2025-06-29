from flask import Flask, request, render_template
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load the pre-trained model
def load_model():
    model_path = 'heart_disease_model.pkl'
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        raise FileNotFoundError("Model file 'heart_disease_model.pkl' not found. Please train and save the model locally first.")
    return model

# Load the model
model = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        input_data = [
            float(request.form['age']),
            float(request.form['sex']),
            float(request.form['cp']),
            float(request.form['trestbps']),
            float(request.form['chol']),
            float(request.form['fbs']),
            float(request.form['restecg']),
            float(request.form['thalach']),
            float(request.form['exang']),
            float(request.form['oldpeak']),
            float(request.form['slope']),
            float(request.form['ca']),
            float(request.form['thal'])
        ]
        
        # Convert input data to numpy array and reshape
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_data_reshaped)
        
        # Determine the result
        result = 'The Person does not have Heart Disease' if prediction[0] == 0 else 'The Person has Heart Disease'
        
        return render_template('index.html', prediction_text=result)
    
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)