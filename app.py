from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

app = Flask(__name__)

# Load or train the model
def load_or_train_model():
    model_path = 'heart_disease_model.pkl'
    if os.path.exists(model_path):
        # Load the pre-trained model
        model = joblib.load(model_path)
    else:
        # Load the dataset
        heart_data = pd.read_csv('heart_disease_data.csv')
        
        # Split features and target
        X = heart_data.drop(columns='target', axis=1)
        Y = heart_data['target']
        
        # Split the data into training and test sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
        
        # Train the logistic regression model
        model = LogisticRegression(max_iter=1000)  # Increased max_iter to avoid convergence warning
        model.fit(X_train, Y_train)
        
        # Save the trained model
        joblib.dump(model, model_path)
    
    return model

# Load the model
model = load_or_train_model()

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