from sklearn.linear_model import LogisticRegression
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
heart_data = pd.read_csv('heart_disease_data.csv')
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# Save with compression
joblib.dump(model, 'heart_disease_model.pkl', compress=3)
print("Model saved with compression level 3.")