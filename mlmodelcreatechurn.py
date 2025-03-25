import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from flask import Flask, request, jsonify

# Load dataset (Ensure the dataset has at least 100,000 rows)
df = pd.read_csv("C:/nsjanvi/organizations-100000.csv")

# Display first few rows
print(df.head())

# Check the number of unique values in each column
high_cardinality_cols = [col for col in df.columns if df[col].nunique() > 1000]

print("High Cardinality Columns:", high_cardinality_cols)

df.drop(columns=high_cardinality_cols, inplace=True)

df.columns = df.columns.str.strip()  # Remove spaces before/after column names
print(df.columns)

df['Churn'] = np.random.choice([0, 1], size=len(df))  # Random 0s and 1s

print(df.columns)
# Display first few rows
print(df.head())




#df.dropna(inplace=True)  # Remove missing values

df = pd.get_dummies(df, drop_first=True)

X = df.drop('Churn', axis=1)  # Features
y = df['Churn']  # Target (0 = No Churn, 1 = Churn)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Get feature importance
importances = model.feature_importances_
features = X.columns

# Plot
sns.barplot(x=importances, y=features)
plt.xlabel("Feature Importance")
plt.ylabel("Feature Name")
plt.title("Feature Importance in Random Forest")
plt.show()

# Load the model
loaded_model = joblib.load("churn_model.pkl")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['features']
    prediction = loaded_model.predict([data])
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)