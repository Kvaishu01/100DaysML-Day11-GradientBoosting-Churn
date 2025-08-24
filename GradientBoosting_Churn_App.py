# GradientBoosting_Churn_App.py

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("📊 Gradient Boosting - Customer Churn Prediction")
st.write("Predict whether a customer is likely to churn using **Gradient Boosting Classifier**.")

# ==========================
# Sample Dataset
# ==========================
data = {
    "gender": ["Female", "Male", "Female", "Male", "Female", "Male"],
    "SeniorCitizen": [0, 1, 0, 0, 1, 1],
    "Partner": ["Yes", "No", "Yes", "No", "Yes", "No"],
    "Dependents": ["No", "No", "Yes", "No", "No", "Yes"],
    "tenure": [1, 34, 2, 45, 5, 60],
    "PhoneService": ["Yes", "Yes", "No", "Yes", "Yes", "Yes"],
    "MonthlyCharges": [29.85, 56.95, 53.85, 42.30, 70.70, 99.65],
    "TotalCharges": [29.85, 1889.50, 108.15, 1840.75, 151.65, 5681.10],
    "Churn": ["No", "Yes", "No", "No", "Yes", "No"]
}

df = pd.DataFrame(data)

# Encode categorical variables
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train Model
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# ==========================
# Model Evaluation
# ==========================
st.subheader("📈 Model Performance")
accuracy = accuracy_score(y_test, y_pred)
st.write(f"🔹 **Accuracy:** {accuracy:.2f}")

st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Feature Importance Plot
st.subheader("🔍 Feature Importance")
fig, ax = plt.subplots(figsize=(6,4))
sns.barplot(x=model.feature_importances_, y=X.columns, ax=ax)
st.pyplot(fig)

# ==========================
# User Input Prediction
# ==========================
st.subheader("🧑‍💻 Try Prediction for a New Customer")

