import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Title
st.title("🍎 Nutrition Health Predictor")

# Load dataset
data = pd.read_csv("Nutrition_Analytics_Dataset.csv")

# Clean columns (DO NOT convert to lowercase)
data.columns = data.columns.str.strip()

# Debug (optional)
st.write("Columns:", data.columns)

# Create target column
data['health_label'] = data['Calories'].apply(
    lambda x: 'Healthy' if x < 400 else 'Unhealthy'
)

# Features (MATCH EXACT NAMES)
X = data[['Protein_g','Carbs_g','Fats_g']]
y = data['health_label']

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# User input
st.subheader("Enter Nutritional Values")

protein = st.number_input("Protein (g)", min_value=0.0)
carbs = st.number_input("Carbs (g)", min_value=0.0)
fat = st.number_input("Fat (g)", min_value=0.0)

# Prediction
if st.button("Predict"):
    prediction = model.predict([[protein, carbs, fat]])
    st.success(f"Prediction: {prediction[0]}")
