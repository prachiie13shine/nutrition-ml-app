import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Title
st.title("🍎 Nutrition Health Predictor")

# Load data
data = pd.read_csv("Nutrition_Analytics_Dataset.csv")

# Clean columns
data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

# Create target (FIXED)
data['health_label'] = data['calories'].apply(
    lambda x: 'Healthy' if x < 400 else 'Unhealthy'
)

# Features
X = data[['protein_g','carbs_g','Fat_g']]
y = data['health_label']

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# User input
protein = st.number_input("Protein (g)")
carbs = st.number_input("Carbs (g)")
fat = st.number_input("Fat (g)")

# Predict
if st.button("Predict"):
    result = model.predict([[protein, carbs, fat]])
    st.success(f"Result: {result[0]}")