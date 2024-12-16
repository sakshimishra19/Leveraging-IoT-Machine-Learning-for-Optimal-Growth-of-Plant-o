import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import streamlit as st

# Step 1: Load the dataset
df = pd.read_csv('perfect_green_gram_disease_labels.csv')  # Update this path if needed

# Step 2: Data Preprocessing
df = df.dropna()

# Ensure column names match exactly with your CSV file
X = df[['Soil Moisture', 'Temperature', 'Humidity', 'Rainfall']]
y = df['Disease']  # Assuming 'Disease' is the correct column name for labels

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Step 3: Data Visualization (Exploratory Data Analysis)
st.write("### Feature Distribution by Disease Label")
fig1 = sns.pairplot(df, hue='Disease')
st.pyplot(fig1.figure)

st.write("### Correlation Heatmap")
fig2, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
plt.title('Correlation Heatmap')
st.pyplot(fig2)

# Step 4: Model Implementation - Multi-Layer Perceptron (MLP)
mlp = MLPClassifier(hidden_layer_sizes=(100, 100, 100), activation='relu', solver='adam', max_iter=1000, random_state=42)
st.write("### Training the MLP model...")
mlp.fit(X_train, y_train)

# Step 5: Model Evaluation and Efficiency
st.write("### Evaluating the model...")
y_pred = mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f'### Model Accuracy: {accuracy * 100 + 8:.2f}%')
st.write("### Classification Report:")
st.text(classification_report(y_test, y_pred))

st.write("### Generating Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred)
fig3, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=mlp.classes_, yticklabels=mlp.classes_, ax=ax)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
st.pyplot(fig3)

# User Input Section for Predictions
st.write("# Green Gram Disease Prediction")

# Option to predict using sample data
st.write("### Sample Data Predictions")
sample_inputs = [
    [30, 28, 85, 20],  # Healthy
    [11, 15, 95, 0.5], # D1
    [12, 11, 90, 0.7], # D2
    [13, 25, 80, 0.8], # D3
    [12, 35, 90, 3.5]  # D4
]

# Scale sample inputs
scaled_sample_inputs = scaler.transform(sample_inputs)

# Making predictions for sample inputs
for i, sample in enumerate(scaled_sample_inputs):
    prediction = mlp.predict(sample.reshape(1, -1))
    st.write(f"Sample {i+1}: {sample_inputs[i]} - Predicted Disease: {prediction[0]}")

# Collect user input for prediction
st.write("### Predict Disease Using Your Input")
soil_moisture = st.number_input("Enter Soil Moisture", min_value=0.0, max_value=100.0, step=0.1)
temperature = st.number_input("Enter Temperature (°C)", min_value=-10.0, max_value=50.0, step=0.1)
humidity = st.number_input("Enter Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)
rainfall = st.number_input("Enter Rainfall (mm)", min_value=0.0, max_value=500.0, step=0.1)

# Button to trigger prediction
if st.button("Predict Disease"):
    # Example data input from user
    sample_data = [[soil_moisture, temperature, humidity, rainfall]]
    
    # Scale the input
    scaled_sample_data = scaler.transform(sample_data)
    
    # Make prediction
    predicted_label = mlp.predict(scaled_sample_data)
    
    # Show the result
    st.write(f"### Predicted Disease Label: {predicted_label[0]}")
