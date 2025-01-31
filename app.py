import streamlit as st
import pandas as pd
import xgboost
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Load the models
xgb_model = xgboost.Booster()
xgb_model.load_model('models/xgboost_model.json')

rf_model = joblib.load('models/random_forest_model.pkl')

# Load weather data
df = pd.read_csv("data/weather_data.csv")

def preprocess_data(data):
    # Preprocess the data (same as your project)
    scaler = MinMaxScaler()
    data[['humidity', 'pressure']] = scaler.fit_transform(data[['humidity', 'pressure']])
    return data

def predict_temperature(model, data):
    # Make predictions using the trained model
    features = ['humidity', 'hour', 'day', 'month', 'pressure']
    return model.predict(data[features])

# Streamlit Sidebar
st.sidebar.header("Weather Analysis Project")
option = st.sidebar.selectbox("Choose an option:", ["Data Analysis", "Temperature Prediction"])

if option == "Data Analysis":
    st.title("Weather Data Analysis")
    
    # Display Basic Stats
    st.subheader("Basic Statistics")
    st.write(df.describe())

    # Plot Temperature Distribution
    st.subheader("Temperature Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['temperature'], kde=True, color='blue', ax=ax)
    st.pyplot(fig)

    # Plot Humidity Distribution
    st.subheader("Humidity Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['humidity'], kde=True, color='green', ax=ax)
    st.pyplot(fig)

elif option == "Temperature Prediction":
    st.title("Predict Temperature")
    
    # User input for prediction
    city = st.selectbox("Select City", df['city'].unique())
    hour = st.slider("Hour", 0, 23)
    day = st.slider("Day", 1, 31)
    month = st.slider("Month", 1, 12)
    humidity = st.slider("Humidity", int(df['humidity'].min()), int(df['humidity'].max()))
    pressure = st.slider("Pressure", int(df['pressure'].min()), int(df['pressure'].max()))
    
    input_data = pd.DataFrame({
        'city': [city],
        'hour': [hour],
        'day': [day],
        'month': [month],
        'humidity': [humidity],
        'pressure': [pressure]
    })

    # Preprocess the input data
    input_data = preprocess_data(input_data)

    # Predict temperature using both models
    xgb_prediction = predict_temperature(xgb_model, input_data)
    rf_prediction = rf_model.predict(input_data[['humidity', 'hour', 'day', 'month', 'pressure']])
    
    st.subheader("Predictions")
    st.write(f"XGBoost Prediction: {xgb_prediction[0]}")
    st.write(f"Random Forest Prediction: {rf_prediction[0]}")

