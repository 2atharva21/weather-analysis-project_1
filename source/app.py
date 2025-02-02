import pandas as pd
import numpy as np
import streamlit as st
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

# Load the dataset
df = pd.read_csv('data/weather_data.csv')

# Sidebar: Set up UI components
st.sidebar.title("Weather Prediction App")
st.sidebar.markdown("This app predicts the temperature and weather conditions based on various parameters.")

# Show Raw Data Option
if st.sidebar.checkbox('Show Raw Data'):
    st.subheader("Raw Weather Data")
    st.write(df)

# City Selector for prediction
city = st.sidebar.selectbox('Select City', df['city'].unique())

# Weather Parameter Sliders
st.sidebar.header("Weather Parameters")
humidity = st.slider("Humidity (%)", 0, 100, 50)
hour = st.slider("Hour of the Day", 0, 23, 12)
day = st.slider("Day of the Month", 1, 31, 1)
month = st.slider("Month", 1, 12, 1)
pressure = st.slider("Pressure (hPa)", 980, 1025, 1013)

# Data Preprocessing
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day'] = df['timestamp'].dt.day
df['month'] = df['timestamp'].dt.month

# Clean data and scale features
df_clean = df.dropna()
scaler = MinMaxScaler()
df_clean[['humidity', 'pressure', 'hour', 'day', 'month']] = scaler.fit_transform(df_clean[['humidity', 'pressure', 'hour', 'day', 'month']])

# Feature Selection for the Model
X = df_clean[['humidity', 'hour', 'day', 'month', 'pressure']]
y = df_clean['temperature']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the XGBoost Model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
xgb_model.fit(X_train, y_train)

# Model Evaluation - Mean Absolute Error
y_pred = xgb_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

# Actual vs Predicted Plot
st.subheader("Actual vs Predicted Temperature")
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_test, y_pred, alpha=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', lw=2)
ax.set_title('Actual vs Predicted Temperature')
ax.set_xlabel('Actual Temperature')
ax.set_ylabel('Predicted Temperature')
st.pyplot(fig)

# Correlation Heatmap
st.subheader("Correlation Heatmap of Weather Data")
corr = df_clean[['temperature', 'humidity', 'pressure', 'hour', 'day', 'month']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=1)
plt.title('Correlation Heatmap')
st.pyplot()

# Weather Prediction Section
st.subheader(f"Weather Prediction for {city}")

# Filter data based on selected city
city_data = df[df['city'] == city]

# Prepare input data for prediction based on user parameters
input_data = np.array([[humidity, hour, day, month, pressure]])

# Scale input data (remember the scaler is already fitted on the data)
input_data_scaled = scaler.transform(input_data)

# Make Prediction for the selected city
predicted_temp = xgb_model.predict(input_data_scaled)[0]

# Display Prediction Result with Weather Information
st.write(f"Predicted Temperature for {city}: {predicted_temp:.2f}°C")

# Display Weather Condition based on the predicted temperature
if predicted_temp > 30:
    weather_condition = "Hot"
    weather_icon = "☀️"
elif 20 <= predicted_temp <= 30:
    weather_condition = "Warm"
    weather_icon = "🌤️"
elif 10 <= predicted_temp < 20:
    weather_condition = "Cool"
    weather_icon = "⛅"
else:
    weather_condition = "Cold"
    weather_icon = "❄️"

st.write(f"Weather Condition: {weather_condition} {weather_icon}")

# Optionally show a background image based on weather condition
if weather_condition == "Hot":
    st.markdown('<style>body {background-image: url("https://link-to-hot-weather-image.jpg"); background-size: cover;}</style>', unsafe_allow_html=True)
elif weather_condition == "Warm":
    st.markdown('<style>body {background-image: url("https://link-to-warm-weather-image.jpg"); background-size: cover;}</style>', unsafe_allow_html=True)
elif weather_condition == "Cool":
    st.markdown('<style>body {background-image: url("https://link-to-cool-weather-image.jpg"); background-size: cover;}</style>', unsafe_allow_html=True)
else:
    st.markdown('<style>body {background-image: url("https://link-to-cold-weather-image.jpg"); background-size: cover;}</style>', unsafe_allow_html=True)

# Display Input Parameters for Clarity
st.write(f"Input values: Humidity = {humidity}%, Hour = {hour}, Day = {day}, Month = {month}, Pressure = {pressure} hPa")
