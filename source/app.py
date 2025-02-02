import pandas as pd
import numpy as np
import streamlit as st
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

# Streamlit App Title
st.title("🌦️ Weather Prediction App")

# Load the dataset with error handling
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/weather_data.csv')
        return df
    except FileNotFoundError:
        st.error("⚠️ Weather data file not found. Please check the path.")
        st.stop()

df = load_data()

# Sidebar UI
st.sidebar.title("🔍 Weather Prediction Settings")
st.sidebar.markdown("This app predicts temperature based on weather parameters.")

# Show raw data
if st.sidebar.checkbox('Show Raw Data'):
    st.subheader("📊 Raw Weather Data")
    st.write(df)

# City selection
city = st.sidebar.selectbox('🌍 Select City', df['city'].unique())

# User Input Parameters
st.sidebar.header("🌡️ Weather Parameters")
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 50)
hour = st.sidebar.slider("Hour of the Day", 0, 23, 12)
day = st.sidebar.slider("Day of the Month", 1, 31, 1)
month = st.sidebar.slider("Month", 1, 12, 1)
pressure = st.sidebar.slider("Pressure (hPa)", 980, 1025, 1013)

# Convert timestamp to features
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day'] = df['timestamp'].dt.day
df['month'] = df['timestamp'].dt.month

# Data Cleaning
df_clean = df.dropna().copy()

# Feature Scaling
scaler = MinMaxScaler()
features = ['humidity', 'pressure', 'hour', 'day', 'month']
df_clean[features] = scaler.fit_transform(df_clean[features])

# Define Features and Target
X = df_clean[features]
y = df_clean['temperature']

# Train-test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training with Caching
@st.cache_resource
def train_model():
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X_train, y_train)
    return model

# Show loading spinner while training
with st.spinner("🔄 Training the model... Please wait."):
    xgb_model = train_model()

# Model Evaluation
y_pred = xgb_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

st.sidebar.success(f"✅ Model trained successfully! MAE: {mae:.2f}")

# Prediction Section
st.subheader(f"📌 Weather Prediction for {city}")

# Filter city-specific data
city_data = df[df['city'] == city]
if city_data.empty:
    st.warning(f"No data available for {city}. Prediction may not be accurate.")

# Prepare input data
input_data = np.array([[humidity, hour, day, month, pressure]])
input_data_scaled = scaler.transform(input_data)

# Predict Temperature
predicted_temp = xgb_model.predict(input_data_scaled)[0]

# Display Prediction
st.write(f"🌡️ **Predicted Temperature:** {predicted_temp:.2f}°C")

# Weather Condition
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

st.write(f"🌍 **Weather Condition:** {weather_condition} {weather_icon}")

# Correlation Heatmap
st.subheader("📊 Correlation Heatmap of Weather Data")
corr = df_clean[['temperature', 'humidity', 'pressure', 'hour', 'day', 'month']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=1)
plt.title('Correlation Heatmap')
st.pyplot()

# Actual vs Predicted Plot
st.subheader("📈 Actual vs Predicted Temperature")
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_test, y_pred, alpha=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', lw=2)
ax.set_title('Actual vs Predicted Temperature')
ax.set_xlabel('Actual Temperature')
ax.set_ylabel('Predicted Temperature')
st.pyplot(fig)

# Display Input Values
st.subheader("📝 Input Values")
st.write(f"- Humidity: **{humidity}%**")
st.write(f"- Hour: **{hour}**")
st.write(f"- Day: **{day}**")
st.write(f"- Month: **{month}**")
st.write(f"- Pressure: **{pressure} hPa**")
