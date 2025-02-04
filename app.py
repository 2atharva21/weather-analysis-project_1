import os
import pandas as pd
import numpy as np
import streamlit as st
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

# Check if dataset exists
DATA_FILE = 'weather_data.csv'
if not os.path.exists(DATA_FILE):
    st.error("⚠️ Weather data file not found! Please place 'weather_data.csv' in the same folder.")
    st.stop()

# Load the dataset
df = pd.read_csv(DATA_FILE)

# Convert timestamp column to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day'] = df['timestamp'].dt.day
df['month'] = df['timestamp'].dt.month

# Drop missing values
df_clean = df.dropna()

# Scale features
scaler = MinMaxScaler()
df_clean[['humidity', 'hour', 'day', 'month']] = scaler.fit_transform(
    df_clean[['humidity', 'hour', 'day', 'month']]
)

# Prepare data for training
X = df_clean[['humidity', 'hour', 'day', 'month']]
y = df_clean['temperature']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost Model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
xgb_model.fit(X_train, y_train)

# Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate models
y_pred_xgb = xgb_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
mae_rf = mean_absolute_error(y_test, y_pred_rf)

# Sidebar
st.sidebar.title("🌦️ Weather Prediction App")
st.sidebar.markdown("Predicts temperature based on weather conditions.")

# Introduction to the Project
st.title("🌦️ Weather Prediction App by Atharva Zare")

st.markdown(""" 
# Introduction

Welcome to the **Weather Prediction App**! This app is designed to predict the temperature based on weather parameters such as humidity, pressure, time of the day, and more. By leveraging machine learning models like **XGBoost** and **Random Forest**, we aim to forecast the temperature accurately.

### Key Features:
- **Prediction**: Predicts the temperature based on user inputs.
- **Data Visualization**: Displays various charts like scatter plots, heatmaps, and distributions.
- **XGBoost and Random Forest Models**: The app uses two different models to train on historical weather data and make predictions.
- **Interactive Interface**: You can adjust the input parameters and instantly see the predicted temperature and weather condition.
- **Detailed Insights**: View actual vs predicted temperature, correlation heatmaps, and the distribution of temperature and humidity.

---

### How It Works:
1. **User Input**: Select values for weather parameters like humidity, hour, day, month.
2. **Prediction**: The app uses either XGBoost or Random Forest model to predict the temperature.
3. **Weather Condition**: Categorize the predicted temperature as **Hot**, **Warm**, **Cool**, or **Cold**.
4. **Visualizations**: View visualizations to better understand the data and the models' performance.

--- 

This app is a fun and interactive way to explore weather data and machine learning!

""")

# Show Raw Data Option
if st.sidebar.checkbox('Show Raw Data'):
    st.subheader("📊 Raw Weather Data")
    st.write(df)

# Model Selection
model_choice = st.sidebar.selectbox("Select Prediction Model", ("XGBoost", "Random Forest"))

# City Selector
city = st.sidebar.selectbox('🏙️ Select City', df['city'].unique())

# User Input for Prediction
st.sidebar.header("📌 Weather Parameters")
humidity = st.sidebar.slider("💧 Humidity (%)", 0, 100, 50)
hour = st.sidebar.slider("⏳ Hour of the Day", 0, 23, 12)
day = st.sidebar.slider("📅 Day of the Month", 1, 31, 1)
month = st.sidebar.slider("📆 Month", 1, 12, 1)

# Scatter Plot (Actual vs Predicted)
st.subheader("📉 Actual vs Predicted Temperature")
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_test, y_pred_xgb, alpha=0.5, label="XGBoost")
ax.scatter(y_test, y_pred_rf, alpha=0.5, label="Random Forest")
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', lw=2)
ax.set_title('Actual vs Predicted Temperature')
ax.set_xlabel('Actual Temperature')
ax.set_ylabel('Predicted Temperature')
ax.legend()
st.pyplot(fig)



# Temperature Distribution
st.subheader("🌡️ 1. Temperature Distribution")
fig1, ax1 = plt.subplots(figsize=(8, 5))
sns.histplot(df['temperature'], kde=True, color='blue', ax=ax1)
ax1.set_title('Temperature Distribution')
ax1.set_xlabel('Temperature (°C)')
ax1.set_ylabel('Frequency')
plt.tight_layout()
st.pyplot(fig1)
st.markdown("""
### Purpose:This plot shows the distribution of temperature values in your dataset.

Histogram: The blue bars represent the frequency of temperature values falling within specific ranges.

KDE (Kernel Density Estimate): A smooth curve overlaid on the histogram (shown 
by kde=True) to show the probability density of the temperature distribution.

X-axis: Represents the temperature in degrees Celsius.

Y-axis: Represents the frequency, which shows how often a certain temperature value occurs.

Purpose of sns.histplot: It creates the histogram and the smooth KDE curve for the temperature distribution.""")

# Humidity Distribution
st.subheader("💧 2. Humidity Distribution")
fig2, ax2 = plt.subplots(figsize=(8, 5))
sns.histplot(df['humidity'], kde=True, color='green', ax=ax2)
ax2.set_title('Humidity Distribution')
ax2.set_xlabel('Humidity (%)')
ax2.set_ylabel('Frequency')
plt.tight_layout()
st.pyplot(fig2)
st.markdown("""
### Explanation for Figure 2: Humidity Distribution 
 
Purpose: This plot shows the distribution of humidity values in your dataset.

Histogram: The green bars represent the frequency of humidity values within specific intervals.

KDE (Kernel Density Estimate): A smooth curve (overlayed on the histogram) that shows the probability density of humidity values.

X-axis: Humidity percentage.
Y-axis: Frequency of humidity values.    
            """)


# Distribution Plot: Humidity Distribution (with custom bins)
st.subheader("3. Distribution Plot: Humidity Distribution")
fig4, ax4 = plt.subplots(figsize=(8, 5))
sns.histplot(df['humidity'], kde=True, color='blue', bins=20, ax=ax4)
ax4.set_title('Humidity Distribution')
ax4.set_xlabel('Humidity (%)')
ax4.set_ylabel('Frequency')
plt.tight_layout()
st.pyplot(fig4)
st.markdown("""         
### Explanation for Figure 3: Humidity Distribution (with custom bins)

Purpose: Similar to the previous humidity plot, but with custom bins for frequency intervals.
            
Bins: Specifies bins=20, which divides the data into 20 bars.

KDE: The smooth curve represents the distribution of humidity values.

X-axis: Humidity percentage (grouped by custom bins).

Y-axis: Frequency of humidity values.            
            """)

# Heatmap of Correlation Matrix
st.subheader("4. Heatmap of Correlation Matrix")
fig5, ax5 = plt.subplots(figsize=(8, 6))
correlation_matrix = df[['temperature', 'humidity']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax5)
ax5.set_title('Correlation Heatmap: Temperature and Humidity')
plt.tight_layout()
st.pyplot(fig5)
st.markdown("""
### Explanation for Figure 4: Correlation Heatmap
Purpose: Shows the relationship (correlation) between temperature and humidity.

Correlation Matrix: Measures how closely temperature and humidity are related:

Positive correlation: Values close to 1 (both increase together).

Negative correlation: Values close to -1 (one increases while the other 
decreases).
No correlation: Values close to 0.

Heatmap: Uses color to represent the strength of correlation (coolwarm color 

map: blue for negative, red for positive).

Annotations (annot=True): Displays correlation values inside the heatmap cells.

X and Y-axis: Variables being correlated (temperature vs humidity)""")

# Boxplot: Temperature Data
st.subheader("5. Boxplot: Temperature Data")
fig6, ax6 = plt.subplots(figsize=(8, 5))
sns.boxplot(x=df['temperature'], color='red', ax=ax6)
ax6.set_title('Temperature Boxplot')
ax6.set_xlabel('Temperature (°C)')
plt.tight_layout()
st.pyplot(fig6)
st.markdown("""
### Explanation for Figure 5: Temperature Boxplot
Purpose: Summarizes the temperature data distribution.

Boxplot: The red box represents the range within which most of the temperature data lies:
The median value is the line inside the box.
The upper and lower quartiles are represented by the edges of the box.
The "whiskers" show the range of the data (excluding outliers).
Outliers are plotted as points outside the whiskers.

X-axis: Temperature values.""")

# Boxplot: Humidity Data
st.subheader("6. Boxplot: Humidity Data")
fig7, ax7 = plt.subplots(figsize=(8, 5))
sns.boxplot(x=df['humidity'], color='blue', ax=ax7)
ax7.set_title('Humidity Boxplot')
ax7.set_xlabel('Humidity (%)')
plt.tight_layout()
st.pyplot(fig7)
st.markdown("""
### Explanation for Figure 6: Humidity Boxplot
Purpose: Similar to the previous boxplot, but this one is for humidity data.

Boxplot: The blue box represents the distribution of humidity values:

The median, upper and lower quartiles, whiskers, and outliers are visualized.

X-axis: Humidity percentage.""")

# Correlation Heatmap
st.subheader("🛠️ 7 Correlation Heatmap of Weather Data")
corr = df_clean[['temperature', 'humidity', 'hour', 'day', 'month']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=1)
plt.title('Correlation Heatmap')
st.pyplot()
# Prediction
st.subheader(f"🌍 Weather Prediction for {city}")

# Prepare input data for prediction
input_data = np.array([[humidity, hour, day, month]])
input_data_scaled = scaler.transform(input_data)

# Make Prediction Based on Selected Model
if model_choice == "XGBoost":
    predicted_temp = xgb_model.predict(input_data_scaled)[0]
    model_name = "XGBoost"
else:
    predicted_temp = rf_model.predict(input_data_scaled)[0]
    model_name = "Random Forest"

# Display Result
st.write(f"🌡️ **Predicted Temperature for {city}: {predicted_temp:.2f}°C**")
st.write(f"📊 **Model Used: {model_name}**")

# Weather Condition
if predicted_temp > 30:
    weather_condition = "Hot ☀️"
elif 20 <= predicted_temp <= 30:
    weather_condition = "Warm 🌤️"
elif 10 <= predicted_temp < 20:
    weather_condition = "Cool ⛅"
else:
    weather_condition = "Cold ❄️"

st.write(f"🌍 **Weather Condition: {weather_condition}**")

# Display Input Parameters
st.write(f"🔹 **Input values:** Humidity = {humidity}%, Hour = {hour}, Day = {day}, Month = {month}")
