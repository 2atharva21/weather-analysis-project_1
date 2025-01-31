import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import joblib

# Load the dataset (replace this with your file path or dataset)
df = pd.read_csv('data/weather_data.csv')

# Check for missing values
print("Missing values in each column:")
print(df.isnull().sum())

# Drop rows with NaN values
df = df.dropna()

# Check for infinite or very large values
print("Checking for infinite or very large values:")
print(df.isin([np.inf, -np.inf]).sum())

# Handle any infinite values if present
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.dropna()

# Feature engineering (add more features for better performance)
df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
df['day'] = pd.to_datetime(df['timestamp']).dt.day
df['month'] = pd.to_datetime(df['timestamp']).dt.month
df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
df['year'] = pd.to_datetime(df['timestamp']).dt.year

# Select features and target variable
X = df[['humidity', 'hour', 'day', 'month', 'pressure']]  # Features
y = df['temperature']  # Target variable (temperature)

# Normalize/scale the features
scaler = MinMaxScaler()
X[['humidity', 'pressure']] = scaler.fit_transform(X[['humidity', 'pressure']].astype(float))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)

# Initialize the RandomForest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the XGBoost model on the training set
xgb_model.fit(X_train, y_train)

# Train the RandomForest model on the training set
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred_xgb = xgb_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

# Calculate R² scores on test set
r2_xgb = r2_score(y_test, y_pred_xgb)
r2_rf = r2_score(y_test, y_pred_rf)

# Print R² scores
print(f"XGBoost R² score on test set: {r2_xgb}")
print(f"RandomForest R² score on test set: {r2_rf}")

# Save both trained models
xgb_model.save_model("models/xgboost_model.json")
joblib.dump(rf_model, "models/random_forest_model.pkl")

print("XGBoost and RandomForest models trained and saved.")
