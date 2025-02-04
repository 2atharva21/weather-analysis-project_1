import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load dataset
try:
    df = pd.read_csv('weather_data.csv')
except FileNotFoundError:
    print("Error: The file 'weather_data.csv' was not found. Please check the file path.")
    exit()

# Checking for missing values
print("Missing values in each column:")
print(df.isnull().sum())

# Checking for infinite values only in numeric columns
numeric_cols = df.select_dtypes(include=[np.number])
print("\nChecking for infinite values:")
print(np.isinf(numeric_cols).sum())

# Preprocessing: Handle missing or infinite values
df = df.replace([np.inf, -np.inf], np.nan)  # Replace infinite values with NaN
df = df.dropna()  # Drop rows with missing values

# Feature Engineering
# For this example, let's assume 'temperature' is our target variable.
# You can modify the target variable based on your dataset.
X = df.drop(['temperature'], axis=1)  # Features (excluding 'temperature')
y = df['temperature']  # Target (temperature)

# Convert categorical columns to numeric if any (e.g., 'city', 'weather' etc.)
X = pd.get_dummies(X, drop_first=True)  # One-hot encoding for categorical variables

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
rf_model = RandomForestRegressor(random_state=42)

# Manual cross-validation for XGBoost
kf = KFold(n_splits=5, shuffle=True, random_state=42)
xgb_r2_scores = []
for train_index, val_index in kf.split(X_train_scaled):
    X_train_cv, X_val_cv = X_train_scaled[train_index], X_train_scaled[val_index]
    y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[val_index]
    
    # Hyperparameters (tuning manually)
    xgb_model.set_params(n_estimators=200, learning_rate=0.1, max_depth=5)
    
    xgb_model.fit(X_train_cv, y_train_cv)
    y_pred = xgb_model.predict(X_val_cv)
    r2 = r2_score(y_val_cv, y_pred)
    xgb_r2_scores.append(r2)

print(f"\nXGBoost R² (manual CV): {np.mean(xgb_r2_scores)}")

# Manual cross-validation for RandomForest
rf_r2_scores = []
for train_index, val_index in kf.split(X_train_scaled):
    X_train_cv, X_val_cv = X_train_scaled[train_index], X_train_scaled[val_index]
    y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[val_index]
    
    rf_model.fit(X_train_cv, y_train_cv)
    y_pred = rf_model.predict(X_val_cv)
    r2 = r2_score(y_val_cv, y_pred)
    rf_r2_scores.append(r2)

print(f"RandomForest R² (manual CV): {np.mean(rf_r2_scores)}")

# Save the best model and scaler
joblib.dump(xgb_model, 'models/xgb_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("\nModels and scaler saved in 'models' directory.")
