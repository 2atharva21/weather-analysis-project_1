import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import r2_score
import numpy as np

# Load your data
df = pd.read_csv("data/weather_data.csv")

# Feature engineering (add more features for better performance)
X = df[['humidity', 'hour', 'day', 'month']]  # Using more features
y = df['temperature']  # Target variable (temperature)

# Initialize the XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)

# Hyperparameter tuning (optional, with GridSearchCV)
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search.fit(X, y)

print(f"Best Hyperparameters: {grid_search.best_params_}")

# Use the best model from grid search
best_model = grid_search.best_estimator_

# Perform manual cross-validation with the best model
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # Split the data into 5 parts
cv_scores = []

# Train and evaluate the model using cross-validation
for train_index, val_index in kf.split(X):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_val)
    
    r2 = r2_score(y_val, y_pred)
    cv_scores.append(r2)

# Print the R² scores for each fold
print(f"R² scores from cross-validation: {cv_scores}")
print(f"Average R² score: {np.mean(cv_scores)}")

# Train the best model on the entire dataset
best_model.fit(X, y)

# Save the trained model
best_model.save_model("models/xgboost_model_best.json")
print("XGBoost model trained and saved with best hyperparameters.")
