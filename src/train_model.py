import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import pickle

CSV_FILE = "data/processed_weather_data.csv"
RANDOM_FOREST_MODEL_FILE = "models/random_forest_model.pkl"
XGBOOST_MODEL_FILE = "models/xgboost_model.pkl"

def train_models():
    try:
        # Load the preprocessed CSV file
        df = pd.read_csv(CSV_FILE)

        # Features and target variable
        X = df[["humidity"]]
        y = df["temperature"]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # RandomForest Model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_predictions = rf_model.predict(X_test)
        rf_rmse = mean_squared_error(y_test, rf_predictions, squared=False)
        print(f"RandomForest RMSE: {rf_rmse:.2f}")

        # Save the RandomForest model
        with open(RANDOM_FOREST_MODEL_FILE, "wb") as file:
            pickle.dump(rf_model, file)
        print(f"RandomForest model saved to {RANDOM_FOREST_MODEL_FILE}")

        # XGBoost Model
        xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        xgb_model.fit(X_train, y_train)
        xgb_predictions = xgb_model.predict(X_test)
        xgb_rmse = mean_squared_error(y_test, xgb_predictions, squared=False)
        print(f"XGBoost RMSE: {xgb_rmse:.2f}")

        # Save the XGBoost model
        with open(XGBOOST_MODEL_FILE, "wb") as file:
            pickle.dump(xgb_model, file)
        print(f"XGBoost model saved to {XGBOOST_MODEL_FILE}")

    except FileNotFoundError:
        print(f"File {CSV_FILE} not found. Please preprocess the data first.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    train_models()
