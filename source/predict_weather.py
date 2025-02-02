import xgboost as xgb
import joblib
import pandas as pd
import logging
import os

# Configure logging for detailed output
logging.basicConfig(level=logging.INFO)

class WeatherPredictor:
    def __init__(self, xgb_model_path, rf_model_path):
        # Check if model files exist
        if not os.path.exists(xgb_model_path) or not os.path.exists(rf_model_path):
            logging.error("Model files not found. Please check the file paths.")
            raise FileNotFoundError("One or more model files are missing.")
        
        # Load the trained XGBoost model (Booster)
        self.xgb_model = xgb.Booster()
        self.xgb_model.load_model(xgb_model_path)
        
        # Load the trained RandomForest model
        self.rf_model = joblib.load(rf_model_path)
        
        logging.info("Models loaded successfully.")

    def prepare_input_data(self, humidity, hour, day, month, pressure):
        # Prepare input data for prediction in the same format as the training data
        input_data = pd.DataFrame([[humidity, hour, day, month, pressure]],
                                  columns=['humidity', 'hour', 'day', 'month', 'pressure'])
        logging.info("Input data prepared for prediction.")
        return input_data

    def make_predictions(self, humidity, hour, day, month, pressure):
        # Prepare the input data
        input_data = self.prepare_input_data(humidity, hour, day, month, pressure)

        # Make predictions using XGBoost (via DMatrix)
        dmatrix = xgb.DMatrix(input_data)
        predicted_temperature_xgb = self.xgb_model.predict(dmatrix)
        logging.info(f"Prediction made using XGBoost: {predicted_temperature_xgb[0]}°C")

        # Make predictions using RandomForest
        predicted_temperature_rf = self.rf_model.predict(input_data)
        logging.info(f"Prediction made using RandomForest: {predicted_temperature_rf[0]}°C")

        # Return the predictions in a readable format
        result = {
            "XGBoost Prediction": predicted_temperature_xgb[0],
            "RandomForest Prediction": predicted_temperature_rf[0]
        }
        
        # Log predictions summary
        logging.info(f"Prediction Summary: {result}")
        
        return result

if __name__ == "__main__":
    try:
        # Initialize WeatherPredictor with model paths
        predictor = WeatherPredictor(xgb_model_path="models/xgboost_model.json", 
                                     rf_model_path="models/random_forest_model.pkl")

        # Example inputs for prediction
        humidity = 60
        hour = 14
        day = 20
        month = 5
        pressure = 1015

        # Make predictions and print the results
        predictions = predictor.make_predictions(humidity, hour, day, month, pressure)
        
        # Print the prediction results in a detailed and readable way
        print("\n--- Weather Prediction Results ---")
        print(f"Input Data: Humidity = {humidity}%, Hour = {hour}, Day = {day}, Month = {month}, Pressure = {pressure} hPa")
        print(f"XGBoost Model Prediction: {predictions['XGBoost Prediction']}°C")
        print(f"RandomForest Model Prediction: {predictions['RandomForest Prediction']}°C")

        # Provide explanation on the results
        print("\n--- Explanation ---")
        print("1. XGBoost Model Prediction: This prediction is made using the XGBoost algorithm.")
        print("   It uses gradient boosting to predict the temperature based on input features.")
        print(f"   The predicted temperature from XGBoost is {predictions['XGBoost Prediction']}°C.")
        print("2. RandomForest Model Prediction: This prediction is made using the RandomForest algorithm.")
        print("   It uses multiple decision trees and averages their predictions.")
        print(f"   The predicted temperature from RandomForest is {predictions['RandomForest Prediction']}°C.")
        print("3. Both models give slightly different predictions, which is normal.")
        print("4. You can either use one model's prediction or combine them for a final estimate.")
        print(f"   Average temperature prediction: {(predictions['XGBoost Prediction'] + predictions['RandomForest Prediction']) / 2}°C")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
