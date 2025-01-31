import pandas as pd
import pickle

RANDOM_FOREST_MODEL_FILE = "models/random_forest_model.pkl"
XGBOOST_MODEL_FILE = "models/xgboost_model.pkl"

def predict_weather(humidity, model_type="RandomForest"):
    try:
        # Load the selected model
        model_file = RANDOM_FOREST_MODEL_FILE if model_type == "RandomForest" else XGBOOST_MODEL_FILE
        with open(model_file, "rb") as file:
            model = pickle.load(file)

        # Create a DataFrame for the input
        input_data = pd.DataFrame({"humidity": [humidity]})

        # Make predictions
        predicted_temp = model.predict(input_data)[0]
        print(f"Predicted Temperature: {predicted_temp:.2f}°C using {model_type}")
    except FileNotFoundError:
        print(f"{model_type} model file not found. Please train the model first.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    humidity_input = float(input("Enter the humidity percentage: "))
    model_choice = input("Enter model type (RandomForest/XGBoost): ").strip()
    predict_weather(humidity_input, model_type=model_choice)
