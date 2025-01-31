import pandas as pd

CSV_FILE = "data/processed_weather_data.csv"

def analyze_weather():
    try:
        df = pd.read_csv(CSV_FILE)
        print("Basic Statistics:")
        print(df.describe())

        print("\nCity-wise Analysis:")
        print(df.groupby("city")[["temperature", "humidity"]].mean())
    except FileNotFoundError:
        print(f"File {CSV_FILE} not found. Please preprocess the data first.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    analyze_weather()
