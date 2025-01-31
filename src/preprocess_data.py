import pandas as pd

CSV_FILE = "data/weather_data.csv"
PROCESSED_FILE = "data/processed_weather_data.csv"

def preprocess_data():
    try:
        df = pd.read_csv(CSV_FILE)
        print("Original Data:")
        print(df)

        # Ensure there are no missing values
        df = df.dropna()

        # Convert the timestamp to a datetime object
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Sort the data by city and timestamp
        df = df.sort_values(by=["city", "timestamp"])

        # Save the cleaned data to a new CSV file
        df.to_csv(PROCESSED_FILE, index=False)
        print(f"Preprocessed data saved to {PROCESSED_FILE}")
    except FileNotFoundError:
        print(f"File {CSV_FILE} not found. Please ensure it exists.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    preprocess_data()
