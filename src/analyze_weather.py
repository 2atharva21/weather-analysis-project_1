import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CSV_FILE = "data/weather_data.csv"

def analyze_weather():
    try:
        df = pd.read_csv(CSV_FILE)
        
        # Basic Statistics
        print("Basic Statistics:")
        print(df.describe())
        
        # Missing Values Check
        print("\nMissing Values Check:")
        print(df.isnull().sum())

        # City-wise Analysis (Extended)
        print("\nCity-wise Analysis (Extended):")
        print(df.groupby("city")[["temperature", "humidity"]].agg(['mean', 'std', 'min', 'max']))

        # Visualizations
        plt.figure(figsize=(10, 6))
        sns.histplot(df['temperature'], kde=True, color='blue')
        plt.title('Temperature Distribution')
        plt.xlabel('Temperature')
        plt.ylabel('Frequency')
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.histplot(df['humidity'], kde=True, color='green')
        plt.title('Humidity Distribution')
        plt.xlabel('Humidity')
        plt.ylabel('Frequency')
        plt.show()

        # Check cities with missing data
        print("\nCities with Missing Data:")
        print(df[df.isnull().any(axis=1)]['city'].unique())

    except FileNotFoundError:
        print(f"File {CSV_FILE} not found. Please preprocess the data first.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    analyze_weather()
