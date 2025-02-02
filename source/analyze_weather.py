import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CSV_FILE = "weather_data.csv"

def analyze_weather():
    try:
        df = pd.read_csv(CSV_FILE)

        # Check if required columns are present
        required_columns = ['temperature', 'humidity', 'city']
        if not all(col in df.columns for col in required_columns):
            print(f"Missing required columns. Expected columns: {required_columns}")
            return
        
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
        sns.histplot(df['temperature'].dropna(), kde=True, color='blue')  # Drop NaN values for visualization
        plt.title('Temperature Distribution')
        plt.xlabel('Temperature')
        plt.ylabel('Frequency')
        plt.tight_layout()  # Ensure proper layout
        plt.savefig('temperature_distribution.png')  # Save the plot
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.histplot(df['humidity'].dropna(), kde=True, color='green')  # Drop NaN values for visualization
        plt.title('Humidity Distribution')
        plt.xlabel('Humidity')
        plt.ylabel('Frequency')
        plt.tight_layout()  # Ensure proper layout
        plt.savefig('humidity_distribution.png')  # Save the plot
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
