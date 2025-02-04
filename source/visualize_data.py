import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_data():
    try:
        # Read the weather data CSV file
        df = pd.read_csv("weather_data.csv")

        # Print the column names to ensure you have the 'timestamp' column
        print("Columns in the dataset:", df.columns)

        # If 'timestamp' is present, proceed with the visualization
        if 'timestamp' in df.columns:
            # Convert to datetime format
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Handle any missing data by dropping rows with NaN values in specific columns
            df.dropna(subset=['temperature', 'humidity'], inplace=True)

            # Optional: Down-sample the data if it's too large to visualize
            df = df.sample(frac=0.1, random_state=42)  # Use 10% of the data for quicker plotting

            # 1. Time Series Plot: Temperature and Humidity over Time
            plt.figure(figsize=(10, 6))
            plt.plot(df['timestamp'], df['temperature'], label='Temperature (°C)', color='red')
            plt.plot(df['timestamp'], df['humidity'], label='Humidity (%)', color='blue')
            plt.title('1. Temperature and Humidity Over Time')
            plt.xlabel('Time')
            plt.ylabel('Temperature (°C) / Humidity (%)')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()  # Ensures everything fits without overlap
            plt.show()

            # 2. Distribution Plot: Temperature Distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(df['temperature'], kde=True, color='red', bins=30)
            plt.title('2. Temperature Distribution')
            plt.xlabel('Temperature (°C)')
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.show()

            # 3. Boxplot: Temperature vs Weather Condition
            if 'weather' in df.columns:  # Check if weather column exists
                plt.figure(figsize=(10, 6))
                sns.boxplot(x='weather', y='temperature', data=df, palette='Set2')
                plt.title('3. Temperature vs Weather Condition')
                plt.xlabel('Weather Condition')
                plt.ylabel('Temperature (°C)')
                plt.tight_layout()
                plt.show()

            # 4. Distribution Plot: Humidity Distribution
            plt.figure(figsize=(8, 5))
            sns.histplot(df['humidity'], kde=True, color='blue', bins=20)
            plt.title('4. Humidity Distribution')
            plt.xlabel('Humidity (%)')
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.show()

            # 5. Heatmap of Correlation Matrix
            plt.figure(figsize=(8, 6))
            correlation_matrix = df[['temperature', 'humidity']].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
            plt.title('5. Correlation Heatmap: Temperature and Humidity')
            plt.tight_layout()
            plt.show()

            # 6. Boxplot: Temperature Data
            plt.figure(figsize=(8, 5))
            sns.boxplot(x=df['temperature'], color='red')
            plt.title('6. Temperature Boxplot')
            plt.xlabel('Temperature (°C)')
            plt.tight_layout()
            plt.show()

            # 7. Boxplot: Humidity Data
            plt.figure(figsize=(8, 5))
            sns.boxplot(x=df['humidity'], color='blue')
            plt.title('7. Humidity Boxplot')
            plt.xlabel('Humidity (%)')
            plt.tight_layout()
            plt.show()

        else:
            print("Error: 'timestamp' column not found in the dataset.")
    except FileNotFoundError:
        print("Error: The file 'weather_data.csv' could not be found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Call the visualization function
visualize_data()
