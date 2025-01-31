import pandas as pd
import matplotlib.pyplot as plt

CSV_FILE = "data/processed_weather_data.csv"

def visualize_data():
    try:
        df = pd.read_csv(CSV_FILE)

        # City-wise temperature bar chart
        city_avg_temp = df.groupby("city")["temperature"].mean()
        city_avg_temp.plot(kind="bar", color="skyblue")
        plt.title("Average Temperature by City")
        plt.xlabel("City")
        plt.ylabel("Temperature (°C)")
        plt.show()

        # Temperature vs Humidity scatter plot
        plt.scatter(df["humidity"], df["temperature"], color="green")
        plt.title("Temperature vs Humidity")
        plt.xlabel("Humidity (%)")
        plt.ylabel("Temperature (°C)")
        plt.show()
    except FileNotFoundError:
        print(f"File {CSV_FILE} not found. Please preprocess the data first.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    visualize_data()
