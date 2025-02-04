import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import stats

def preprocess_data():
    try:
        # Load the dataset
        df = pd.read_csv("weather_data.csv")
    except FileNotFoundError:
        print("Error: The file 'weather_data.csv' was not found.")
        return
    
    # Check if necessary columns exist
    required_columns = ['timestamp', 'temperature', 'humidity']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: The dataset must contain the following columns: {required_columns}")
        return
    
    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    if df['timestamp'].isnull().any():
        print("Warning: Some timestamps could not be converted and have been set to NaT.")
    
    # Feature Engineering: Extract time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['weekday'] = df['timestamp'].dt.weekday
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)

    # Drop rows with NaT timestamps
    df = df.dropna(subset=['timestamp'])
    
    # Removing outliers before imputation
    z_scores = np.abs(stats.zscore(df[['temperature', 'humidity']]))
    df = df[(z_scores < 3).all(axis=1)]

    # Handling missing values (mean imputation)
    imputer = SimpleImputer(strategy='mean')
    df[['temperature', 'humidity']] = imputer.fit_transform(df[['temperature', 'humidity']])
    
    # Splitting data
    X = df[['humidity', 'hour', 'day', 'month', 'weekday', 'is_weekend']]
    y = df['temperature']  # Target variable: temperature

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardization
    scaler = StandardScaler()
    X_train.loc[:, ['humidity']] = scaler.fit_transform(X_train[['humidity']])
    X_test.loc[:, ['humidity']] = scaler.transform(X_test[['humidity']])
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Save processed data
    df.to_csv("processed_weather_data.csv", index=False)
    print("Data preprocessing completed and saved to 'processed_weather_data.csv'.")

if __name__ == "__main__":
    preprocess_data()
