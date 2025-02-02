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
    
    # Handle missing values using imputation (mean for numerical columns)
    imputer = SimpleImputer(strategy='mean')
    df['temperature'] = imputer.fit_transform(df[['temperature']])
    df['humidity'] = imputer.fit_transform(df[['humidity']])
    
    # Feature engineering: Extract time-based features from timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    if df['timestamp'].isnull().any():
        print("Warning: Some timestamps could not be converted and have been set to NaT.")
    
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['weekday'] = df['timestamp'].dt.weekday  # Add weekday feature
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)  # Add is_weekend feature
    
    # Optional: Removing outliers (using Z-score method for simplicity)
    z_scores = np.abs(stats.zscore(df[['temperature', 'humidity']]))
    df = df[(z_scores < 3).all(axis=1)]  # Keep only rows where Z-score < 3
    
    # Split the data into train and test sets (important for scaling after split)
    X = df[['temperature', 'humidity', 'hour', 'day', 'month', 'weekday', 'is_weekend']]
    y = df['temperature']  # Assuming temperature is the target variable
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling numerical features (standardization)
    scaler = StandardScaler()
    X_train[['temperature', 'humidity']] = scaler.fit_transform(X_train[['temperature', 'humidity']])
    X_test[['temperature', 'humidity']] = scaler.transform(X_test[['temperature', 'humidity']])
    
    # Handle categorical features (if any, for example, weather condition or other features)
    # df = pd.get_dummies(df, columns=['weather'])

    # Remove duplicates if any
    df = df.drop_duplicates()
    
    # Saving the cleaned and processed data to a new CSV
    df.to_csv("processed_weather_data.csv", index=False)
    print("Data preprocessing completed and saved to 'processed_weather_data.csv'.")

if __name__ == "__main__":
    preprocess_data()
