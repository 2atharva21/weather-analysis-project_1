import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import stats

def preprocess_data():
    # Load the dataset
    df = pd.read_csv("data/weather_data.csv")
    
    # Handle missing values using imputation (mean for numerical columns)
    imputer = SimpleImputer(strategy='mean')
    df['temperature'] = imputer.fit_transform(df[['temperature']])
    df['humidity'] = imputer.fit_transform(df[['humidity']])
    
    # Feature engineering: Extract time-based features from timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['weekday'] = df['timestamp'].dt.weekday  # Add weekday feature
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)  # Add is_weekend feature
    
    # Optional: Removing outliers (using Z-score method for simplicity)
    # Calculate Z-scores for temperature and humidity
    z_scores = np.abs(stats.zscore(df[['temperature', 'humidity']]))
    df = df[(z_scores < 3).all(axis=1)]  # Keep only rows where Z-score < 3
    
    # Scaling numerical features (standardization)
    scaler = StandardScaler()
    df[['temperature', 'humidity']] = scaler.fit_transform(df[['temperature', 'humidity']])
    
    # Handle categorical features (if any)
    # Example: df = pd.get_dummies(df, columns=['weather'])
    
    # Remove duplicates if any
    df = df.drop_duplicates()
    
    # Saving the cleaned and processed data to a new CSV
    df.to_csv("data/processed_weather_data.csv", index=False)
    print("Data preprocessing completed and saved to 'processed_weather_data.csv'.")

if __name__ == "__main__":
    preprocess_data()
