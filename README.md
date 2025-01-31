Weather Analysis Project
This project analyzes weather data and performs machine learning model training for temperature prediction using different regression models.

Features
Weather Data Analysis: Data cleaning, missing value handling, and city-wise analysis.
Visualization: Histograms and plots for temperature and humidity distribution.
Machine Learning Models: Training of XGBoost and Random Forest models to predict temperature.
Model Saving: The trained models are saved for later use.
GitHub LFS: Large files (like .dll files) are handled using Git Large File Storage.
Requirements
Python Libraries
To run this project, you need the following Python libraries:

pandas
numpy
xgboost
scikit-learn
joblib
matplotlib
seaborn

weather-analysis-project/
│
├── data/
│   └── weather_data.csv
│
├── models/
│   ├── xgboost_model.json
│   └── random_forest_model.pkl
│
├── source/
│   ├── analyze_weather.py
│   ├── train_model.py
│
├── temperature_distribution.png
├── requirements.txt
└── README.md

Setup and Usage

git clone https://github.com/2atharva21/weather-analysis-project_1.git

Navigate to the project directory:

cd weather-analysis-project
