import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the trained model
with open('models/random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load or define the label encoder for 'weather' feature
# Assuming the LabelEncoder was saved or previously fitted during training
label_encoder = LabelEncoder()
label_encoder.fit(['Clear', 'Cloudy', 'Rain', 'Foggy'])  # Or load if saved

# Sample input for prediction (ensure these values match the training data features)
input_data = pd.DataFrame({
    'humidity': [0.3],
    'hour': [12],
    'day': [2],
    'month': [1],
    'weather': [label_encoder.transform(['Clear'])[0]]  # Transforming 'Clear' to its encoded value
})

# Predict the temperature
predicted_temperature = model.predict(input_data)

# Print the result
print(f"Predicted Temperature: {predicted_temperature[0]}°C")
