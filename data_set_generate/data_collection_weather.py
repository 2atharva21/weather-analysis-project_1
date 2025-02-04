import pandas as pd
from random import randint, choice
from datetime import datetime, timedelta

# List of states in India with their cities
states_and_cities = {
    "Andhra Pradesh": ["Visakhapatnam", "Vijayawada", "Guntur", "Tirupati", "Kakinada"],
    "Arunachal Pradesh": ["Itanagar", "Tawang", "Pasighat", "Ziro", "Aalo"],
    "Assam": ["Guwahati", "Jorhat", "Dibrugarh", "Nagaon", "Tinsukia"],
    "Bihar": ["Patna", "Gaya", "Bhagalpur", "Muzaffarpur", "Darbhanga"],
    "Chhattisgarh": ["Raipur", "Bilaspur", "Korba", "Durg", "Raigarh"],
    "Goa": ["Panaji", "Margao", "Vasco da Gama", "Mapusa", "Ponda"],
    "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Bhavnagar"],
    "Haryana": ["Chandigarh", "Faridabad", "Gurugram", "Panipat", "Ambala"],
    "Himachal Pradesh": ["Shimla", "Manali", "Kullu", "Dharamsala", "Solan"],
    "Jharkhand": ["Ranchi", "Jamshedpur", "Dhanbad", "Deoghar", "Bokaro"],
    "Karnataka": ["Bangalore", "Mysore", "Hubli", "Mangalore", "Belagavi"],
    "Kerala": ["Kochi", "Thiruvananthapuram", "Kozhikode", "Kottayam", "Thrissur"],
    "Madhya Pradesh": ["Bhopal", "Indore", "Gwalior", "Ujjain", "Jabalpur"],
    "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Nashik", "Aurangabad"],
    "Manipur": ["Imphal", "Thoubal", "Churachandpur", "Kakching", "Jiribam"],
    "Meghalaya": ["Shillong", "Tura", "Nongstoin", "Jowai", "Williamnagar"],
    "Mizoram": ["Aizawl", "Lunglei", "Champhai", "Kolasib", "Serchhip"],
    "Nagaland": ["Kohima", "Dimapur", "Mokokchung", "Tuensang", "Zunheboto"],
    "Odisha": ["Bhubaneswar", "Cuttack", "Berhampur", "Sambalpur", "Rourkela"],
    "Punjab": ["Amritsar", "Chandigarh", "Ludhiana", "Patiala", "Jalandhar"],
    "Rajasthan": ["Jaipur", "Udaipur", "Jodhpur", "Ajmer", "Kota"],
    "Sikkim": ["Gangtok", "Namchi", "Mangan", "Pakyong", "Jorethang"],
    "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Trichy", "Salem"],
    "Telangana": ["Hyderabad", "Warangal", "Khammam", "Karimnagar", "Nizamabad"],
    "Tripura": ["Agartala", "Udaipur", "Dharmanagar", "Amarpur", "Belonia"],
    "Uttar Pradesh": ["Lucknow", "Agra", "Kanpur", "Varanasi", "Meerut"],
    "Uttarakhand": ["Dehradun", "Nainital", "Haridwar", "Rishikesh", "Roorkee"],
    "West Bengal": ["Kolkata", "Siliguri", "Durgapur", "Asansol", "Howrah"]
}

# Possible weather conditions
weather_conditions = ["Clear", "Cloudy", "Foggy", "Sunny", "Rainy", "Stormy"]

# Start date for timestamps
start_date = datetime(2025, 1, 1)

# Create 1000 rows of data
data = []
for i in range(1000):
    # Pick a random state and city from the list
    state = choice(list(states_and_cities.keys()))
    city = choice(states_and_cities[state])
    
    # Randomly generate temperature, humidity, and weather
    temperature = randint(10, 35)  # Random temperature between 10°C and 35°C
    humidity = round(randint(40, 90) / 100, 2)  # Random humidity between 40% and 90%
    weather = choice(weather_conditions)
    
    # Generate timestamp (random incrementing from the start date)
    timestamp = (start_date + timedelta(days=i)).strftime('%Y-%m-%d %H:%M:%S')
    
    # Add the data to the list
    data.append([timestamp, state, city, temperature, humidity, weather])

# Convert the data into a DataFrame with the required columns
df = pd.DataFrame(data, columns=["timestamp", "state", "city", "temperature", "humidity", "weather"])

# Save this data into a CSV file
df.to_csv('weather_data.csv', index=False)

# Show the first few rows of the DataFrame
print(df.head())
