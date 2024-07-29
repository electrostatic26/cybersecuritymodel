
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('cyber_data.csv')

# Feature engineering
data['day_of_week'] = pd.to_datetime(data['Attack-Date']).dt.dayofweek
data['month'] = pd.to_datetime(data['Attack-Date']).dt.month

# Select features and target
features = ['day_of_week', 'month', 'Country', 'Spam', 'Ransomware', 'Local Infection', 'Exploit', 'Malicious Mail']
target = 'Number of Attacks'  # Hypothetical target column

# Encode categorical variables
encoder = OneHotEncoder()
encoded_features = encoder.fit_transform(data[['Country']])

# Scale numerical features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[['Spam', 'Ransomware', 'Local Infection', 'Exploit', 'Malicious Mail']])

# Combine features
X = pd.concat([pd.DataFrame(encoded_features.toarray()), pd.DataFrame(scaled_features), data[['day_of_week', 'month']]], axis=1)
y = data[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
