# ================================
# AIR QUALITY PREDICTION PIPELINE
# ================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# -------------------------------
# 1. LOAD DATA
# -------------------------------
# Replace with your dataset path
data = pd.read_csv("air_quality_data.csv")

print("Dataset Loaded Successfully")
print(data.head())

# -------------------------------
# 2. PREPROCESSING
# -------------------------------

# Selecting features and target
features = ['PM2.5', 'PM10', 'CO', 'NO2', 'temperature', 'humidity', 'wind_speed']
target = 'AQI'

X = data[features]
y = data[target]

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# -------------------------------
# 3. TRAIN-TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 4. MODEL TRAINING
# -------------------------------
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    random_state=42
)

model.fit(X_train, y_train)

print("Model Training Completed")

# -------------------------------
# 5. PREDICTION
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# 6. EVALUATION
# -------------------------------
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("RMSE:", round(rmse, 2))
print("MAE:", round(mae, 2))
print("R2 Score:", round(r2, 2))

# -------------------------------
# 7. SAMPLE PREDICTION
# -------------------------------
sample = np.array([[50, 80, 0.5, 30, 28, 65, 10]])  # example input
sample = imputer.transform(sample)
sample = scaler.transform(sample)

prediction = model.predict(sample)
print("\nSample AQI Prediction:", prediction[0])