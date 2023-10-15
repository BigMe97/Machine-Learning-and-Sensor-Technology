import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from scipy import stats
import math

# Load the dataset
housingData = pd.read_csv('C:/Users/magnu\/OneDrive - USN/Machine Learning and Sensor Technology/Assignment 2/Housing.csv')

# Section numerical columns
numericalColumns = housingData.select_dtypes(include=['number']).columns
housingData[numericalColumns] = housingData[numericalColumns]
# Section categorical columns
categorical_columns = housingData.select_dtypes(include=['object']).columns
housingData[categorical_columns] = housingData[categorical_columns]

# Encode categorical columns
encoder = OneHotEncoder(drop='first', sparse=False)
encoded_categorical = encoder.fit_transform(housingData[categorical_columns])
encoded_categorical_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names(categorical_columns))
housingData = pd.concat([housingData[numericalColumns], encoded_categorical_df], axis=1)

# Feature Selection
# housingData = housingData.drop(['hotwaterheating_yes'], axis=1)
housingData = housingData.drop(['basement_yes'], axis=1)
housingData = housingData.drop(['mainroad_yes'], axis=1)
housingData = housingData.drop(['guestroom_yes'], axis=1)
# housingData = housingData.drop(['stories'], axis=1)
# housingData = housingData.drop(['furnishingstatus_semi-furnished',
#                                 'furnishingstatus_unfurnished'], axis=1)

print(housingData.columns)

# Train-Test Split
X = housingData.drop(columns=['price'])
y = housingData['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Step 9: Feature Engineering (optional)

# Model Training
print('Training')
# model = RandomForestRegressor(n_estimators=86, random_state=42)
# model = Ridge(alpha=0.0000000001, random_state=12)
model = LinearRegression()
# model = XGBRegressor(n_estimators=45, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
print('Evaluate')
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

y_test = np.array(y_test)
y_pred = np.array(y_pred)
y_pred = np.round(y_pred, -3)

error = y_test - y_pred
plt.figure(2)
plt.plot(y_test, 'g', y_pred, 'r', error, 'y')
plt.title('Test VS Predicted')
plt.legend(['Test', 'Predicted', 'Difference'])

print(f"Mean Absolute Error: {mae}")
plt.show()
