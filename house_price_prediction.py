"""
House Price Prediction with Linear Regression
--------------------------------------------
This script uses the Boston Housing dataset to predict house prices 
using a linear regression model.
"""

import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Data Loading
boston = load_boston()
data = pd.DataFrame(boston.data, columns=boston.feature_names)
data['PRICE'] = boston.target

# 2. Data Exploration (optional: print first 5 rows)
print("First 5 rows of the dataset:")
print(data.head(), '\n')

# 3. Data Preprocessing
# Check for missing values
print("Missing values in each column:\n", data.isnull().sum(), '\n')

X = data.drop('PRICE', axis=1)
y = data['PRICE']

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Model Building
model = LinearRegression()
model.fit(X_train, y_train)

print(f"Model intercept: {model.intercept_}")
print(f"Model coefficients: {model.coef_}\n")

# 5. Model Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# 6. Save results (optional)
# pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).to_csv('predictions.csv', index=False)
