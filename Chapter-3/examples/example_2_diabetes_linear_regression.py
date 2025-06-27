"""
Example 2 - Multiple Linear Regression with Diabetes Dataset
This example demonstrates multiple linear regression using the diabetes dataset
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error

print("=== Multiple Linear Regression with Diabetes Dataset ===")

# Load diabetes dataset
X, y = load_diabetes(return_X_y=True)
print(f"Dataset shape: X={X.shape}, y={y.shape}")
print(f"Features: 10 physiological variables for each person")

# Split dataset into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Training set size: {xtrain.shape}")
print(f"Testing set size: {xtest.shape}")

# Create and train linear regression model
model = LinearRegression()
model.fit(xtrain, ytrain)

# Make predictions
yp = model.predict(xtest)

# Calculate performance metrics
mse = mean_squared_error(ytest, yp)

print(f"\n=== Model Results ===")
print(f'Mean Squared Error = {mse:.3f}')
print(f'Intercept (W0) = {model.intercept_:.3f}')

print(f"\n=== Feature Weights ===")
feature_names = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
for i, (name, coef) in enumerate(zip(feature_names, model.coef_)):
    print(f'W{i+1} ({name}): {coef:.3f}')

# Show prediction examples
print(f"\n=== Sample Predictions ===")
for i in range(5):
    print(f"Actual: {ytest[i]:.1f}, Predicted: {yp[i]:.1f}, Error: {abs(ytest[i] - yp[i]):.1f}")

print(f"\n=== Model Performance ===")
print(f"Root Mean Squared Error: {np.sqrt(mse):.3f}")
print(f"Mean Absolute Error: {np.mean(np.abs(ytest - yp)):.3f}")
