"""
Example 1 - Simple Linear Regression
This example shows implementation of simple linear regression with visualization
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

print("=== Simple Linear Regression Example ===")

# Create simple regression dataset
X, y, _ = datasets.make_regression(n_samples=200, n_features=1, n_informative=1, 
                                   noise=15, coef=True, random_state=0)

print(f"Dataset shape: X={X.shape}, y={y.shape}")

# Create and train linear regression model
model = linear_model.LinearRegression()
model.fit(X, y)

# Make predictions
yp = model.predict(X)

# Calculate performance metrics
mse = mean_squared_error(y, yp)

print(f'Weight (W) = {model.coef_[0]:.3f}')
print(f'Intercept (W0) = {model.intercept_:.3f}')
print(f'Mean Squared Error = {mse:.3f}')

# Create visualization
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color="red", marker=".", alpha=0.6, label="Actual data (y)")
plt.plot(X, yp, color='blue', linewidth=2, label='Predicted line (yp)')
plt.xlabel("Input Feature (X)", fontsize=12)
plt.ylabel("Target Value (y)", fontsize=12)
plt.title("Simple Linear Regression", fontsize=14)
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n=== Model Equation ===")
print(f"y = {model.coef_[0]:.3f} * x + {model.intercept_:.3f}")
