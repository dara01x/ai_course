"""
Example 3 - Regularization Comparison: OLS vs Ridge vs Lasso
This example compares the performance of OLS, Ridge, and Lasso regression
"""

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import numpy as np
from sklearn.metrics import mean_squared_error

print("=== Regularization Comparison: OLS vs Ridge vs Lasso ===")

# Load diabetes dataset
X, y = load_diabetes(return_X_y=True)
print(f"Dataset shape: X={X.shape}, y={y.shape}")

# Standardize features
scaler = StandardScaler().fit(X)
XS = scaler.transform(X)

print("\n=== OLS (Ordinary Least Squares) ===")
print("="*50)

# OLS without standardization
model = LinearRegression()
model.fit(X, y)
yp = model.predict(X)
mse = mean_squared_error(y, yp)
print(f'Linear - Sum of weights = {sum(model.coef_):.3f}, MSE = {mse:.3f}')
print(f'W = {model.coef_}')

# OLS with standardization
model.fit(XS, y)
yp = model.predict(XS)
mse = mean_squared_error(y, yp)
print(f'Normalized - Sum of weights = {sum(model.coef_):.3f}, MSE = {mse:.3f}')
print(f'W = {model.coef_}')

print("\n=== RIDGE REGRESSION ===")
print("="*50)

# Ridge without standardization
model = Ridge(alpha=0.5)
model.fit(X, y)
yp = model.predict(X)
mse = mean_squared_error(y, yp)
print(f'Linear - Sum of weights = {sum(model.coef_):.3f}, MSE = {mse:.3f}')
print(f'W = {model.coef_}')

# Ridge with standardization
model.fit(XS, y)
yp = model.predict(XS)
mse = mean_squared_error(y, yp)
print(f'Normalized - Sum of weights = {sum(model.coef_):.3f}, MSE = {mse:.3f}')
print(f'W = {model.coef_}')

print("\n=== LASSO REGRESSION ===")
print("="*50)

# Lasso without standardization
model = Lasso(alpha=0.5)
model.fit(X, y)
yp = model.predict(X)
mse = mean_squared_error(y, yp)
print(f'Linear - Sum of weights = {sum(model.coef_):.3f}, MSE = {mse:.3f}')
print(f'W = {model.coef_}')

# Lasso with standardization
model.fit(XS, y)
yp = model.predict(XS)
mse = mean_squared_error(y, yp)
print(f'Normalized - Sum of weights = {sum(model.coef_):.3f}, MSE = {mse:.3f}')
print(f'W = {model.coef_}')

print("\n=== Key Observations ===")
print("• Ridge regression reduces coefficient magnitudes without setting them to zero")
print("• Lasso regression sets some coefficients to exactly zero (feature selection)")
print("• Standardization significantly affects Ridge and Lasso performance")
print("• Lasso performs automatic feature selection by zeroing out less important features")
