"""
Example 4 - Polynomial Regression for Non-linear Relationships
This example demonstrates how to use PolynomialFeatures with LinearRegression for non-linear data
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

print("=== Polynomial Regression Example ===")

# Create non-linear synthetic data
x = 1 + np.linspace(0.0, 2 * np.pi, 100)
y = 5 * np.sin(x * np.pi) * np.exp(-x/4)
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

print(f"Dataset shape: x={x.shape}, y={y.shape}")

# Linear regression baseline
model = LinearRegression()
model.fit(x, y)
yp_linear = model.predict(x)
mse_linear = mean_squared_error(y, yp_linear)
print(f'Linear Regression MSE = {mse_linear:.3f}')

# Create visualization
plt.figure(figsize=(12, 8))
plt.plot(x, y, color="red", linewidth=3, label="True function")
plt.plot(x, yp_linear, color="blue", linewidth=2, label="Linear regression")

print(f"\n=== Polynomial Regression Results ===")
print("Degree\tFeatures\tMSE")
print("-" * 30)

# Test different polynomial degrees
degrees = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
colors = ['green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow', 'black']

for i, deg in enumerate(degrees):
    # Create polynomial features
    poly = PolynomialFeatures(degree=deg)
    xp = poly.fit_transform(x)
    
    # Fit polynomial regression
    model.fit(xp, y)
    yp_poly = model.predict(xp)
    mse = mean_squared_error(y, yp_poly)
    
    print(f"{deg}\t{xp.shape[1]}\t\t{mse:.3f}")
    
    # Plot only selected degrees to avoid clutter
    if deg in [2, 5, 9, 12]:
        plt.plot(x, yp_poly, color=colors[i % len(colors)], 
                linewidth=2, label=f"Polynomial degree {deg}")

plt.xlabel('x', fontsize=14)
plt.ylabel("Output", fontsize=14)
plt.title("Polynomial Regression: Effect of Degree", fontsize=16)
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\n=== Key Observations ===")
print("• Higher degree polynomials can fit complex non-linear relationships")
print("• Very high degrees may lead to overfitting (degree 12: MSE ≈ 0.001)")
print("• The optimal degree balances bias and variance")
print("• Polynomial features transform the problem into linear regression")
