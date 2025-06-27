"""
Example 5 - Basic Logistic Regression
This example demonstrates binary classification using logistic regression
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("=== Basic Logistic Regression Example ===")

# Create simple binary classification dataset
x = np.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

print(f"Dataset shape: x={x.shape}, y={y.shape}")
print(f"Classes: {np.unique(y)} (0: negative class, 1: positive class)")

# Create and train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(x, y)

# Make predictions
yp = model.predict(x)
yp_proba = model.predict_proba(x)

# Calculate accuracy
accuracy = accuracy_score(y, yp)
print(f'\nAccuracy = {accuracy:.4f} ({accuracy*100:.1f}%)')

# Display results
print(f"\n=== Model Parameters ===")
print(f"Weight (coefficient): {model.coef_[0][0]:.4f}")
print(f"Intercept: {model.intercept_[0]:.4f}")

print(f"\n=== Predictions vs Actual ===")
print("x\tActual\tPred\tProb(0)\tProb(1)")
print("-" * 40)
for i in range(len(x)):
    print(f"{x[i][0]:.2f}\t{y[i]}\t{yp[i]}\t{yp_proba[i][0]:.3f}\t{yp_proba[i][1]:.3f}")

# Create visualization
plt.figure(figsize=(10, 6))

# Plot data points
plt.scatter(x[y==0], y[y==0], color='red', marker='o', s=100, label='Class 0', alpha=0.7)
plt.scatter(x[y==1], y[y==1], color='blue', marker='s', s=100, label='Class 1', alpha=0.7)

# Plot decision boundary and probability curve
x_range = np.linspace(x.min() - 1, x.max() + 1, 300).reshape(-1, 1)
y_proba_range = model.predict_proba(x_range)[:, 1]

plt.plot(x_range, y_proba_range, 'g-', linewidth=2, label='P(Class=1)')
plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Decision boundary (0.5)')

plt.xlabel('Feature Value', fontsize=12)
plt.ylabel('Class / Probability', fontsize=12)
plt.title('Logistic Regression: Binary Classification', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\n=== Key Concepts ===")
print("• Logistic regression uses sigmoid function to map any real value to (0,1)")
print("• Decision boundary is typically at probability = 0.5")
print("• Output represents probability of belonging to positive class")
print("• Unlike linear regression, logistic regression is used for classification")
