"""
Example 2 - Create Regression Dataset
Write Python code to create regression dataset with 500 samples and 2 features
"""

from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import numpy as np

# Create regression dataset with 500 samples and 2 features
X, y = make_regression(
    n_samples=500,      # Number of samples
    n_features=2,       # Number of features
    noise=10,           # Amount of noise
    random_state=42     # For reproducibility
)

print("=== Regression Dataset ===")
print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"First 5 feature samples:\n{X[:5]}")
print(f"First 5 target values: {y[:5]}")

# Visualize the dataset
fig = plt.figure(figsize=(12, 4))

# Plot feature 1 vs target
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], y, alpha=0.6)
plt.xlabel('Feature 1')
plt.ylabel('Target')
plt.title('Feature 1 vs Target')

# Plot feature 2 vs target
plt.subplot(1, 3, 2)
plt.scatter(X[:, 1], y, alpha=0.6)
plt.xlabel('Feature 2')
plt.ylabel('Target')
plt.title('Feature 2 vs Target')

# 3D plot of both features vs target
ax = fig.add_subplot(1, 3, 3, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y, alpha=0.6)
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Target')
ax.set_title('3D View: Features vs Target')

plt.tight_layout()
plt.show()

# Save the dataset
np.save('regression_dataset_X.npy', X)
np.save('regression_dataset_y.npy', y)
print("\nDataset saved as 'regression_dataset_X.npy' and 'regression_dataset_y.npy'")
