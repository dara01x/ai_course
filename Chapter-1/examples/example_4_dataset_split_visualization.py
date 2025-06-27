"""
Example 4 - Dataset Splitting with Visualization
Advanced example showing dataset splitting with visualization
"""

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Create a binary classification dataset
X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_classes=2,
    n_clusters_per_class=1,
    random_state=42
)

print("=== Dataset Splitting with Visualization ===")
print(f"Dataset shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Class distribution: {np.bincount(y)}")

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create visualization
plt.figure(figsize=(15, 5))

# Plot 1: Original dataset
plt.subplot(1, 3, 1)
colors = ['red', 'blue']
for i in range(2):
    mask = y == i
    plt.scatter(X[mask, 0], X[mask, 1], c=colors[i], 
                label=f'Class {i}', alpha=0.6)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Original Dataset')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Training set
plt.subplot(1, 3, 2)
for i in range(2):
    mask = y_train == i
    plt.scatter(X_train[mask, 0], X_train[mask, 1], c=colors[i], 
                label=f'Class {i}', alpha=0.6)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title(f'Training Set (n={len(X_train)})')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Test set
plt.subplot(1, 3, 3)
for i in range(2):
    mask = y_test == i
    plt.scatter(X_test[mask, 0], X_test[mask, 1], c=colors[i], 
                label=f'Class {i}', alpha=0.6)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title(f'Test Set (n={len(X_test)})')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print statistics
print(f"\nDataset Statistics:")
print(f"Total samples: {len(X)}")
print(f"Training samples: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Testing samples: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

print(f"\nClass Distribution:")
print(f"Original - Class 0: {np.sum(y == 0)}, Class 1: {np.sum(y == 1)}")
print(f"Training - Class 0: {np.sum(y_train == 0)}, Class 1: {np.sum(y_train == 1)}")
print(f"Testing - Class 0: {np.sum(y_test == 0)}, Class 1: {np.sum(y_test == 1)}")
