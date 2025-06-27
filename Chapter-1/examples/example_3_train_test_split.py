"""
Example 3 - Training and Testing Set Split
Demonstrates how to split dataset into training and testing sets
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

print("=== Dataset Split Example ===")
print(f"Original dataset shape: {X.shape}")
print(f"Original target shape: {y.shape}")

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% for testing
    random_state=42,    # For reproducibility
    stratify=y          # Maintain class distribution
)

print(f"\nAfter split:")
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")
print(f"Training target shape: {y_train.shape}")
print(f"Testing target shape: {y_test.shape}")

# Check class distribution
print(f"\nOriginal class distribution: {np.bincount(y)}")
print(f"Training class distribution: {np.bincount(y_train)}")
print(f"Testing class distribution: {np.bincount(y_test)}")

# Calculate split percentages
train_percent = (len(X_train) / len(X)) * 100
test_percent = (len(X_test) / len(X)) * 100

print(f"\nSplit percentages:")
print(f"Training: {train_percent:.1f}%")
print(f"Testing: {test_percent:.1f}%")

# Alternative: Split into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"\n=== Three-way Split (60% train, 20% val, 20% test) ===")
print(f"Training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")
print(f"Testing set shape: {X_test.shape}")

train_percent = (len(X_train) / len(X)) * 100
val_percent = (len(X_val) / len(X)) * 100
test_percent = (len(X_test) / len(X)) * 100

print(f"\nSplit percentages:")
print(f"Training: {train_percent:.1f}%")
print(f"Validation: {val_percent:.1f}%")
print(f"Testing: {test_percent:.1f}%")
