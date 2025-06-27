"""
Example 1 - Feature Normalization
This example demonstrates how to transform features into [0, 1] range and [-1, 1] range
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Sample data array
x = np.array([10, 20, 30, 40, 50])

print("=== Feature Normalization Example ===")
print(f"Original array: {x}")

# Manual normalization to [0, 1]
def normalize_to_01(data):
    """Normalize data to [0, 1] range"""
    x_min = np.min(data)
    x_max = np.max(data)
    return (data - x_min) / (x_max - x_min)

# Manual normalization to [-1, 1]
def normalize_to_minus1_plus1(data):
    """Normalize data to [-1, 1] range"""
    x_min = np.min(data)
    x_max = np.max(data)
    # First normalize to [0, 1], then scale to [-1, 1]
    normalized_01 = (data - x_min) / (x_max - x_min)
    return 2 * normalized_01 - 1

# Apply manual normalization
x_normalized_01 = normalize_to_01(x)
x_normalized_minus1_plus1 = normalize_to_minus1_plus1(x)

print(f"\nManual normalization [0, 1]: {x_normalized_01}")
print(f"Manual normalization [-1, 1]: {x_normalized_minus1_plus1}")

# Using scikit-learn MinMaxScaler
scaler_01 = MinMaxScaler(feature_range=(0, 1))
scaler_minus1_plus1 = MinMaxScaler(feature_range=(-1, 1))

# Reshape for sklearn (needs 2D array)
x_reshaped = x.reshape(-1, 1)

x_sklearn_01 = scaler_01.fit_transform(x_reshaped).flatten()
x_sklearn_minus1_plus1 = scaler_minus1_plus1.fit_transform(x_reshaped).flatten()

print(f"\nScikit-learn normalization [0, 1]: {x_sklearn_01}")
print(f"Scikit-learn normalization [-1, 1]: {x_sklearn_minus1_plus1}")

# Verification
print(f"\nVerification:")
print(f"Manual [0,1] == Sklearn [0,1]: {np.allclose(x_normalized_01, x_sklearn_01)}")
print(f"Manual [-1,1] == Sklearn [-1,1]: {np.allclose(x_normalized_minus1_plus1, x_sklearn_minus1_plus1)}")

# Show the formula
print(f"\nFormula: x_normalized = (x - x_min) / (x_max - x_min)")
print(f"For [-1, 1]: x_normalized = 2 * ((x - x_min) / (x_max - x_min)) - 1")
