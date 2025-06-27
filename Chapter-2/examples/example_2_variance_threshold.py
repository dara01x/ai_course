"""
Example 2 - Variance Threshold Feature Selection
This example demonstrates removing features with standard deviation less than 0.5 from iris dataset
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.feature_selection import VarianceThreshold

# Load iris dataset
iris = load_iris()
X = iris.data
feature_names = iris.feature_names

print("=== Variance Threshold Feature Selection ===")
print(f"Original dataset shape: {X.shape}")
print(f"Feature names: {feature_names}")

# Calculate standard deviation for each feature
std_devs = np.std(X, axis=0)
print(f"\nStandard deviations:")
for i, (name, std) in enumerate(zip(feature_names, std_devs)):
    print(f"  {name}: {std:.3f}")

# Manual threshold filtering (std < 0.5)
threshold = 0.5
manual_mask = std_devs >= threshold
X_manual = X[:, manual_mask]
selected_features_manual = [feature_names[i] for i in range(len(feature_names)) if manual_mask[i]]

print(f"\nManual selection (std >= {threshold}):")
print(f"  Selected features: {selected_features_manual}")
print(f"  New shape: {X_manual.shape}")

# Using scikit-learn VarianceThreshold
# Note: VarianceThreshold uses variance, not std. variance = std²
variance_threshold = threshold ** 2
selector = VarianceThreshold(threshold=variance_threshold)
X_sklearn = selector.fit_transform(X)

# Get selected feature names
sklearn_mask = selector.get_support()
selected_features_sklearn = [feature_names[i] for i in range(len(feature_names)) if sklearn_mask[i]]

print(f"\nScikit-learn VarianceThreshold (variance >= {variance_threshold}):")
print(f"  Selected features: {selected_features_sklearn}")
print(f"  New shape: {X_sklearn.shape}")

# Show variances
variances = np.var(X, axis=0)
print(f"\nVariances:")
for i, (name, var) in enumerate(zip(feature_names, variances)):
    status = "KEPT" if var >= variance_threshold else "REMOVED"
    print(f"  {name}: {var:.3f} ({status})")

# Verification
print(f"\nVerification:")
print(f"Manual and sklearn results match: {np.allclose(X_manual, X_sklearn)}")

# Answer: Which feature should be removed if keeping only 2?
print(f"\nIf keeping only 2 features:")
sorted_features = sorted(zip(feature_names, std_devs), key=lambda x: x[1], reverse=True)
print("Features ranked by standard deviation (highest first):")
for name, std in sorted_features:
    print(f"  {name}: {std:.3f}")
print(f"Answer: Remove '{sorted_features[-1][0]}' and '{sorted_features[-2][0]}'")
