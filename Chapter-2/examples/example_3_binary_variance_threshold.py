"""
Example 3 - Binary Feature Variance Threshold
This example demonstrates removing binary features with probability = 0.8 using variance threshold
"""

import numpy as np
from sklearn.feature_selection import VarianceThreshold

# Create binary features with different probabilities
np.random.seed(42)
n_samples = 1000

# Create binary features with different probabilities
feature_1 = np.random.binomial(1, 0.5, n_samples)    # p=0.5, variance = 0.5*0.5 = 0.25
feature_2 = np.random.binomial(1, 0.8, n_samples)    # p=0.8, variance = 0.8*0.2 = 0.16  
feature_3 = np.random.binomial(1, 0.2, n_samples)    # p=0.2, variance = 0.2*0.8 = 0.16
feature_4 = np.random.binomial(1, 0.9, n_samples)    # p=0.9, variance = 0.9*0.1 = 0.09

# Combine features
X = np.column_stack([feature_1, feature_2, feature_3, feature_4])
feature_names = ['Feature_1 (p=0.5)', 'Feature_2 (p=0.8)', 'Feature_3 (p=0.2)', 'Feature_4 (p=0.9)']

print("=== Binary Feature Variance Threshold ===")
print(f"Dataset shape: {X.shape}")

# Calculate actual probabilities and variances
print("\nActual statistics:")
for i, name in enumerate(feature_names):
    prob = np.mean(X[:, i])
    variance = np.var(X[:, i])
    theoretical_var = prob * (1 - prob)
    print(f"{name}:")
    print(f"  Probability: {prob:.3f}")
    print(f"  Actual variance: {variance:.3f}")
    print(f"  Theoretical variance: {theoretical_var:.3f}")

# For binary features with probability p, variance = p * (1-p)
# For p=0.8, variance = 0.8 * 0.2 = 0.16
target_prob = 0.8
target_variance = target_prob * (1 - target_prob)

print(f"\nTarget: Remove features with probability ≈ {target_prob}")
print(f"Corresponding variance threshold: {target_variance}")

# Apply variance threshold
# We want to remove features with variance ≈ 0.16, so we set threshold slightly higher
threshold = 0.17
selector = VarianceThreshold(threshold=threshold)
X_selected = selector.fit_transform(X)

# Show results
selected_mask = selector.get_support()
print(f"\nResults with threshold = {threshold}:")
for i, (name, selected) in enumerate(zip(feature_names, selected_mask)):
    status = "KEPT" if selected else "REMOVED"
    variance = np.var(X[:, i])
    print(f"  {name}: variance = {variance:.3f} ({status})")

print(f"\nOriginal shape: {X.shape}")
print(f"After selection: {X_selected.shape}")

# Manual approach: target features with p=0.8 specifically
print(f"\nManual approach - target features with p ≈ {target_prob}:")
for i, name in enumerate(feature_names):
    prob = np.mean(X[:, i])
    if abs(prob - target_prob) < 0.05:  # Within 5% of target probability
        print(f"  {name} has probability {prob:.3f} - WOULD BE REMOVED")
    else:
        print(f"  {name} has probability {prob:.3f} - would be kept")
