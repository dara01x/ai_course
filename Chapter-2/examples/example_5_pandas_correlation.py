"""
Example 5 - Pandas Correlation Analysis
This example repeats the previous correlation analysis using pandas library for efficiency
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes

# Load diabetes dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
feature_names = diabetes.feature_names

print("=== Pandas Correlation Analysis ===")
print(f"Dataset shape: {X.shape}")

# Create pandas DataFrame
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# 1. Correlation between features and target
print("\n1. Feature-Target Correlations:")
target_corr = df.corr()['target'].drop('target').sort_values(key=abs, ascending=False)
for feature, corr in target_corr.items():
    print(f"   {feature}: {corr:.3f}")

# 2. Correlation matrix between all features
print("\n2. Feature-Feature Correlation Matrix:")
feature_corr = df[feature_names].corr()
print(feature_corr.round(3))

# 3. Identify highly correlated feature pairs
print("\n3. Highly Correlated Feature Pairs (|r| > 0.5):")
high_corr_pairs = []
for i in range(len(feature_names)):
    for j in range(i+1, len(feature_names)):
        corr = feature_corr.iloc[i, j]
        if abs(corr) > 0.5:
            high_corr_pairs.append((feature_names[i], feature_names[j], corr))
            print(f"   {feature_names[i]} ↔ {feature_names[j]}: {corr:.3f}")

if not high_corr_pairs:
    print("   No feature pairs with |correlation| > 0.5")

# 4. Summary statistics
print(f"\n4. Summary Statistics:")
print(f"   Strongest positive correlation with target: {target_corr.max():.3f} ({target_corr.idxmax()})")
print(f"   Strongest negative correlation with target: {target_corr.min():.3f} ({target_corr.idxmin()})")
print(f"   Average absolute correlation with target: {abs(target_corr).mean():.3f}")

# 5. Why pandas correlation is efficient
print(f"\n5. Why Pandas Correlation is Efficient:")
print(f"   ✓ Computes correlation between features and target in one operation")
print(f"   ✓ Computes correlation matrix between all features simultaneously")
print(f"   ✓ Handles missing values automatically")
print(f"   ✓ Provides easy sorting and filtering capabilities")
print(f"   ✓ Built on optimized NumPy operations")

# 6. Identify redundant features
print(f"\n6. Redundant Feature Detection:")
redundant_threshold = 0.7
redundant_features = []
for i in range(len(feature_names)):
    for j in range(i+1, len(feature_names)):
        corr = abs(feature_corr.iloc[i, j])
        if corr > redundant_threshold:
            # Keep the one with higher correlation to target
            corr_i = abs(target_corr[feature_names[i]])
            corr_j = abs(target_corr[feature_names[j]])
            if corr_i > corr_j:
                redundant_features.append(feature_names[j])
            else:
                redundant_features.append(feature_names[i])

if redundant_features:
    print(f"   Features to consider removing (correlation > {redundant_threshold}): {redundant_features}")
else:
    print(f"   No redundant features found (threshold = {redundant_threshold})")

# 7. Feature importance ranking
print(f"\n7. Feature Importance Ranking (by absolute correlation with target):")
ranked_features = target_corr.abs().sort_values(ascending=False)
for rank, (feature, corr) in enumerate(ranked_features.items(), 1):
    print(f"   {rank}. {feature}: {corr:.3f}")
