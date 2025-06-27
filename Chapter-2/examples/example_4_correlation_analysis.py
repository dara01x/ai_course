"""
Example 4 - Correlation Analysis in Diabetes Dataset
This example demonstrates computing correlation between features and target in diabetes dataset
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes

# Load diabetes dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
feature_names = diabetes.feature_names

print("=== Correlation Analysis - Diabetes Dataset ===")
print(f"Dataset shape: {X.shape}")
print(f"Feature names: {list(feature_names)}")

# Manual correlation calculation using numpy
def manual_correlation(x, y):
    """Calculate Pearson correlation coefficient manually"""
    # Center the data (subtract mean)
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    
    # Calculate correlation
    numerator = np.sum(x_centered * y_centered)
    denominator = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
    
    return numerator / denominator

print("\nManual correlation calculation (feature vs target):")
manual_correlations = []
for i, feature_name in enumerate(feature_names):
    corr = manual_correlation(X[:, i], y)
    manual_correlations.append(corr)
    print(f"  {feature_name}: {corr:.3f}")

# Using numpy's corrcoef function
print("\nUsing numpy.corrcoef:")
numpy_correlations = []
for i, feature_name in enumerate(feature_names):
    corr_matrix = np.corrcoef(X[:, i], y)
    corr = corr_matrix[0, 1]  # correlation between feature and target
    numpy_correlations.append(corr)
    print(f"  {feature_name}: {corr:.3f}")

# Using pandas for comprehensive analysis
print("\nUsing pandas (more detailed analysis):")
# Create DataFrame
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# Correlation with target
target_correlations = df.corr()['target'].drop('target')
print("Feature-target correlations:")
for feature_name, corr in target_correlations.items():
    print(f"  {feature_name}: {corr:.3f}")

# Show strongest correlations
print(f"\nStrongest positive correlation: {target_correlations.idxmax()} ({target_correlations.max():.3f})")
print(f"Strongest negative correlation: {target_correlations.idxmin()} ({target_correlations.min():.3f})")

# Feature-feature correlation matrix (to identify multicollinearity)
print(f"\nFeature-feature correlation matrix:")
feature_corr_matrix = df[feature_names].corr()
print(feature_corr_matrix.round(3))

# Find highly correlated feature pairs
print(f"\nHighly correlated feature pairs (|correlation| > 0.3):")
for i in range(len(feature_names)):
    for j in range(i+1, len(feature_names)):
        corr = feature_corr_matrix.iloc[i, j]
        if abs(corr) > 0.3:
            print(f"  {feature_names[i]} - {feature_names[j]}: {corr:.3f}")

# Verification
print(f"\nVerification:")
print(f"Manual vs numpy correlations match: {np.allclose(manual_correlations, numpy_correlations)}")
print(f"Numpy vs pandas correlations match: {np.allclose(numpy_correlations, target_correlations.values)}")

# Formula explanation
print(f"\nPearson correlation formula:")
print(f"r = Σ((xi - μx)(yi - μy)) / √(Σ(xi - μx)² * Σ(yi - μy)²)")
print(f"where μx and μy are the means of x and y respectively")
