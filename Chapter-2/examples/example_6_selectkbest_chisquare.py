"""
Example 6 - SelectKBest Feature Selection with Chi-square
This example demonstrates keeping K highest scoring features in iris dataset using chi-square test
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.preprocessing import MinMaxScaler

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

print("=== SelectKBest Feature Selection ===")
print(f"Original dataset shape: {X.shape}")
print(f"Feature names: {list(feature_names)}")
print(f"Target classes: {iris.target_names}")

# Chi-square test requires non-negative features
# Scale features to positive range [0, 1]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

print(f"\nFeatures scaled to [0, 1] range for chi-square test")

# Method 1: Chi-square test (for classification with positive features)
print(f"\n1. Chi-square Test:")
k = 2  # Select top 2 features
selector_chi2 = SelectKBest(score_func=chi2, k=k)
X_chi2 = selector_chi2.fit_transform(X_scaled, y)

# Get chi-squared scores
chi2_scores = selector_chi2.scores_
chi2_pvalues = selector_chi2.pvalues_

print(f"Chi-square scores and p-values:")
for i, (name, score, pval) in enumerate(zip(feature_names, chi2_scores, chi2_pvalues)):
    selected = "✓" if selector_chi2.get_support()[i] else "✗"
    print(f"  {selected} {name}: score={score:.3f}, p-value={pval:.6f}")

selected_features_chi2 = [feature_names[i] for i in range(len(feature_names)) if selector_chi2.get_support()[i]]
print(f"Selected features (Chi-square): {selected_features_chi2}")

# Method 2: F-classif test (ANOVA F-test)
print(f"\n2. ANOVA F-test:")
selector_f = SelectKBest(score_func=f_classif, k=k)
X_f = selector_f.fit_transform(X, y)  # Can use original data (no scaling required)

# Get F-scores
f_scores = selector_f.scores_
f_pvalues = selector_f.pvalues_

print(f"F-scores and p-values:")
for i, (name, score, pval) in enumerate(zip(feature_names, f_scores, f_pvalues)):
    selected = "✓" if selector_f.get_support()[i] else "✗"
    print(f"  {selected} {name}: score={score:.3f}, p-value={pval:.6f}")

selected_features_f = [feature_names[i] for i in range(len(feature_names)) if selector_f.get_support()[i]]
print(f"Selected features (F-test): {selected_features_f}")

# Compare methods
print(f"\n3. Comparison:")
print(f"Chi-square selected: {selected_features_chi2}")
print(f"F-test selected:     {selected_features_f}")
print(f"Methods agree: {set(selected_features_chi2) == set(selected_features_f)}")

# Show feature ranking
print(f"\n4. Feature Rankings:")
print("By Chi-square score (descending):")
chi2_ranking = sorted(zip(feature_names, chi2_scores), key=lambda x: x[1], reverse=True)
for i, (name, score) in enumerate(chi2_ranking, 1):
    print(f"  {i}. {name}: {score:.3f}")

print("By F-score (descending):")
f_ranking = sorted(zip(feature_names, f_scores), key=lambda x: x[1], reverse=True)
for i, (name, score) in enumerate(f_ranking, 1):
    print(f"  {i}. {name}: {score:.3f}")

# Statistical significance
print(f"\n5. Statistical Significance (p < 0.05):")
significant_chi2 = [name for name, pval in zip(feature_names, chi2_pvalues) if pval < 0.05]
significant_f = [name for name, pval in zip(feature_names, f_pvalues) if pval < 0.05]

print(f"Significant features (Chi-square): {significant_chi2}")
print(f"Significant features (F-test): {significant_f}")

print(f"\nFinal shapes:")
print(f"Original: {X.shape}")
print(f"After Chi-square selection: {X_chi2.shape}")
print(f"After F-test selection: {X_f.shape}")
