"""
Example 8 - RFE with Large Dataset
This example demonstrates RFE on a synthetic dataset with 100 features and 1000 samples
"""

from sklearn.datasets import make_classification
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import numpy as np

# Create synthetic dataset
print("=== RFE with Large Synthetic Dataset ===")

# Generate dataset with 1000 samples, 100 features, but only 2 are actually informative
X, y = make_classification(
    n_samples=1000, 
    n_features=100, 
    n_informative=2,  # Only 2 features are actually useful
    n_redundant=5,    # 5 features are linear combinations of informative features
    n_clusters_per_class=1,
    random_state=42
)

print(f"Dataset created:")
print(f"  Samples: {X.shape[0]}")
print(f"  Total features: {X.shape[1]}")
print(f"  Informative features: 2")
print(f"  Redundant features: 5")
print(f"  Random features: {100 - 2 - 5}")

# Apply RFE to select top 5 features
estimator = LogisticRegression(max_iter=500, random_state=42)
rfe = RFE(estimator=estimator, n_features_to_select=5, step=1)
X_selected = rfe.fit_transform(X, y)

print(f"\nRFE Results:")
print(f"  Original shape: {X.shape}")
print(f"  Selected shape: {X_selected.shape}")

# Get feature rankings
rankings = rfe.ranking_
selected_features = np.where(rfe.get_support())[0]

print(f"\nTop 5 selected features (indices): {selected_features}")
print(f"Their rankings: {[rankings[i] for i in selected_features]}")

# Show distribution of rankings
unique_ranks = np.unique(rankings)
print(f"\nRanking distribution:")
for rank in sorted(unique_ranks):
    count = np.sum(rankings == rank)
    if rank == 1:
        print(f"  Rank {rank} (SELECTED): {count} features")
    else:
        print(f"  Rank {rank}: {count} features")

# Analyze feature importance using the final model
final_model = rfe.estimator_
feature_importance = np.abs(final_model.coef_[0])

print(f"\nFeature importance (absolute coefficients) for selected features:")
for i, (feat_idx, importance) in enumerate(zip(selected_features, feature_importance)):
    print(f"  Feature {feat_idx}: {importance:.4f}")

# Performance comparison
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

# Train on all features
model_all = LogisticRegression(max_iter=500, random_state=42)
scores_all = cross_val_score(model_all, X, y, cv=5, scoring='accuracy')

# Train on selected features
model_selected = LogisticRegression(max_iter=500, random_state=42)
scores_selected = cross_val_score(model_selected, X_selected, y, cv=5, scoring='accuracy')

print(f"\nPerformance Comparison (5-fold CV):")
print(f"  All features (100):     {scores_all.mean():.4f} ± {scores_all.std():.4f}")
print(f"  Selected features (5):  {scores_selected.mean():.4f} ± {scores_selected.std():.4f}")
print(f"  Feature reduction:      {((100-5)/100)*100:.1f}%")
print(f"  Performance change:     {((scores_selected.mean() - scores_all.mean())/scores_all.mean())*100:+.2f}%")

# Show most and least important features
print(f"\nFeature Analysis:")
print(f"Most important features (rank 1): {np.where(rankings == 1)[0]}")
print(f"Least important features (highest rank): {np.where(rankings == rankings.max())[0][:5]}")

print(f"\nConclusion:")
if scores_selected.mean() >= scores_all.mean() * 0.95:  # Within 5% of original performance
    print(f"  ✓ RFE successfully reduced features by 95% while maintaining performance!")
else:
    print(f"  ⚠ RFE reduced features but with some performance loss.")
