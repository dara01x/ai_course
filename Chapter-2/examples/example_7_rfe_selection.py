"""
Example 7 - Recursive Feature Elimination (RFE)
This example demonstrates using RFE to remove least significant features from iris dataset
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

print("=== Recursive Feature Elimination (RFE) ===")
print(f"Original dataset shape: {X.shape}")
print(f"Feature names: {list(feature_names)}")

# RFE with Logistic Regression
print(f"\n1. RFE with Logistic Regression:")
estimator_lr = LogisticRegression(max_iter=1000, random_state=42)
rfe_lr = RFE(estimator=estimator_lr, n_features_to_select=2)
X_rfe_lr = rfe_lr.fit_transform(X, y)

# Get feature rankings and selected features
lr_rankings = rfe_lr.ranking_
lr_selected = rfe_lr.get_support()
lr_selected_features = [feature_names[i] for i in range(len(feature_names)) if lr_selected[i]]

print(f"Feature rankings (1=best, higher=worse):")
for name, rank, selected in zip(feature_names, lr_rankings, lr_selected):
    status = "SELECTED" if selected else "REMOVED"
    print(f"  {name}: rank {rank} ({status})")

print(f"Selected features: {lr_selected_features}")

# RFE with Random Forest
print(f"\n2. RFE with Random Forest:")
estimator_rf = RandomForestClassifier(n_estimators=100, random_state=42)
rfe_rf = RFE(estimator=estimator_rf, n_features_to_select=2)
X_rfe_rf = rfe_rf.fit_transform(X, y)

rf_rankings = rfe_rf.ranking_
rf_selected = rfe_rf.get_support()
rf_selected_features = [feature_names[i] for i in range(len(feature_names)) if rf_selected[i]]

print(f"Feature rankings:")
for name, rank, selected in zip(feature_names, rf_rankings, rf_selected):
    status = "SELECTED" if selected else "REMOVED"
    print(f"  {name}: rank {rank} ({status})")

print(f"Selected features: {rf_selected_features}")

# RFE with SVM
print(f"\n3. RFE with SVM:")
estimator_svm = SVC(kernel='linear', random_state=42)
rfe_svm = RFE(estimator=estimator_svm, n_features_to_select=2)
X_rfe_svm = rfe_svm.fit_transform(X, y)

svm_rankings = rfe_svm.ranking_
svm_selected = rfe_svm.get_support()
svm_selected_features = [feature_names[i] for i in range(len(feature_names)) if svm_selected[i]]

print(f"Feature rankings:")
for name, rank, selected in zip(feature_names, svm_rankings, svm_selected):
    status = "SELECTED" if selected else "REMOVED"
    print(f"  {name}: rank {rank} ({status})")

print(f"Selected features: {svm_selected_features}")

# Compare results across estimators
print(f"\n4. Comparison Across Estimators:")
print(f"Logistic Regression: {lr_selected_features}")
print(f"Random Forest:       {rf_selected_features}")
print(f"SVM:                 {svm_selected_features}")

# Find consensus features (selected by all methods)
all_selected = [lr_selected_features, rf_selected_features, svm_selected_features]
consensus_features = set(all_selected[0])
for selected in all_selected[1:]:
    consensus_features = consensus_features.intersection(set(selected))

print(f"Consensus features (selected by all): {list(consensus_features)}")

# Step-by-step elimination demonstration
print(f"\n5. Step-by-step RFE Process (Logistic Regression):")
estimator_demo = LogisticRegression(max_iter=1000, random_state=42)
rfe_demo = RFE(estimator=estimator_demo, n_features_to_select=1, step=1)
rfe_demo.fit(X, y)

print(f"Elimination order (worst to best):")
elimination_order = []
for rank in range(len(feature_names), 0, -1):
    eliminated_features = [feature_names[i] for i, r in enumerate(rfe_demo.ranking_) if r == rank]
    elimination_order.extend(eliminated_features)
    if rank == len(feature_names):
        print(f"  Step 1: Remove {eliminated_features[0]} (least important)")
    elif rank == len(feature_names) - 1:
        print(f"  Step 2: Remove {eliminated_features[0]}")
    elif rank == len(feature_names) - 2:
        print(f"  Step 3: Remove {eliminated_features[0]}")
    else:
        print(f"  Final: Keep {eliminated_features[0]} (most important)")

print(f"\nFinal shapes:")
print(f"Original: {X.shape}")
print(f"After RFE: {X_rfe_lr.shape}")
