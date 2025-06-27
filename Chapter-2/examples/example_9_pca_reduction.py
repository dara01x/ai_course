"""
Example 9 - Principal Component Analysis (PCA) 
This example demonstrates using PCA to reduce the dimensionality of iris dataset
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print("=== Principal Component Analysis (PCA) ===")
print(f"Original dataset shape: {X.shape}")
print(f"Features: {list(feature_names)}")

# Step 1: Standardize the data
print(f"\n1. Data Standardization:")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Original data statistics:")
print(f"  Means: {np.mean(X, axis=0).round(3)}")
print(f"  Std devs: {np.std(X, axis=0).round(3)}")

print(f"Standardized data statistics:")
print(f"  Means: {np.mean(X_scaled, axis=0).round(3)}")
print(f"  Std devs: {np.std(X_scaled, axis=0).round(3)}")

# Step 2: Apply PCA
print(f"\n2. PCA Analysis:")
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Get explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

print(f"Explained variance ratio by component:")
for i, (ratio, cumulative) in enumerate(zip(explained_variance_ratio, cumulative_variance)):
    print(f"  PC{i+1}: {ratio:.4f} ({ratio*100:.2f}%) - Cumulative: {cumulative:.4f} ({cumulative*100:.2f}%)")

# Step 3: Dimensionality reduction to 2D
print(f"\n3. Dimensionality Reduction to 2D:")
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)

print(f"Reduced dataset shape: {X_pca_2d.shape}")
print(f"Variance explained by 2 PCs: {sum(pca_2d.explained_variance_ratio_):.4f} ({sum(pca_2d.explained_variance_ratio_)*100:.2f}%)")

# Step 4: Show principal components (loadings)
print(f"\n4. Principal Component Loadings:")
components = pca.components_
print(f"PC1 loadings:")
for feature, loading in zip(feature_names, components[0]):
    print(f"  {feature}: {loading:.4f}")

print(f"PC2 loadings:")
for feature, loading in zip(feature_names, components[1]):
    print(f"  {feature}: {loading:.4f}")

# Step 5: Manual PCA calculation verification
print(f"\n5. Manual PCA Verification:")
# Calculate covariance matrix
cov_matrix = np.cov(X_scaled.T)
print(f"Covariance matrix shape: {cov_matrix.shape}")

# Calculate eigenvalues and eigenvectors  
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort by eigenvalues (descending)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print(f"Manual eigenvalues: {eigenvalues.round(4)}")
print(f"Sklearn eigenvalues: {pca.explained_variance_.round(4)}")
print(f"Eigenvalues match: {np.allclose(eigenvalues, pca.explained_variance_)}")

# Step 6: Determine optimal number of components
print(f"\n6. Optimal Number of Components:")
variance_thresholds = [0.80, 0.90, 0.95, 0.99]
for threshold in variance_thresholds:
    n_components = np.argmax(cumulative_variance >= threshold) + 1
    print(f"  For {threshold*100:.0f}% variance: {n_components} components")

# Step 7: Visualization preparation
print(f"\n7. Data Transformation Summary:")
print(f"Original features: {feature_names}")
print(f"Principal components: ['PC1', 'PC2', 'PC3', 'PC4']")
print(f"PC1 explains {explained_variance_ratio[0]*100:.1f}% of variance")
print(f"PC2 explains {explained_variance_ratio[1]*100:.1f}% of variance")
print(f"First 2 PCs explain {sum(explained_variance_ratio[:2])*100:.1f}% of total variance")

# Step 8: Show first few transformed samples
print(f"\n8. Sample Transformation:")
print(f"First 3 samples in original space:")
for i in range(3):
    print(f"  Sample {i+1}: {X_scaled[i].round(3)}")

print(f"First 3 samples in PC space:")
for i in range(3):
    print(f"  Sample {i+1}: {X_pca[i].round(3)}")

print(f"\nPCA Benefits:")
print(f"  ✓ Reduced dimensionality while preserving {sum(explained_variance_ratio[:2])*100:.1f}% of variance")
print(f"  ✓ Principal components are uncorrelated")
print(f"  ✓ Enables visualization of high-dimensional data")
print(f"  ✓ Can help with noise reduction and feature extraction")
