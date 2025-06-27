"""
Example 1 - K-means Clustering
This example demonstrates K-means clustering algorithm with synthetic data
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

print("=== K-means Clustering Example ===")

# Create synthetic dataset with 4 centers
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
print(f"Dataset shape: {X.shape}")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"True number of clusters: {len(np.unique(y_true))}")

# Create and fit K-means model
k = 4
kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
kmeans.fit(X)

# Get results
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
inertia = kmeans.inertia_

print(f"\n=== K-means Results ===")
print(f"Number of clusters (K): {k}")
print(f"Initialization method: {kmeans.init}")
print(f"Maximum iterations: {kmeans.max_iter}")
print(f"Number of initializations: {kmeans.n_init}")
print(f"Final inertia (WCSS): {inertia:.3f}")
print(f"Number of iterations to converge: {kmeans.n_iter_}")

# Calculate silhouette score
silhouette_avg = silhouette_score(X, labels)
print(f"Silhouette score: {silhouette_avg:.3f}")

print(f"\n=== Cluster Centers (Centroids) ===")
for i, centroid in enumerate(centroids):
    print(f"Cluster {i}: ({centroid[0]:.3f}, {centroid[1]:.3f})")

# Analyze cluster sizes
unique_labels, cluster_sizes = np.unique(labels, return_counts=True)
print(f"\n=== Cluster Analysis ===")
for label, size in zip(unique_labels, cluster_sizes):
    print(f"Cluster {label}: {size} points ({size/len(X)*100:.1f}%)")

# Create visualization
plt.figure(figsize=(15, 5))

# Plot 1: Original data with true clusters
plt.subplot(1, 3, 1)
colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']
for i in range(len(np.unique(y_true))):
    mask = y_true == i
    plt.scatter(X[mask, 0], X[mask, 1], c=colors[i], alpha=0.6, s=50, label=f'True Cluster {i}')
plt.title('Original Data (True Clusters)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: K-means clustering results
plt.subplot(1, 3, 2)
scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6, s=50)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=300, color='red', 
           edgecolors='black', linewidths=2, label='Centroids')
plt.title(f'K-means Clustering (K={k})')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True, alpha=0.3)
plt.colorbar(scatter)

# Plot 3: Elbow method for optimal K
plt.subplot(1, 3, 3)
k_range = range(1, 11)
inertias = []
silhouette_scores = []

for k_test in k_range:
    kmeans_test = KMeans(n_clusters=k_test, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans_test.fit(X)
    inertias.append(kmeans_test.inertia_)
    
    if k_test > 1:  # Silhouette score requires at least 2 clusters
        sil_score = silhouette_score(X, kmeans_test.labels_)
        silhouette_scores.append(sil_score)
    else:
        silhouette_scores.append(0)

plt.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
plt.axvline(x=4, color='red', linestyle='--', alpha=0.7, label='Optimal K=4')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia (WCSS)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Show silhouette scores for different K values
print(f"\n=== Elbow Method Analysis ===")
print("K\tInertia\t\tSilhouette Score")
print("-" * 40)
for k_val, inert, sil in zip(k_range, inertias, silhouette_scores):
    print(f"{k_val}\t{inert:.3f}\t\t{sil:.3f}")

# Find optimal K based on elbow method (biggest drop in inertia)
inertia_diffs = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
optimal_k_elbow = inertia_diffs.index(max(inertia_diffs)) + 1
print(f"\nOptimal K by elbow method: {optimal_k_elbow}")

# Find optimal K based on silhouette score
optimal_k_silhouette = silhouette_scores.index(max(silhouette_scores[1:])) + 1  # Skip K=1
print(f"Optimal K by silhouette score: {optimal_k_silhouette}")

# Predict new points
print(f"\n=== Prediction Example ===")
new_points = np.array([[0, 0], [2, 2], [-3, -3]])
predictions = kmeans.predict(new_points)

for i, (point, pred) in enumerate(zip(new_points, predictions)):
    centroid = centroids[pred]
    distance = np.linalg.norm(point - centroid)
    print(f"Point {i+1} {point} -> Cluster {pred} (distance to centroid: {distance:.3f})")

print(f"\n=== K-means Key Concepts ===")
print("• K-means partitions data into K clusters using centroids")
print("• Algorithm minimizes within-cluster sum of squares (WCSS/inertia)")
print("• Sensitive to initialization - k-means++ helps with better starting centroids")
print("• Assumes spherical clusters of similar size")
print("• Requires predefined number of clusters (K)")
print("• Elbow method and silhouette analysis help choose optimal K")

print(f"\n=== Algorithm Steps ===")
print("1. Choose number of clusters (K)")
print("2. Initialize K centroids randomly or using k-means++")
print("3. Assign each point to nearest centroid")
print("4. Update centroids as mean of assigned points")
print("5. Repeat steps 3-4 until convergence")
print("6. Final clusters and centroids represent the solution")
