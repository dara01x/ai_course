"""
Example 2 - Mean Shift Clustering
This example demonstrates Mean Shift clustering algorithm for automatic cluster detection
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

print("=== Mean Shift Clustering Example ===")

# Create synthetic dataset
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.7, random_state=0)
print(f"Dataset shape: {X.shape}")
print(f"Number of samples: {X.shape[0]}")
print(f"True number of clusters: {len(np.unique(y_true))}")

# Estimate bandwidth automatically
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
print(f"\n=== Bandwidth Estimation ===")
print(f"Estimated bandwidth: {bandwidth:.3f}")
print(f"Quantile used for estimation: 0.2")

# Create and fit Mean Shift model
ms = MeanShift(bandwidth=bandwidth)
ms.fit(X)

# Get results
labels = ms.labels_
cluster_centers = ms.cluster_centers_
n_clusters = len(np.unique(labels))

print(f"\n=== Mean Shift Results ===")
print(f"Number of estimated clusters: {n_clusters}")
print(f"Number of cluster centers found: {len(cluster_centers)}")

# Calculate silhouette score
if n_clusters > 1:
    silhouette_avg = silhouette_score(X, labels)
    print(f"Silhouette score: {silhouette_avg:.3f}")
else:
    print("Silhouette score: Not applicable (only 1 cluster)")

print(f"\n=== Cluster Centers ===")
for i, center in enumerate(cluster_centers):
    print(f"Cluster {i}: ({center[0]:.3f}, {center[1]:.3f})")

# Analyze cluster sizes
unique_labels, cluster_sizes = np.unique(labels, return_counts=True)
print(f"\n=== Cluster Analysis ===")
for label, size in zip(unique_labels, cluster_sizes):
    print(f"Cluster {label}: {size} points ({size/len(X)*100:.1f}%)")

# Test different bandwidth values
print(f"\n=== Bandwidth Sensitivity Analysis ===")
bandwidth_range = np.linspace(0.5, 3.0, 8)
bandwidth_results = []

print("Bandwidth\tClusters\tSilhouette")
print("-" * 35)

for bw in bandwidth_range:
    ms_test = MeanShift(bandwidth=bw)
    try:
        ms_test.fit(X)
        n_clust = len(np.unique(ms_test.labels_))
        if n_clust > 1:
            sil_score = silhouette_score(X, ms_test.labels_)
        else:
            sil_score = 0.0
        bandwidth_results.append((bw, n_clust, sil_score))
        print(f"{bw:.2f}\t\t{n_clust}\t\t{sil_score:.3f}")
    except:
        print(f"{bw:.2f}\t\tError\t\tN/A")

# Create comprehensive visualization
plt.figure(figsize=(20, 12))

# Plot 1: Original data with true clusters
plt.subplot(2, 4, 1)
colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']
for i in range(len(np.unique(y_true))):
    mask = y_true == i
    plt.scatter(X[mask, 0], X[mask, 1], c=colors[i], alpha=0.6, s=50, label=f'True Cluster {i}')
plt.title('Original Data (True Clusters)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Mean Shift clustering results
plt.subplot(2, 4, 2)
scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6, s=50)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='*', s=300, color='red',
           edgecolors='black', linewidths=2, label='Cluster Centers')
plt.title(f'Mean Shift Clustering\n({n_clusters} clusters)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True, alpha=0.3)
plt.colorbar(scatter)

# Plot 3: Bandwidth sensitivity - number of clusters
plt.subplot(2, 4, 3)
bw_vals, n_clusters_vals, sil_vals = zip(*bandwidth_results) if bandwidth_results else ([], [], [])
plt.plot(bw_vals, n_clusters_vals, 'bo-', linewidth=2, markersize=8)
plt.axvline(x=bandwidth, color='red', linestyle='--', alpha=0.7, label=f'Used BW: {bandwidth:.2f}')
plt.title('Bandwidth vs Number of Clusters')
plt.xlabel('Bandwidth')
plt.ylabel('Number of Clusters')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Bandwidth sensitivity - silhouette score
plt.subplot(2, 4, 4)
plt.plot(bw_vals, sil_vals, 'go-', linewidth=2, markersize=8)
plt.axvline(x=bandwidth, color='red', linestyle='--', alpha=0.7, label=f'Used BW: {bandwidth:.2f}')
plt.title('Bandwidth vs Silhouette Score')
plt.xlabel('Bandwidth')
plt.ylabel('Silhouette Score')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 5-8: Different bandwidth examples
example_bandwidths = [0.8, 1.5, 2.5, 3.5]
for i, test_bw in enumerate(example_bandwidths):
    plt.subplot(2, 4, 5 + i)
    try:
        ms_example = MeanShift(bandwidth=test_bw)
        ms_example.fit(X)
        example_labels = ms_example.labels_
        example_centers = ms_example.cluster_centers_
        n_example_clusters = len(np.unique(example_labels))
        
        plt.scatter(X[:, 0], X[:, 1], c=example_labels, cmap='viridis', alpha=0.6, s=30)
        plt.scatter(example_centers[:, 0], example_centers[:, 1], marker='*', s=200, 
                   color='red', edgecolors='black', linewidths=1)
        plt.title(f'BW={test_bw:.1f} ({n_example_clusters} clusters)')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.grid(True, alpha=0.3)
    except:
        plt.text(0.5, 0.5, f'BW={test_bw:.1f}\nError', transform=plt.gca().transAxes,
                ha='center', va='center', fontsize=12)
        plt.title(f'BW={test_bw:.1f} (Error)')

plt.tight_layout()
plt.show()

# Demonstrate automatic cluster number detection
print(f"\n=== Comparison with K-means ===")
from sklearn.cluster import KMeans

# K-means with true number of clusters
kmeans_true = KMeans(n_clusters=len(np.unique(y_true)), random_state=0)
kmeans_labels_true = kmeans_true.fit_predict(X)
kmeans_sil_true = silhouette_score(X, kmeans_labels_true)

# K-means with Mean Shift detected number of clusters
if n_clusters > 1:
    kmeans_detected = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans_labels_detected = kmeans_detected.fit_predict(X)
    kmeans_sil_detected = silhouette_score(X, kmeans_labels_detected)
else:
    kmeans_sil_detected = 0.0

print(f"True clusters: {len(np.unique(y_true))}")
print(f"Mean Shift detected: {n_clusters}")
print(f"K-means (true K) silhouette: {kmeans_sil_true:.3f}")
print(f"K-means (detected K) silhouette: {kmeans_sil_detected:.3f}")
if n_clusters > 1:
    print(f"Mean Shift silhouette: {silhouette_avg:.3f}")

# Demonstrate mode-seeking behavior
print(f"\n=== Mode-Seeking Demonstration ===")
# Show how points converge to modes
sample_points = X[:5]  # Take first 5 points
print("Sample points and their cluster assignments:")
for i, (point, label) in enumerate(zip(sample_points, labels[:5])):
    center = cluster_centers[label]
    distance = np.linalg.norm(point - center)
    print(f"Point {i+1}: {point} -> Cluster {label}, Distance to center: {distance:.3f}")

print(f"\n=== Mean Shift Key Concepts ===")
print("• Mean Shift finds modes (peaks) in the density distribution")
print("• Automatically determines the number of clusters")
print("• Can handle arbitrarily shaped clusters")
print("• Robust to outliers")
print("• Bandwidth parameter controls the size of the search window")
print("• Points converge to the same mode belong to the same cluster")

print(f"\n=== Algorithm Steps ===")
print("1. Place a window around each data point")
print("2. Calculate the mean of points within the window")
print("3. Shift the window center to the calculated mean")
print("4. Repeat until convergence (window stops moving)")
print("5. Points that converge to the same location form a cluster")
print("6. The convergence points become the cluster centers")

print(f"\n=== Advantages and Disadvantages ===")
print("Advantages:")
print("  • No need to specify number of clusters")
print("  • Can find arbitrarily shaped clusters")
print("  • Robust to outliers")
print("  • Theoretically well-founded")
print("\nDisadvantages:")
print("  • Sensitive to bandwidth parameter")
print("  • Computationally expensive for large datasets")
print("  • Performance degrades in high-dimensional spaces")
