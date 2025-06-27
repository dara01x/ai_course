"""
Example 3 - DBSCAN Clustering
This example demonstrates DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

print("=== DBSCAN Clustering Example ===")

# Create synthetic dataset with some noise
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.5, random_state=0)

# Add some noise points
np.random.seed(0)
noise_points = np.random.uniform(X.min() - 1, X.max() + 1, (20, 2))
X_with_noise = np.vstack([X, noise_points])
y_true_with_noise = np.hstack([y_true, [-1] * 20])  # -1 for noise points

print(f"Dataset shape: {X_with_noise.shape}")
print(f"Number of samples: {X_with_noise.shape[0]}")
print(f"Number of noise points added: 20")
print(f"True number of clusters: {len(np.unique(y_true))}")

# DBSCAN parameters
eps = 0.3
min_samples = 10

print(f"\n=== DBSCAN Parameters ===")
print(f"eps (epsilon): {eps}")
print(f"min_samples: {min_samples}")

# Create and fit DBSCAN model
db = DBSCAN(eps=eps, min_samples=min_samples)
db.fit(X_with_noise)

# Get results
labels = db.labels_
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"\n=== DBSCAN Results ===")
print(f"Estimated number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")
print(f"Percentage of noise points: {n_noise/len(X_with_noise)*100:.1f}%")

# Calculate silhouette score (excluding noise points)
if n_clusters > 1:
    # Only calculate silhouette for non-noise points
    non_noise_mask = labels != -1
    if np.sum(non_noise_mask) > 0:
        silhouette_avg = silhouette_score(X_with_noise[non_noise_mask], labels[non_noise_mask])
        print(f"Silhouette score (excluding noise): {silhouette_avg:.3f}")
    else:
        print("Silhouette score: Not applicable (no non-noise points)")
else:
    print("Silhouette score: Not applicable (less than 2 clusters)")

# Analyze cluster composition
unique_labels = set(labels)
print(f"\n=== Cluster Analysis ===")
for label in sorted(unique_labels):
    if label == -1:
        print(f"Noise points: {list(labels).count(label)} points")
    else:
        cluster_size = list(labels).count(label)
        print(f"Cluster {label}: {cluster_size} points ({cluster_size/len(X_with_noise)*100:.1f}%)")

# Categorize points
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

print(f"\n=== Point Categories ===")
print(f"Core points: {len(db.core_sample_indices_)}")
print(f"Border points: {np.sum((labels != -1) & (~core_samples_mask))}")
print(f"Noise points: {n_noise}")

# Parameter sensitivity analysis
print(f"\n=== Parameter Sensitivity Analysis ===")
eps_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
min_samples_range = [5, 10, 15, 20]

print("eps\tmin_samples\tClusters\tNoise\tSilhouette")
print("-" * 50)

param_results = []
for eps_test in eps_range:
    for min_samples_test in min_samples_range:
        db_test = DBSCAN(eps=eps_test, min_samples=min_samples_test)
        labels_test = db_test.fit_predict(X_with_noise)
        
        n_clusters_test = len(set(labels_test)) - (1 if -1 in labels_test else 0)
        n_noise_test = list(labels_test).count(-1)
        
        # Calculate silhouette score
        if n_clusters_test > 1:
            non_noise_mask_test = labels_test != -1
            if np.sum(non_noise_mask_test) > 0:
                sil_test = silhouette_score(X_with_noise[non_noise_mask_test], labels_test[non_noise_mask_test])
            else:
                sil_test = 0.0
        else:
            sil_test = 0.0
        
        param_results.append((eps_test, min_samples_test, n_clusters_test, n_noise_test, sil_test))
        print(f"{eps_test:.1f}\t{min_samples_test}\t\t{n_clusters_test}\t\t{n_noise_test}\t{sil_test:.3f}")

# Create comprehensive visualization
plt.figure(figsize=(20, 15))

# Plot 1: Original data with true clusters
plt.subplot(3, 4, 1)
colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']
for i in range(len(np.unique(y_true))):
    mask = y_true == i
    plt.scatter(X[mask, 0], X[mask, 1], c=colors[i], alpha=0.6, s=50, label=f'True Cluster {i}')
plt.scatter(noise_points[:, 0], noise_points[:, 1], c='black', marker='x', s=50, label='Added Noise')
plt.title('Original Data with Added Noise')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: DBSCAN clustering results
plt.subplot(3, 4, 2)
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise
        col = [0, 0, 0, 1]
    
    class_member_mask = (labels == k)
    xy = X_with_noise[class_member_mask]
    
    if k == -1:
        plt.scatter(xy[:, 0], xy[:, 1], c='black', marker='x', s=50, label='Noise', alpha=0.6)
    else:
        plt.scatter(xy[:, 0], xy[:, 1], c=[col], s=50, alpha=0.6, label=f'Cluster {k}')

plt.title(f'DBSCAN Clustering\n(eps={eps}, min_samples={min_samples})')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Core, border, and noise points
plt.subplot(3, 4, 3)
# Core points
core_points = X_with_noise[core_samples_mask]
plt.scatter(core_points[:, 0], core_points[:, 1], c='red', s=50, alpha=0.6, label='Core Points')

# Border points (non-noise, non-core)
border_mask = (labels != -1) & (~core_samples_mask)
border_points = X_with_noise[border_mask]
plt.scatter(border_points[:, 0], border_points[:, 1], c='blue', s=50, alpha=0.6, label='Border Points')

# Noise points
noise_mask = labels == -1
noise_points_found = X_with_noise[noise_mask]
plt.scatter(noise_points_found[:, 0], noise_points_found[:, 1], c='black', marker='x', s=50, label='Noise Points')

plt.title('Point Categories in DBSCAN')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Parameter sensitivity heatmap (eps vs clusters)
plt.subplot(3, 4, 4)
eps_vals = sorted(set([r[0] for r in param_results]))
min_samples_vals = sorted(set([r[1] for r in param_results]))
cluster_matrix = np.zeros((len(min_samples_vals), len(eps_vals)))

for eps_test, min_samples_test, n_clusters_test, _, _ in param_results:
    i = min_samples_vals.index(min_samples_test)
    j = eps_vals.index(eps_test)
    cluster_matrix[i, j] = n_clusters_test

im = plt.imshow(cluster_matrix, cmap='viridis', aspect='auto')
plt.colorbar(im, label='Number of Clusters')
plt.xticks(range(len(eps_vals)), eps_vals)
plt.yticks(range(len(min_samples_vals)), min_samples_vals)
plt.xlabel('eps')
plt.ylabel('min_samples')
plt.title('Parameter Sensitivity:\nNumber of Clusters')

# Plots 5-8: Different parameter combinations
example_params = [(0.2, 5), (0.3, 10), (0.4, 15), (0.5, 20)]
for i, (eps_ex, min_samples_ex) in enumerate(example_params):
    plt.subplot(3, 4, 5 + i)
    
    db_ex = DBSCAN(eps=eps_ex, min_samples=min_samples_ex)
    labels_ex = db_ex.fit_predict(X_with_noise)
    
    unique_labels_ex = set(labels_ex)
    colors_ex = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels_ex)))
    
    for k, col in zip(unique_labels_ex, colors_ex):
        if k == -1:
            col = [0, 0, 0, 1]
        
        class_member_mask = (labels_ex == k)
        xy = X_with_noise[class_member_mask]
        
        if k == -1:
            plt.scatter(xy[:, 0], xy[:, 1], c='black', marker='x', s=30, alpha=0.6)
        else:
            plt.scatter(xy[:, 0], xy[:, 1], c=[col], s=30, alpha=0.6)
    
    n_clusters_ex = len(set(labels_ex)) - (1 if -1 in labels_ex else 0)
    n_noise_ex = list(labels_ex).count(-1)
    plt.title(f'eps={eps_ex}, min_samples={min_samples_ex}\n{n_clusters_ex} clusters, {n_noise_ex} noise')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True, alpha=0.3)

# Plots 9-12: Silhouette score analysis
plt.subplot(3, 4, 9)
eps_for_plot = [r[0] for r in param_results if r[1] == 10]  # min_samples=10
sil_for_plot = [r[4] for r in param_results if r[1] == 10]
plt.plot(eps_for_plot, sil_for_plot, 'bo-', linewidth=2, markersize=8)
plt.axvline(x=eps, color='red', linestyle='--', alpha=0.7, label=f'Used eps: {eps}')
plt.title('eps vs Silhouette Score\n(min_samples=10)')
plt.xlabel('eps')
plt.ylabel('Silhouette Score')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Demonstrate noise detection capability
print(f"\n=== Noise Detection Analysis ===")
# Check how many of the artificially added noise points were correctly identified
noise_mask = labels == -1
noise_points_detected = X_with_noise[noise_mask]

# Calculate how many of the last 20 points (our added noise) were detected as noise
original_data_size = len(X)
artificial_noise_detected = np.sum(labels[original_data_size:] == -1)
print(f"Artificial noise points correctly detected: {artificial_noise_detected}/20 ({artificial_noise_detected/20*100:.1f}%)")

# Check how many original points were misclassified as noise
original_points_as_noise = np.sum(labels[:original_data_size] == -1)
print(f"Original points misclassified as noise: {original_points_as_noise}/{original_data_size} ({original_points_as_noise/original_data_size*100:.1f}%)")

print(f"\n=== DBSCAN Key Concepts ===")
print("• DBSCAN groups points that are closely packed together")
print("• Points in low-density regions are marked as outliers/noise")
print("• Does not require specifying the number of clusters beforehand")
print("• Can find arbitrarily shaped clusters")
print("• Robust to outliers and noise")
print("• Two key parameters: eps (neighborhood radius) and min_samples")

print(f"\n=== Point Categories ===")
print("• Core points: Have at least min_samples points within eps distance")
print("• Border points: Within eps distance of a core point but not core themselves")
print("• Noise points: Neither core nor border points")

print(f"\n=== Algorithm Steps ===")
print("1. For each point, find all points within eps distance")
print("2. If a point has >= min_samples neighbors, mark it as core point")
print("3. For each core point, create a cluster with all its neighbors")
print("4. Merge clusters that share core points")
print("5. Mark non-core points not in any cluster as noise")

print(f"\n=== Parameter Selection Guidelines ===")
print("• eps: Use k-distance graph or domain knowledge")
print("• min_samples: Typically 2*dimensions or higher")
print("• Smaller eps: More clusters, more noise points")
print("• Larger min_samples: Fewer clusters, more noise points")

print(f"\n=== Advantages and Disadvantages ===")
print("Advantages:")
print("  • No need to specify number of clusters")
print("  • Can find arbitrarily shaped clusters")
print("  • Robust to outliers (marks them as noise)")
print("  • Works well with spatial data")
print("\nDisadvantages:")
print("  • Sensitive to eps and min_samples parameters")
print("  • Struggles with varying densities")
print("  • Can be computationally expensive for large datasets")
print("  • Performance degrades in high-dimensional spaces")
