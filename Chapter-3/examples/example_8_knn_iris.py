"""
Example 8 - K-Nearest Neighbors (kNN) Algorithm on Iris Dataset
This example demonstrates kNN for finding similar samples and understanding the algorithm
"""

from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import numpy as np

print("=== K-Nearest Neighbors (kNN) Algorithm Example ===")

# Load iris dataset
iris = load_iris()
features = iris.data
target = iris.target

print(f"Dataset shape: features={features.shape}, target={target.shape}")
print(f"Feature names: {iris.feature_names}")
print(f"Classes: {iris.target_names}")
print(f"Class distribution: {np.bincount(target)}")

# Standardize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
print(f"\nFeatures standardized for distance calculation")

# Create kNN model
k = 5
model = NearestNeighbors(n_neighbors=k)
model.fit(scaled_features)

print(f"\n=== kNN Model Information ===")
print(f"Number of neighbors (k): {k}")
print(f"Distance metric: Euclidean (default)")

# Test with a new sample
xnew = np.array([5.0, 4.0, 1.5, 0.5]).reshape(1, -1)
scaled_xnew = scaler.transform(xnew)

print(f"\n=== Finding Neighbors for New Sample ===")
print(f"New sample: {xnew[0]}")

# Find k nearest neighbors
distances, indices = model.kneighbors(scaled_xnew)

print(f"\nNearest neighbors (indices): {indices[0]}")
print(f"Distances: {distances[0]}")

print(f"\n=== Neighbor Details ===")
print("Rank\tIndex\tDistance\tFeatures\t\t\t\tClass")
print("-" * 80)
for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
    neighbor_features = features[idx]
    neighbor_class = iris.target_names[target[idx]]
    print(f"{i+1}\t{idx}\t{dist:.3f}\t\t{neighbor_features}\t{neighbor_class}")

# Predict class based on majority vote
neighbor_classes = target[indices[0]]
predicted_class = np.bincount(neighbor_classes).argmax()
print(f"\n=== Prediction ===")
print(f"Neighbor classes: {neighbor_classes}")
print(f"Class counts: {np.bincount(neighbor_classes)}")
print(f"Predicted class: {predicted_class} ({iris.target_names[predicted_class]})")

# Additional analysis: Try another sample
print(f"\n" + "="*60)
print("=== Testing with Another Sample ===")
xnew2 = np.array([6.9, 3.1, 6.0, 2.1]).reshape(1, -1)
scaled_xnew2 = scaler.transform(xnew2)

print(f"New sample: {xnew2[0]}")

distances2, indices2 = model.kneighbors(scaled_xnew2)
neighbor_classes2 = target[indices2[0]]
predicted_class2 = np.bincount(neighbor_classes2).argmax()

print(f"Neighbor classes: {neighbor_classes2}")
print(f"Predicted class: {predicted_class2} ({iris.target_names[predicted_class2]})")

print(f"\n=== Key kNN Concepts ===")
print("• kNN is a lazy learning algorithm (no training phase)")
print("• Prediction based on majority vote of k nearest neighbors")
print("• Distance metric (usually Euclidean) determines similarity")
print("• k should be odd to avoid ties in binary classification")
print("• Feature scaling is important for fair distance calculation")
print("• Performance depends on choice of k and distance metric")
