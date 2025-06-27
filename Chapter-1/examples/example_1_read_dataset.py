"""
Example 1 - Reading Dataset
This example shows how to read various datasets from sklearn

NOTE: This example has been modified from the original lecture version.
CHANGE: Removed 'load_boston' from the imports
REASON: load_boston was deprecated and removed from scikit-learn 1.2+ due to ethical concerns
        about the dataset's racial bias. The function now raises an ImportError.
"""

from sklearn.datasets import (
    load_breast_cancer, load_diabetes, 
    load_digits, load_iris, fetch_california_housing
)

# Load Iris dataset (example of supervised dataset)
iris_data = load_iris()

print("=== Iris Dataset ===")
print(f"Dataset shape: {iris_data.data.shape}")
print(f"Target shape: {iris_data.target.shape}")
print(f"Feature names: {iris_data.feature_names}")
print(f"Target names: {iris_data.target_names}")
print(f"Description: {iris_data.DESCR[:200]}...")

# Display first 5 samples
print("\nFirst 5 samples:")
print("Features:")
print(iris_data.data[:5])
print("Targets:")
print(iris_data.target[:5])
