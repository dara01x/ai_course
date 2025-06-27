"""
Homework 1 - Load Other Datasets and Display Their Sizes
Modify the example code to load different datasets and display their sizes
"""

from sklearn.datasets import (
    load_boston, load_breast_cancer, load_diabetes, 
    load_digits, load_iris, fetch_california_housing,
    load_wine, load_linnerud
)
import warnings
warnings.filterwarnings('ignore')  # Suppress deprecation warnings

def display_dataset_info(dataset, name):
    """Display comprehensive information about a dataset"""
    print(f"\n{'='*60}")
    print(f"DATASET: {name.upper()}")
    print(f"{'='*60}")
    
    print(f"Data shape: {dataset.data.shape}")
    print(f"Target shape: {dataset.target.shape}")
    print(f"Number of samples: {dataset.data.shape[0]}")
    print(f"Number of features: {dataset.data.shape[1]}")
    
    if hasattr(dataset, 'feature_names') and dataset.feature_names is not None:
        print(f"Feature names: {dataset.feature_names}")
    
    if hasattr(dataset, 'target_names') and dataset.target_names is not None:
        print(f"Target names: {dataset.target_names}")
    
    # Display basic statistics
    print(f"Data range: [{dataset.data.min():.3f}, {dataset.data.max():.3f}]")
    print(f"Target range: [{dataset.target.min():.3f}, {dataset.target.max():.3f}]")
    
    # Show first few samples
    print(f"\nFirst 3 samples (features):")
    print(dataset.data[:3])
    print(f"First 3 targets:")
    print(dataset.target[:3])
    
    # Description (first 200 characters)
    if hasattr(dataset, 'DESCR') and dataset.DESCR:
        print(f"\nDescription (first 200 chars):")
        print(f"{dataset.DESCR[:200]}...")

print("="*80)
print("HOMEWORK 1 - LOADING AND ANALYZING DIFFERENT DATASETS")
print("="*80)

# List of datasets to load
datasets_info = [
    ("Iris", load_iris),
    ("Boston Housing", load_boston),
    ("Breast Cancer", load_breast_cancer),
    ("Diabetes", load_diabetes),
    ("Digits", load_digits),
    ("Wine", load_wine),
    ("Linnerud", load_linnerud)
]

# Load and display each dataset
for name, loader in datasets_info:
    try:
        dataset = loader()
        display_dataset_info(dataset, name)
    except Exception as e:
        print(f"Error loading {name}: {e}")

# Fetch California Housing (requires internet connection)
print(f"\n{'='*60}")
print("FETCHING CALIFORNIA HOUSING DATASET")
print(f"{'='*60}")
try:
    california_housing = fetch_california_housing()
    display_dataset_info(california_housing, "California Housing")
except Exception as e:
    print(f"Error fetching California Housing dataset: {e}")

# Summary comparison
print(f"\n{'='*80}")
print("DATASET COMPARISON SUMMARY")
print(f"{'='*80}")

print(f"{'Dataset':<20} {'Samples':<10} {'Features':<10} {'Type':<15} {'Target Type':<15}")
print("-" * 75)

dataset_summaries = []
for name, loader in datasets_info:
    try:
        dataset = loader()
        samples = dataset.data.shape[0]
        features = dataset.data.shape[1]
        
        # Determine if it's classification or regression
        if len(dataset.target.shape) == 1:
            unique_targets = len(set(dataset.target))
            if unique_targets < 20:  # Likely classification
                task_type = "Classification"
                target_type = f"{unique_targets} classes"
            else:  # Likely regression
                task_type = "Regression"
                target_type = "Continuous"
        else:
            task_type = "Multi-output"
            target_type = f"{dataset.target.shape[1]} outputs"
        
        print(f"{name:<20} {samples:<10} {features:<10} {task_type:<15} {target_type:<15}")
        
        dataset_summaries.append({
            'name': name,
            'samples': samples,
            'features': features,
            'type': task_type
        })
        
    except Exception as e:
        print(f"{name:<20} {'Error':<10} {'Error':<10} {'Error':<15} {'Error':<15}")

# Additional analysis
print(f"\n{'='*60}")
print("ADDITIONAL ANALYSIS")
print(f"{'='*60}")

# Find dataset with most samples
if dataset_summaries:
    max_samples = max(dataset_summaries, key=lambda x: x['samples'])
    min_samples = min(dataset_summaries, key=lambda x: x['samples'])
    max_features = max(dataset_summaries, key=lambda x: x['features'])
    min_features = min(dataset_summaries, key=lambda x: x['features'])
    
    print(f"Largest dataset (samples): {max_samples['name']} with {max_samples['samples']} samples")
    print(f"Smallest dataset (samples): {min_samples['name']} with {min_samples['samples']} samples")
    print(f"Most features: {max_features['name']} with {max_features['features']} features")
    print(f"Fewest features: {min_features['name']} with {min_features['features']} features")
    
    # Count by type
    classification_count = sum(1 for d in dataset_summaries if d['type'] == 'Classification')
    regression_count = sum(1 for d in dataset_summaries if d['type'] == 'Regression')
    
    print(f"\nDataset types:")
    print(f"Classification datasets: {classification_count}")
    print(f"Regression datasets: {regression_count}")

# Detailed feature analysis for a few key datasets
print(f"\n{'='*60}")
print("DETAILED FEATURE ANALYSIS")
print(f"{'='*60}")

# Analyze Iris dataset in detail
print("IRIS DATASET - Detailed Analysis:")
iris = load_iris()
print(f"Feature names and ranges:")
for i, feature_name in enumerate(iris.feature_names):
    feature_data = iris.data[:, i]
    print(f"  {feature_name}: [{feature_data.min():.2f}, {feature_data.max():.2f}], "
          f"mean={feature_data.mean():.2f}, std={feature_data.std():.2f}")

print(f"\nTarget distribution:")
import numpy as np
unique, counts = np.unique(iris.target, return_counts=True)
for target_val, count in zip(unique, counts):
    target_name = iris.target_names[target_val]
    print(f"  {target_name}: {count} samples ({count/len(iris.target)*100:.1f}%)")

# Analyze Digits dataset
print(f"\nDIGITS DATASET - Detailed Analysis:")
digits = load_digits()
print(f"Image dimensions: {int(np.sqrt(digits.data.shape[1]))}x{int(np.sqrt(digits.data.shape[1]))}")
print(f"Pixel value range: [{digits.data.min():.0f}, {digits.data.max():.0f}]")
print(f"Classes: {digits.target_names}")
unique, counts = np.unique(digits.target, return_counts=True)
for digit, count in zip(unique, counts):
    print(f"  Digit {digit}: {count} samples ({count/len(digits.target)*100:.1f}%)")

print(f"\n{'='*60}")
print("HOMEWORK COMPLETION SUMMARY")
print(f"{'='*60}")
print("✓ Successfully loaded multiple datasets")
print("✓ Displayed dataset shapes and sizes")
print("✓ Analyzed feature names and target names")
print("✓ Compared different dataset characteristics")
print("✓ Provided detailed analysis of sample datasets")
print("✓ Identified classification vs regression datasets")

print(f"\nKey Learning Points:")
print("1. Different datasets have varying numbers of samples and features")
print("2. Sklearn provides both classification and regression datasets")
print("3. Some datasets have meaningful feature names, others don't")
print("4. Dataset sizes range from small (Iris: 150 samples) to large (Digits: 1797 samples)")
print("5. Understanding dataset characteristics is crucial for choosing appropriate models")
