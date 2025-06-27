"""
Example 13 - SVM Hyperparameter Tuning using Grid Search
This example demonstrates hyperparameter optimization for SVM using grid search
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np

print("=== SVM Hyperparameter Tuning using Grid Search ===")

# Load iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

print(f"Dataset shape: X={X.shape}, y={y.shape}")
print(f"Classes: {iris.target_names}")

# Standardize features (important for SVM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
xtrain, xtest, ytrain, ytest = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(f"Training set: {xtrain.shape}")
print(f"Testing set: {xtest.shape}")

# Define parameter grid for SVM
param_grid = {
    'C': [0.1, 1, 10, 100], 
    'kernel': ['linear', 'rbf', 'poly'], 
    'gamma': ['scale', 'auto', 0.1, 1, 10]
}

print(f"\n=== Parameter Grid ===")
print(f"C values: {param_grid['C']}")
print(f"Kernels: {param_grid['kernel']}")
print(f"Gamma values: {param_grid['gamma']}")
print(f"Total combinations: {len(param_grid['C']) * len(param_grid['kernel']) * len(param_grid['gamma'])}")

# Create SVM model
svm_model = SVC()

# Create GridSearchCV object
print(f"\n=== Running Grid Search ===")
print("This may take a moment...")
grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=5, 
                          scoring='accuracy', n_jobs=-1, verbose=1)

# Fit grid search
grid_search.fit(xtrain, ytrain)

# Get best parameters and score
best_params = grid_search.best_params_
best_cv_score = grid_search.best_score_
best_model = grid_search.best_estimator_

print(f"\n=== Grid Search Results ===")
print(f"Best parameters: {best_params}")
print(f"Best cross-validation score: {best_cv_score:.4f} ({best_cv_score*100:.2f}%)")

# Test the best model
test_accuracy = best_model.score(xtest, ytest)
print(f"Test accuracy with best parameters: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Detailed results analysis
results_df = grid_search.cv_results_
print(f"\n=== Top 10 Parameter Combinations ===")
print("Rank\tMean CV Score\tStd\tParameters")
print("-" * 80)

# Sort by mean test score
sorted_indices = np.argsort(results_df['mean_test_score'])[::-1]
for i in range(min(10, len(sorted_indices))):
    idx = sorted_indices[i]
    mean_score = results_df['mean_test_score'][idx]
    std_score = results_df['std_test_score'][idx]
    params = results_df['params'][idx]
    print(f"{i+1}\t{mean_score:.4f}\t\t{std_score:.4f}\t{params}")

# Analyze parameter importance
print(f"\n=== Parameter Analysis ===")

# Analyze C parameter
print("C parameter performance:")
c_scores = {}
for c_val in param_grid['C']:
    scores = [results_df['mean_test_score'][i] for i, params in enumerate(results_df['params']) 
              if params['C'] == c_val]
    c_scores[c_val] = np.mean(scores) if scores else 0
    print(f"C={c_val}: Average score = {c_scores[c_val]:.4f}")

best_c = max(c_scores, key=c_scores.get)
print(f"Best C value: {best_c}")

# Analyze kernel performance
print(f"\nKernel performance:")
kernel_scores = {}
for kernel in param_grid['kernel']:
    scores = [results_df['mean_test_score'][i] for i, params in enumerate(results_df['params']) 
              if params['kernel'] == kernel]
    kernel_scores[kernel] = np.mean(scores) if scores else 0
    print(f"Kernel={kernel}: Average score = {kernel_scores[kernel]:.4f}")

best_kernel = max(kernel_scores, key=kernel_scores.get)
print(f"Best kernel: {best_kernel}")

# Compare with default SVM
print(f"\n=== Comparison with Default SVM ===")
default_svm = SVC()
default_svm.fit(xtrain, ytrain)
default_accuracy = default_svm.score(xtest, ytest)

print(f"Default SVM accuracy: {default_accuracy:.4f} ({default_accuracy*100:.2f}%)")
print(f"Optimized SVM accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Improvement: {test_accuracy - default_accuracy:.4f} ({(test_accuracy - default_accuracy)*100:.2f}%)")

# Model complexity analysis
print(f"\n=== Best Model Analysis ===")
print(f"Number of support vectors: {best_model.n_support_}")
print(f"Total support vectors: {np.sum(best_model.n_support_)}")
print(f"Support vector ratio: {np.sum(best_model.n_support_) / len(xtrain):.3f}")

print(f"\n=== Key Hyperparameter Tuning Insights ===")
print("• Grid search exhaustively tests all parameter combinations")
print("• Cross-validation provides robust performance estimates")
print("• Different kernels work better for different data patterns")
print("• C parameter controls trade-off between margin and misclassification")
print("• Gamma parameter affects the influence of single training examples")
print("• Hyperparameter tuning can significantly improve model performance")
