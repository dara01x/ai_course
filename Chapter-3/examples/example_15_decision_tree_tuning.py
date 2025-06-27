"""
Example 15 - Decision Tree Hyperparameter Tuning
This example demonstrates hyperparameter optimization for Decision Tree using grid search
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

print("=== Decision Tree Hyperparameter Tuning ===")

# Load iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

print(f"Dataset shape: X={X.shape}, y={y.shape}")
print(f"Classes: {iris.target_names}")

# Split dataset
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set: {xtrain.shape}")
print(f"Testing set: {xtest.shape}")

# Define parameter grid for Decision Tree
param_grid = {
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'criterion': ['gini', 'entropy']
}

print(f"\n=== Parameter Grid ===")
for param, values in param_grid.items():
    print(f"{param}: {values}")

total_combinations = 1
for param, values in param_grid.items():
    total_combinations *= len(values)
print(f"Total combinations: {total_combinations}")

# Create Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)

# Create GridSearchCV object
print(f"\n=== Running Grid Search ===")
grid_search = GridSearchCV(estimator=dt_model, param_grid=param_grid, cv=5, 
                          scoring='accuracy', n_jobs=-1)

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

# Compare with default Decision Tree
print(f"\n=== Comparison with Default Decision Tree ===")
default_dt = DecisionTreeClassifier(random_state=42)
default_dt.fit(xtrain, ytrain)
default_accuracy = default_dt.score(xtest, ytest)

print(f"Default Decision Tree accuracy: {default_accuracy:.4f} ({default_accuracy*100:.2f}%)")
print(f"Optimized Decision Tree accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Improvement: {test_accuracy - default_accuracy:.4f} ({(test_accuracy - default_accuracy)*100:.2f}%)")

# Detailed model analysis
print(f"\n=== Best Model Analysis ===")
print(f"Tree depth: {best_model.get_depth()}")
print(f"Number of leaves: {best_model.get_n_leaves()}")
print(f"Number of nodes: {best_model.tree_.node_count}")

print(f"\nDefault Model Analysis:")
print(f"Tree depth: {default_dt.get_depth()}")
print(f"Number of leaves: {default_dt.get_n_leaves()}")
print(f"Number of nodes: {default_dt.tree_.node_count}")

# Feature importance comparison
print(f"\n=== Feature Importance Comparison ===")
print("Feature\t\t\tDefault\tOptimized")
print("-" * 45)
for i, name in enumerate(iris.feature_names):
    default_imp = default_dt.feature_importances_[i]
    best_imp = best_model.feature_importances_[i]
    print(f"{name:<20}\t{default_imp:.3f}\t{best_imp:.3f}")

# Top parameter combinations
results_df = grid_search.cv_results_
print(f"\n=== Top 10 Parameter Combinations ===")
print("Rank\tCV Score\tStd\tParameters")
print("-" * 80)

sorted_indices = np.argsort(results_df['mean_test_score'])[::-1]
for i in range(min(10, len(sorted_indices))):
    idx = sorted_indices[i]
    mean_score = results_df['mean_test_score'][idx]
    std_score = results_df['std_test_score'][idx]
    params = results_df['params'][idx]
    print(f"{i+1}\t{mean_score:.4f}\t\t{std_score:.4f}\t{params}")

# Parameter analysis
print(f"\n=== Parameter Impact Analysis ===")

# Analyze max_depth
print("Max depth performance:")
depth_scores = {}
for depth in param_grid['max_depth']:
    scores = [results_df['mean_test_score'][i] for i, params in enumerate(results_df['params']) 
              if params['max_depth'] == depth]
    depth_scores[depth] = np.mean(scores) if scores else 0
    print(f"max_depth={depth}: Average score = {depth_scores[depth]:.4f}")

# Analyze criterion
print(f"\nCriterion performance:")
criterion_scores = {}
for criterion in param_grid['criterion']:
    scores = [results_df['mean_test_score'][i] for i, params in enumerate(results_df['params']) 
              if params['criterion'] == criterion]
    criterion_scores[criterion] = np.mean(scores) if scores else 0
    print(f"criterion={criterion}: Average score = {criterion_scores[criterion]:.4f}")

# Analyze min_samples_split
print(f"\nMin samples split performance:")
split_scores = {}
for split in param_grid['min_samples_split']:
    scores = [results_df['mean_test_score'][i] for i, params in enumerate(results_df['params']) 
              if params['min_samples_split'] == split]
    split_scores[split] = np.mean(scores) if scores else 0
    print(f"min_samples_split={split}: Average score = {split_scores[split]:.4f}")

# Classification report for best model
print(f"\n=== Classification Report (Best Model) ===")
yp_best = best_model.predict(xtest)
print(classification_report(ytest, yp_best, target_names=iris.target_names))

# Overfitting analysis
train_accuracy_best = best_model.score(xtrain, ytrain)
train_accuracy_default = default_dt.score(xtrain, ytrain)

print(f"\n=== Overfitting Analysis ===")
print(f"Default DT - Train: {train_accuracy_default:.4f}, Test: {default_accuracy:.4f}, Gap: {train_accuracy_default - default_accuracy:.4f}")
print(f"Best DT - Train: {train_accuracy_best:.4f}, Test: {test_accuracy:.4f}, Gap: {train_accuracy_best - test_accuracy:.4f}")

if train_accuracy_best - test_accuracy < train_accuracy_default - default_accuracy:
    print("✓ Hyperparameter tuning reduced overfitting")
else:
    print("⚠ Hyperparameter tuning did not reduce overfitting significantly")

print(f"\n=== Key Decision Tree Tuning Insights ===")
print("• max_depth controls tree complexity and overfitting")
print("• min_samples_split prevents splitting on too few samples")
print("• min_samples_leaf ensures minimum samples in leaf nodes")
print("• criterion (gini vs entropy) affects splitting decisions")
print("• Grid search helps find optimal balance between bias and variance")
print("• Cross-validation provides robust performance estimates")
