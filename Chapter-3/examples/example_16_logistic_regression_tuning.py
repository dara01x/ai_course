"""
Example 16 - Logistic Regression Regularization Parameter Tuning
This example demonstrates hyperparameter optimization for Logistic Regression regularization
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

print("=== Logistic Regression Regularization Parameter Tuning ===")

# Create synthetic classification dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, 
                          n_classes=2, random_state=0)

print(f"Dataset shape: X={X.shape}, y={y.shape}")
print(f"Class distribution: {np.bincount(y)}")

# Split dataset
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set: {xtrain.shape}")
print(f"Testing set: {xtest.shape}")

# Standardize features (important for regularization)
scaler = StandardScaler()
xtrain_scaled = scaler.fit_transform(xtrain)
xtest_scaled = scaler.transform(xtest)

# Define parameter grid for Logistic Regression
param_grid = {
    'C': np.logspace(-3, 3, 7),  # [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']  # Required for l1 penalty
}

print(f"\n=== Parameter Grid ===")
print(f"C values: {param_grid['C']}")
print(f"Penalty types: {param_grid['penalty']}")
print(f"Solver: {param_grid['solver']}")
print(f"Total combinations: {len(param_grid['C']) * len(param_grid['penalty'])}")

# Create Logistic Regression model
lr_model = LogisticRegression(max_iter=1000, random_state=42)

# Create GridSearchCV object
print(f"\n=== Running Grid Search ===")
grid_search = GridSearchCV(estimator=lr_model, param_grid=param_grid, cv=5, 
                          scoring='accuracy', n_jobs=-1)

# Fit grid search
grid_search.fit(xtrain_scaled, ytrain)

# Get best parameters and score
best_params = grid_search.best_params_
best_cv_score = grid_search.best_score_
best_model = grid_search.best_estimator_

print(f"\n=== Grid Search Results ===")
print(f"Best parameters: {best_params}")
print(f"Best cross-validation score: {best_cv_score:.4f} ({best_cv_score*100:.2f}%)")

# Test the best model
test_accuracy = best_model.score(xtest_scaled, ytest)
print(f"Test accuracy with best parameters: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Compare with default Logistic Regression
print(f"\n=== Comparison with Default Logistic Regression ===")
default_lr = LogisticRegression(max_iter=1000, random_state=42)
default_lr.fit(xtrain_scaled, ytrain)
default_accuracy = default_lr.score(xtest_scaled, ytest)

print(f"Default LR accuracy: {default_accuracy:.4f} ({default_accuracy*100:.2f}%)")
print(f"Optimized LR accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Improvement: {test_accuracy - default_accuracy:.4f} ({(test_accuracy - default_accuracy)*100:.2f}%)")

# Detailed results analysis
results_df = grid_search.cv_results_
print(f"\n=== Detailed Results ===")
print("C\tPenalty\tCV Score\tStd")
print("-" * 35)

# Sort by mean test score
sorted_indices = np.argsort(results_df['mean_test_score'])[::-1]
for i in range(len(sorted_indices)):
    idx = sorted_indices[i]
    params = results_df['params'][idx]
    mean_score = results_df['mean_test_score'][idx]
    std_score = results_df['std_test_score'][idx]
    print(f"{params['C']:.3f}\t{params['penalty']}\t{mean_score:.4f}\t\t{std_score:.4f}")

# Analyze C parameter impact
print(f"\n=== C Parameter Analysis ===")
c_values = param_grid['C']

l1_scores = []
l2_scores = []

for c_val in c_values:
    # L1 scores
    l1_score = [results_df['mean_test_score'][i] for i, params in enumerate(results_df['params']) 
                if params['C'] == c_val and params['penalty'] == 'l1'][0]
    l1_scores.append(l1_score)
    
    # L2 scores
    l2_score = [results_df['mean_test_score'][i] for i, params in enumerate(results_df['params']) 
                if params['C'] == c_val and params['penalty'] == 'l2'][0]
    l2_scores.append(l2_score)

# Plot regularization path
plt.figure(figsize=(12, 5))

# Plot 1: C parameter vs accuracy
plt.subplot(1, 2, 1)
plt.semilogx(c_values, l1_scores, 'o-', label='L1 (Lasso)', marker='o')
plt.semilogx(c_values, l2_scores, 's-', label='L2 (Ridge)', marker='s')
plt.axvline(best_params['C'], color='red', linestyle='--', alpha=0.7, 
           label=f'Best C: {best_params["C"]:.3f}')
plt.xlabel('C (Regularization Strength)')
plt.ylabel('Cross-Validation Accuracy')
plt.title('Regularization Parameter vs Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Feature coefficients for different C values
plt.subplot(1, 2, 2)
c_test_values = [0.01, 0.1, 1.0, 10.0]
colors = ['red', 'blue', 'green', 'orange']

for i, c_val in enumerate(c_test_values):
    temp_model = LogisticRegression(C=c_val, penalty='l1', solver='liblinear', 
                                   max_iter=1000, random_state=42)
    temp_model.fit(xtrain_scaled, ytrain)
    
    feature_indices = np.arange(len(temp_model.coef_[0]))
    plt.scatter(feature_indices, temp_model.coef_[0], 
               alpha=0.7, label=f'C={c_val}', color=colors[i])

plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.title('Feature Coefficients for Different C Values (L1)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Feature selection analysis
print(f"\n=== Feature Selection Analysis (L1 Regularization) ===")
print("C Value\tActive Features\tAccuracy")
print("-" * 40)

for c_val in [0.001, 0.01, 0.1, 1.0, 10.0]:
    temp_model = LogisticRegression(C=c_val, penalty='l1', solver='liblinear', 
                                   max_iter=1000, random_state=42)
    temp_model.fit(xtrain_scaled, ytrain)
    temp_accuracy = temp_model.score(xtest_scaled, ytest)
    active_features = np.sum(np.abs(temp_model.coef_[0]) > 1e-10)
    print(f"{c_val:.3f}\t\t{active_features}\t\t{temp_accuracy:.4f}")

# Classification report for best model
print(f"\n=== Classification Report (Best Model) ===")
yp_best = best_model.predict(xtest_scaled)
print(classification_report(ytest, yp_best))

# Regularization strength interpretation
print(f"\n=== Best Model Analysis ===")
print(f"Best C value: {best_params['C']:.3f}")
print(f"Best penalty: {best_params['penalty']}")

if best_params['C'] < 1:
    print("→ Strong regularization: Model prefers simplicity")
elif best_params['C'] > 1:
    print("→ Weak regularization: Model can fit more complex patterns")
else:
    print("→ Balanced regularization: Default strength works well")

if best_params['penalty'] == 'l1':
    active_features = np.sum(np.abs(best_model.coef_[0]) > 1e-10)
    print(f"→ L1 penalty performs feature selection: {active_features}/{X.shape[1]} features active")
else:
    print("→ L2 penalty shrinks all coefficients but keeps all features")

print(f"\n=== Key Logistic Regression Tuning Insights ===")
print("• C parameter controls regularization strength (inverse relationship)")
print("• Lower C = stronger regularization = simpler model")
print("• L1 penalty (Lasso) performs automatic feature selection")
print("• L2 penalty (Ridge) shrinks coefficients but keeps all features")
print("• Feature scaling is crucial for regularization effectiveness")
print("• Cross-validation helps find optimal bias-variance trade-off")
