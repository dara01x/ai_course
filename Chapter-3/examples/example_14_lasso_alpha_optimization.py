"""
Example 14 - Lasso Alpha Optimization using Cross-Validation
This example demonstrates automatic alpha parameter tuning for Lasso regression
"""

from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

print("=== Lasso Alpha Optimization using Cross-Validation ===")

# Load diabetes dataset
X, y = load_diabetes(return_X_y=True)
print(f"Dataset shape: X={X.shape}, y={y.shape}")

# Split dataset
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set: {xtrain.shape}")
print(f"Testing set: {xtest.shape}")

print(f"\n=== Lasso without Scaling ===")
# Create LassoCV model (automatically selects best alpha)
model = LassoCV(cv=5, random_state=42)
model.fit(xtrain, ytrain)

# Make predictions
yp = model.predict(xtest)
mse = mean_squared_error(ytest, yp)

print(f'Best alpha: {model.alpha_:.6f}')
print(f'Mean Squared Error: {mse:.3f}')
print(f'Root Mean Squared Error: {np.sqrt(mse):.3f}')

# Show coefficients
print(f'\n=== Feature Coefficients (without scaling) ===')
feature_names = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
non_zero_features = 0
for i, (name, coef) in enumerate(zip(feature_names, model.coef_)):
    if abs(coef) > 1e-10:
        print(f'{name}: {coef:.3f}')
        non_zero_features += 1
    else:
        print(f'{name}: 0.000 (eliminated)')

print(f'\nFeatures selected: {non_zero_features}/{len(feature_names)}')

print(f"\n" + "="*60)
print("=== Lasso with Feature Scaling ===")

# Scale features
scaler = StandardScaler()
xtrain_scaled = scaler.fit_transform(xtrain)
xtest_scaled = scaler.transform(xtest)

# Create LassoCV model with scaled features
model_scaled = LassoCV(cv=5, random_state=42)
model_scaled.fit(xtrain_scaled, ytrain)

# Make predictions
yp_scaled = model_scaled.predict(xtest_scaled)
mse_scaled = mean_squared_error(ytest, yp_scaled)

print(f'Best alpha (scaled): {model_scaled.alpha_:.6f}')
print(f'Mean Squared Error (scaled): {mse_scaled:.3f}')
print(f'Root Mean Squared Error (scaled): {np.sqrt(mse_scaled):.3f}')

# Show coefficients for scaled features
print(f'\n=== Feature Coefficients (with scaling) ===')
non_zero_features_scaled = 0
for i, (name, coef) in enumerate(zip(feature_names, model_scaled.coef_)):
    if abs(coef) > 1e-10:
        print(f'{name}: {coef:.3f}')
        non_zero_features_scaled += 1
    else:
        print(f'{name}: 0.000 (eliminated)')

print(f'\nFeatures selected: {non_zero_features_scaled}/{len(feature_names)}')

# Compare performance
print(f"\n=== Performance Comparison ===")
print(f"Without scaling - MSE: {mse:.3f}, Features: {non_zero_features}")
print(f"With scaling - MSE: {mse_scaled:.3f}, Features: {non_zero_features_scaled}")
print(f"MSE improvement: {mse - mse_scaled:.3f}")

# Alpha selection analysis
print(f"\n=== Alpha Selection Analysis ===")
print(f"Alpha range tested: {model_scaled.alphas_.min():.6f} to {model_scaled.alphas_.max():.6f}")
print(f"Number of alphas tested: {len(model_scaled.alphas_)}")

# Plot alpha vs MSE
plt.figure(figsize=(12, 5))

# Plot 1: Alpha path
plt.subplot(1, 2, 1)
plt.plot(model_scaled.alphas_, model_scaled.mse_path_.mean(axis=1), 'b-', alpha=0.7)
plt.fill_between(model_scaled.alphas_, 
                 model_scaled.mse_path_.mean(axis=1) - model_scaled.mse_path_.std(axis=1),
                 model_scaled.mse_path_.mean(axis=1) + model_scaled.mse_path_.std(axis=1),
                 alpha=0.2)
plt.axvline(model_scaled.alpha_, color='red', linestyle='--', label=f'Best alpha: {model_scaled.alpha_:.4f}')
plt.xlabel('Alpha')
plt.ylabel('Mean Squared Error')
plt.title('Lasso Cross-Validation: Alpha vs MSE')
plt.xscale('log')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Feature coefficients
plt.subplot(1, 2, 2)
feature_indices = np.arange(len(feature_names))
colors = ['red' if abs(coef) < 1e-10 else 'blue' for coef in model_scaled.coef_]
bars = plt.bar(feature_indices, model_scaled.coef_, color=colors, alpha=0.7)
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.title('Lasso Feature Coefficients (Scaled Data)')
plt.xticks(feature_indices, feature_names, rotation=45)
plt.grid(True, alpha=0.3)

# Add legend
red_bar = plt.Rectangle((0,0),1,1, color='red', alpha=0.7)
blue_bar = plt.Rectangle((0,0),1,1, color='blue', alpha=0.7)
plt.legend([red_bar, blue_bar], ['Eliminated', 'Selected'])

plt.tight_layout()
plt.show()

# Regularization path analysis
print(f"\n=== Regularization Path Analysis ===")
alphas_to_show = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
print("Alpha\t\tMSE\t\tActive Features")
print("-" * 45)

for alpha in alphas_to_show:
    from sklearn.linear_model import Lasso
    temp_model = Lasso(alpha=alpha)
    temp_model.fit(xtrain_scaled, ytrain)
    temp_pred = temp_model.predict(xtest_scaled)
    temp_mse = mean_squared_error(ytest, temp_pred)
    active_features = np.sum(np.abs(temp_model.coef_) > 1e-10)
    print(f"{alpha:.4f}\t\t{temp_mse:.3f}\t\t{active_features}")

print(f"\n=== Key Lasso Insights ===")
print("• LassoCV automatically selects optimal alpha using cross-validation")
print("• Feature scaling significantly affects Lasso performance")
print("• Lasso performs automatic feature selection by setting coefficients to zero")
print("• Higher alpha values lead to more regularization and fewer features")
print("• Cross-validation helps balance bias-variance trade-off")
