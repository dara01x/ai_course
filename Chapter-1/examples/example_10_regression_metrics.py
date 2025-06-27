"""
Example 10 - Regression Evaluation Metrics
Demonstrate how to use various metrics to assess regression algorithm performance
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error, r2_score
from sklearn.preprocessing import StandardScaler

print("=== Regression Evaluation Metrics Example ===")

# Generate synthetic regression dataset
X, y = make_regression(
    n_samples=100,
    n_features=1,
    noise=10,
    random_state=42
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print("Dataset Information:")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Feature range: [{X.min():.2f}, {X.max():.2f}]")
print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")

# Calculate metrics using sklearn
print(f"\n{'='*60}")
print("SKLEARN METRICS RESULTS")
print(f"{'='*60}")

# Training metrics
mse_train = mean_squared_error(y_train, y_pred_train)
mae_train = mean_absolute_error(y_train, y_pred_train)
max_err_train = max_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)
rmse_train = np.sqrt(mse_train)

print("Training Set Metrics:")
print(f"  MSE:  {mse_train:.4f}")
print(f"  RMSE: {rmse_train:.4f}")
print(f"  MAE:  {mae_train:.4f}")
print(f"  Max Error: {max_err_train:.4f}")
print(f"  R² Score:  {r2_train:.4f}")

# Test metrics
mse_test = mean_squared_error(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
max_err_test = max_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)
rmse_test = np.sqrt(mse_test)

print("\nTesting Set Metrics:")
print(f"  MSE:  {mse_test:.4f}")
print(f"  RMSE: {rmse_test:.4f}")
print(f"  MAE:  {mae_test:.4f}")
print(f"  Max Error: {max_err_test:.4f}")
print(f"  R² Score:  {r2_test:.4f}")

# Manual calculation for verification
print(f"\n{'='*60}")
print("MANUAL CALCULATIONS (for verification)")
print(f"{'='*60}")

def calculate_metrics_manual(y_true, y_pred):
    """Calculate metrics manually"""
    n = len(y_true)
    errors = y_true - y_pred
    
    # MSE = (1/N) * Σ(y_true - y_pred)²
    mse = np.sum(errors**2) / n
    
    # RMSE = √MSE
    rmse = np.sqrt(mse)
    
    # MAE = (1/N) * Σ|y_true - y_pred|
    mae = np.sum(np.abs(errors)) / n
    
    # Max Error = max(|y_true - y_pred|)
    max_err = np.max(np.abs(errors))
    
    # R² = 1 - (SS_res / SS_tot)
    ss_res = np.sum(errors**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    return mse, rmse, mae, max_err, r2

# Manual calculation for test set
mse_manual, rmse_manual, mae_manual, max_err_manual, r2_manual = calculate_metrics_manual(y_test, y_pred_test)

print("Manual Test Set Calculations:")
print(f"  MSE:  {mse_manual:.4f}")
print(f"  RMSE: {rmse_manual:.4f}")
print(f"  MAE:  {mae_manual:.4f}")
print(f"  Max Error: {max_err_manual:.4f}")
print(f"  R² Score:  {r2_manual:.4f}")

print("\nVerification (sklearn vs manual):")
print(f"  MSE match:  {np.isclose(mse_test, mse_manual)}")
print(f"  RMSE match: {np.isclose(rmse_test, rmse_manual)}")
print(f"  MAE match:  {np.isclose(mae_test, mae_manual)}")
print(f"  Max Error match: {np.isclose(max_err_test, max_err_manual)}")
print(f"  R² match:  {np.isclose(r2_test, r2_manual)}")

# Visualization
plt.figure(figsize=(16, 12))

# Plot 1: Actual vs Predicted
plt.subplot(3, 3, 1)
plt.scatter(y_test, y_pred_test, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.grid(True, alpha=0.3)

# Add perfect prediction line
plt.text(0.05, 0.95, f'R² = {r2_test:.3f}', transform=plt.gca().transAxes, 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 2: Residuals
plt.subplot(3, 3, 2)
residuals = y_test - y_pred_test
plt.scatter(y_pred_test, residuals, alpha=0.6, color='red')
plt.axhline(y=0, color='black', linestyle='--', alpha=0.7)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(True, alpha=0.3)

# Plot 3: Error distribution
plt.subplot(3, 3, 3)
plt.hist(residuals, bins=15, alpha=0.7, color='green', edgecolor='black')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residual Distribution')
plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
plt.grid(True, alpha=0.3)

# Add statistics
plt.text(0.05, 0.95, f'Mean: {np.mean(residuals):.3f}\nStd: {np.std(residuals):.3f}', 
         transform=plt.gca().transAxes, 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 4: Metrics comparison
plt.subplot(3, 3, 4)
metrics_names = ['MSE', 'RMSE', 'MAE', 'Max Error']
train_metrics = [mse_train, rmse_train, mae_train, max_err_train]
test_metrics = [mse_test, rmse_test, mae_test, max_err_test]

x_pos = np.arange(len(metrics_names))
width = 0.35

plt.bar(x_pos - width/2, train_metrics, width, label='Training', alpha=0.7)
plt.bar(x_pos + width/2, test_metrics, width, label='Testing', alpha=0.7)

plt.xlabel('Metrics')
plt.ylabel('Error Value')
plt.title('Training vs Testing Metrics')
plt.xticks(x_pos, metrics_names, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 5: Individual predictions analysis
plt.subplot(3, 3, 5)
sample_indices = np.arange(len(y_test))
plt.plot(sample_indices, y_test, 'o-', label='Actual', alpha=0.7)
plt.plot(sample_indices, y_pred_test, 's-', label='Predicted', alpha=0.7)
plt.fill_between(sample_indices, y_test, y_pred_test, alpha=0.3, color='red', label='Error')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title('Individual Predictions')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 6: Regression line with data
plt.subplot(3, 3, 6)
plt.scatter(X_test, y_test, alpha=0.6, color='blue', label='Actual')
plt.scatter(X_test, y_pred_test, alpha=0.6, color='red', label='Predicted')

# Plot regression line
X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_line = model.predict(X_line)
plt.plot(X_line, y_line, 'g-', linewidth=2, label='Regression Line')

plt.xlabel('Feature Value')
plt.ylabel('Target Value')
plt.title('Regression Line Fit')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 7: Error magnitude analysis
plt.subplot(3, 3, 7)
abs_errors = np.abs(residuals)
sorted_errors = np.sort(abs_errors)
percentiles = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100

plt.plot(percentiles, sorted_errors, 'b-', linewidth=2)
plt.axhline(y=mae_test, color='red', linestyle='--', label=f'MAE = {mae_test:.2f}')
plt.axhline(y=rmse_test, color='green', linestyle='--', label=f'RMSE = {rmse_test:.2f}')
plt.xlabel('Percentile')
plt.ylabel('Absolute Error')
plt.title('Error Distribution Analysis')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 8: Metrics sensitivity analysis
plt.subplot(3, 3, 8)
# Add artificial outliers to show metric sensitivity
y_test_outlier = y_test.copy()
y_test_outlier[-1] = y_test_outlier[-1] + 5 * np.std(y_test)  # Add one outlier

mse_outlier = mean_squared_error(y_test_outlier, y_pred_test)
mae_outlier = mean_absolute_error(y_test_outlier, y_pred_test)
rmse_outlier = np.sqrt(mse_outlier)

metrics_original = [mse_test, rmse_test, mae_test]
metrics_outlier = [mse_outlier, rmse_outlier, mae_outlier]
metric_names = ['MSE', 'RMSE', 'MAE']

x_pos = np.arange(len(metric_names))
width = 0.35

plt.bar(x_pos - width/2, metrics_original, width, label='Original', alpha=0.7)
plt.bar(x_pos + width/2, metrics_outlier, width, label='With Outlier', alpha=0.7)

plt.xlabel('Metrics')
plt.ylabel('Error Value')
plt.title('Outlier Sensitivity')
plt.xticks(x_pos, metric_names)
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 9: R² interpretation
plt.subplot(3, 3, 9)
# Show what R² means visually
y_mean = np.mean(y_test)
ss_tot = np.sum((y_test - y_mean)**2)
ss_res = np.sum((y_test - y_pred_test)**2)

plt.scatter(range(len(y_test)), y_test, alpha=0.6, label='Actual', color='blue')
plt.scatter(range(len(y_test)), y_pred_test, alpha=0.6, label='Predicted', color='red')
plt.axhline(y=y_mean, color='green', linestyle='--', label=f'Mean = {y_mean:.2f}')

# Show residual lines
for i in range(len(y_test)):
    plt.plot([i, i], [y_test[i], y_pred_test[i]], 'k-', alpha=0.3)

plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title(f'R² = {r2_test:.3f} Interpretation')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Detailed step-by-step calculation example
print(f"\n{'='*60}")
print("STEP-BY-STEP CALCULATION EXAMPLE")
print(f"{'='*60}")

# Take first 5 test samples for detailed calculation
n_examples = 5
print(f"Using first {n_examples} test samples:")
print(f"{'Sample':<8} {'Actual':<10} {'Predicted':<10} {'Error':<10} {'Abs Error':<10} {'Squared Error':<15}")
print("-" * 70)

total_error = 0
total_abs_error = 0
total_squared_error = 0

for i in range(n_examples):
    actual = y_test[i]
    predicted = y_pred_test[i]
    error = actual - predicted
    abs_error = abs(error)
    squared_error = error**2
    
    total_error += error
    total_abs_error += abs_error
    total_squared_error += squared_error
    
    print(f"{i+1:<8} {actual:<10.3f} {predicted:<10.3f} {error:<10.3f} {abs_error:<10.3f} {squared_error:<15.3f}")

print("-" * 70)
print(f"{'Totals:':<8} {'':<10} {'':<10} {total_error:<10.3f} {total_abs_error:<10.3f} {total_squared_error:<15.3f}")

# Calculate metrics for these examples
mse_example = total_squared_error / n_examples
mae_example = total_abs_error / n_examples
rmse_example = np.sqrt(mse_example)

print(f"\nCalculated metrics for {n_examples} samples:")
print(f"MSE  = {total_squared_error:.3f} / {n_examples} = {mse_example:.3f}")
print(f"MAE  = {total_abs_error:.3f} / {n_examples} = {mae_example:.3f}")
print(f"RMSE = √{mse_example:.3f} = {rmse_example:.3f}")

# Compare different model complexities
print(f"\n{'='*60}")
print("COMPARISON OF DIFFERENT MODEL COMPLEXITIES")
print(f"{'='*60}")

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

models = {
    'Linear': LinearRegression(),
    'Polynomial (degree 2)': Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('linear', LinearRegression())
    ]),
    'Polynomial (degree 3)': Pipeline([
        ('poly', PolynomialFeatures(degree=3)),
        ('linear', LinearRegression())
    ])
}

print(f"{'Model':<20} {'MSE':<10} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
print("-" * 60)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"{name:<20} {mse:<10.4f} {rmse:<10.4f} {mae:<10.4f} {r2:<10.4f}")

print(f"\nKey Insights:")
print(f"1. MSE penalizes large errors more heavily than MAE")
print(f"2. RMSE is in the same units as the target variable")
print(f"3. R² measures the proportion of variance explained by the model")
print(f"4. Max error shows the worst-case prediction error")
print(f"5. Lower error metrics indicate better model performance")
