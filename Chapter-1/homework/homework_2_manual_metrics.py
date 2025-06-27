"""
Homework 2 - Compute Regression Metrics by Hand
Manual calculation of MSE, MAE, and Max Error for the regression example
"""

import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error

print("="*80)
print("HOMEWORK 2 - MANUAL COMPUTATION OF REGRESSION METRICS")
print("="*80)

# Generate the same dataset as in Example 10
np.random.seed(42)  # For reproducibility
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Dataset Information:")
print(f"Test samples: {len(y_test)}")
print(f"Model coefficients: {model.coef_[0]:.4f}")
print(f"Model intercept: {model.intercept_:.4f}")

# Take first 10 samples for detailed manual calculation
n_samples = 10
print(f"\n{'='*70}")
print(f"MANUAL CALCULATION FOR FIRST {n_samples} TEST SAMPLES")
print(f"{'='*70}")

print(f"{'Sample':<8} {'Actual':<12} {'Predicted':<12} {'Error':<12} {'|Error|':<12} {'Error²':<12}")
print("-" * 75)

# Initialize accumulators
total_squared_error = 0
total_absolute_error = 0
max_absolute_error = 0

# Calculate for each sample
errors = []
for i in range(n_samples):
    actual = y_test[i]
    predicted = y_pred[i]
    error = actual - predicted
    abs_error = abs(error)
    squared_error = error ** 2
    
    # Update accumulators
    total_squared_error += squared_error
    total_absolute_error += abs_error
    max_absolute_error = max(max_absolute_error, abs_error)
    
    errors.append(error)
    
    print(f"{i+1:<8} {actual:<12.4f} {predicted:<12.4f} {error:<12.4f} {abs_error:<12.4f} {squared_error:<12.4f}")

print("-" * 75)
print(f"{'TOTALS:':<8} {'':<12} {'':<12} {sum(errors):<12.4f} {total_absolute_error:<12.4f} {total_squared_error:<12.4f}")

# Calculate metrics manually
mse_manual = total_squared_error / n_samples
mae_manual = total_absolute_error / n_samples
rmse_manual = np.sqrt(mse_manual)
max_error_manual = max_absolute_error

print(f"\n{'='*60}")
print("MANUAL METRIC CALCULATIONS")
print(f"{'='*60}")

print(f"Mean Squared Error (MSE):")
print(f"  Formula: MSE = (1/N) × Σ(y_true - y_pred)²")
print(f"  Calculation: MSE = {total_squared_error:.4f} / {n_samples}")
print(f"  Result: MSE = {mse_manual:.4f}")

print(f"\nRoot Mean Squared Error (RMSE):")
print(f"  Formula: RMSE = √MSE")
print(f"  Calculation: RMSE = √{mse_manual:.4f}")
print(f"  Result: RMSE = {rmse_manual:.4f}")

print(f"\nMean Absolute Error (MAE):")
print(f"  Formula: MAE = (1/N) × Σ|y_true - y_pred|")
print(f"  Calculation: MAE = {total_absolute_error:.4f} / {n_samples}")
print(f"  Result: MAE = {mae_manual:.4f}")

print(f"\nMaximum Error:")
print(f"  Formula: Max Error = max(|y_true - y_pred|)")
print(f"  Result: Max Error = {max_error_manual:.4f}")

# Now calculate for ALL test samples
print(f"\n{'='*60}")
print("MANUAL CALCULATION FOR ALL TEST SAMPLES")
print(f"{'='*60}")

# Calculate metrics for all test samples manually
all_errors = y_test - y_pred
all_squared_errors = all_errors ** 2
all_absolute_errors = np.abs(all_errors)

mse_all_manual = np.sum(all_squared_errors) / len(y_test)
mae_all_manual = np.sum(all_absolute_errors) / len(y_test)
rmse_all_manual = np.sqrt(mse_all_manual)
max_error_all_manual = np.max(all_absolute_errors)

print(f"Manual calculations for all {len(y_test)} test samples:")
print(f"  MSE  = {np.sum(all_squared_errors):.4f} / {len(y_test)} = {mse_all_manual:.4f}")
print(f"  MAE  = {np.sum(all_absolute_errors):.4f} / {len(y_test)} = {mae_all_manual:.4f}")
print(f"  RMSE = √{mse_all_manual:.4f} = {rmse_all_manual:.4f}")
print(f"  Max Error = {max_error_all_manual:.4f}")

# Verify with sklearn
mse_sklearn = mean_squared_error(y_test, y_pred)
mae_sklearn = mean_absolute_error(y_test, y_pred)
rmse_sklearn = np.sqrt(mse_sklearn)
max_error_sklearn = max_error(y_test, y_pred)

print(f"\n{'='*60}")
print("VERIFICATION WITH SKLEARN")
print(f"{'='*60}")

print(f"{'Metric':<12} {'Manual':<15} {'Sklearn':<15} {'Difference':<15} {'Match'}")
print("-" * 70)
print(f"{'MSE':<12} {mse_all_manual:<15.6f} {mse_sklearn:<15.6f} {abs(mse_all_manual - mse_sklearn):<15.6f} {np.isclose(mse_all_manual, mse_sklearn)}")
print(f"{'RMSE':<12} {rmse_all_manual:<15.6f} {rmse_sklearn:<15.6f} {abs(rmse_all_manual - rmse_sklearn):<15.6f} {np.isclose(rmse_all_manual, rmse_sklearn)}")
print(f"{'MAE':<12} {mae_all_manual:<15.6f} {mae_sklearn:<15.6f} {abs(mae_all_manual - mae_sklearn):<15.6f} {np.isclose(mae_all_manual, mae_sklearn)}")
print(f"{'Max Error':<12} {max_error_all_manual:<15.6f} {max_error_sklearn:<15.6f} {abs(max_error_all_manual - max_error_sklearn):<15.6f} {np.isclose(max_error_all_manual, max_error_sklearn)}")

# Detailed step-by-step calculation breakdown
print(f"\n{'='*70}")
print("STEP-BY-STEP CALCULATION BREAKDOWN")
print(f"{'='*70}")

print("Understanding the formulas:")
print("\n1. Mean Squared Error (MSE):")
print("   • Purpose: Measures average squared difference between actual and predicted values")
print("   • Formula: MSE = (1/N) × Σ(y_actual - y_predicted)²")
print("   • Why squared? Penalizes larger errors more heavily, always positive")
print(f"   • Our calculation: Sum of squared errors = {np.sum(all_squared_errors):.4f}")
print(f"                     Number of samples = {len(y_test)}")
print(f"                     MSE = {np.sum(all_squared_errors):.4f} / {len(y_test)} = {mse_all_manual:.4f}")

print("\n2. Root Mean Squared Error (RMSE):")
print("   • Purpose: Same as MSE but in original units (more interpretable)")
print("   • Formula: RMSE = √MSE")
print("   • Advantage: Same units as the target variable")
print(f"   • Our calculation: RMSE = √{mse_all_manual:.4f} = {rmse_all_manual:.4f}")

print("\n3. Mean Absolute Error (MAE):")
print("   • Purpose: Measures average absolute difference between actual and predicted")
print("   • Formula: MAE = (1/N) × Σ|y_actual - y_predicted|")
print("   • Advantage: More robust to outliers than MSE")
print(f"   • Our calculation: Sum of absolute errors = {np.sum(all_absolute_errors):.4f}")
print(f"                      MAE = {np.sum(all_absolute_errors):.4f} / {len(y_test)} = {mae_all_manual:.4f}")

print("\n4. Maximum Error:")
print("   • Purpose: Identifies the worst-case prediction error")
print("   • Formula: Max Error = max(|y_actual - y_predicted|)")
print("   • Use case: Understanding model's worst performance")
print(f"   • Our calculation: Max Error = {max_error_all_manual:.4f}")

# Error analysis
print(f"\n{'='*60}")
print("ERROR ANALYSIS")
print(f"{'='*60}")

# Find the samples with largest errors
error_indices = np.argsort(np.abs(all_errors))[::-1]  # Sort by absolute error, descending

print("Top 5 samples with largest errors:")
print(f"{'Rank':<6} {'Sample':<8} {'Actual':<12} {'Predicted':<12} {'Error':<12} {'|Error|':<12}")
print("-" * 65)

for rank, idx in enumerate(error_indices[:5], 1):
    actual = y_test[idx]
    predicted = y_pred[idx]
    error = all_errors[idx]
    abs_error = abs(error)
    
    print(f"{rank:<6} {idx+1:<8} {actual:<12.4f} {predicted:<12.4f} {error:<12.4f} {abs_error:<12.4f}")

# Distribution analysis
print(f"\nError distribution analysis:")
print(f"Mean error: {np.mean(all_errors):.4f} (should be close to 0 for unbiased model)")
print(f"Standard deviation of errors: {np.std(all_errors):.4f}")
print(f"Minimum error: {np.min(all_errors):.4f}")
print(f"Maximum error: {np.max(all_errors):.4f}")
print(f"25th percentile error: {np.percentile(all_errors, 25):.4f}")
print(f"75th percentile error: {np.percentile(all_errors, 75):.4f}")

# Comparison of metrics
print(f"\n{'='*60}")
print("METRIC COMPARISON AND INTERPRETATION")
print(f"{'='*60}")

print(f"MSE = {mse_all_manual:.4f}")
print(f"MAE = {mae_all_manual:.4f}")
print(f"RMSE = {rmse_all_manual:.4f}")

print(f"\nComparison insights:")
if mse_all_manual > mae_all_manual**2:
    print("• MSE > MAE², indicating presence of some large errors (outliers)")
else:
    print("• MSE ≤ MAE², indicating relatively uniform error distribution")

if rmse_all_manual > mae_all_manual:
    print(f"• RMSE ({rmse_all_manual:.4f}) > MAE ({mae_all_manual:.4f}): Large errors have significant impact")
    print(f"  The difference ({rmse_all_manual - mae_all_manual:.4f}) indicates the penalty for large errors")
else:
    print("• RMSE ≈ MAE: Errors are relatively uniform")

print(f"• Max Error ({max_error_all_manual:.4f}) is {max_error_all_manual/mae_all_manual:.1f}x larger than MAE")
print(f"  This shows the difference between typical and worst-case errors")

# Mathematical verification using different approaches
print(f"\n{'='*60}")
print("MATHEMATICAL VERIFICATION USING DIFFERENT APPROACHES")
print(f"{'='*60}")

# Method 1: Using numpy built-in functions
mse_numpy = np.mean((y_test - y_pred)**2)
mae_numpy = np.mean(np.abs(y_test - y_pred))
rmse_numpy = np.sqrt(mse_numpy)
max_error_numpy = np.max(np.abs(y_test - y_pred))

# Method 2: Using list comprehension
mse_list = sum([(actual - pred)**2 for actual, pred in zip(y_test, y_pred)]) / len(y_test)
mae_list = sum([abs(actual - pred) for actual, pred in zip(y_test, y_pred)]) / len(y_test)
rmse_list = np.sqrt(mse_list)
max_error_list = max([abs(actual - pred) for actual, pred in zip(y_test, y_pred)])

print("Verification using different calculation methods:")
print(f"{'Method':<20} {'MSE':<12} {'MAE':<12} {'RMSE':<12} {'Max Error':<12}")
print("-" * 75)
print(f"{'Manual (loops)':<20} {mse_all_manual:<12.6f} {mae_all_manual:<12.6f} {rmse_all_manual:<12.6f} {max_error_all_manual:<12.6f}")
print(f"{'NumPy vectorized':<20} {mse_numpy:<12.6f} {mae_numpy:<12.6f} {rmse_numpy:<12.6f} {max_error_numpy:<12.6f}")
print(f"{'List comprehension':<20} {mse_list:<12.6f} {mae_list:<12.6f} {rmse_list:<12.6f} {max_error_list:<12.6f}")
print(f"{'Sklearn':<20} {mse_sklearn:<12.6f} {mae_sklearn:<12.6f} {rmse_sklearn:<12.6f} {max_error_sklearn:<12.6f}")

print(f"\n{'='*60}")
print("HOMEWORK COMPLETION SUMMARY")
print(f"{'='*60}")
print("✓ Manually calculated MSE using the formula: (1/N) × Σ(y_true - y_pred)²")
print("✓ Manually calculated MAE using the formula: (1/N) × Σ|y_true - y_pred|")
print("✓ Manually calculated RMSE using the formula: √MSE")
print("✓ Manually calculated Max Error using the formula: max(|y_true - y_pred|)")
print("✓ Verified all calculations against sklearn implementations")
print("✓ Provided step-by-step breakdown of calculations")
print("✓ Analyzed error distribution and characteristics")
print("✓ Demonstrated multiple calculation approaches")

print(f"\nKey takeaways:")
print("1. Manual calculations match sklearn implementations exactly")
print("2. MSE penalizes large errors more than MAE due to squaring")
print("3. RMSE provides interpretable error magnitude in original units")
print("4. Max Error reveals worst-case model performance")
print("5. Understanding these metrics helps in model evaluation and selection")
