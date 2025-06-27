"""
Homework 3 - What is R² Score?
Comprehensive explanation and analysis of R² (R-squared) score
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

print("="*80)
print("HOMEWORK 3 - R² SCORE (R-SQUARED) COMPREHENSIVE ANALYSIS")
print("="*80)

def calculate_r2_manual(y_true, y_pred):
    """Calculate R² score manually"""
    # Total Sum of Squares (TSS) - variation from mean
    y_mean = np.mean(y_true)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    
    # Residual Sum of Squares (RSS) - variation from predictions
    ss_res = np.sum((y_true - y_pred) ** 2)
    
    # R² = 1 - (RSS/TSS)
    r2 = 1 - (ss_res / ss_tot)
    
    return r2, ss_tot, ss_res, y_mean

# Generate sample data
np.random.seed(42)
X, y = make_regression(n_samples=100, n_features=1, noise=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("WHAT IS R² SCORE?")
print("="*50)
print("R² (R-squared), also known as the coefficient of determination, measures")
print("the proportion of variance in the dependent variable that is predictable")
print("from the independent variable(s).")
print()
print("Formula: R² = 1 - (SS_res / SS_tot)")
print("Where:")
print("  • SS_res = Σ(y_actual - y_predicted)² (Residual Sum of Squares)")
print("  • SS_tot = Σ(y_actual - y_mean)² (Total Sum of Squares)")
print()
print("Interpretation:")
print("  • R² = 1.0: Perfect predictions (all variance explained)")
print("  • R² = 0.0: Model is no better than predicting the mean")
print("  • R² < 0.0: Model is worse than predicting the mean")

# Calculate R² manually
r2_manual, ss_tot, ss_res, y_mean = calculate_r2_manual(y_test, y_pred)
r2_sklearn = r2_score(y_test, y_pred)

print(f"\n{'='*60}")
print("MANUAL R² CALCULATION")
print(f"{'='*60}")

print(f"Test samples: {len(y_test)}")
print(f"Mean of actual values: {y_mean:.4f}")
print()
print("Step 1: Calculate Total Sum of Squares (SS_tot)")
print(f"  SS_tot = Σ(y_actual - y_mean)²")
print(f"  SS_tot = Σ(y_actual - {y_mean:.4f})²")
print(f"  SS_tot = {ss_tot:.4f}")
print(f"  (This represents total variance in the data)")
print()
print("Step 2: Calculate Residual Sum of Squares (SS_res)")
print(f"  SS_res = Σ(y_actual - y_predicted)²")
print(f"  SS_res = {ss_res:.4f}")
print(f"  (This represents unexplained variance)")
print()
print("Step 3: Calculate R²")
print(f"  R² = 1 - (SS_res / SS_tot)")
print(f"  R² = 1 - ({ss_res:.4f} / {ss_tot:.4f})")
print(f"  R² = 1 - {ss_res/ss_tot:.4f}")
print(f"  R² = {r2_manual:.4f}")

# Verification
print(f"\nVerification with sklearn: {r2_sklearn:.4f}")
print(f"Match: {np.isclose(r2_manual, r2_sklearn)}")

# Detailed explanation with examples
print(f"\n{'='*60}")
print("DETAILED BREAKDOWN WITH FIRST 10 SAMPLES")
print(f"{'='*60}")

print(f"{'Sample':<8} {'Actual':<10} {'Predicted':<10} {'y-mean':<10} {'(y-mean)²':<12} {'y-pred':<10} {'(y-pred)²':<12}")
print("-" * 80)

for i in range(min(10, len(y_test))):
    actual = y_test[i]
    predicted = y_pred[i]
    diff_from_mean = actual - y_mean
    diff_from_mean_sq = diff_from_mean ** 2
    diff_from_pred = actual - predicted
    diff_from_pred_sq = diff_from_pred ** 2
    
    print(f"{i+1:<8} {actual:<10.3f} {predicted:<10.3f} {diff_from_mean:<10.3f} {diff_from_mean_sq:<12.3f} {diff_from_pred:<10.3f} {diff_from_pred_sq:<12.3f}")

# Visual explanation
plt.figure(figsize=(16, 12))

# Plot 1: Actual vs Predicted with R²
plt.subplot(3, 3, 1)
plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'Actual vs Predicted (R² = {r2_manual:.3f})')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Residuals vs Fitted
plt.subplot(3, 3, 2)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.6, color='red')
plt.axhline(y=0, color='black', linestyle='--', alpha=0.7)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(True, alpha=0.3)

# Plot 3: Components of R² calculation
plt.subplot(3, 3, 3)
sample_indices = range(len(y_test))
plt.plot(sample_indices, y_test, 'o-', label='Actual', alpha=0.7)
plt.axhline(y=y_mean, color='red', linestyle='--', label=f'Mean = {y_mean:.2f}', alpha=0.7)
plt.plot(sample_indices, y_pred, 's-', label='Predicted', alpha=0.7)

# Show some residual lines
for i in range(0, len(y_test), 5):
    plt.plot([i, i], [y_test[i], y_mean], 'r-', alpha=0.3, linewidth=1)  # Total deviation
    plt.plot([i, i], [y_test[i], y_pred[i]], 'b-', alpha=0.3, linewidth=1)  # Residual

plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title('R² Components Visualization')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Different R² scenarios
plt.subplot(3, 3, 4)
# Generate data with different R² values
x_demo = np.linspace(0, 10, 50)
y_perfect = 2 * x_demo + 1  # Perfect relationship
y_good = y_perfect + np.random.normal(0, 2, len(x_demo))  # Good relationship
y_poor = y_perfect + np.random.normal(0, 8, len(x_demo))  # Poor relationship
y_random = np.random.normal(np.mean(y_perfect), np.std(y_perfect), len(x_demo))  # No relationship

r2_perfect = r2_score(y_perfect, y_perfect)
r2_good = r2_score(y_perfect, y_good)
r2_poor = r2_score(y_perfect, y_poor)
r2_random = r2_score(y_perfect, y_random)

plt.scatter(x_demo, y_perfect, alpha=0.6, label=f'Perfect (R²={r2_perfect:.2f})', s=30)
plt.scatter(x_demo, y_good, alpha=0.6, label=f'Good (R²={r2_good:.2f})', s=30)
plt.scatter(x_demo, y_poor, alpha=0.6, label=f'Poor (R²={r2_poor:.2f})', s=30)
plt.scatter(x_demo, y_random, alpha=0.6, label=f'Random (R²={r2_random:.2f})', s=30)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Different R² Scenarios')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 5: R² interpretation ranges
plt.subplot(3, 3, 5)
r2_ranges = ['Excellent\n(0.9-1.0)', 'Very Good\n(0.8-0.9)', 'Good\n(0.7-0.8)', 
             'Moderate\n(0.5-0.7)', 'Poor\n(0.3-0.5)', 'Very Poor\n(0.0-0.3)', 'Negative\n(<0.0)']
r2_values = [0.95, 0.85, 0.75, 0.6, 0.4, 0.15, -0.2]
colors = ['darkgreen', 'green', 'lightgreen', 'yellow', 'orange', 'red', 'darkred']

bars = plt.bar(range(len(r2_ranges)), r2_values, color=colors, alpha=0.7)
plt.xticks(range(len(r2_ranges)), r2_ranges, rotation=45, ha='right')
plt.ylabel('R² Value')
plt.title('R² Interpretation Guide')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
plt.grid(True, alpha=0.3)

# Add value labels
for bar, value in zip(bars, r2_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02 if value >= 0 else bar.get_height() - 0.05, 
             f'{value:.2f}', ha='center', va='bottom' if value >= 0 else 'top', fontweight='bold')

# Plot 6: Model complexity vs R²
plt.subplot(3, 3, 6)
degrees = range(1, 11)
r2_train_scores = []
r2_test_scores = []

for degree in degrees:
    poly_model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    poly_model.fit(X_train, y_train)
    
    r2_train = poly_model.score(X_train, y_train)
    r2_test = poly_model.score(X_test, y_test)
    
    r2_train_scores.append(r2_train)
    r2_test_scores.append(r2_test)

plt.plot(degrees, r2_train_scores, 'o-', label='Training R²', linewidth=2)
plt.plot(degrees, r2_test_scores, 's-', label='Testing R²', linewidth=2)
plt.xlabel('Polynomial Degree')
plt.ylabel('R² Score')
plt.title('Model Complexity vs R²')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 7: Components breakdown
plt.subplot(3, 3, 7)
variance_explained = 1 - (ss_res / ss_tot)
variance_unexplained = ss_res / ss_tot

labels = ['Explained\nVariance', 'Unexplained\nVariance']
sizes = [variance_explained, variance_unexplained]
colors = ['lightblue', 'lightcoral']
explode = (0.1, 0)

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90)
plt.title(f'Variance Breakdown\n(R² = {r2_manual:.3f})')

# Plot 8: R² vs other metrics
plt.subplot(3, 3, 8)
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Generate models with different performance
models_performance = []
noise_levels = [1, 5, 10, 20, 30]

for noise in noise_levels:
    X_noise, y_noise = make_regression(n_samples=100, n_features=1, noise=noise, random_state=42)
    X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(X_noise, y_noise, test_size=0.3, random_state=42)
    
    model_n = LinearRegression()
    model_n.fit(X_train_n, y_train_n)
    y_pred_n = model_n.predict(X_test_n)
    
    r2_n = r2_score(y_test_n, y_pred_n)
    mse_n = mean_squared_error(y_test_n, y_pred_n)
    mae_n = mean_absolute_error(y_test_n, y_pred_n)
    
    models_performance.append((noise, r2_n, mse_n, mae_n))

noise_vals, r2_vals, mse_vals, mae_vals = zip(*models_performance)

# Normalize MSE and MAE for comparison
mse_norm = np.array(mse_vals) / max(mse_vals)
mae_norm = np.array(mae_vals) / max(mae_vals)

plt.plot(noise_vals, r2_vals, 'o-', label='R² Score', linewidth=2)
plt.plot(noise_vals, 1 - mse_norm, 's-', label='1 - Normalized MSE', linewidth=2)
plt.plot(noise_vals, 1 - mae_norm, '^-', label='1 - Normalized MAE', linewidth=2)
plt.xlabel('Noise Level')
plt.ylabel('Score')
plt.title('R² vs Other Metrics')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 9: Mathematical intuition
plt.subplot(3, 3, 9)
# Show what happens to R² with different prediction quality
sample_y = y_test[:20]  # First 20 samples
perfect_pred = sample_y.copy()
good_pred = sample_y + np.random.normal(0, np.std(sample_y) * 0.1, len(sample_y))
poor_pred = np.full_like(sample_y, np.mean(sample_y))  # Just predict the mean

r2_perfect = r2_score(sample_y, perfect_pred)
r2_good = r2_score(sample_y, good_pred)
r2_poor = r2_score(sample_y, poor_pred)

x_pos = range(len(sample_y))
plt.plot(x_pos, sample_y, 'ko-', label='Actual', linewidth=2, markersize=6)
plt.plot(x_pos, perfect_pred, 'g^-', label=f'Perfect (R²={r2_perfect:.2f})', alpha=0.7)
plt.plot(x_pos, good_pred, 'bs-', label=f'Good (R²={r2_good:.2f})', alpha=0.7)
plt.plot(x_pos, poor_pred, 'r--', label=f'Mean Prediction (R²={r2_poor:.2f})', alpha=0.7)

plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title('R² with Different Prediction Quality')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Mathematical properties of R²
print(f"\n{'='*60}")
print("MATHEMATICAL PROPERTIES OF R²")
print(f"{'='*60}")

print("1. Range and Interpretation:")
print("   • R² = 1.0: Perfect model (SS_res = 0)")
print("   • R² = 0.0: Model as good as predicting the mean")
print("   • R² < 0.0: Model worse than predicting the mean")
print("   • Higher R² indicates better fit")

print("\n2. Relationship to Correlation:")
print("   • For simple linear regression: R² = r² (correlation coefficient squared)")
print("   • For multiple regression: R² measures overall model fit")

# Demonstrate correlation relationship
from scipy.stats import pearsonr
correlation, _ = pearsonr(y_test.flatten(), y_pred.flatten())
print(f"   • Our example - Correlation (r): {correlation:.4f}")
print(f"   • Correlation squared (r²): {correlation**2:.4f}")
print(f"   • R² from formula: {r2_manual:.4f}")
print(f"   • Match: {np.isclose(correlation**2, r2_manual)}")

print("\n3. Alternative Formulations:")
ss_exp = ss_tot - ss_res  # Explained sum of squares
print(f"   • R² = SS_explained / SS_total = {ss_exp:.4f} / {ss_tot:.4f} = {ss_exp/ss_tot:.4f}")
print(f"   • R² = 1 - SS_residual / SS_total = 1 - {ss_res:.4f} / {ss_tot:.4f} = {1 - ss_res/ss_tot:.4f}")

print("\n4. Limitations:")
print("   • Can be artificially inflated by adding more variables")
print("   • Doesn't indicate if model is appropriate")
print("   • High R² doesn't guarantee good predictions on new data")
print("   • Can be misleading with nonlinear relationships")

# Demonstrate limitations
print(f"\n{'='*60}")
print("DEMONSTRATING R² LIMITATIONS")
print(f"{'='*60}")

# Overfitting example
print("Overfitting Example:")
high_degree_model = Pipeline([
    ('poly', PolynomialFeatures(degree=15)),
    ('linear', LinearRegression())
])
high_degree_model.fit(X_train, y_train)

r2_train_overfit = high_degree_model.score(X_train, y_train)
r2_test_overfit = high_degree_model.score(X_test, y_test)

print(f"  High-degree polynomial (degree 15):")
print(f"  Training R²: {r2_train_overfit:.4f}")
print(f"  Testing R²:  {r2_test_overfit:.4f}")
print(f"  Difference:  {r2_train_overfit - r2_test_overfit:.4f}")
print("  → High training R² but poor generalization!")

# Anscombe's quartet demonstration concept
print(f"\nNon-linear relationship limitation:")
print("  R² might be same for different relationships:")
print("  • Linear relationship")
print("  • Quadratic relationship")  
print("  • Relationship with outliers")
print("  → Always visualize your data!")

print(f"\n{'='*60}")
print("WHEN TO USE R² AND ALTERNATIVES")
print(f"{'='*60}")

print("Use R² when:")
print("  ✓ You want to understand proportion of variance explained")
print("  ✓ Comparing models with same number of parameters")
print("  ✓ Working with linear relationships")
print("  ✓ You need an interpretable goodness-of-fit measure")

print("\nConsider alternatives when:")
print("  • Comparing models with different numbers of parameters → Adjusted R²")
print("  • Dealing with nonlinear relationships → Other metrics")
print("  • Focused on prediction accuracy → MSE, MAE, RMSE")
print("  • Working with time series → Specialized metrics")

print(f"\nAdjusted R² formula:")
n = len(y_test)
p = X_test.shape[1]  # number of predictors
adj_r2 = 1 - ((1 - r2_manual) * (n - 1) / (n - p - 1))
print(f"  Adjusted R² = 1 - ((1 - R²) × (n-1) / (n-p-1))")
print(f"  Adjusted R² = 1 - ((1 - {r2_manual:.4f}) × ({n}-1) / ({n}-{p}-1))")
print(f"  Adjusted R² = {adj_r2:.4f}")
print(f"  (Penalizes adding more variables)")

print(f"\n{'='*60}")
print("HOMEWORK COMPLETION SUMMARY")
print(f"{'='*60}")
print("✓ Explained what R² score measures")
print("✓ Provided mathematical formula and derivation")
print("✓ Demonstrated manual calculation step-by-step")
print("✓ Verified calculations against sklearn")
print("✓ Visualized R² components and interpretation")
print("✓ Discussed range and meaning of R² values")
print("✓ Showed relationship to correlation coefficient")
print("✓ Demonstrated limitations and when to use alternatives")
print("✓ Provided practical interpretation guidelines")

print(f"\nKey Takeaways:")
print("1. R² measures proportion of variance explained by the model")
print("2. Formula: R² = 1 - (SS_residual / SS_total)")
print("3. Range: -∞ to 1.0, where 1.0 is perfect fit")
print("4. Higher R² generally indicates better model fit")
print("5. Can be misleading with overfitting or nonlinear relationships")
print("6. Should be used alongside other metrics and visualization")
print("7. Adjusted R² is better for comparing models with different complexity")
