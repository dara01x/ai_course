"""
Example 6 - Binary Cross-Entropy (BCE) Loss
Demonstrate how to use BCE in binary classification

NOTE: This example has been enhanced from the original lecture version.
CHANGES:
1. Fixed array broadcasting issue in probability distribution visualization
2. Enhanced probability generation for good vs bad model comparison
REASONS:
- The original np.where usage caused broadcasting errors with incompatible array shapes
- Fixed by using proper indexing with boolean masks for different classes
- Ensures the visualization works correctly for all sample sizes
"""

import numpy as np
import matplotlib.pyplot as plt

def binary_cross_entropy(y_true, y_pred):
    """
    Calculate Binary Cross-Entropy Loss
    BCE = -1/N * Σ[y*log(p) + (1-y)*log(1-p)]
    """
    # Avoid log(0) by adding small epsilon
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    bce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return bce

def binary_cross_entropy_single(y, p):
    """Calculate BCE for single prediction"""
    epsilon = 1e-15
    p = np.clip(p, epsilon, 1 - epsilon)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))

# Example from the chapter
print("=== Example from Chapter ===")
y = np.array([1, 0, 1])  # True labels
p = np.array([0.9, 0.1, 0.8])  # Predicted probabilities

print(f"True labels (y): {y}")
print(f"Predicted probabilities (p): {p}")

# Calculate BCE step by step
bce_manual = 0
for i in range(len(y)):
    single_bce = binary_cross_entropy_single(y[i], p[i])
    print(f"Sample {i+1}: y={y[i]}, p={p[i]:.1f}, BCE={single_bce:.4f}")
    bce_manual += single_bce

bce_manual /= len(y)
print(f"\nManual BCE calculation: {bce_manual:.4f}")

# Using the function
bce_function = binary_cross_entropy(y, p)
print(f"Function BCE calculation: {bce_function:.4f}")

# Verify with the chapter calculation
chapter_bce = -1/3 * (1*np.log(0.9) + 0*np.log(1-0.1) + 1*np.log(0.8))
print(f"Chapter calculation: {chapter_bce:.4f}")

print("\n" + "="*60)
print("DETAILED ANALYSIS OF BCE BEHAVIOR")
print("="*60)

# Analyze BCE behavior for different prediction values
y_true_cases = [0, 1]  # True labels
p_range = np.linspace(0.001, 0.999, 1000)  # Predicted probabilities

plt.figure(figsize=(15, 10))

# Plot BCE for both cases
for i, y_true in enumerate(y_true_cases):
    plt.subplot(2, 3, i+1)
    bce_values = [binary_cross_entropy_single(y_true, p) for p in p_range]
    plt.plot(p_range, bce_values, linewidth=2)
    plt.xlabel('Predicted Probability (p)')
    plt.ylabel('BCE Loss')
    plt.title(f'BCE Loss when True Label = {y_true}')
    plt.grid(True, alpha=0.3)
    
    # Add annotations for key points
    if y_true == 1:
        plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='p=0.5 (uncertain)')
        plt.axvline(x=1.0, color='green', linestyle='--', alpha=0.7, label='p=1.0 (perfect)')
    else:
        plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='p=0.5 (uncertain)')
        plt.axvline(x=0.0, color='green', linestyle='--', alpha=0.7, label='p=0.0 (perfect)')
    plt.legend()

# Compare good vs bad predictions
plt.subplot(2, 3, 3)
scenarios = [
    ("Perfect predictions", [1, 0, 1], [0.99, 0.01, 0.99]),
    ("Good predictions", [1, 0, 1], [0.8, 0.2, 0.9]),
    ("Poor predictions", [1, 0, 1], [0.6, 0.4, 0.7]),
    ("Random predictions", [1, 0, 1], [0.5, 0.5, 0.5]),
    ("Opposite predictions", [1, 0, 1], [0.1, 0.9, 0.2])
]

scenario_names = []
bce_values = []

for name, y_true, y_pred in scenarios:
    bce = binary_cross_entropy(np.array(y_true), np.array(y_pred))
    scenario_names.append(name)
    bce_values.append(bce)
    print(f"{name}: BCE = {bce:.4f}")

plt.bar(range(len(scenario_names)), bce_values, color=['green', 'lightgreen', 'yellow', 'orange', 'red'])
plt.xlabel('Prediction Quality')
plt.ylabel('BCE Loss')
plt.title('BCE Loss for Different Prediction Qualities')
plt.xticks(range(len(scenario_names)), [name.split()[0] for name in scenario_names], rotation=45)
plt.grid(True, alpha=0.3)

# Probability distribution visualization
plt.subplot(2, 3, 4)
n_samples = 100
y_true = np.random.choice([0, 1], n_samples)

# Create good predictions: high probability for class 1, low for class 0
y_pred_good = np.zeros(n_samples)
mask_positive = y_true == 1
mask_negative = y_true == 0
y_pred_good[mask_positive] = np.random.beta(8, 2, np.sum(mask_positive))
y_pred_good[mask_negative] = np.random.beta(2, 8, np.sum(mask_negative))

y_pred_bad = np.random.uniform(0.3, 0.7, n_samples)

bce_good = binary_cross_entropy(y_true, y_pred_good)
bce_bad = binary_cross_entropy(y_true, y_pred_bad)

plt.hist(y_pred_good, bins=20, alpha=0.7, label=f'Good Model (BCE={bce_good:.3f})', color='green')
plt.hist(y_pred_bad, bins=20, alpha=0.7, label=f'Bad Model (BCE={bce_bad:.3f})', color='red')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.title('Prediction Distribution Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

# Effect of class imbalance
plt.subplot(2, 3, 5)
class_ratios = np.arange(0.1, 1.0, 0.1)
bce_balanced = []
bce_imbalanced = []

for ratio in class_ratios:
    n_pos = int(100 * ratio)
    n_neg = 100 - n_pos
    
    # Balanced model (same performance on both classes)
    y_true = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    y_pred_balanced = np.concatenate([np.random.beta(7, 3, n_pos), np.random.beta(3, 7, n_neg)])
    
    # Imbalanced model (biased towards majority class)
    y_pred_imbalanced = np.full(100, 0.5)  # Always predicts 50%
    
    bce_balanced.append(binary_cross_entropy(y_true, y_pred_balanced))
    bce_imbalanced.append(binary_cross_entropy(y_true, y_pred_imbalanced))

plt.plot(class_ratios, bce_balanced, 'o-', label='Good Model', linewidth=2)
plt.plot(class_ratios, bce_imbalanced, 's-', label='Always 50% Model', linewidth=2)
plt.xlabel('Positive Class Ratio')
plt.ylabel('BCE Loss')
plt.title('BCE vs Class Imbalance')
plt.legend()
plt.grid(True, alpha=0.3)

# Decision boundary visualization
plt.subplot(2, 3, 6)
thresholds = np.arange(0.1, 0.9, 0.1)
y_true_example = np.array([1, 0, 1, 0, 1])
y_pred_example = np.array([0.8, 0.3, 0.9, 0.2, 0.6])

accuracies = []
for threshold in thresholds:
    y_pred_binary = (y_pred_example >= threshold).astype(int)
    accuracy = np.mean(y_true_example == y_pred_binary)
    accuracies.append(accuracy)

plt.plot(thresholds, accuracies, 'o-', linewidth=2, markersize=8)
plt.xlabel('Decision Threshold')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Decision Threshold')
plt.grid(True, alpha=0.3)

# Add text showing the example
textstr = f'Example:\nTrue: {y_true_example}\nPred: {y_pred_example}'
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

print(f"\nKey Insights:")
print(f"1. BCE loss approaches 0 when predictions are perfect")
print(f"2. BCE loss approaches infinity when predictions are completely wrong")
print(f"3. BCE is steeper near extreme values, providing stronger gradients")
print(f"4. The loss is symmetric around p=0.5 for balanced datasets")
