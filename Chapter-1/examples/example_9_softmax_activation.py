"""
Example 9 - Softmax Activation Function
Demonstrate softmax function for multi-class classification
Chapter example: z = [5, 1, 0, 2]

NOTE: This example has been enhanced from the original lecture version.
CHANGES:
1. Added numerical stability by subtracting max value before exponentiation
2. Enhanced with comprehensive visualizations and practical examples
3. Added temperature scaling and multi-class classification examples
REASONS:
- The original softmax can cause overflow for large input values
- Subtracting max(z) prevents numerical overflow while maintaining mathematical correctness
- Enhanced educational content shows practical applications and variations
"""

import numpy as np
import matplotlib.pyplot as plt

def softmax(z):
    """
    Softmax activation function: f(z_i) = e^(z_i) / Σ(e^(z_j))
    """
    # Subtract max for numerical stability
    z_stable = z - np.max(z)
    exp_z = np.exp(z_stable)
    return exp_z / np.sum(exp_z)

def softmax_derivative(z):
    """
    Derivative of softmax (Jacobian matrix)
    """
    s = softmax(z)
    jacobian = np.zeros((len(s), len(s)))
    for i in range(len(s)):
        for j in range(len(s)):
            if i == j:
                jacobian[i, j] = s[i] * (1 - s[i])
            else:
                jacobian[i, j] = -s[i] * s[j]
    return jacobian

# Chapter Example
print("=== Chapter Example ===")
z = np.array([5, 1, 0, 2])
print(f"Input logits (z): {z}")

# Step-by-step calculation as shown in chapter
exp_z = np.exp(z)
print(f"Exponentials (e^z): {exp_z}")
print(f"  e^5 = {exp_z[0]:.2f}")
print(f"  e^1 = {exp_z[1]:.2f}")
print(f"  e^0 = {exp_z[2]:.2f}")
print(f"  e^2 = {exp_z[3]:.2f}")

sum_exp_z = np.sum(exp_z)
print(f"Sum of exponentials: {sum_exp_z:.2f}")

probabilities = exp_z / sum_exp_z
print(f"Softmax probabilities: {probabilities}")
print(f"  Class 0: {probabilities[0]:.3f} ({probabilities[0]*100:.1f}%)")
print(f"  Class 1: {probabilities[1]:.3f} ({probabilities[1]*100:.1f}%)")
print(f"  Class 2: {probabilities[2]:.3f} ({probabilities[2]*100:.1f}%)")
print(f"  Class 3: {probabilities[3]:.3f} ({probabilities[3]*100:.1f}%)")

print(f"\nPredicted class: {np.argmax(probabilities)} (highest probability)")
print(f"Probability sum: {np.sum(probabilities):.6f} (should be 1.0)")

# Verify with softmax function
softmax_result = softmax(z)
print(f"Function result: {softmax_result}")
print(f"Match chapter calculation: {np.allclose(probabilities, softmax_result)}")

print(f"\n{'='*70}")
print("COMPREHENSIVE SOFTMAX ANALYSIS")
print(f"{'='*70}")

plt.figure(figsize=(16, 12))

# Plot 1: Chapter example visualization
plt.subplot(3, 4, 1)
classes = ['Class 0', 'Class 1', 'Class 2', 'Class 3']
colors = ['red', 'blue', 'green', 'orange']
bars = plt.bar(classes, probabilities, color=colors, alpha=0.7)
plt.ylabel('Probability')
plt.title('Chapter Example: Softmax Output')
plt.ylim(0, 1)

# Add value labels on bars
for bar, prob in zip(bars, probabilities):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')

plt.grid(True, alpha=0.3)

# Plot 2: Effect of temperature (scaling)
plt.subplot(3, 4, 2)
temperatures = [0.5, 1.0, 2.0, 5.0]
for temp in temperatures:
    scaled_probs = softmax(z / temp)
    plt.plot(classes, scaled_probs, 'o-', linewidth=2, label=f'T={temp}')

plt.ylabel('Probability')
plt.title('Temperature Scaling Effect')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Plot 3: Logit values vs probabilities
plt.subplot(3, 4, 3)
logit_range = np.linspace(-2, 6, 100)
prob_curves = []

for i, base_logit in enumerate([0, 1, 2]):  # Fix other logits
    probs_for_varying = []
    for varying_logit in logit_range:
        temp_z = z.copy()
        temp_z[0] = varying_logit  # Vary first logit
        temp_probs = softmax(temp_z)
        probs_for_varying.append(temp_probs[0])  # Track first class probability
    
    plt.plot(logit_range, probs_for_varying, linewidth=2, 
             label=f'P(Class 0) when others fixed')
    break  # Just show one example

plt.xlabel('Logit Value')
plt.ylabel('Probability')
plt.title('Logit vs Probability Relationship')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Comparison with other normalizations
plt.subplot(3, 4, 4)
# L1 normalization
l1_norm = np.abs(z) / np.sum(np.abs(z))
# L2 normalization  
l2_norm = z**2 / np.sum(z**2)
# Simple normalization (min-max)
simple_norm = (z - np.min(z)) / (np.max(z) - np.min(z))

x_pos = np.arange(len(z))
width = 0.2

plt.bar(x_pos - 1.5*width, probabilities, width, label='Softmax', alpha=0.8)
plt.bar(x_pos - 0.5*width, l1_norm, width, label='L1 Norm', alpha=0.8)
plt.bar(x_pos + 0.5*width, l2_norm, width, label='L2 Norm', alpha=0.8)
plt.bar(x_pos + 1.5*width, simple_norm, width, label='Min-Max', alpha=0.8)

plt.xlabel('Class Index')
plt.ylabel('Normalized Value')
plt.title('Different Normalization Methods')
plt.xticks(x_pos, [f'C{i}' for i in range(len(z))])
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 5: Numerical stability demonstration
plt.subplot(3, 4, 5)
large_z = np.array([1000, 1001, 999, 1002])  # Large values that could cause overflow
print(f"\nNumerical Stability Test:")
print(f"Large logits: {large_z}")

# Naive implementation (would overflow)
try:
    naive_exp = np.exp(large_z)
    print(f"Naive exp(): {naive_exp} (likely overflow)")
except:
    print("Naive implementation causes overflow!")

# Stable implementation
stable_probs = softmax(large_z)
print(f"Stable softmax: {stable_probs}")

# Demonstrate the stability trick
large_z_stable = large_z - np.max(large_z)
print(f"After subtracting max: {large_z_stable}")

categories = ['Naive\n(overflow)', 'Stable\n(works)']
heights = [0, 1]  # 0 for failed, 1 for success
colors_stable = ['red', 'green']
bars = plt.bar(categories, heights, color=colors_stable, alpha=0.7)
plt.ylabel('Success')
plt.title('Numerical Stability')
plt.ylim(0, 1.2)

for bar, height in zip(bars, heights):
    status = 'FAILS' if height == 0 else 'WORKS'
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.05, 
             status, ha='center', va='bottom', fontweight='bold')

# Plot 6: Cross-entropy loss with softmax
plt.subplot(3, 4, 6)
def cross_entropy_loss(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-15))

# Example: true class is class 0
y_true = np.array([1, 0, 0, 0])  # One-hot encoded

# Vary the logit of true class and see how loss changes
true_class_logits = np.linspace(-5, 10, 100)
losses = []

for true_logit in true_class_logits:
    temp_z = np.array([true_logit, 1, 0, 2])  # Vary true class logit
    y_pred = softmax(temp_z)
    loss = cross_entropy_loss(y_true, y_pred)
    losses.append(loss)

plt.plot(true_class_logits, losses, linewidth=3, color='purple')
plt.xlabel('True Class Logit')
plt.ylabel('Cross-Entropy Loss')
plt.title('Loss vs True Class Confidence')
plt.grid(True, alpha=0.3)

# Mark the chapter example point
chapter_loss = cross_entropy_loss(y_true, probabilities)
plt.axvline(x=z[0], color='red', linestyle='--', alpha=0.7)
plt.axhline(y=chapter_loss, color='red', linestyle='--', alpha=0.7)
plt.plot(z[0], chapter_loss, 'ro', markersize=10, label=f'Chapter example\nLoss={chapter_loss:.3f}')
plt.legend()

# Plot 7: Softmax derivative visualization
plt.subplot(3, 4, 7)
jacobian = softmax_derivative(z)
im = plt.imshow(jacobian, cmap='RdBu', aspect='auto')
plt.colorbar(im, shrink=0.6)
plt.xlabel('j (output index)')
plt.ylabel('i (input index)')
plt.title('Softmax Jacobian Matrix')
plt.xticks(range(len(z)), [f'z{i}' for i in range(len(z))])
plt.yticks(range(len(z)), [f'∂/∂z{i}' for i in range(len(z))])

# Add text annotations
for i in range(len(z)):
    for j in range(len(z)):
        plt.text(j, i, f'{jacobian[i,j]:.3f}', ha='center', va='center', 
                fontsize=8, color='white' if abs(jacobian[i,j]) > 0.1 else 'black')

# Plot 8: Multi-class classification example
plt.subplot(3, 4, 8)
# Simulate a 3-class classification problem
np.random.seed(42)
n_samples = 300
n_classes = 3

# Create some 2D data for visualization
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0,
                          n_informative=2, n_classes=n_classes, n_clusters_per_class=1,
                          random_state=42)

# Train a simple classifier and get logits
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(multi_class='multinomial', solver='lbfgs')
clf.fit(X, y)

# Get decision function (logits) for visualization
xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 50),
                     np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, 50))
grid_points = np.c_[xx.ravel(), yy.ravel()]
logits = clf.decision_function(grid_points)

# Apply softmax to get probabilities
probs = np.array([softmax(logit) for logit in logits])
predicted_class = np.argmax(probs, axis=1).reshape(xx.shape)

# Plot decision regions
plt.contourf(xx, yy, predicted_class, alpha=0.4, cmap='viridis')
colors = ['red', 'blue', 'green']
for i in range(n_classes):
    mask = y == i
    plt.scatter(X[mask, 0], X[mask, 1], c=colors[i], label=f'Class {i}', alpha=0.6)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Multi-class Classification\nwith Softmax')
plt.legend()

# Plot 9: Confidence visualization
plt.subplot(3, 4, 9)
# Show confidence (max probability) across the decision space
max_probs = np.max(probs, axis=1).reshape(xx.shape)
contour = plt.contourf(xx, yy, max_probs, levels=20, cmap='viridis', alpha=0.6)
plt.colorbar(contour, shrink=0.6, label='Confidence')
plt.contour(xx, yy, max_probs, levels=[0.5, 0.7, 0.9], colors='white', linestyles='--')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Prediction Confidence Map')

# Plot 10: Comparison of different scenarios
plt.subplot(3, 4, 10)
scenarios = [
    ("Confident", [5, 1, 0, 2]),      # Chapter example
    ("Less confident", [2, 1.5, 1, 1.8]),
    ("Very uncertain", [1, 1, 1, 1]),
    ("Extreme confident", [10, 0, 0, 0])
]

x_pos = np.arange(len(scenarios))
max_probs_scenarios = []
entropies = []

for name, logits in scenarios:
    probs = softmax(np.array(logits))
    max_prob = np.max(probs)
    entropy = -np.sum(probs * np.log(probs + 1e-15))
    max_probs_scenarios.append(max_prob)
    entropies.append(entropy)

# Plot max probability (confidence)
bars1 = plt.bar(x_pos - 0.2, max_probs_scenarios, 0.4, label='Max Probability', alpha=0.7)
# Plot entropy (uncertainty) - normalize to [0,1] for comparison
normalized_entropies = np.array(entropies) / np.log(4)  # Normalize by max possible entropy
bars2 = plt.bar(x_pos + 0.2, normalized_entropies, 0.4, label='Normalized Entropy', alpha=0.7)

plt.xlabel('Scenario')
plt.ylabel('Value')
plt.title('Confidence vs Uncertainty')
plt.xticks(x_pos, [s[0] for s in scenarios], rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

# Add value labels
for bars, values in [(bars1, max_probs_scenarios), (bars2, normalized_entropies)]:
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)

# Plot 11: Softmax vs other activation combinations
plt.subplot(3, 4, 11)
# Compare softmax with individual sigmoids (not normalized)
sigmoid_outputs = 1 / (1 + np.exp(-z))
sigmoid_normalized = sigmoid_outputs / np.sum(sigmoid_outputs)

x_pos = np.arange(len(z))
width = 0.35

plt.bar(x_pos - width/2, probabilities, width, label='Softmax', alpha=0.8, color='blue')
plt.bar(x_pos + width/2, sigmoid_normalized, width, label='Normalized Sigmoid', alpha=0.8, color='red')

plt.xlabel('Class')
plt.ylabel('Probability')
plt.title('Softmax vs Normalized Sigmoid')
plt.xticks(x_pos, [f'C{i}' for i in range(len(z))])
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 12: Real-world interpretation
plt.subplot(3, 4, 12)
# Example: Image classification scores
class_names = ['Cat', 'Dog', 'Bird', 'Fish']
# Simulate CNN output logits
cnn_logits = z  # Use chapter example

probs = softmax(cnn_logits)
sorted_indices = np.argsort(probs)[::-1]  # Sort by probability (descending)

# Create horizontal bar chart
y_pos = np.arange(len(class_names))
colors_sorted = plt.cm.viridis(np.linspace(0, 1, len(class_names)))

bars = plt.barh(y_pos, [probs[i] for i in sorted_indices], 
                color=colors_sorted, alpha=0.7)

plt.yticks(y_pos, [class_names[i] for i in sorted_indices])
plt.xlabel('Probability')
plt.title('Image Classification Results')
plt.xlim(0, 1)

# Add percentage labels
for i, (bar, idx) in enumerate(zip(bars, sorted_indices)):
    prob = probs[idx]
    plt.text(prob + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{prob*100:.1f}%', va='center', fontweight='bold')

plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Additional analysis
print(f"\n{'='*70}")
print("DETAILED SOFTMAX PROPERTIES")
print(f"{'='*70}")

print(f"Chapter example analysis:")
print(f"Input logits: {z}")
print(f"Output probabilities: {probabilities}")
print(f"Winner-takes-all prediction: Class {np.argmax(probabilities)}")
print(f"Confidence (max probability): {np.max(probabilities):.1%}")
print(f"Entropy (uncertainty): {-np.sum(probabilities * np.log(probabilities)):.3f}")
print(f"Max possible entropy: {np.log(len(z)):.3f}")

print(f"\nSoftmax key properties:")
print(f"1. Sum of outputs: {np.sum(probabilities):.6f} (always 1.0)")
print(f"2. All outputs positive: {np.all(probabilities > 0)}")
print(f"3. Monotonic: higher logit → higher probability")
print(f"4. Differentiable everywhere")
print(f"5. Temperature sensitive (sharpness controllable)")

# Temperature analysis
print(f"\nTemperature scaling analysis:")
temperatures = [0.1, 0.5, 1.0, 2.0, 10.0]
for T in temperatures:
    scaled_probs = softmax(z / T)
    entropy = -np.sum(scaled_probs * np.log(scaled_probs + 1e-15))
    print(f"T = {T:4.1f}: max_prob = {np.max(scaled_probs):.3f}, entropy = {entropy:.3f}")

print(f"\nGradient analysis:")
jacobian = softmax_derivative(z)
print(f"Jacobian diagonal (∂p_i/∂z_i): {np.diag(jacobian)}")
print(f"Jacobian off-diagonal (∂p_i/∂z_j, i≠j): {jacobian[0,1]:.4f} (example)")
print(f"Max gradient magnitude: {np.max(np.abs(jacobian)):.4f}")
print(f"Gradient norm: {np.linalg.norm(jacobian):.4f}")
