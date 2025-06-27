"""
Example 7 - Sigmoid Activation Function in Binary Classification
Demonstrate how to use sigmoid activation function in binary classification
"""

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    """
    Sigmoid activation function: f(z) = 1 / (1 + e^(-z))
    """
    # Prevent overflow by clipping z
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    """
    Derivative of sigmoid: f'(z) = f(z) * (1 - f(z))
    """
    s = sigmoid(z)
    return s * (1 - s)

# Example from the chapter
print("=== Chapter Example ===")
z = 0.5
sigmoid_output = sigmoid(z)
print(f"Input (z): {z}")
print(f"Sigmoid output f(z): {sigmoid_output:.3f}")
print(f"Since f(z) = {sigmoid_output:.3f} >= 0.5, predicted class: 1")

# Binary classification decision rule
def predict_binary(z, threshold=0.5):
    """Make binary prediction using sigmoid with threshold"""
    prob = sigmoid(z)
    prediction = 1 if prob >= threshold else 0
    return prediction, prob

print("\n" + "="*60)
print("SIGMOID FUNCTION ANALYSIS")
print("="*60)

# Analyze sigmoid behavior for different inputs
z_values = np.linspace(-10, 10, 1000)
sigmoid_values = sigmoid(z_values)
derivative_values = sigmoid_derivative(z_values)

plt.figure(figsize=(15, 12))

# Plot 1: Sigmoid function
plt.subplot(3, 3, 1)
plt.plot(z_values, sigmoid_values, 'b-', linewidth=3, label='σ(z) = 1/(1+e^(-z))')
plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Decision Boundary')
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
plt.xlabel('Input (z)')
plt.ylabel('Sigmoid Output')
plt.title('Sigmoid Activation Function')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Sigmoid derivative
plt.subplot(3, 3, 2)
plt.plot(z_values, derivative_values, 'g-', linewidth=3, label="σ'(z) = σ(z)(1-σ(z))")
plt.xlabel('Input (z)')
plt.ylabel('Derivative')
plt.title('Sigmoid Derivative')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Examples of binary classification
plt.subplot(3, 3, 3)
example_z = [-3, -1, 0, 0.5, 1, 3]
example_probs = [sigmoid(z) for z in example_z]
example_preds = [predict_binary(z)[0] for z in example_z]

colors = ['red' if pred == 0 else 'blue' for pred in example_preds]
plt.scatter(example_z, example_probs, c=colors, s=100, alpha=0.7)
plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.7, label='Threshold = 0.5')

for i, (z, prob, pred) in enumerate(zip(example_z, example_probs, example_preds)):
    plt.annotate(f'z={z}\np={prob:.3f}\nclass={pred}', 
                 (z, prob), textcoords="offset points", 
                 xytext=(0,20), ha='center', fontsize=8)

plt.xlabel('Input (z)')
plt.ylabel('Probability')
plt.title('Binary Classification Examples')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Compare different activation functions
plt.subplot(3, 3, 4)
def tanh_func(z):
    return np.tanh(z)

def relu_func(z):
    return np.maximum(0, z)

plt.plot(z_values, sigmoid_values, 'b-', linewidth=2, label='Sigmoid')
plt.plot(z_values, tanh_func(z_values), 'r-', linewidth=2, label='Tanh')
plt.plot(z_values, relu_func(z_values), 'g-', linewidth=2, label='ReLU')
plt.xlabel('Input (z)')
plt.ylabel('Output')
plt.title('Activation Functions Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 5: Sigmoid with different parameters (scaled sigmoid)
plt.subplot(3, 3, 5)
scales = [0.5, 1, 2, 5]
for scale in scales:
    scaled_sigmoid = sigmoid(scale * z_values)
    plt.plot(z_values, scaled_sigmoid, linewidth=2, label=f'σ({scale}z)')

plt.xlabel('Input (z)')
plt.ylabel('Sigmoid Output')
plt.title('Sigmoid with Different Scales')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 6: Effect of bias term
plt.subplot(3, 3, 6)
biases = [-2, -1, 0, 1, 2]
for bias in biases:
    biased_sigmoid = sigmoid(z_values + bias)
    plt.plot(z_values, biased_sigmoid, linewidth=2, label=f'σ(z+{bias})')

plt.xlabel('Input (z)')
plt.ylabel('Sigmoid Output')
plt.title('Sigmoid with Different Bias Terms')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 7: Probability interpretation
plt.subplot(3, 3, 7)
# Generate sample data
np.random.seed(42)
n_samples = 100
z_positive = np.random.normal(2, 1, n_samples//2)  # Positive class centered at z=2
z_negative = np.random.normal(-2, 1, n_samples//2)  # Negative class centered at z=-2

prob_positive = sigmoid(z_positive)
prob_negative = sigmoid(z_negative)

plt.hist(prob_positive, bins=20, alpha=0.7, label='Positive Class', color='blue')
plt.hist(prob_negative, bins=20, alpha=0.7, label='Negative Class', color='red')
plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.7, label='Decision Boundary')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.title('Probability Distribution by Class')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 8: Gradient behavior
plt.subplot(3, 3, 8)
# Show how gradient changes with input
z_gradient = np.linspace(-6, 6, 100)
grad_values = sigmoid_derivative(z_gradient)

plt.plot(z_gradient, grad_values, 'purple', linewidth=3)
plt.fill_between(z_gradient, grad_values, alpha=0.3, color='purple')
plt.xlabel('Input (z)')
plt.ylabel('Gradient')
plt.title('Sigmoid Gradient (Vanishing Gradient)')
plt.grid(True, alpha=0.3)

# Add annotation for vanishing gradient
max_grad_idx = np.argmax(grad_values)
plt.annotate(f'Max gradient: {grad_values[max_grad_idx]:.3f}\nat z = {z_gradient[max_grad_idx]:.1f}',
             xy=(z_gradient[max_grad_idx], grad_values[max_grad_idx]),
             xytext=(2, 0.2), fontsize=10,
             arrowprops=dict(arrowstyle='->', color='red'))

# Plot 9: Real-world example
plt.subplot(3, 3, 9)
# Simulate logistic regression on 2D data
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, 
                          n_informative=2, n_clusters_per_class=1, random_state=42)

# Fit logistic regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
lr = LogisticRegression()
lr.fit(X_scaled, y)

# Get decision boundary
xx, yy = np.meshgrid(np.linspace(X_scaled[:, 0].min()-1, X_scaled[:, 0].max()+1, 100),
                     np.linspace(X_scaled[:, 1].min()-1, X_scaled[:, 1].max()+1, 100))

# Calculate probabilities for each point
grid_points = np.c_[xx.ravel(), yy.ravel()]
probabilities = lr.predict_proba(grid_points)[:, 1].reshape(xx.shape)

# Plot
plt.contourf(xx, yy, probabilities, levels=50, alpha=0.6, cmap='RdYlBu')
plt.colorbar(label='P(Class=1)')
plt.contour(xx, yy, probabilities, levels=[0.5], colors='black', linestyles='--', linewidths=2)

# Plot data points
colors = ['red', 'blue']
for i in range(2):
    mask = y == i
    plt.scatter(X_scaled[mask, 0], X_scaled[mask, 1], 
               c=colors[i], label=f'Class {i}', alpha=0.7)

plt.xlabel('Feature 1 (scaled)')
plt.ylabel('Feature 2 (scaled)')
plt.title('Logistic Regression Decision Boundary')
plt.legend()

plt.tight_layout()
plt.show()

print(f"\nDetailed Examples:")
test_cases = [
    (-5, "Very negative input"),
    (-2, "Negative input"), 
    (-0.5, "Slightly negative"),
    (0, "Zero input"),
    (0.5, "Chapter example"),
    (2, "Positive input"),
    (5, "Very positive input")
]

print(f"{'Input (z)':<12} {'Sigmoid(z)':<12} {'Prediction':<12} {'Description':<20}")
print("-" * 60)

for z, description in test_cases:
    prob = sigmoid(z)
    pred = 1 if prob >= 0.5 else 0
    print(f"{z:<12} {prob:<12.6f} {pred:<12} {description:<20}")

print(f"\nKey Properties of Sigmoid:")
print(f"1. Output range: (0, 1) - suitable for probabilities")
print(f"2. S-shaped curve - smooth transition between classes")
print(f"3. Derivative has maximum at z=0, causing vanishing gradient problem")
print(f"4. Saturates at extreme values (approaches 0 or 1)")
print(f"5. Computationally stable with proper implementation")
