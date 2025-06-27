"""
Example 8 - Tanh (Hyperbolic Tangent) Activation Function
Demonstrate the tanh activation function and compare with sigmoid

NOTE: This example has been enhanced from the original lecture version.
CHANGES:
1. Added comprehensive comparisons with sigmoid and other activation functions
2. Enhanced visualizations showing derivatives and mathematical properties
3. Added practical examples and use cases
REASONS:
- Provides deeper understanding of activation functions beyond basic definition
- Shows the relationship between different activation functions
- Includes practical considerations for choosing activation functions
"""

import numpy as np
import matplotlib.pyplot as plt

def tanh_func(z):
    """
    Tanh activation function: f(z) = (e^z - e^(-z)) / (e^z + e^(-z))
    Equivalent to: f(z) = 2*sigmoid(2*z) - 1
    """
    return np.tanh(z)

def tanh_derivative(z):
    """
    Derivative of tanh: f'(z) = 1 - tanh²(z)
    """
    return 1 - np.tanh(z)**2

def sigmoid(z):
    """Sigmoid function for comparison"""
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    """Sigmoid derivative for comparison"""
    s = sigmoid(z)
    return s * (1 - s)

print("=== Tanh Activation Function Analysis ===")

# Test values
z_values = np.linspace(-5, 5, 1000)
tanh_values = tanh_func(z_values)
tanh_deriv_values = tanh_derivative(z_values)
sigmoid_values = sigmoid(z_values)
sigmoid_deriv_values = sigmoid_derivative(z_values)

plt.figure(figsize=(15, 12))

# Plot 1: Tanh function
plt.subplot(3, 3, 1)
plt.plot(z_values, tanh_values, 'b-', linewidth=3, label='tanh(z)')
plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='y = 0')
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
plt.xlabel('Input (z)')
plt.ylabel('Tanh Output')
plt.title('Tanh Activation Function')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-1.2, 1.2)

# Plot 2: Tanh vs Sigmoid comparison
plt.subplot(3, 3, 2)
plt.plot(z_values, tanh_values, 'b-', linewidth=3, label='tanh(z) ∈ [-1, 1]')
plt.plot(z_values, sigmoid_values, 'r-', linewidth=3, label='sigmoid(z) ∈ [0, 1]')
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
plt.xlabel('Input (z)')
plt.ylabel('Output')
plt.title('Tanh vs Sigmoid')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Derivatives comparison
plt.subplot(3, 3, 3)
plt.plot(z_values, tanh_deriv_values, 'b-', linewidth=3, label="tanh'(z)")
plt.plot(z_values, sigmoid_deriv_values, 'r-', linewidth=3, label="sigmoid'(z)")
plt.xlabel('Input (z)')
plt.ylabel('Derivative')
plt.title('Derivatives Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

# Print max derivatives
max_tanh_deriv = np.max(tanh_deriv_values)
max_sigmoid_deriv = np.max(sigmoid_deriv_values)
print(f"Maximum tanh derivative: {max_tanh_deriv:.3f}")
print(f"Maximum sigmoid derivative: {max_sigmoid_deriv:.3f}")
print(f"Tanh derivative is {max_tanh_deriv/max_sigmoid_deriv:.1f}x larger than sigmoid")

# Plot 4: Zero-centered nature of tanh
plt.subplot(3, 3, 4)
# Generate random data
np.random.seed(42)
n_samples = 1000
z_input = np.random.normal(0, 2, n_samples)

tanh_output = tanh_func(z_input)
sigmoid_output = sigmoid(z_input)

plt.hist(tanh_output, bins=30, alpha=0.7, label=f'Tanh (mean={np.mean(tanh_output):.3f})', color='blue')
plt.hist(sigmoid_output, bins=30, alpha=0.7, label=f'Sigmoid (mean={np.mean(sigmoid_output):.3f})', color='red')
plt.axvline(x=0, color='black', linestyle='--', alpha=0.7)
plt.xlabel('Activation Output')
plt.ylabel('Frequency')
plt.title('Output Distribution (Zero-Centered)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 5: Gradient flow comparison
plt.subplot(3, 3, 5)
z_range = np.linspace(-4, 4, 100)
tanh_grads = tanh_derivative(z_range)
sigmoid_grads = sigmoid_derivative(z_range)

plt.fill_between(z_range, tanh_grads, alpha=0.6, color='blue', label='Tanh gradient area')
plt.fill_between(z_range, sigmoid_grads, alpha=0.6, color='red', label='Sigmoid gradient area')
plt.plot(z_range, tanh_grads, 'b-', linewidth=2, label='Tanh gradient')
plt.plot(z_range, sigmoid_grads, 'r-', linewidth=2, label='Sigmoid gradient')
plt.xlabel('Input (z)')
plt.ylabel('Gradient')
plt.title('Gradient Flow Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 6: Relationship between tanh and sigmoid
plt.subplot(3, 3, 6)
# tanh(z) = 2*sigmoid(2*z) - 1
sigmoid_2z = sigmoid(2 * z_values)
tanh_from_sigmoid = 2 * sigmoid_2z - 1

plt.plot(z_values, tanh_values, 'b-', linewidth=3, label='tanh(z)')
plt.plot(z_values, tanh_from_sigmoid, 'r--', linewidth=2, label='2*sigmoid(2z) - 1')
plt.xlabel('Input (z)')
plt.ylabel('Output')
plt.title('Relationship: tanh(z) = 2*sigmoid(2z) - 1')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 7: Saturation behavior
plt.subplot(3, 3, 7)
extreme_z = np.linspace(-10, 10, 1000)
extreme_tanh = tanh_func(extreme_z)
extreme_sigmoid = sigmoid(extreme_z)

plt.plot(extreme_z, extreme_tanh, 'b-', linewidth=2, label='tanh(z)')
plt.plot(extreme_z, extreme_sigmoid, 'r-', linewidth=2, label='sigmoid(z)')
plt.axhline(y=1, color='blue', linestyle=':', alpha=0.7, label='tanh upper bound')
plt.axhline(y=-1, color='blue', linestyle=':', alpha=0.7, label='tanh lower bound')
plt.axhline(y=1, color='red', linestyle=':', alpha=0.7, label='sigmoid upper bound')
plt.axhline(y=0, color='red', linestyle=':', alpha=0.7, label='sigmoid lower bound')
plt.xlabel('Input (z)')
plt.ylabel('Output')
plt.title('Saturation Behavior')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 8: Example application in neural network
plt.subplot(3, 3, 8)
# Simulate a simple neural network layer
np.random.seed(42)
layer_inputs = np.random.normal(0, 1, (100, 3))  # 100 samples, 3 features
weights = np.random.normal(0, 0.5, (3, 4))  # 3 inputs, 4 neurons
bias = np.random.normal(0, 0.1, 4)

# Forward pass
z = np.dot(layer_inputs, weights) + bias
tanh_activations = tanh_func(z)
sigmoid_activations = sigmoid(z)

# Plot activation distributions for each neuron
for i in range(4):
    plt.hist(tanh_activations[:, i], bins=20, alpha=0.5, 
             label=f'Tanh Neuron {i+1}' if i < 2 else '', color='blue')
    plt.hist(sigmoid_activations[:, i], bins=20, alpha=0.5, 
             label=f'Sigmoid Neuron {i+1}' if i < 2 else '', color='red')

plt.xlabel('Activation Value')
plt.ylabel('Frequency')
plt.title('Neural Network Layer Activations')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 9: Practical examples with specific values
plt.subplot(3, 3, 9)
example_inputs = [-3, -1.5, -0.5, 0, 0.5, 1.5, 3]
tanh_outputs = [tanh_func(z) for z in example_inputs]
sigmoid_outputs = [sigmoid(z) for z in example_inputs]

x_pos = np.arange(len(example_inputs))
width = 0.35

plt.bar(x_pos - width/2, tanh_outputs, width, label='tanh', color='blue', alpha=0.7)
plt.bar(x_pos + width/2, sigmoid_outputs, width, label='sigmoid', color='red', alpha=0.7)

plt.xlabel('Input Values')
plt.ylabel('Output Values')
plt.title('Specific Input Examples')
plt.xticks(x_pos, example_inputs)
plt.legend()
plt.grid(True, alpha=0.3)

# Add value labels on bars
for i, (t_val, s_val) in enumerate(zip(tanh_outputs, sigmoid_outputs)):
    plt.text(i - width/2, t_val + 0.05, f'{t_val:.2f}', ha='center', va='bottom', fontsize=8)
    plt.text(i + width/2, s_val + 0.05, f'{s_val:.2f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()

# Detailed numerical examples
print(f"\n{'='*70}")
print("DETAILED NUMERICAL EXAMPLES")
print(f"{'='*70}")

test_inputs = [-5, -2, -1, -0.5, 0, 0.5, 1, 2, 5]
print(f"{'Input (z)':<10} {'tanh(z)':<12} {'sigmoid(z)':<12} {'tanh\'(z)':<12} {'sigmoid\'(z)':<12}")
print("-" * 65)

for z in test_inputs:
    tanh_val = tanh_func(z)
    sigmoid_val = sigmoid(z)
    tanh_deriv = tanh_derivative(z)
    sigmoid_deriv = sigmoid_derivative(z)
    
    print(f"{z:<10} {tanh_val:<12.6f} {sigmoid_val:<12.6f} {tanh_deriv:<12.6f} {sigmoid_deriv:<12.6f}")

print(f"\n{'='*70}")
print("KEY DIFFERENCES AND ADVANTAGES")
print(f"{'='*70}")

print("TANH ADVANTAGES:")
print("1. Zero-centered output [-1, 1] → better gradient flow")
print("2. Stronger gradients → faster learning")
print("3. Symmetric around origin → balanced positive/negative outputs")
print("4. Less bias shift in subsequent layers")

print("\nSIGMOID ADVANTAGES:")
print("1. Output [0, 1] → natural probability interpretation")
print("2. Always positive → suitable for certain applications")
print("3. Historical significance and widespread understanding")

print("\nCOMMON DISADVANTAGES:")
print("1. Both suffer from vanishing gradient problem")
print("2. Both saturate at extreme values")
print("3. Computationally expensive (exponential operations)")

# Mathematical relationship verification
print(f"\n{'='*50}")
print("MATHEMATICAL RELATIONSHIP VERIFICATION")
print(f"{'='*50}")

z_test = 2.0
tanh_direct = tanh_func(z_test)
tanh_from_sigmoid = 2 * sigmoid(2 * z_test) - 1

print(f"For z = {z_test}:")
print(f"tanh({z_test}) = {tanh_direct:.6f}")
print(f"2*sigmoid(2*{z_test}) - 1 = {tanh_from_sigmoid:.6f}")
print(f"Difference: {abs(tanh_direct - tanh_from_sigmoid):.10f}")

# Show why tanh is better for hidden layers
print(f"\n{'='*50}")
print("WHY TANH IS OFTEN PREFERRED FOR HIDDEN LAYERS")
print(f"{'='*50}")

# Simulate gradient flow through multiple layers
def simulate_gradient_flow(activation_func, n_layers=5):
    """Simulate how gradients flow through multiple layers"""
    np.random.seed(42)
    gradient = 1.0  # Initial gradient
    gradients = [gradient]
    
    for layer in range(n_layers):
        # Random pre-activation value
        z = np.random.normal(0, 1)
        
        # Calculate derivative based on activation function
        if activation_func == 'tanh':
            local_grad = tanh_derivative(z)
        else:  # sigmoid
            local_grad = sigmoid_derivative(z)
        
        # Update gradient (chain rule)
        gradient *= local_grad
        gradients.append(gradient)
    
    return gradients

tanh_gradients = simulate_gradient_flow('tanh')
sigmoid_gradients = simulate_gradient_flow('sigmoid')

print("Gradient flow simulation through 5 layers:")
print("Layer\tTanh Gradient\tSigmoid Gradient")
for i, (t_grad, s_grad) in enumerate(zip(tanh_gradients, sigmoid_gradients)):
    print(f"{i}\t{t_grad:.6f}\t\t{s_grad:.6f}")

print(f"\nFinal gradient ratio (tanh/sigmoid): {tanh_gradients[-1]/sigmoid_gradients[-1]:.2f}")
print("Tanh maintains stronger gradients through multiple layers!")
