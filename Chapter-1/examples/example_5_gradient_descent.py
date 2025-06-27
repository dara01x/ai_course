"""
Example 5 - Gradient Descent
Minimize f(x) = x³ - 3x² - x using gradient descent with α = 0.05
"""

import numpy as np
import matplotlib.pyplot as plt

def f(x):
    """Loss function: f(x) = x³ - 3x² - x"""
    return x**3 - 3*x**2 - x

def df_dx(x):
    """Derivative of f(x): f'(x) = 3x² - 6x - 1"""
    return 3*x**2 - 6*x - 1

def gradient_descent(start_x, learning_rate, max_iterations=100, tolerance=1e-6):
    """
    Perform gradient descent to minimize f(x)
    """
    x = start_x
    history = {'x': [x], 'f_x': [f(x)]}
    
    print(f"Starting gradient descent with:")
    print(f"Initial x: {x}")
    print(f"Learning rate (α): {learning_rate}")
    print(f"Initial f(x): {f(x):.6f}")
    print(f"Initial f'(x): {df_dx(x):.6f}")
    print("-" * 50)
    
    for i in range(max_iterations):
        # Calculate gradient
        gradient = df_dx(x)
        
        # Update x using gradient descent rule: x_new = x_old - α * f'(x)
        x_new = x - learning_rate * gradient
        
        # Calculate function values
        f_x = f(x)
        f_x_new = f(x_new)
        
        print(f"Iteration {i+1:2d}: x = {x:8.6f}, f(x) = {f_x:8.6f}, f'(x) = {gradient:8.6f}")
        
        # Store history
        history['x'].append(x_new)
        history['f_x'].append(f_x_new)
        
        # Check for convergence
        if abs(x_new - x) < tolerance:
            print(f"Converged after {i+1} iterations!")
            break
            
        x = x_new
    
    print("-" * 50)
    print(f"Final x: {x:.6f}")
    print(f"Final f(x): {f(x):.6f}")
    print(f"Final f'(x): {df_dx(x):.6f}")
    
    return x, history

# Run gradient descent with different starting points
starting_points = [2.0, -1.0, 4.0]
learning_rate = 0.05

plt.figure(figsize=(15, 10))

# Plot the function
x_range = np.linspace(-2, 5, 1000)
y_range = f(x_range)

for idx, start_x in enumerate(starting_points):
    print(f"\n{'='*60}")
    print(f"GRADIENT DESCENT STARTING FROM x = {start_x}")
    print(f"{'='*60}")
    
    final_x, history = gradient_descent(start_x, learning_rate)
    
    # Plot results
    plt.subplot(2, 2, idx + 1)
    plt.plot(x_range, y_range, 'b-', linewidth=2, label='f(x) = x³ - 3x² - x')
    
    # Plot gradient descent path
    x_history = np.array(history['x'])
    y_history = np.array(history['f_x'])
    
    plt.plot(x_history, y_history, 'ro-', markersize=6, linewidth=2, 
             label=f'GD path (start: {start_x})')
    plt.plot(start_x, f(start_x), 'go', markersize=10, label='Start')
    plt.plot(final_x, f(final_x), 'rs', markersize=10, label='End')
    
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'Gradient Descent from x₀ = {start_x}')
    plt.legend()
    plt.grid(True, alpha=0.3)

# Plot all paths together
plt.subplot(2, 2, 4)
plt.plot(x_range, y_range, 'b-', linewidth=3, label='f(x) = x³ - 3x² - x')

colors = ['red', 'green', 'orange']
for idx, start_x in enumerate(starting_points):
    _, history = gradient_descent(start_x, learning_rate)
    x_history = np.array(history['x'])
    y_history = np.array(history['f_x'])
    
    plt.plot(x_history, y_history, f'{colors[idx][0]}o-', markersize=4, 
             linewidth=1.5, alpha=0.7, label=f'Start: {start_x}')

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('All Gradient Descent Paths')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Analyze the function to find actual minima
print(f"\n{'='*60}")
print("ANALYTICAL ANALYSIS")
print(f"{'='*60}")
print("To find critical points, solve f'(x) = 3x² - 6x - 1 = 0")
print("Using quadratic formula: x = (6 ± √(36 + 12)) / 6 = (6 ± √48) / 6")

discriminant = 36 + 12
x1 = (6 + np.sqrt(discriminant)) / 6
x2 = (6 - np.sqrt(discriminant)) / 6

print(f"Critical points: x₁ = {x1:.6f}, x₂ = {x2:.6f}")
print(f"f(x₁) = {f(x1):.6f}, f(x₂) = {f(x2):.6f}")

# Second derivative test
def d2f_dx2(x):
    return 6*x - 6

print(f"f''(x₁) = {d2f_dx2(x1):.6f} ({'minimum' if d2f_dx2(x1) > 0 else 'maximum'})")
print(f"f''(x₂) = {d2f_dx2(x2):.6f} ({'minimum' if d2f_dx2(x2) > 0 else 'maximum'})")
