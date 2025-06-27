# Chapter 1 - Introduction to Machine Learning

This directory contains comprehensive implementations of all examples and homework assignments from Chapter 1 of the AI course.

## Directory Structure

```
Chapter-1/
├── examples/           # All example implementations
├── homework/          # All homework solutions
├── C1_ML-introduction.pdf  # Original course material
└── README.md         # This file
```

## Examples

### Example 1: Reading Dataset (`example_1_read_dataset.py`)
- **Topic**: Loading and exploring sklearn datasets
- **Demonstrates**: Basic dataset loading, exploring data shapes, feature names
- **Key Learning**: Understanding dataset structure and properties

### Example 2: Create Regression Dataset (`example_2_create_regression_dataset.py`)
- **Topic**: Creating synthetic regression datasets
- **Demonstrates**: Using `make_regression()`, data visualization, saving datasets
- **Key Learning**: How to generate controlled datasets for testing

### Example 3: Training and Testing Set Split (`example_3_train_test_split.py`)
- **Topic**: Data splitting for machine learning
- **Demonstrates**: `train_test_split()`, stratified splitting, class distribution
- **Key Learning**: Importance of proper data splitting

### Example 4: Dataset Splitting with Visualization (`example_4_dataset_split_visualization.py`)
- **Topic**: Advanced data splitting with visual analysis
- **Demonstrates**: Visualization of split datasets, class distribution analysis
- **Key Learning**: Visual verification of data splits

### Example 5: Gradient Descent (`example_5_gradient_descent.py`)
- **Topic**: Optimization algorithm implementation
- **Demonstrates**: Manual gradient descent, convergence analysis, multiple starting points
- **Chapter Problem**: Minimize f(x) = x³ - 3x² - x with α = 0.05
- **Key Learning**: How gradient descent finds minima

### Example 6: Binary Cross-Entropy (`example_6_binary_cross_entropy.py`)
- **Topic**: Loss function for binary classification
- **Demonstrates**: Manual BCE calculation, behavior analysis, confidence interpretation
- **Chapter Problem**: Calculate BCE for y=[1,0,1], p=[0.9,0.1,0.8]
- **Key Learning**: Understanding loss function behavior

### Example 7: Sigmoid Activation Function (`example_7_sigmoid_activation.py`)
- **Topic**: Activation function for binary classification
- **Demonstrates**: Sigmoid properties, decision boundaries, gradient analysis
- **Chapter Problem**: Sigmoid output for z=0.5, binary classification decision
- **Key Learning**: How activation functions work in neural networks

### Example 8: Tanh Activation Function (`example_8_tanh_activation.py`)
- **Topic**: Hyperbolic tangent activation function
- **Demonstrates**: Tanh vs sigmoid comparison, zero-centered outputs, gradient flow
- **Key Learning**: Advantages of tanh over sigmoid

### Example 9: Softmax Activation Function (`example_9_softmax_activation.py`)
- **Topic**: Multi-class classification activation
- **Demonstrates**: Softmax calculation, temperature scaling, probability interpretation
- **Chapter Problem**: Softmax for z=[5,1,0,2], probability distribution
- **Key Learning**: Converting logits to probabilities

### Example 10: Regression Metrics (`example_10_regression_metrics.py`)
- **Topic**: Evaluation metrics for regression
- **Demonstrates**: MSE, MAE, RMSE, Max Error calculations and interpretations
- **Key Learning**: How to evaluate regression model performance

### Example 11: Confusion Matrix (`example_11_confusion_matrix.py`)
- **Topic**: Classification evaluation metrics
- **Demonstrates**: Confusion matrix, accuracy, precision, recall, F1-score
- **Key Learning**: Comprehensive classification evaluation

## Homework Solutions

### Homework 1: Load Other Datasets (`homework_1_load_datasets.py`)
- **Task**: Modify example code to load different datasets and display sizes
- **Solution**: Comprehensive analysis of multiple sklearn datasets
- **Features**: 
  - Loads 7+ different datasets
  - Compares dataset characteristics
  - Provides detailed feature analysis
  - Categorizes by problem type (classification/regression)

### Homework 2: Manual Metrics Calculation (`homework_2_manual_metrics.py`)
- **Task**: Compute regression metrics by hand for the previous example
- **Solution**: Step-by-step manual calculation with verification
- **Features**:
  - Manual MSE, MAE, RMSE calculations
  - Detailed mathematical breakdowns
  - Verification against sklearn
  - Error analysis and interpretation

### Homework 3: R² Score (`homework_3_r2_score.py`)
- **Task**: What is R² score?
- **Solution**: Comprehensive explanation and analysis
- **Features**:
  - Mathematical derivation and formula
  - Manual calculation examples
  - Visual interpretation
  - Comparison with other metrics
  - Discussion of limitations and alternatives

### Homework 4: Manual Confusion Matrix (`homework_4_manual_confusion_matrix.py`)
- **Task**: Repeat confusion matrix example manually
- **Solution**: Complete manual implementation with detailed analysis
- **Features**:
  - Manual counting of TP, TN, FP, FN
  - Step-by-step metric calculations
  - Comprehensive visualizations
  - Error type analysis
  - Performance interpretation

## Key Concepts Covered

### Machine Learning Fundamentals
- Types of ML: Supervised, Unsupervised, Semi-supervised, Reinforcement
- Dataset handling and preprocessing
- Training/validation/test splits

### Mathematical Foundations
- Gradient descent optimization
- Loss functions (MSE, MAE, Binary Cross-Entropy)
- Activation functions (Sigmoid, Tanh, Softmax)
- Evaluation metrics

### Model Evaluation
- Regression metrics: MSE, MAE, RMSE, R²
- Classification metrics: Accuracy, Precision, Recall, F1-score
- Confusion matrix interpretation
- Overfitting and underfitting concepts

## Running the Code

### Prerequisites
```bash
pip install numpy matplotlib scikit-learn seaborn scipy
```

### Running Examples
```bash
python examples/example_1_read_dataset.py
python examples/example_5_gradient_descent.py
# ... etc
```

### Running Homework
```bash
python homework/homework_1_load_datasets.py
python homework/homework_2_manual_metrics.py
# ... etc
```

## File Dependencies

All files are self-contained but require the following libraries:
- `numpy`: Numerical computations
- `matplotlib`: Plotting and visualization
- `scikit-learn`: Machine learning algorithms and datasets
- `seaborn`: Enhanced visualizations
- `scipy`: Scientific computing (for some advanced examples)

## Learning Objectives

By working through these examples and homework, you will:

1. **Understand ML Fundamentals**: Learn the basic types and workflow of machine learning
2. **Master Data Handling**: Know how to load, split, and prepare datasets
3. **Implement Core Algorithms**: Build gradient descent and understand optimization
4. **Comprehend Loss Functions**: Calculate and interpret different loss functions
5. **Use Activation Functions**: Understand how different activations work
6. **Evaluate Models**: Calculate and interpret evaluation metrics manually
7. **Analyze Performance**: Use confusion matrices and various metrics effectively

## Tips for Learning

1. **Run Code Step by Step**: Don't just read - execute and experiment
2. **Modify Parameters**: Change values and see how results change
3. **Visualize Everything**: Use the provided plots to understand concepts
4. **Verify Manually**: The manual calculations help build intuition
5. **Compare Methods**: See how different approaches give same results

## Chapter Summary

This chapter provides a comprehensive foundation in machine learning, covering:
- **Theory**: Core concepts and mathematical foundations
- **Practice**: Hands-on implementation of key algorithms
- **Evaluation**: Proper metrics and analysis techniques
- **Visualization**: Clear plots and explanations

Each file is extensively commented and includes both theoretical explanations and practical implementations, making it suitable for both learning and reference.
