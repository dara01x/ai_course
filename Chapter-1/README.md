# Chapter 1 - Introduction to Machine Learning

This directory contains comprehensive implementations of all examples and homework assignments from Chapter 1 of the AI course.


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

Each file is extensively commented and includes both theoretical explanations and practical implementations, making it suitable for both learning and reference.