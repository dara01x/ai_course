# About Chapter Two - Feature Engineering

This directory contains examples and exercises from Chapter Two of the AI course, focusing on Feature Engineering techniques.

**Note: Run any Python file to see immediate educational output with clear explanations!**

## Key Concepts
- **Feature Imputation**: Handling missing values using statistical measures (mean, median, mode)
- **Feature Encoding**: Converting categorical data to numerical (Label encoding, One-hot encoding, Ordinal encoding)
- **Feature Scaling**: Normalization, Standardization, Maximum Absolute Scaling, and Transformations
- **Feature Selection**: Filter methods (Variance threshold, P-value, Correlation, Chi-square), Wrapper methods (RFE), Embedded methods
- **Feature Extraction**: Creating new meaningful features from existing ones
- **Dimensionality Reduction**: PCA for regression, LDA for classification

## Quick Start

**Prerequisites:** `pip install numpy pandas matplotlib scikit-learn`

**Run Examples:**
```bash
python examples/example_1_normalization.py           # Feature normalization
python examples/example_2_variance_threshold.py     # Remove low-variance features
python examples/example_4_correlation_analysis.py   # Feature correlation analysis
python examples/example_7_rfe_selection.py          # Recursive feature elimination
python examples/example_9_pca_reduction.py          # Principal Component Analysis
```

## Learning Goals
Master feature engineering techniques essential for improving ML model performance. Learn to preprocess data, select relevant features, and reduce dimensionality while preserving important information. Each example includes step-by-step implementations with visualizations and comparisons to scikit-learn functions.

## Chapter Structure
- **Data Preprocessing**: Imputation, scaling, and encoding techniques
- **Feature Selection**: Statistical and model-based methods for choosing relevant features
- **Feature Extraction & Reduction**: Creating new features and reducing dimensionality
- **Practical Applications**: Real-world examples using standard datasets (Iris, Diabetes, etc.)
