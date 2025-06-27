"""
Example 6 - Logistic Regression with Make_Classification Dataset
This example demonstrates logistic regression on a larger synthetic dataset
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification
import numpy as np

print("=== Logistic Regression with Make_Classification Dataset ===")

# Create synthetic binary classification dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, 
                          n_classes=2, random_state=42)

print(f"Dataset shape: X={X.shape}, y={y.shape}")
print(f"Number of features: {X.shape[1]}")
print(f"Number of informative features: 5")
print(f"Classes distribution: {np.bincount(y)}")

# Split dataset into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {xtrain.shape}")
print(f"Testing set size: {xtest.shape}")

# Create and train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(xtrain, ytrain)

# Make predictions
yp = model.predict(xtest)
yp_proba = model.predict_proba(xtest)

# Calculate accuracy
accuracy = accuracy_score(ytest, yp)
print(f'\nTest Accuracy = {accuracy:.4f} ({accuracy*100:.1f}%)')

# Training accuracy
train_pred = model.predict(xtrain)
train_accuracy = accuracy_score(ytrain, train_pred)
print(f'Training Accuracy = {train_accuracy:.4f} ({train_accuracy*100:.1f}%)')

print(f"\n=== Model Parameters ===")
print(f"Intercept: {model.intercept_[0]:.4f}")
print(f"\nFeature weights:")
for i, coef in enumerate(model.coef_[0]):
    print(f"Feature {i+1}: {coef:.4f}")

print(f"\n=== Classification Report ===")
print(classification_report(ytest, yp))

print(f"\n=== Sample Predictions ===")
print("Actual\tPredicted\tProb(Class 0)\tProb(Class 1)")
print("-" * 50)
for i in range(min(10, len(ytest))):
    print(f"{ytest[i]}\t{yp[i]}\t\t{yp_proba[i][0]:.3f}\t\t{yp_proba[i][1]:.3f}")

print(f"\n=== Model Performance Analysis ===")
correct_predictions = np.sum(ytest == yp)
total_predictions = len(ytest)
print(f"Correct predictions: {correct_predictions}/{total_predictions}")
print(f"Misclassified samples: {total_predictions - correct_predictions}")

# Confidence analysis
high_confidence = np.max(yp_proba, axis=1) > 0.8
print(f"High confidence predictions (>80%): {np.sum(high_confidence)}/{len(yp_proba)}")
print(f"Low confidence predictions (<80%): {np.sum(~high_confidence)}/{len(yp_proba)}")
