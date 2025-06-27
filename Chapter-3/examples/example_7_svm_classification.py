"""
Example 7 - Support Vector Machine (SVM) Classification
This example demonstrates SVM classification on the breast cancer dataset
"""

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

print("=== Support Vector Machine (SVM) Classification ===")

# Load breast cancer dataset
breast = datasets.load_breast_cancer()
X = breast.data
y = breast.target

print(f"Dataset shape: X={X.shape}, y={y.shape}")
print(f"Features: {X.shape[1]} features (tumor characteristics)")
print(f"Classes: {breast.target_names} ({np.bincount(y)})")

# Standardize features (important for SVM)
scaler = StandardScaler()
X = scaler.fit_transform(X)
print(f"\nFeatures standardized (mean=0, std=1)")

# Split dataset
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set: {xtrain.shape}")
print(f"Testing set: {xtest.shape}")

# Create and train SVM model
model = SVC(kernel='linear', C=1, gamma='scale')
model.fit(xtrain, ytrain)

# Make predictions
yp = model.predict(xtest)

# Calculate accuracies
train_accuracy = model.score(xtrain, ytrain)
test_accuracy = accuracy_score(ytest, yp)

print(f"\n=== Model Performance ===")
print(f'Training Accuracy = {train_accuracy*100:.2f}%')
print(f'Test Accuracy = {test_accuracy*100:.2f}%')

print(f"\n=== SVM Model Information ===")
print(f"Kernel: {model.kernel}")
print(f"C parameter: {model.C}")
print(f"Number of support vectors: {model.n_support_}")
print(f"Total support vectors: {np.sum(model.n_support_)}")

print(f"\n=== Classification Report ===")
print(classification_report(ytest, yp, target_names=breast.target_names))

print(f"\n=== Confusion Matrix ===")
cm = confusion_matrix(ytest, yp)
print(cm)
print(f"\nTrue Negatives: {cm[0,0]}")
print(f"False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]}")
print(f"True Positives: {cm[1,1]}")

print(f"\n=== Key SVM Concepts ===")
print("• SVM finds optimal hyperplane that maximizes margin between classes")
print("• Support vectors are the critical data points closest to decision boundary")
print("• C parameter controls trade-off between margin width and classification errors")
print("• Linear kernel works well when data is linearly separable")
print("• Feature standardization is crucial for SVM performance")

print(f"\n=== Sample Predictions ===")
print("Actual\tPredicted\tClass Name")
print("-" * 35)
for i in range(min(10, len(ytest))):
    actual_name = breast.target_names[ytest[i]]
    pred_name = breast.target_names[yp[i]]
    print(f"{ytest[i]}\t{yp[i]}\t\t{actual_name} -> {pred_name}")
