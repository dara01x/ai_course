"""
Example 12 - Random Forest Classifier for Iris Dataset
This example demonstrates ensemble learning using Random Forest on the iris dataset
"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

print("=== Random Forest Classifier - Iris Dataset ===")

# Load iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

print(f"Dataset shape: X={X.shape}, y={y.shape}")
print(f"Feature names: {iris.feature_names}")
print(f"Class names: {iris.target_names}")
print(f"Class distribution: {np.bincount(y)}")

# Split dataset
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set: {xtrain.shape}")
print(f"Testing set: {xtest.shape}")

# Create and train Random Forest model
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(xtrain, ytrain)

# Make predictions
yp = model.predict(xtest)

print(f"\n=== Model Configuration ===")
print(f"Number of trees (n_estimators): {model.n_estimators}")
print(f"Criterion: {model.criterion}")
print(f"Max depth: {model.max_depth}")
print(f"Random state: {model.random_state}")

# Calculate accuracy
accuracy = accuracy_score(ytest, yp)
print(f'\nAccuracy = {accuracy:.3f} ({accuracy*100:.1f}%)')

# Feature importance
feature_importance = model.feature_importances_
print(f'\n=== Feature Importance ===')
for i, (name, importance) in enumerate(zip(iris.feature_names, feature_importance)):
    print(f"{name}: {importance:.3f}")

# Sort features by importance
importance_order = np.argsort(feature_importance)[::-1]
print(f"\nFeatures ranked by importance:")
for i, idx in enumerate(importance_order):
    print(f"{i+1}. {iris.feature_names[idx]}: {feature_importance[idx]:.3f}")

print(f'\n=== Classification Report ===')
print(classification_report(ytest, yp, target_names=iris.target_names))

print(f'\n=== Confusion Matrix ===')
cm = confusion_matrix(ytest, yp)
print(cm)

# Individual tree analysis
print(f'\n=== Individual Tree Analysis ===')
print("Tree\tDepth\tLeaves\tAccuracy")
print("-" * 35)
individual_accuracies = []
for i, tree in enumerate(model.estimators_):
    tree_pred = tree.predict(xtest)
    tree_accuracy = accuracy_score(ytest, tree_pred)
    individual_accuracies.append(tree_accuracy)
    print(f"{i+1}\t{tree.get_depth()}\t{tree.get_n_leaves()}\t{tree_accuracy:.3f}")

print(f"\nIndividual tree accuracy stats:")
print(f"Mean: {np.mean(individual_accuracies):.3f}")
print(f"Std: {np.std(individual_accuracies):.3f}")
print(f"Min: {np.min(individual_accuracies):.3f}")
print(f"Max: {np.max(individual_accuracies):.3f}")

# Probability predictions
yp_proba = model.predict_proba(xtest)
print(f'\n=== Probability Predictions (First 10 samples) ===')
print("Sample\tActual\tPredicted\tSetosa\t\tVersicolor\tVirginica")
print("-" * 70)
for i in range(min(10, len(ytest))):
    actual_name = iris.target_names[ytest[i]]
    pred_name = iris.target_names[yp[i]]
    print(f"{i+1}\t{actual_name[:4]}\t{pred_name[:4]}\t\t{yp_proba[i][0]:.3f}\t\t{yp_proba[i][1]:.3f}\t\t{yp_proba[i][2]:.3f}")

# Voting analysis
print(f'\n=== Ensemble Voting Analysis ===')
# Get predictions from all trees
all_tree_predictions = np.array([tree.predict(xtest) for tree in model.estimators_])

print("Sample\tActual\tPredicted\tVoting Pattern")
print("-" * 50)
for i in range(min(5, len(ytest))):
    votes = all_tree_predictions[:, i]
    vote_counts = np.bincount(votes, minlength=3)
    actual_name = iris.target_names[ytest[i]]
    pred_name = iris.target_names[yp[i]]
    print(f"{i+1}\t{actual_name[:4]}\t{pred_name[:4]}\t\t{vote_counts}")

# Feature importance visualization
plt.figure(figsize=(10, 6))
indices = np.argsort(feature_importance)[::-1]
plt.bar(range(len(feature_importance)), feature_importance[indices])
plt.title('Feature Importance in Random Forest')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(range(len(feature_importance)), [iris.feature_names[i] for i in indices], rotation=45)
plt.tight_layout()
plt.show()

print(f'\n=== Random Forest Key Concepts ===')
print("• Random Forest combines multiple decision trees")
print("• Each tree is trained on different subset of data (bootstrap sampling)")
print("• Final prediction is majority vote of all trees")
print("• Reduces overfitting compared to single decision tree")
print("• Feature importance is averaged across all trees")
print("• Generally more robust and accurate than individual trees")

print(f'\n=== Ensemble vs Individual Performance ===')
print(f"Random Forest accuracy: {accuracy:.3f}")
print(f"Average individual tree accuracy: {np.mean(individual_accuracies):.3f}")
print(f"Improvement: {accuracy - np.mean(individual_accuracies):.3f}")
