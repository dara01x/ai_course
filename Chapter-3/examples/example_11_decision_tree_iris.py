"""
Example 11 - Decision Tree Classification with Iris Dataset
This example demonstrates decision tree classification on the iris dataset
"""

from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np

print("=== Decision Tree Classification - Iris Dataset ===")

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

print(f"Dataset shape: X={X.shape}, y={y.shape}")
print(f"Feature names: {iris.feature_names}")
print(f"Class names: {iris.target_names}")
print(f"Class distribution: {np.bincount(y)}")

# Split dataset
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20, random_state=42)

print(f"\nTraining set: {xtrain.shape}")
print(f"Testing set: {xtest.shape}")

# Create and train decision tree model
model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(xtrain, ytrain)

# Make predictions
yp = model.predict(xtest)

# Calculate accuracy
accuracy = accuracy_score(ytest, yp)
print(f'\nAccuracy = {accuracy:.3f} ({accuracy*100:.1f}%)')

print(f"\n=== Model Information ===")
print(f"Criterion: {model.criterion}")
print(f"Tree depth: {model.get_depth()}")
print(f"Number of leaves: {model.get_n_leaves()}")
print(f"Number of nodes: {model.tree_.node_count}")

# Feature importance
feature_importance = model.feature_importances_
print(f"\n=== Feature Importance ===")
for i, (name, importance) in enumerate(zip(iris.feature_names, feature_importance)):
    print(f"{name}: {importance:.3f}")

# Most important feature
most_important_idx = np.argmax(feature_importance)
print(f"\nMost important feature: {iris.feature_names[most_important_idx]} ({feature_importance[most_important_idx]:.3f})")

# Display tree structure in text format
print(f"\n=== Decision Tree Rules (First 20 lines) ===")
tree_rules = export_text(model, feature_names=iris.feature_names)
print('\n'.join(tree_rules.split('\n')[:20]))
print("... (truncated)")

# Prediction analysis
print(f"\n=== Prediction Analysis ===")
print("Actual\tPredicted\tClass Names")
print("-" * 40)
correct_predictions = 0
for i in range(len(ytest)):
    actual_name = iris.target_names[ytest[i]]
    predicted_name = iris.target_names[yp[i]]
    correct = "✓" if ytest[i] == yp[i] else "✗"
    print(f"{ytest[i]}\t{yp[i]}\t\t{actual_name} -> {predicted_name} {correct}")
    if ytest[i] == yp[i]:
        correct_predictions += 1

print(f"\nCorrect predictions: {correct_predictions}/{len(ytest)}")

# Class-wise accuracy
print(f"\n=== Class-wise Performance ===")
for class_idx, class_name in enumerate(iris.target_names):
    # Find indices for this class in test set
    class_mask = ytest == class_idx
    class_predictions = yp[class_mask]
    class_actual = ytest[class_mask]
    
    if len(class_actual) > 0:
        class_accuracy = np.sum(class_predictions == class_actual) / len(class_actual)
        print(f"{class_name}: {class_accuracy:.3f} ({len(class_actual)} samples)")

# Visualize the decision tree
plt.figure(figsize=(25, 20))
tree.plot_tree(model, filled=True, feature_names=iris.feature_names, 
               class_names=iris.target_names, rounded=True, 
               proportion=True, precision=2, fontsize=12)
plt.title("Decision Tree for Iris Dataset", fontsize=16)
plt.tight_layout()
plt.show()

print(f"\n=== Decision Tree Insights ===")
print("• Petal length and petal width are most discriminative features")
print("• Tree structure shows clear decision boundaries")
print("• Each path from root to leaf represents a classification rule")
print("• Decision trees are highly interpretable models")
print("• Pure nodes (gini=0.0) indicate perfect class separation")

# Show decision path for a sample
sample_idx = 0
sample = xtest[sample_idx].reshape(1, -1)
decision_path = model.decision_path(sample)
leaf_id = model.apply(sample)

print(f"\n=== Decision Path for Sample {sample_idx} ===")
print(f"Sample features: {xtest[sample_idx]}")
print(f"Actual class: {iris.target_names[ytest[sample_idx]]}")
print(f"Predicted class: {iris.target_names[yp[sample_idx]]}")
print(f"Leaf node ID: {leaf_id[0]}")
