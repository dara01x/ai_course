"""
Example 10 - Decision Tree Implementation with Tennis Dataset
This example implements decision tree classifier using scikit-learn on the tennis dataset
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn import tree, preprocessing
import matplotlib.pyplot as plt

print("=== Decision Tree Implementation - Tennis Dataset ===")

# Create the tennis dataset
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 
                'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 
                   'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 
                'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Windy': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 
             'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 
                  'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)
print("Tennis Dataset:")
print(df)
print(f"\nDataset shape: {df.shape}")

# Encode categorical variables to numerical
le = preprocessing.LabelEncoder()
df_encoded = df.copy()

print(f"\n=== Label Encoding ===")
for column in df.columns:
    df_encoded[column] = le.fit_transform(df[column])
    print(f"{column}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

print(f"\nEncoded dataset:")
print(df_encoded)

# Split features and target
X = df_encoded.drop('PlayTennis', axis=1)
y = df_encoded['PlayTennis']

print(f"\nFeatures (X): {list(X.columns)}")
print(f"Target (y): PlayTennis")

# Split dataset (for demonstration, we'll use a small test set)
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20, random_state=42)

print(f"\nTraining set: {xtrain.shape}")
print(f"Testing set: {xtest.shape}")

# Create and train decision tree model
model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(xtrain, ytrain)

# Make predictions
yp = model.predict(xtest)

# Calculate accuracy
accuracy = sum(yp == ytest) / len(ytest)
print(f'\nAccuracy = {accuracy:.3f} ({accuracy*100:.1f}%)')

print(f"\n=== Model Information ===")
print(f"Criterion: {model.criterion}")
print(f"Tree depth: {model.get_depth()}")
print(f"Number of leaves: {model.get_n_leaves()}")

# Show feature importance
feature_importance = model.feature_importances_
print(f"\n=== Feature Importance ===")
feature_names = list(X.columns)
for i, importance in enumerate(feature_importance):
    print(f"{feature_names[i]}: {importance:.3f}")

# Display tree structure in text format
print(f"\n=== Decision Tree Structure ===")
tree_rules = export_text(model, feature_names=feature_names)
print(tree_rules)

# Test predictions on full dataset
y_full_pred = model.predict(X)
full_accuracy = sum(y_full_pred == y) / len(y)
print(f"\nFull dataset accuracy: {full_accuracy:.3f} ({full_accuracy*100:.1f}%)")

# Show some predictions
print(f"\n=== Sample Predictions ===")
print("Features\t\t\t\tActual\tPredicted")
print("-" * 60)
for i in range(min(5, len(xtest))):
    features_str = f"{list(xtest.iloc[i].values)}"
    print(f"{features_str:<40}\t{ytest.iloc[i]}\t{yp[i]}")

# Visualize the decision tree
plt.figure(figsize=(20, 15))
tree.plot_tree(model, filled=True, feature_names=feature_names, 
               class_names=['No', 'Yes'], rounded=True, fontsize=10)
plt.title("Decision Tree for Tennis Dataset", fontsize=16)
plt.tight_layout()
plt.show()

print(f"\n=== Key Decision Tree Concepts ===")
print("• Decision tree splits data based on feature values")
print("• Each internal node represents a test on an attribute")
print("• Each leaf represents a class prediction")
print("• Entropy criterion measures information gain")
print("• Tree automatically handles categorical features after encoding")
