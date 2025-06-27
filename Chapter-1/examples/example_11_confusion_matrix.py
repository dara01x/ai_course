"""
Example 11 - Confusion Matrix for Classification Evaluation
Demonstrate confusion matrix and classification metrics calculation
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns

print("=== Confusion Matrix and Classification Metrics ===")

# Generate binary classification dataset
X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_classes=2,
    n_clusters_per_class=1,
    random_state=42
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class

print("Dataset Information:")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Positive class (1): {np.sum(y_test)} samples ({np.mean(y_test)*100:.1f}%)")
print(f"Negative class (0): {np.sum(y_test == 0)} samples ({np.mean(y_test == 0)*100:.1f}%)")

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\n{'='*50}")
print("CONFUSION MATRIX")
print(f"{'='*50}")

print("Confusion Matrix:")
print(cm)
print(f"\nMatrix breakdown:")
print(f"True Negatives (TN):  {cm[0, 0]}")
print(f"False Positives (FP): {cm[0, 1]}")
print(f"False Negatives (FN): {cm[1, 0]}")
print(f"True Positives (TP):  {cm[1, 1]}")

# Extract TP, TN, FP, FN
tn = cm[0, 0]
fp = cm[0, 1]
fn = cm[1, 0]
tp = cm[1, 1]

# Calculate metrics manually
accuracy_manual = (tp + tn) / (tp + tn + fp + fn)
precision_manual = tp / (tp + fp) if (tp + fp) > 0 else 0
recall_manual = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_manual = 2 * (precision_manual * recall_manual) / (precision_manual + recall_manual) if (precision_manual + recall_manual) > 0 else 0

print(f"\n{'='*50}")
print("MANUAL METRIC CALCULATIONS")
print(f"{'='*50}")

print(f"Accuracy  = (TP + TN) / (TP + TN + FP + FN)")
print(f"         = ({tp} + {tn}) / ({tp} + {tn} + {fp} + {fn})")
print(f"         = {tp + tn} / {tp + tn + fp + fn}")
print(f"         = {accuracy_manual:.4f}")

print(f"\nPrecision = TP / (TP + FP)")
print(f"         = {tp} / ({tp} + {fp})")
print(f"         = {tp} / {tp + fp}")
print(f"         = {precision_manual:.4f}")

print(f"\nRecall    = TP / (TP + FN)")
print(f"         = {tp} / ({tp} + {fn})")
print(f"         = {tp} / {tp + fn}")
print(f"         = {recall_manual:.4f}")

print(f"\nF1-Score  = 2 * (Precision * Recall) / (Precision + Recall)")
print(f"         = 2 * ({precision_manual:.4f} * {recall_manual:.4f}) / ({precision_manual:.4f} + {recall_manual:.4f})")
print(f"         = 2 * {precision_manual * recall_manual:.4f} / {precision_manual + recall_manual:.4f}")
print(f"         = {f1_manual:.4f}")

# Verify with sklearn
accuracy_sklearn = accuracy_score(y_test, y_pred)
precision_sklearn = precision_score(y_test, y_pred)
recall_sklearn = recall_score(y_test, y_pred)
f1_sklearn = f1_score(y_test, y_pred)

print(f"\n{'='*50}")
print("SKLEARN VERIFICATION")
print(f"{'='*50}")

print(f"{'Metric':<12} {'Manual':<12} {'Sklearn':<12} {'Match':<8}")
print("-" * 48)
print(f"{'Accuracy':<12} {accuracy_manual:<12.4f} {accuracy_sklearn:<12.4f} {np.isclose(accuracy_manual, accuracy_sklearn)}")
print(f"{'Precision':<12} {precision_manual:<12.4f} {precision_sklearn:<12.4f} {np.isclose(precision_manual, precision_sklearn)}")
print(f"{'Recall':<12} {recall_manual:<12.4f} {recall_sklearn:<12.4f} {np.isclose(recall_manual, recall_sklearn)}")
print(f"{'F1-Score':<12} {f1_manual:<12.4f} {f1_sklearn:<12.4f} {np.isclose(f1_manual, f1_sklearn)}")

# Visualization
plt.figure(figsize=(16, 12))

# Plot 1: Confusion Matrix Heatmap
plt.subplot(3, 4, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Plot 2: Normalized Confusion Matrix
plt.subplot(3, 4, 2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Normalized Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Plot 3: Metrics Comparison
plt.subplot(3, 4, 3)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [accuracy_manual, precision_manual, recall_manual, f1_manual]
colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']

bars = plt.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
plt.ylabel('Score')
plt.title('Classification Metrics')
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

# Plot 4: Classification Decision Boundary
plt.subplot(3, 4, 4)
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, levels=50, alpha=0.6, cmap='RdYlBu')
plt.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='--', linewidths=2)

# Plot test points
colors = ['red', 'blue']
for i in range(2):
    mask = y_test == i
    plt.scatter(X_test[mask, 0], X_test[mask, 1], c=colors[i], 
                label=f'Class {i}', alpha=0.7, edgecolor='black')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary')
plt.legend()

# Plot 5: Prediction Errors Analysis
plt.subplot(3, 4, 5)
# Identify different types of predictions
correct_positive = (y_test == 1) & (y_pred == 1)  # TP
correct_negative = (y_test == 0) & (y_pred == 0)  # TN
false_positive = (y_test == 0) & (y_pred == 1)    # FP
false_negative = (y_test == 1) & (y_pred == 0)    # FN

plt.scatter(X_test[correct_positive, 0], X_test[correct_positive, 1], 
           c='green', marker='o', s=50, label=f'TP ({np.sum(correct_positive)})', alpha=0.7)
plt.scatter(X_test[correct_negative, 0], X_test[correct_negative, 1], 
           c='blue', marker='s', s=50, label=f'TN ({np.sum(correct_negative)})', alpha=0.7)
plt.scatter(X_test[false_positive, 0], X_test[false_positive, 1], 
           c='red', marker='^', s=50, label=f'FP ({np.sum(false_positive)})', alpha=0.7)
plt.scatter(X_test[false_negative, 0], X_test[false_negative, 1], 
           c='orange', marker='v', s=50, label=f'FN ({np.sum(false_negative)})', alpha=0.7)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Prediction Types')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 6: Probability Distribution
plt.subplot(3, 4, 6)
# Separate probabilities by true class
prob_class_0 = y_pred_proba[y_test == 0]
prob_class_1 = y_pred_proba[y_test == 1]

plt.hist(prob_class_0, bins=20, alpha=0.7, label='True Class 0', color='red', density=True)
plt.hist(prob_class_1, bins=20, alpha=0.7, label='True Class 1', color='blue', density=True)
plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.7, label='Threshold')

plt.xlabel('Predicted Probability')
plt.ylabel('Density')
plt.title('Probability Distribution by True Class')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 7: Threshold Analysis
plt.subplot(3, 4, 7)
thresholds = np.linspace(0.1, 0.9, 50)
accuracies = []
precisions = []
recalls = []
f1_scores = []

for threshold in thresholds:
    y_pred_thresh = (y_pred_proba >= threshold).astype(int)
    
    # Avoid division by zero
    if np.sum(y_pred_thresh) == 0:  # No positive predictions
        precision = 0
    else:
        precision = precision_score(y_test, y_pred_thresh, zero_division=0)
    
    recall = recall_score(y_test, y_pred_thresh, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred_thresh)
    f1 = f1_score(y_test, y_pred_thresh, zero_division=0)
    
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

plt.plot(thresholds, accuracies, label='Accuracy', linewidth=2)
plt.plot(thresholds, precisions, label='Precision', linewidth=2)
plt.plot(thresholds, recalls, label='Recall', linewidth=2)
plt.plot(thresholds, f1_scores, label='F1-Score', linewidth=2)
plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.7, label='Default Threshold')

plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Metrics vs Threshold')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 8: ROC-like curve (simplified)
plt.subplot(3, 4, 8)
# Calculate TPR and FPR for different thresholds
tprs = []
fprs = []

for threshold in thresholds:
    y_pred_thresh = (y_pred_proba >= threshold).astype(int)
    
    # Calculate confusion matrix for this threshold
    cm_thresh = confusion_matrix(y_test, y_pred_thresh)
    if cm_thresh.shape == (2, 2):
        tn_t, fp_t, fn_t, tp_t = cm_thresh.ravel()
    else:
        # Handle edge cases where only one class is predicted
        if len(np.unique(y_pred_thresh)) == 1:
            if y_pred_thresh[0] == 0:  # All predicted as negative
                tn_t, fp_t, fn_t, tp_t = np.sum(y_test == 0), 0, np.sum(y_test == 1), 0
            else:  # All predicted as positive
                tn_t, fp_t, fn_t, tp_t = 0, np.sum(y_test == 0), 0, np.sum(y_test == 1)
        else:
            continue
    
    tpr = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0  # Sensitivity/Recall
    fpr = fp_t / (fp_t + tn_t) if (fp_t + tn_t) > 0 else 0  # 1 - Specificity
    
    tprs.append(tpr)
    fprs.append(fpr)

plt.plot(fprs, tprs, 'b-', linewidth=2, label='ROC Curve')
plt.plot([0, 1], [0, 1], 'r--', alpha=0.7, label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 9: Multi-class example
plt.subplot(3, 4, 9)
# Generate multi-class dataset
X_multi, y_multi = make_classification(
    n_samples=300, n_features=2, n_redundant=0, n_informative=2,
    n_classes=3, n_clusters_per_class=1, random_state=42
)

X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X_multi, y_multi, test_size=0.3, random_state=42, stratify=y_multi
)

model_multi = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42)
model_multi.fit(X_train_multi, y_train_multi)
y_pred_multi = model_multi.predict(X_test_multi)

cm_multi = confusion_matrix(y_test_multi, y_pred_multi)
sns.heatmap(cm_multi, annot=True, fmt='d', cmap='Blues')
plt.title('Multi-class Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Plot 10: Precision-Recall breakdown per class (multi-class)
plt.subplot(3, 4, 10)
report_dict = classification_report(y_test_multi, y_pred_multi, output_dict=True)

classes = ['Class 0', 'Class 1', 'Class 2']
precisions_multi = [report_dict['0']['precision'], report_dict['1']['precision'], report_dict['2']['precision']]
recalls_multi = [report_dict['0']['recall'], report_dict['1']['recall'], report_dict['2']['recall']]
f1s_multi = [report_dict['0']['f1-score'], report_dict['1']['f1-score'], report_dict['2']['f1-score']]

x_pos = np.arange(len(classes))
width = 0.25

plt.bar(x_pos - width, precisions_multi, width, label='Precision', alpha=0.7)
plt.bar(x_pos, recalls_multi, width, label='Recall', alpha=0.7)
plt.bar(x_pos + width, f1s_multi, width, label='F1-Score', alpha=0.7)

plt.xlabel('Classes')
plt.ylabel('Score')
plt.title('Multi-class Metrics by Class')
plt.xticks(x_pos, classes)
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 11: Classification Report Visualization
plt.subplot(3, 4, 11)
# Get classification report as dictionary
report = classification_report(y_test, y_pred, output_dict=True)

# Extract metrics for visualization
metrics_data = {
    'Class 0': [report['0']['precision'], report['0']['recall'], report['0']['f1-score']],
    'Class 1': [report['1']['precision'], report['1']['recall'], report['1']['f1-score']],
    'Macro Avg': [report['macro avg']['precision'], report['macro avg']['recall'], report['macro avg']['f1-score']],
    'Weighted Avg': [report['weighted avg']['precision'], report['weighted avg']['recall'], report['weighted avg']['f1-score']]
}

# Create heatmap
metrics_array = np.array(list(metrics_data.values()))
sns.heatmap(metrics_array, annot=True, fmt='.3f', cmap='YlOrRd',
            xticklabels=['Precision', 'Recall', 'F1-Score'],
            yticklabels=list(metrics_data.keys()))
plt.title('Classification Report Heatmap')

# Plot 12: Confusion Matrix with Percentages
plt.subplot(3, 4, 12)
cm_percent = cm.astype('float') / cm.sum() * 100
labels = np.asarray([f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)' 
                    for i in range(cm.shape[0]) 
                    for j in range(cm.shape[1])]).reshape(cm.shape)

sns.heatmap(cm, annot=labels, fmt='', cmap='Blues',
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix with Percentages')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

plt.tight_layout()
plt.show()

# Print detailed classification report
print(f"\n{'='*50}")
print("DETAILED CLASSIFICATION REPORT")
print(f"{'='*50}")
print(classification_report(y_test, y_pred))

# Manual calculation example for first few samples
print(f"\n{'='*60}")
print("STEP-BY-STEP EXAMPLE FOR UNDERSTANDING")
print(f"{'='*60}")

print("First 10 test samples:")
print(f"{'Sample':<8} {'True':<8} {'Predicted':<10} {'Probability':<12} {'Correct':<8}")
print("-" * 50)

for i in range(10):
    true_label = y_test[i]
    pred_label = y_pred[i]
    pred_prob = y_pred_proba[i]
    correct = true_label == pred_label
    
    print(f"{i+1:<8} {true_label:<8} {pred_label:<10} {pred_prob:<12.3f} {correct}")

print(f"\n{'='*50}")
print("KEY INSIGHTS")
print(f"{'='*50}")

print("Confusion Matrix Interpretation:")
print(f"• True Negatives (TN): {tn} - Correctly predicted negative cases")
print(f"• False Positives (FP): {fp} - Incorrectly predicted as positive (Type I error)")
print(f"• False Negatives (FN): {fn} - Incorrectly predicted as negative (Type II error)")
print(f"• True Positives (TP): {tp} - Correctly predicted positive cases")

print(f"\nMetric Interpretations:")
print(f"• Accuracy ({accuracy_manual:.3f}): Overall correctness - good for balanced datasets")
print(f"• Precision ({precision_manual:.3f}): Of positive predictions, how many were correct?")
print(f"• Recall ({recall_manual:.3f}): Of actual positives, how many were found?")
print(f"• F1-Score ({f1_manual:.3f}): Harmonic mean of precision and recall")

specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
print(f"• Specificity ({specificity:.3f}): Of actual negatives, how many were correctly identified?")

print(f"\nWhen to prioritize each metric:")
print(f"• High Precision: When false positives are costly (e.g., spam detection)")
print(f"• High Recall: When false negatives are costly (e.g., cancer detection)")
print(f"• High F1-Score: When you need balance between precision and recall")
print(f"• High Accuracy: When classes are balanced and all errors are equally important")
