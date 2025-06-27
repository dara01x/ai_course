"""
Homework 4 - Repeat Confusion Matrix Example Manually
Manual calculation of confusion matrix and all related metrics
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

print("="*80)
print("HOMEWORK 4 - MANUAL CONFUSION MATRIX AND METRICS CALCULATION")
print("="*80)

# Generate a simple binary classification dataset
np.random.seed(42)
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                          n_informative=2, n_classes=2, n_clusters_per_class=1, 
                          random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train logistic regression
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Dataset Information:")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Features: {X_test.shape[1]}")

# Display actual predictions for first 20 samples
print(f"\n{'='*60}")
print("FIRST 20 TEST SAMPLES - PREDICTIONS")
print(f"{'='*60}")

print(f"{'Sample':<8} {'Actual':<8} {'Predicted':<10} {'Correct':<8}")
print("-" * 40)

for i in range(min(20, len(y_test))):
    actual = y_test[i]
    predicted = y_pred[i]
    correct = "✓" if actual == predicted else "✗"
    print(f"{i+1:<8} {actual:<8} {predicted:<10} {correct:<8}")

# Manual confusion matrix calculation
print(f"\n{'='*60}")
print("MANUAL CONFUSION MATRIX CALCULATION")
print(f"{'='*60}")

print("Step 1: Count each type of prediction")
print("Going through each test sample and categorizing...")

# Initialize counters
true_positive = 0    # Actual: 1, Predicted: 1
true_negative = 0    # Actual: 0, Predicted: 0
false_positive = 0   # Actual: 0, Predicted: 1
false_negative = 0   # Actual: 1, Predicted: 0

# Count each category manually
tp_samples = []
tn_samples = []
fp_samples = []
fn_samples = []

for i in range(len(y_test)):
    actual = y_test[i]
    predicted = y_pred[i]
    
    if actual == 1 and predicted == 1:
        true_positive += 1
        tp_samples.append(i+1)
    elif actual == 0 and predicted == 0:
        true_negative += 1
        tn_samples.append(i+1)
    elif actual == 0 and predicted == 1:
        false_positive += 1
        fp_samples.append(i+1)
    elif actual == 1 and predicted == 0:
        false_negative += 1
        fn_samples.append(i+1)

print(f"\nCounting results:")
print(f"True Positives (TP):  {true_positive} samples {tp_samples[:10]}{'...' if len(tp_samples) > 10 else ''}")
print(f"True Negatives (TN):  {true_negative} samples {tn_samples[:10]}{'...' if len(tn_samples) > 10 else ''}")
print(f"False Positives (FP): {false_positive} samples {fp_samples[:10]}{'...' if len(fp_samples) > 10 else ''}")
print(f"False Negatives (FN): {false_negative} samples {fn_samples[:10]}{'...' if len(fn_samples) > 10 else ''}")

print(f"\nTotal samples check: {true_positive + true_negative + false_positive + false_negative} = {len(y_test)}")

# Create confusion matrix manually
print(f"\n{'='*50}")
print("CONFUSION MATRIX CONSTRUCTION")
print(f"{'='*50}")

print("Confusion Matrix Layout:")
print("                 Predicted")
print("              |   0   |   1   |")
print("           ---+-------+-------+")
print("Actual  0  |  TN   |  FP   |")
print("        1  |  FN   |  TP   |")
print()

manual_cm = np.array([[true_negative, false_positive],
                      [false_negative, true_positive]])

print("Manual Confusion Matrix:")
print(f"[[{true_negative:2d}, {false_positive:2d}]")
print(f" [{false_negative:2d}, {true_positive:2d}]]")

print(f"\nInterpretation:")
print(f"• True Negatives (TN):  {true_negative} - Correctly predicted as class 0")
print(f"• False Positives (FP): {false_positive} - Incorrectly predicted as class 1 (Type I error)")
print(f"• False Negatives (FN): {false_negative} - Incorrectly predicted as class 0 (Type II error)")
print(f"• True Positives (TP):  {true_positive} - Correctly predicted as class 1")

# Verify with sklearn
sklearn_cm = confusion_matrix(y_test, y_pred)
print(f"\nVerification with sklearn:")
print(f"Sklearn confusion matrix:")
print(sklearn_cm)
print(f"Manual matches sklearn: {np.array_equal(manual_cm, sklearn_cm)}")

# Manual metric calculations
print(f"\n{'='*60}")
print("MANUAL METRICS CALCULATIONS")
print(f"{'='*60}")

print("1. ACCURACY")
print("   Formula: Accuracy = (TP + TN) / (TP + TN + FP + FN)")
print(f"   Calculation: ({true_positive} + {true_negative}) / ({true_positive} + {true_negative} + {false_positive} + {false_negative})")
print(f"   = {true_positive + true_negative} / {true_positive + true_negative + false_positive + false_negative}")

accuracy_manual = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
print(f"   = {accuracy_manual:.4f}")
print(f"   Interpretation: {accuracy_manual*100:.1f}% of predictions were correct")

print(f"\n2. PRECISION")
print("   Formula: Precision = TP / (TP + FP)")
print(f"   Calculation: {true_positive} / ({true_positive} + {false_positive})")
if (true_positive + false_positive) == 0:
    precision_manual = 0
    print(f"   = undefined (no positive predictions made)")
else:
    precision_manual = true_positive / (true_positive + false_positive)
    print(f"   = {true_positive} / {true_positive + false_positive}")
    print(f"   = {precision_manual:.4f}")
    print(f"   Interpretation: {precision_manual*100:.1f}% of positive predictions were correct")

print(f"\n3. RECALL (Sensitivity)")
print("   Formula: Recall = TP / (TP + FN)")
print(f"   Calculation: {true_positive} / ({true_positive} + {false_negative})")
if (true_positive + false_negative) == 0:
    recall_manual = 0
    print(f"   = undefined (no actual positives in dataset)")
else:
    recall_manual = true_positive / (true_positive + false_negative)
    print(f"   = {true_positive} / {true_positive + false_negative}")
    print(f"   = {recall_manual:.4f}")
    print(f"   Interpretation: {recall_manual*100:.1f}% of actual positives were found")

print(f"\n4. SPECIFICITY")
print("   Formula: Specificity = TN / (TN + FP)")
print(f"   Calculation: {true_negative} / ({true_negative} + {false_positive})")
if (true_negative + false_positive) == 0:
    specificity_manual = 0
    print(f"   = undefined (no actual negatives in dataset)")
else:
    specificity_manual = true_negative / (true_negative + false_positive)
    print(f"   = {true_negative} / {true_negative + false_positive}")
    print(f"   = {specificity_manual:.4f}")
    print(f"   Interpretation: {specificity_manual*100:.1f}% of actual negatives were correctly identified")

print(f"\n5. F1-SCORE")
print("   Formula: F1 = 2 × (Precision × Recall) / (Precision + Recall)")
if precision_manual == 0 and recall_manual == 0:
    f1_manual = 0
    print(f"   = undefined (both precision and recall are 0)")
else:
    f1_manual = 2 * (precision_manual * recall_manual) / (precision_manual + recall_manual)
    print(f"   Calculation: 2 × ({precision_manual:.4f} × {recall_manual:.4f}) / ({precision_manual:.4f} + {recall_manual:.4f})")
    print(f"   = 2 × {precision_manual * recall_manual:.4f} / {precision_manual + recall_manual:.4f}")
    print(f"   = {2 * precision_manual * recall_manual:.4f} / {precision_manual + recall_manual:.4f}")
    print(f"   = {f1_manual:.4f}")
    print(f"   Interpretation: Harmonic mean of precision and recall")

# Verification with sklearn
print(f"\n{'='*60}")
print("VERIFICATION WITH SKLEARN")
print(f"{'='*60}")

accuracy_sklearn = accuracy_score(y_test, y_pred)
precision_sklearn = precision_score(y_test, y_pred, zero_division=0)
recall_sklearn = recall_score(y_test, y_pred, zero_division=0)
f1_sklearn = f1_score(y_test, y_pred, zero_division=0)

print(f"{'Metric':<12} {'Manual':<12} {'Sklearn':<12} {'Difference':<12} {'Match'}")
print("-" * 65)
print(f"{'Accuracy':<12} {accuracy_manual:<12.6f} {accuracy_sklearn:<12.6f} {abs(accuracy_manual - accuracy_sklearn):<12.6f} {np.isclose(accuracy_manual, accuracy_sklearn)}")
print(f"{'Precision':<12} {precision_manual:<12.6f} {precision_sklearn:<12.6f} {abs(precision_manual - precision_sklearn):<12.6f} {np.isclose(precision_manual, precision_sklearn)}")
print(f"{'Recall':<12} {recall_manual:<12.6f} {recall_sklearn:<12.6f} {abs(recall_manual - recall_sklearn):<12.6f} {np.isclose(recall_manual, recall_sklearn)}")
print(f"{'F1-Score':<12} {f1_manual:<12.6f} {f1_sklearn:<12.6f} {abs(f1_manual - f1_sklearn):<12.6f} {np.isclose(f1_manual, f1_sklearn)}")
print(f"{'Specificity':<12} {specificity_manual:<12.6f} {'N/A':<12} {'N/A':<12} {'Manual calc'}")

# Step-by-step walkthrough for understanding
print(f"\n{'='*60}")
print("STEP-BY-STEP WALKTHROUGH OF FIRST 10 SAMPLES")
print(f"{'='*60}")

print(f"{'Sample':<8} {'Actual':<8} {'Pred':<8} {'Category':<12} {'TP':<4} {'TN':<4} {'FP':<4} {'FN':<4}")
print("-" * 65)

tp_count = 0
tn_count = 0
fp_count = 0
fn_count = 0

for i in range(min(10, len(y_test))):
    actual = y_test[i]
    predicted = y_pred[i]
    
    if actual == 1 and predicted == 1:
        category = "True Pos"
        tp_count += 1
        tp_marker, tn_marker, fp_marker, fn_marker = "1", "0", "0", "0"
    elif actual == 0 and predicted == 0:
        category = "True Neg"
        tn_count += 1
        tp_marker, tn_marker, fp_marker, fn_marker = "0", "1", "0", "0"
    elif actual == 0 and predicted == 1:
        category = "False Pos"
        fp_count += 1
        tp_marker, tn_marker, fp_marker, fn_marker = "0", "0", "1", "0"
    else:  # actual == 1 and predicted == 0
        category = "False Neg"
        fn_count += 1
        tp_marker, tn_marker, fp_marker, fn_marker = "0", "0", "0", "1"
    
    print(f"{i+1:<8} {actual:<8} {predicted:<8} {category:<12} {tp_marker:<4} {tn_marker:<4} {fp_marker:<4} {fn_marker:<4}")

print("-" * 65)
print(f"{'TOTALS:':<32} {tp_count:<4} {tn_count:<4} {fp_count:<4} {fn_count:<4}")

# Calculate running metrics
running_accuracy = (tp_count + tn_count) / min(10, len(y_test))
running_precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
running_recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0

print(f"\nRunning metrics for first 10 samples:")
print(f"Accuracy:  {running_accuracy:.4f}")
print(f"Precision: {running_precision:.4f}")
print(f"Recall:    {running_recall:.4f}")

# Visualization
plt.figure(figsize=(15, 10))

# Plot 1: Confusion Matrix Visualization
plt.subplot(2, 3, 1)
cm_labels = np.array([['TN\n' + str(true_negative), 'FP\n' + str(false_positive)],
                      ['FN\n' + str(false_negative), 'TP\n' + str(true_positive)]])

im = plt.imshow(manual_cm, interpolation='nearest', cmap='Blues')
plt.title('Manual Confusion Matrix')
plt.colorbar(im)

# Add text annotations
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm_labels[i, j], ha='center', va='center', 
                fontsize=12, fontweight='bold')

plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks([0, 1], ['Class 0', 'Class 1'])
plt.yticks([0, 1], ['Class 0', 'Class 1'])

# Plot 2: Metrics Comparison
plt.subplot(2, 3, 2)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
manual_values = [accuracy_manual, precision_manual, recall_manual, f1_manual]
sklearn_values = [accuracy_sklearn, precision_sklearn, recall_sklearn, f1_sklearn]

x_pos = np.arange(len(metrics))
width = 0.35

plt.bar(x_pos - width/2, manual_values, width, label='Manual', alpha=0.7, color='lightblue')
plt.bar(x_pos + width/2, sklearn_values, width, label='Sklearn', alpha=0.7, color='lightcoral')

plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Manual vs Sklearn Metrics')
plt.xticks(x_pos, metrics, rotation=45)
plt.legend()
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)

# Add value labels
for i, (manual, sklearn) in enumerate(zip(manual_values, sklearn_values)):
    plt.text(i - width/2, manual + 0.02, f'{manual:.3f}', ha='center', fontsize=9)
    plt.text(i + width/2, sklearn + 0.02, f'{sklearn:.3f}', ha='center', fontsize=9)

# Plot 3: Sample Classification Results
plt.subplot(2, 3, 3)
colors = ['red', 'blue']
markers = ['o', 's', '^', 'v']
labels = ['True Neg', 'True Pos', 'False Pos', 'False Neg']

# Plot each category
for i in range(len(y_test)):
    actual = y_test[i]
    predicted = y_pred[i]
    
    if actual == 1 and predicted == 1:  # TP
        plt.scatter(X_test[i, 0], X_test[i, 1], c='green', marker='o', s=60, alpha=0.7)
    elif actual == 0 and predicted == 0:  # TN
        plt.scatter(X_test[i, 0], X_test[i, 1], c='blue', marker='s', s=60, alpha=0.7)
    elif actual == 0 and predicted == 1:  # FP
        plt.scatter(X_test[i, 0], X_test[i, 1], c='red', marker='^', s=60, alpha=0.7)
    else:  # FN
        plt.scatter(X_test[i, 0], X_test[i, 1], c='orange', marker='v', s=60, alpha=0.7)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Classification Results')
plt.grid(True, alpha=0.3)

# Create custom legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label=f'TP ({true_positive})'),
                   Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=8, label=f'TN ({true_negative})'),
                   Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=8, label=f'FP ({false_positive})'),
                   Line2D([0], [0], marker='v', color='w', markerfacecolor='orange', markersize=8, label=f'FN ({false_negative})')]
plt.legend(handles=legend_elements)

# Plot 4: Metric Interpretation
plt.subplot(2, 3, 4)
categories = ['Correct\nPredictions', 'Incorrect\nPredictions']
values = [true_positive + true_negative, false_positive + false_negative]
colors_pie = ['lightgreen', 'lightcoral']

plt.pie(values, labels=categories, colors=colors_pie, autopct='%1.1f%%', startangle=90)
plt.title(f'Overall Accuracy: {accuracy_manual:.1%}')

# Plot 5: Error Types
plt.subplot(2, 3, 5)
error_types = ['Type I\n(False Pos)', 'Type II\n(False Neg)']
error_values = [false_positive, false_negative]
error_colors = ['red', 'orange']

if sum(error_values) > 0:
    plt.pie(error_values, labels=error_types, colors=error_colors, autopct='%1.0f', startangle=90)
    plt.title('Error Types Distribution')
else:
    plt.text(0.5, 0.5, 'No Errors!', ha='center', va='center', fontsize=16, 
             transform=plt.gca().transAxes)
    plt.title('Error Types Distribution')

# Plot 6: Performance Summary
plt.subplot(2, 3, 6)
performance_data = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score'],
    'Value': [accuracy_manual, precision_manual, recall_manual, specificity_manual, f1_manual],
    'Interpretation': [
        f'{accuracy_manual:.1%} overall correct',
        f'{precision_manual:.1%} of pos. pred. correct',
        f'{recall_manual:.1%} of actual pos. found',
        f'{specificity_manual:.1%} of actual neg. correct',
        f'{f1_manual:.3f} balance of prec./recall'
    ]
}

# Create a table-like visualization
y_positions = np.arange(len(performance_data['Metric']))
plt.barh(y_positions, performance_data['Value'], color='skyblue', alpha=0.7)

for i, (metric, value, interp) in enumerate(zip(performance_data['Metric'], 
                                               performance_data['Value'], 
                                               performance_data['Interpretation'])):
    plt.text(value + 0.02, i, f'{value:.3f}', va='center', fontweight='bold')

plt.yticks(y_positions, performance_data['Metric'])
plt.xlabel('Score')
plt.title('Performance Summary')
plt.xlim(0, 1.1)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Summary and insights
print(f"\n{'='*60}")
print("DETAILED ANALYSIS AND INSIGHTS")
print(f"{'='*60}")

print(f"Confusion Matrix Summary:")
print(f"┌─────────────────┬──────────┬──────────┐")
print(f"│                 │ Pred: 0  │ Pred: 1  │")
print(f"├─────────────────┼──────────┼──────────┤")
print(f"│ Actual: 0       │   {true_negative:2d}     │   {false_positive:2d}     │")
print(f"│ Actual: 1       │   {false_negative:2d}     │   {true_positive:2d}     │")
print(f"└─────────────────┴──────────┴──────────┘")

total_samples = len(y_test)
print(f"\nBreakdown of {total_samples} test samples:")
print(f"• Correctly classified: {true_positive + true_negative} ({(true_positive + true_negative)/total_samples*100:.1f}%)")
print(f"• Incorrectly classified: {false_positive + false_negative} ({(false_positive + false_negative)/total_samples*100:.1f}%)")

print(f"\nError Analysis:")
if false_positive > 0:
    print(f"• Type I Errors (False Positives): {false_positive}")
    print(f"  - These are cases where we predicted positive but actual was negative")
    print(f"  - Could be due to model being too liberal in positive predictions")

if false_negative > 0:
    print(f"• Type II Errors (False Negatives): {false_negative}")
    print(f"  - These are cases where we predicted negative but actual was positive")  
    print(f"  - Could be due to model being too conservative in positive predictions")

print(f"\nMetric Interpretations:")
print(f"• Accuracy ({accuracy_manual:.3f}): Model is correct {accuracy_manual*100:.1f}% of the time")
print(f"• Precision ({precision_manual:.3f}): When model predicts positive, it's right {precision_manual*100:.1f}% of the time")
print(f"• Recall ({recall_manual:.3f}): Model finds {recall_manual*100:.1f}% of all actual positive cases")
print(f"• Specificity ({specificity_manual:.3f}): Model correctly identifies {specificity_manual*100:.1f}% of actual negative cases")
print(f"• F1-Score ({f1_manual:.3f}): Balanced measure considering both precision and recall")

print(f"\n{'='*60}")
print("HOMEWORK COMPLETION SUMMARY")
print(f"{'='*60}")
print("✓ Manually counted TP, TN, FP, FN from predictions")
print("✓ Constructed confusion matrix by hand")
print("✓ Calculated accuracy using manual formula")
print("✓ Calculated precision using manual formula")
print("✓ Calculated recall using manual formula")
print("✓ Calculated F1-score using manual formula")
print("✓ Calculated specificity (bonus metric)")
print("✓ Verified all calculations against sklearn")
print("✓ Provided step-by-step walkthrough")
print("✓ Visualized results and interpretations")
print("✓ Analyzed error types and model performance")

print(f"\nKey Learning Points:")
print("1. Confusion matrix is built by counting prediction outcomes")
print("2. Each metric focuses on different aspects of performance")
print("3. Manual calculations help understand what metrics really measure")
print("4. Visualization helps identify patterns in misclassifications")
print("5. Different metrics are important for different use cases")
